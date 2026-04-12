"""Tests for ``soma.metacognition.curiosity.CuriosityModule``."""

from __future__ import annotations

import pytest
import torch

from soma.metacognition.curiosity import CuriosityModule


class TestConstruction:
    def test_defaults(self) -> None:
        cm = CuriosityModule(input_dim=16, num_domains=4)
        assert cm.input_dim == 16
        assert cm.num_domains == 4
        assert len(cm.error_histories) == 4
        for h in cm.error_histories:
            assert h.is_empty

    @pytest.mark.parametrize(
        "input_dim, num_domains, window, warmup, match",
        [
            (0, 4, 100, 10, "input_dim"),
            (16, 0, 100, 10, "num_domains"),
            (16, 4, 0, 10, "window_size"),
            (16, 4, 100, 0, "warmup"),
        ],
    )
    def test_validation(
        self, input_dim: int, num_domains: int, window: int, warmup: int, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            CuriosityModule(
                input_dim=input_dim,
                num_domains=num_domains,
                window_size=window,
                warmup=warmup,
            )


class TestClassification:
    def test_classify_returns_valid_index(self) -> None:
        cm = CuriosityModule(input_dim=8, num_domains=4)
        domain = cm.classify_domain(torch.randn(8))
        assert 0 <= domain < 4

    def test_rejects_shape_mismatch(self) -> None:
        cm = CuriosityModule(input_dim=8, num_domains=4)
        with pytest.raises(ValueError, match="shape"):
            cm.classify_domain(torch.randn(5))


class TestCuriosity:
    def test_returns_default_during_warmup(self) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=2, window_size=100, warmup=10)
        # Force a single domain so we accumulate in one bucket.
        with torch.no_grad():
            cm.domain_classifier.weight.zero_()
            cm.domain_classifier.bias.zero_()
        # Below warmup — should return default 1.0.
        for _ in range(5):
            val = cm.compute_curiosity(torch.randn(4), prediction_error=0.5)
            assert val == pytest.approx(1.0)

    def test_high_progress_yields_positive_curiosity(self) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=1, window_size=100, warmup=10)
        with torch.no_grad():
            cm.domain_classifier.weight.zero_()
            cm.domain_classifier.bias.zero_()
        # Warmup phase with high errors.
        for _ in range(15):
            cm.compute_curiosity(torch.zeros(4), prediction_error=2.0)
        # Now post-warmup: submit a rapidly-dropping error trace.
        for _ in range(20):
            cm.compute_curiosity(torch.zeros(4), prediction_error=0.1)
        score = cm.compute_curiosity(torch.zeros(4), prediction_error=0.05)
        assert score > 0.0

    def test_zero_progress_yields_zero_curiosity(self) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=1, window_size=50, warmup=5)
        with torch.no_grad():
            cm.domain_classifier.weight.zero_()
            cm.domain_classifier.bias.zero_()
        # Error stays flat at 1.0 forever -> no learning progress.
        for _ in range(40):
            cm.compute_curiosity(torch.zeros(4), prediction_error=1.0)
        score = cm.compute_curiosity(torch.zeros(4), prediction_error=1.0)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_rejects_negative_error(self) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=2)
        with pytest.raises(ValueError, match="prediction_error"):
            cm.compute_curiosity(torch.zeros(4), prediction_error=-1.0)


class TestIntrospection:
    def test_domain_mean_error(self) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=2)
        with torch.no_grad():
            cm.domain_classifier.weight.zero_()
            cm.domain_classifier.bias.zero_()
        cm.compute_curiosity(torch.zeros(4), prediction_error=0.5)
        cm.compute_curiosity(torch.zeros(4), prediction_error=1.5)
        assert cm.domain_mean_error(0) == pytest.approx(1.0)
        assert cm.domain_mean_error(1) == pytest.approx(0.0)

    def test_domain_out_of_range(self) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=2)
        with pytest.raises(IndexError):
            cm.domain_mean_error(5)

    def test_reset_histories(self) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=2)
        with torch.no_grad():
            cm.domain_classifier.weight.zero_()
            cm.domain_classifier.bias.zero_()
        cm.compute_curiosity(torch.zeros(4), prediction_error=1.0)
        cm.reset_histories()
        assert cm.error_histories[0].is_empty


class TestSerialization:
    def test_history_round_trip(self) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=2, window_size=20)
        with torch.no_grad():
            cm.domain_classifier.weight.zero_()
            cm.domain_classifier.bias.zero_()
        for err in [0.1, 0.2, 0.3]:
            cm.compute_curiosity(torch.zeros(4), prediction_error=err)
        snapshot = cm.to_dict()

        cm2 = CuriosityModule(input_dim=4, num_domains=2, window_size=20)
        cm2.load_histories(snapshot)
        assert cm2.error_histories[0].get_all() == [0.1, 0.2, 0.3]
        assert cm2.error_histories[1].is_empty
