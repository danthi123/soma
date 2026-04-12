"""Tests for ``soma.io.multimodal_curriculum``."""

from __future__ import annotations

import pytest
import torch

from soma.io.multimodal_curriculum import CurriculumWindow, MultimodalCurriculum


class TestCurriculumWindow:
    def test_contains(self) -> None:
        w = CurriculumWindow(0, 100, {"text": 1.0})
        assert w.contains(0)
        assert w.contains(50)
        assert not w.contains(100)  # exclusive end
        assert not w.contains(-1)

    def test_open_ended(self) -> None:
        w = CurriculumWindow(100, None, {"text": 1.0})
        assert w.contains(100)
        assert w.contains(10**9)
        assert not w.contains(99)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"start_step": -1, "end_step": 10, "weights": {"text": 1.0}}, "start_step"),
            ({"start_step": 10, "end_step": 10, "weights": {"text": 1.0}}, "end_step"),
            ({"start_step": 0, "end_step": 10, "weights": {}}, "at least one"),
            ({"start_step": 0, "end_step": 10, "weights": {"": 1.0}}, "non-empty"),
            ({"start_step": 0, "end_step": 10, "weights": {"text": -0.1}}, "non-negative"),
            ({"start_step": 0, "end_step": 10, "weights": {"text": 0.0}}, "Total weights"),
        ],
    )
    def test_validation(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            CurriculumWindow(**kwargs)  # type: ignore[arg-type]


class TestMultimodalCurriculum:
    def test_default_text_image(self) -> None:
        curr = MultimodalCurriculum.default_text_image()
        assert curr.current_weights(0) == {"text": 0.8, "image": 0.2}
        assert curr.current_weights(20_000) == {"text": 0.5, "image": 0.5}
        assert curr.current_weights(100_000) == {
            "text": 0.4,
            "image": 0.3,
            "interleaved": 0.3,
        }

    def test_requires_window_starting_at_zero(self) -> None:
        with pytest.raises(ValueError, match="step 0"):
            MultimodalCurriculum([CurriculumWindow(5, 10, {"text": 1.0})])

    def test_requires_contiguous_schedule(self) -> None:
        with pytest.raises(ValueError, match="gap"):
            MultimodalCurriculum(
                [
                    CurriculumWindow(0, 10, {"text": 1.0}),
                    CurriculumWindow(20, 30, {"text": 1.0}),  # gap at 10..20
                ]
            )

    def test_open_ended_must_be_last(self) -> None:
        with pytest.raises(ValueError, match="final entry"):
            MultimodalCurriculum(
                [
                    CurriculumWindow(0, None, {"text": 1.0}),
                    CurriculumWindow(100, 200, {"text": 1.0}),
                ]
            )

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            MultimodalCurriculum([])

    def test_rejects_negative_step(self) -> None:
        curr = MultimodalCurriculum.default_text_image()
        with pytest.raises(ValueError, match="step"):
            curr.current_weights(-1)

    def test_out_of_range_step_raises(self) -> None:
        curr = MultimodalCurriculum([CurriculumWindow(0, 10, {"text": 1.0})])
        with pytest.raises(ValueError, match="schedule"):
            curr.current_weights(100)

    def test_sample_modality_respects_weights(self) -> None:
        curr = MultimodalCurriculum([CurriculumWindow(0, None, {"text": 1.0, "image": 0.0})])
        rng = torch.Generator().manual_seed(0)
        samples = [curr.sample_modality(0, rng=rng) for _ in range(50)]
        # image weight is 0 -> every sample must be text.
        assert all(s == "text" for s in samples)

    def test_current_window(self) -> None:
        curr = MultimodalCurriculum.default_text_image()
        window = curr.current_window(25_000)
        assert window.start_step == 10_000
        assert window.end_step == 50_000
