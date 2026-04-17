"""Tests for the ablation harness."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from soma.core.config import SOMAConfig
from soma.developmental.ablation import AblationHarness


def _make_config() -> SOMAConfig:
    return SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        seed=42,
    )


class TestAblationHarness:
    def test_init(self) -> None:
        harness = AblationHarness(config=_make_config(), llm_model="test-model")
        assert harness.loop is not None

    @patch("soma.developmental.ablation.requests.post")
    def test_run_ablation(self, mock_post: MagicMock) -> None:
        # Mock alternating responses
        call_count = [0]

        def side_effect(*a: object, **kw: object) -> MagicMock:
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] % 2 == 1:
                resp.json.return_value = {
                    "message": {"content": "With SOMA response"}
                }
            else:
                resp.json.return_value = {
                    "message": {"content": "Without SOMA response"}
                }
            resp.raise_for_status = MagicMock()
            return resp

        mock_post.side_effect = side_effect

        harness = AblationHarness(config=_make_config(), llm_model="test-model")
        harness.loop.train_tokenizer(["hello", "world"])
        results = harness.run(
            inputs=["hello", "world"], references=["greeting", "planet"]
        )
        assert "with_soma" in results
        assert "without_soma" in results
        assert "delta" in results
        assert len(results["with_soma"]) == 2

    def test_empty_state_text(self) -> None:
        harness = AblationHarness(config=_make_config(), llm_model="test-model")
        empty = harness.empty_state_text()
        assert isinstance(empty, str)
        assert "blank" in empty.lower() or "0" in empty
