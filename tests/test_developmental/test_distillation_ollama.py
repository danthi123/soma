"""Integration smoke test for Direction 4a with real Ollama teacher.

Gated on Ollama reachability — skips in CI / offline environments.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.llm.embedders import CachedEmbedder, OllamaEmbedder


def _ollama_alive(url: str = "http://localhost:11434") -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1.0) as r:
            return 200 <= r.status < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


@pytest.mark.skipif(
    not _ollama_alive(),
    reason="Ollama server not reachable at localhost:11434",
)
class TestOllamaDistillEndToEnd:
    def test_three_steps_no_crash_projections_finite(self, tmp_path) -> None:
        """Run three prediction steps with a real Ollama teacher; the
        projections should stay finite and the distillation loss should
        decrease OR at least not explode.
        """
        cfg = SOMAConfig.developmental(
            initial_associator_count=4,
            max_nodes=16,
            projection_mode="learnable",
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=0.5,
        )
        pred = PredictiveSOMA(config=cfg)
        teacher = CachedEmbedder(
            teacher=OllamaEmbedder(model="mxbai-embed-large"),
            cache_dir=str(tmp_path),
        )
        pred.attach_teacher(teacher)

        torch.manual_seed(0)
        x = torch.randn(cfg.sensor_output_dim)
        texts = [
            "the cat sat on the mat",
            "a dog chased a squirrel",
            "the cat sat on the mat",  # repeat: should be cache hit
        ]
        for text in texts:
            result = pred.process_input(x, source_text=text)
            assert torch.isfinite(torch.tensor(result["prediction_error"]))

        # Projections should still be finite
        for p in pred._input_projections.values():
            assert torch.isfinite(p).all()

        # Cache should have exactly 2 entries (3rd was a repeat)
        subdirs = list(tmp_path.iterdir())
        assert subdirs, "expected at least one teacher subdir in cache"
        files = list(subdirs[0].glob("*.pt"))
        assert len(files) == 2
