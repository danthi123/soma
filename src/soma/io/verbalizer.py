"""SOMA → frozen transformer projection contract.

The SomaVerbalizer is the stable interface between SOMA's cognition
(variable over time via structural plasticity) and a swappable small
transformer's verbalization (fixed weights, replaceable). A projector
tied to a specific VerbalizerSpec can be retrained on new paired data
when the transformer is swapped, without touching the SOMA brain.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True)
class VerbalizerSpec:
    """Identity card for a trained verbalizer projector.

    Saved alongside weights so that a future loader can detect
    incompatibility (e.g., different ``llm_hidden_dim``) before silently
    producing a broken prefix.
    """

    soma_output_dim: int
    llm_name: str
    llm_hidden_dim: int
    num_prefix_tokens: int
    proj_hidden_dim: int = 512

    def __post_init__(self) -> None:
        for name in ("soma_output_dim", "llm_hidden_dim", "num_prefix_tokens", "proj_hidden_dim"):
            val = getattr(self, name)
            if val <= 0:
                raise ValueError(f"VerbalizerSpec.{name} must be positive, got {val}")


class SomaVerbalizer(nn.Module):
    """Projects SOMA's OUTPUT-node aggregate into a soft-prompt prefix.

    Training: forward() + LM-loss through a FROZEN LLM backpropagates
    only into this module.

    Inference: forward() produces ``(B, k, llm_hidden_dim)`` tensors
    that a caller concatenates ahead of tokenized text via the LLM's
    ``inputs_embeds`` entrypoint.

    Swap procedure: to pair with a new LLM, construct with a new
    VerbalizerSpec and retrain. SOMA's graph is untouched.
    """

    def __init__(self, spec: VerbalizerSpec) -> None:
        super().__init__()
        self.spec = spec
        self.proj = nn.Sequential(
            nn.Linear(spec.soma_output_dim, spec.proj_hidden_dim),
            nn.LayerNorm(spec.proj_hidden_dim),
            nn.GELU(),
            nn.Linear(
                spec.proj_hidden_dim,
                spec.num_prefix_tokens * spec.llm_hidden_dim,
            ),
        )
        # Near-zero init on the final layer: an untrained projector should
        # emit a near-null prefix so the frozen LLM behaves ~vanilla until
        # the verbalizer is trained on paired SOMA/text data.
        _final = [m for m in self.proj if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(_final.weight, std=1e-3)
        nn.init.zeros_(_final.bias)

    def forward(self, soma_state: torch.Tensor) -> torch.Tensor:
        """Project a SOMA OUTPUT aggregate into a soft-prompt prefix.

        Args:
            soma_state: ``(B, soma_output_dim)`` or ``(soma_output_dim,)``
                aggregated OUTPUT-node activations from SOMA. A 1-D tensor
                is treated as a single unbatched example.

        Returns:
            ``(B, num_prefix_tokens, llm_hidden_dim)`` soft prefix embeddings.

        Raises:
            ValueError: if the last dimension does not match
                ``spec.soma_output_dim``.
        """
        if soma_state.ndim == 1:
            soma_state = soma_state.unsqueeze(0)
        if soma_state.shape[-1] != self.spec.soma_output_dim:
            raise ValueError(
                f"SomaVerbalizer expects last dim "
                f"{self.spec.soma_output_dim} (soma_output_dim), "
                f"got {soma_state.shape[-1]}"
            )
        flat = self.proj(soma_state)  # (B, k*D)
        prefix = flat.view(
            soma_state.shape[0],
            self.spec.num_prefix_tokens,
            self.spec.llm_hidden_dim,
        )
        return cast(torch.Tensor, prefix)

    def save(self, path: Path | str) -> None:
        """Save verbalizer as a directory: spec.json + weights.pt."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "spec.json").write_text(json.dumps(asdict(self.spec), indent=2))
        torch.save(self.state_dict(), str(p / "weights.pt"))

    @classmethod
    def load(cls, path: Path | str) -> SomaVerbalizer:
        p = Path(path)
        spec_path = p / "spec.json"
        weights_path = p / "weights.pt"
        if not spec_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"No verbalizer bundle at {p}")
        spec = VerbalizerSpec(**json.loads(spec_path.read_text()))
        v = cls(spec)
        v.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
        return v


class SomaAggregator:
    """Collapses per-OUTPUT-node activations into a canonical (1, D) vector.

    SOMA produces one activation per OUTPUT node at each step; the verbalizer
    consumes a single D-dim vector. Mean pooling is the v1 strategy — simple,
    order-invariant, and graceful when node count changes via growth.
    """

    @staticmethod
    def collapse(activations: dict[str, torch.Tensor], *, soma_output_dim: int) -> torch.Tensor:
        if not activations:
            return torch.zeros(1, soma_output_dim)
        stacked = []
        for node_id, act in activations.items():
            if act.shape[-1] != soma_output_dim:
                raise ValueError(
                    f"SomaAggregator expected dim {soma_output_dim}, "
                    f"got {act.shape[-1]} for node {node_id}"
                )
            stacked.append(act.view(-1))
        mean = torch.stack(stacked, dim=0).mean(dim=0)
        return mean.unsqueeze(0)
