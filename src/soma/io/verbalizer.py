"""SOMA → frozen transformer projection contract.

The SomaVerbalizer is the stable interface between SOMA's cognition
(variable over time via structural plasticity) and a swappable small
transformer's verbalization (fixed weights, replaceable). A projector
tied to a specific VerbalizerSpec can be retrained on new paired data
when the transformer is swapped, without touching the SOMA brain.
"""

from __future__ import annotations

from dataclasses import dataclass


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
