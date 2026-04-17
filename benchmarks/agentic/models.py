"""Ollama model registry for the agentic benchmark suite.

Each entry maps a human-friendly tier name to the Ollama model tag
and relevant metadata.  Actual tag names will be verified when models
are pulled in Phase 46.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a single Ollama model."""

    name: str  # ollama model tag (e.g. "qwen3.5:27b-q4_K_M")
    display_name: str  # human-readable label
    vram_estimate_gb: float
    context_window: int = 8192  # default context window in tokens
    disable_thinking: bool = True  # disable /think for tool-call consistency
    ollama_options: dict[str, object] = field(default_factory=dict)


MODELS: dict[str, ModelConfig] = {
    "sota-max": ModelConfig(
        "qwen3.5:27b-q4_K_M", "Qwen3.5-27B Q4", 16.0, context_window=32768
    ),
    "sota-alt": ModelConfig(
        "glm4.7-flash:q4_K_M", "GLM-4.7-Flash Q4", 12.0, context_window=32768
    ),
    "sota-high": ModelConfig(
        "qwen3.5:9b-q8_0", "Qwen3.5-9B Q8", 9.0, context_window=32768
    ),
    "sota-mid": ModelConfig(
        "qwen3.5:9b-q4_K_M", "Qwen3.5-9B Q4", 6.0, context_window=32768
    ),
    "reasoning": ModelConfig(
        "phi4-reasoning:14b-q4_K_M", "Phi-4 Reasoning 14B Q4", 9.0, context_window=16384
    ),
    "small-high": ModelConfig(
        "qwen3.5:4b-q8_0", "Qwen3.5-4B Q8", 4.0, context_window=32768
    ),
    "small-low": ModelConfig(
        "qwen3.5:4b-q4_K_M", "Qwen3.5-4B Q4", 3.0, context_window=32768
    ),
    "ultralight": ModelConfig(
        "qwen3:0.6b-q8_0", "Qwen3-0.6B Q8", 0.7, context_window=8192
    ),
}


def get_model(tier: str) -> ModelConfig:
    """Look up a model config by tier name; raise KeyError if unknown."""
    if tier not in MODELS:
        raise KeyError(f"Unknown model tier {tier!r}. Available: {sorted(MODELS)}")
    return MODELS[tier]
