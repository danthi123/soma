"""Model registry for the agentic benchmark suite.

Each entry maps a human-friendly tier name to the model ID as
reported by the inference server and relevant metadata.  Model IDs
are Ollama tags, verified against the Ollama library on 2026-04-17.

The benchmark agents talk to an OpenAI-compatible endpoint
(``/v1/chat/completions``).  Ollama exposes this at ``/v1`` on
port 11434; LM Studio and vLLM also support this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Ollama's OpenAI-compatible endpoint on localhost.
DEFAULT_API_BASE = "http://localhost:11434/v1"


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str  # model ID as returned by /v1/models (Ollama tag)
    display_name: str  # human-readable label
    vram_estimate_gb: float
    context_window: int = 8192  # default context window in tokens
    disable_thinking: bool = True  # disable /think for tool-call consistency
    extra_options: dict[str, object] = field(default_factory=dict)


MODELS: dict[str, ModelConfig] = {
    # ── SOTA tier ───────────────────────────────────────────────
    "sota-max": ModelConfig(
        "qwen3.5:27b-q4_K_M",
        "Qwen3.5-27B Q4", 16.0, context_window=32768,
    ),
    "dense-large": ModelConfig(
        "gemma4:31b-it-q4_K_M",
        "Gemma-4-31B Q4", 18.0, context_window=32768,
    ),
    "moe-large": ModelConfig(
        "gemma4:26b-a4b-it-q4_K_M",
        "Gemma-4-26B-a4b Q4 (MoE)", 15.0, context_window=32768,
    ),
    "sota-alt": ModelConfig(
        "glm4:9b-chat-q4_K_M",
        "GLM-4 9B Q4", 6.0, context_window=32768,
    ),

    # ── Mid tier ────────────────────────────────────────────────
    "reasoning": ModelConfig(
        "phi4-reasoning:14b-q4_K_M",
        "Phi-4 Reasoning 14B Q4", 9.0, context_window=32768,
    ),
    "sota-high": ModelConfig(
        "qwen3.5:9b-q8_0",
        "Qwen3.5-9B Q8", 9.0, context_window=32768,
    ),
    "sota-mid": ModelConfig(
        "qwen3.5:9b-q4_K_M",
        "Qwen3.5-9B Q4", 6.0, context_window=32768,
    ),

    # ── Small tier ──────────────────────────────────────────────
    "moe-small": ModelConfig(
        "gemma4:e4b",
        "Gemma-4-e4b (MoE small)", 4.0, context_window=32768,
    ),
    "small-high": ModelConfig(
        "qwen3.5:4b-q8_0",
        "Qwen3.5-4B Q8", 4.0, context_window=32768,
    ),
    "small-low": ModelConfig(
        "qwen3.5:4b-q4_K_M",
        "Qwen3.5-4B Q4", 3.0, context_window=32768,
    ),
    "small-gemma": ModelConfig(
        "gemma4:e2b",
        "Gemma-4-e2b", 2.0, context_window=32768,
    ),

    # ── Ultralight ──────────────────────────────────────────────
    "ultralight": ModelConfig(
        "qwen3:0.6b",
        "Qwen3-0.6B Q8", 0.7, context_window=8192,
    ),
}


def get_model(tier: str) -> ModelConfig:
    """Look up a model config by tier name; raise KeyError if unknown."""
    if tier not in MODELS:
        raise KeyError(f"Unknown model tier {tier!r}. Available: {sorted(MODELS)}")
    return MODELS[tier]
