"""GGUF inference-only backend for SOMA chat.

Wraps a llama.cpp GGUF model so ``scripts/chat_repl.py`` and
``scripts/demo_chat.py`` can consume a locally-cached GGUF file instead
of downloading HF safetensors. NO gradient flow -- this backend cannot
be used for verbalizer bootstrap or online training. If a verbalizer
was pre-trained against an HF model, its soft-prompt prefix can be
decoded via llama.cpp as text and prepended to the user turn.

For training-capable paths, use the HF factory
(:func:`soma.deploy.chat_head_factory.build_chat_head`).

Optional dependency
-------------------
The ``llama-cpp-python`` package is intentionally NOT a hard dependency
of SOMA -- it ships with several CUDA/CPU build variants that can be
finicky to install. Importing this module is safe without it; only
construction of :class:`GGUFChatHead` actually requires the package
(install via ``pip install -e ".[gguf]"``).

Demo / REPL gap
---------------
``scripts/demo_chat.py`` and ``scripts/chat_repl.py`` currently feed the
HF :class:`~soma.io.chat_head.ChatHead` an ``inputs_embeds`` tensor that
includes the SOMA verbalizer's soft-prompt prefix. GGUF runtimes only
accept text prompts, so wiring a GGUF backend into those scripts is a
nontrivial soft-prompt -> hard-prompt conversion task that's deferred
to a future phase. This module lands the building block (backend +
factory + tests + CLI flag) only; the demo wiring will follow once we
decide on a soft-prompt decoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama

    _HAS_LLAMA_CPP = True
except ImportError:
    Llama = None  # mypy: llama_cpp is in ignore_missing_imports, so Llama: Any
    _HAS_LLAMA_CPP = False


class GGUFChatHead:
    """Inference-only stand-in for :class:`~soma.io.chat_head.ChatHead`.

    Implements a minimal subset of the ChatHead surface: tokenizer-like
    encode/decode and a :meth:`generate_text` that takes a string prompt
    (NOT ``inputs_embeds``). Verbalizer prefix, if present, must be
    converted to text via a separate path (left unimplemented in this
    track; a future phase can wire soft-prompt -> hard-prompt conversion
    once we decide on a decoder).

    Parameters
    ----------
    gguf_path:
        Absolute path to a local ``.gguf`` model file (typically from the
        LM Studio cache at ``~/.cache/lm-studio/models/``).
    n_ctx:
        llama.cpp context window. Default 4096.
    n_gpu_layers:
        Number of transformer layers to offload to GPU. ``-1`` means all
        layers (full GPU offload); ``0`` means CPU-only. Default ``-1``.
    verbose:
        Forwarded to llama.cpp; controls its own log spam.
    """

    def __init__(
        self,
        *,
        gguf_path: Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        if not _HAS_LLAMA_CPP:
            raise ImportError(
                'GGUF backend requires llama-cpp-python. Install via `pip install -e ".[gguf]"`.'
            )
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        # Defense-in-depth: if a test monkeypatches ``_HAS_LLAMA_CPP=True`` but
        # ``Llama`` is still None, raise a clear error rather than a confusing
        # ``TypeError: 'NoneType' object is not callable`` two lines down.
        if Llama is None:  # pragma: no cover -- import guard above prevents this
            raise ImportError("Internal: _HAS_LLAMA_CPP=True but Llama is None.")
        self._llama = Llama(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self._gguf_path = gguf_path

    @property
    def supports_gradients(self) -> bool:
        """Always ``False`` -- GGUF cannot be used for training.

        Used by orchestration code that wants to refuse training paths
        when the backing LLM doesn't expose ``nn.Embedding`` weights.
        """
        return False

    @property
    def gguf_path(self) -> Path:
        """Absolute path to the loaded GGUF file."""
        return self._gguf_path

    def generate_text(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **_ignored: Any,
    ) -> str:
        """Generate text from a string prompt.

        NOT compatible with the HF :meth:`ChatHead.generate_text`
        ``inputs_embeds=`` signature -- callers that want to use GGUF
        must rebuild their prompt as text (no soft-prompt prefixes).
        Unknown keyword arguments are silently ignored so callers that
        pass HF-specific kwargs (e.g. ``attention_mask``) don't crash;
        they just have no effect on the GGUF backend.
        """
        out = self._llama(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        return str(out["choices"][0]["text"])

    def tokenize(self, text: str) -> list[int]:
        """Encode ``text`` to a list of token ids using the GGUF's tokenizer."""
        return [int(t) for t in self._llama.tokenize(text.encode("utf-8"))]

    def detokenize(self, tokens: list[int]) -> str:
        """Decode a list of token ids back to a string.

        Uses ``errors="replace"`` so partial UTF-8 sequences (mid-word
        sampling cuts) decode to the replacement character rather than
        crashing the chat loop.
        """
        return str(self._llama.detokenize(tokens).decode("utf-8", errors="replace"))

    # Explicit rejection of the ChatHead training API so mis-use fails loudly
    # at attribute-access time instead of much later inside a forward pass.
    def __getattr__(self, name: str) -> Any:
        if name in {"model", "get_input_embeddings", "parameters"}:
            raise AttributeError(
                f"GGUFChatHead does not expose {name!r}; GGUF is inference-only. "
                "Use soma.deploy.chat_head_factory.build_chat_head for training."
            )
        raise AttributeError(name)


def build_gguf_chat_head(
    *,
    gguf_path: Path,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    verbose: bool = False,
) -> GGUFChatHead:
    """Factory for a :class:`GGUFChatHead`.

    Raises
    ------
    ImportError
        If ``llama-cpp-python`` is not installed.
    FileNotFoundError
        If ``gguf_path`` does not exist.
    """
    return GGUFChatHead(
        gguf_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
