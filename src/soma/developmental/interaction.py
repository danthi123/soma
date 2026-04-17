"""Main interaction loop: human <-> SOMA <-> LLM."""
from __future__ import annotations

from typing import Any

import requests
import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.developmental.tracker import DevelopmentTracker
from soma.developmental.verbalize import verbalize_state
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

_SYSTEM_PROMPT = """\
You are the voice of a developing mind. Your responses must reflect \
ONLY the internal state provided below. If the state shows high \
novelty, express curiosity and uncertainty. If associations are \
strong, make connections. If working memory is sparse, be brief. \
You are not a knowledgeable assistant — you are a developing \
intelligence expressing what it currently understands and feels.

Do NOT answer from general knowledge. Only reflect what the \
internal state tells you."""


class InteractionLoop:
    """Orchestrates the human <-> SOMA <-> LLM interaction pipeline.

    Connects :class:`PredictiveSOMA`, :func:`verbalize_state`, and
    :class:`DevelopmentTracker` into a single loop that processes text
    input, runs it through SOMA, verbalizes the internal state, and
    optionally sends it to an LLM for a natural-language response.
    """

    def __init__(
        self,
        config: SOMAConfig,
        llm_model: str,
        llm_api_base: str = "http://localhost:11434",
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.device = device or torch.device("cpu")
        self.predictive_soma = PredictiveSOMA(config, device=self.device)
        self.tracker = DevelopmentTracker()
        self._encoder: TextEncoder | None = None

    # ------------------------------------------------------------------
    # Tokenizer / encoding
    # ------------------------------------------------------------------

    def train_tokenizer(
        self, corpus: list[str], vocab_size: int | None = None
    ) -> None:
        """Train a BPE tokenizer on *corpus* and build a :class:`TextEncoder`."""
        vs = vocab_size if vocab_size is not None else self.config.vocab_size
        tokenizer = train_bpe_tokenizer(corpus, vocab_size=vs)
        self._encoder = TextEncoder(
            tokenizer,
            embed_dim=self.config.text_embed_dim,
            device=self.device,
        )

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode *text* into a single mean-pooled embedding vector."""
        if self._encoder is None:
            raise RuntimeError(
                "Tokenizer not initialised — call train_tokenizer() first."
            )
        tokens = self._encoder.encode_batch(text)  # (T, embed_dim)
        if tokens.shape[0] == 0:
            return torch.zeros(self.config.text_embed_dim, device=self.device)
        return tokens.mean(dim=0)

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    def _call_llm(self, soma_state: str) -> str:
        """Send the verbalized SOMA state to the LLM and return its reply."""
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": soma_state},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": 256, "temperature": 0.7},
        }
        resp = requests.post(
            f"{self.llm_api_base}/api/chat", json=payload, timeout=60
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Main loop step
    # ------------------------------------------------------------------

    def process_input(
        self, text: str, *, call_llm: bool = True
    ) -> dict[str, Any]:
        """Run one full interaction step.

        1. Encode *text* into an embedding vector.
        2. Feed it through :class:`PredictiveSOMA`.
        3. Verbalize the resulting SOMA state.
        4. Record metrics in the :class:`DevelopmentTracker`.
        5. Optionally query the LLM with the verbalized state.

        Returns a dict with keys: ``response``, ``soma_state``,
        ``prediction_error``, ``novelty``, ``global_step``.
        """
        input_vec = self.encode_text(text)
        result = self.predictive_soma.process_input(input_vec)
        soma_state = verbalize_state(self.predictive_soma.soma)

        soma = self.predictive_soma.soma
        wm_occupancy = int((soma.working_memory.usage > 0.1).sum().item())
        episodic_count = int(soma.episodic_memory.valid.sum().item())

        self.tracker.record(
            step=result["global_step"],
            prediction_error=result["prediction_error"],
            novelty=result["novelty"],
            num_nodes=result["num_nodes"],
            num_edges=result["num_edges"],
            wm_occupancy=wm_occupancy,
            episodic_count=episodic_count,
        )

        response: str | None = None
        if call_llm:
            response = self._call_llm(soma_state)

        return {
            "response": response,
            "soma_state": soma_state,
            "prediction_error": result["prediction_error"],
            "novelty": result["novelty"],
            "global_step": result["global_step"],
        }
