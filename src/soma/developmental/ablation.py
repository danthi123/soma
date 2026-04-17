"""Ablation harness — measure SOMA's developmental contribution."""
from __future__ import annotations

from typing import Any

import requests

from soma.core.config import SOMAConfig
from soma.developmental.interaction import _SYSTEM_PROMPT, InteractionLoop


class AblationHarness:
    """Run inputs with and without SOMA state to measure its contribution.

    For each input the harness produces two LLM responses:

    * **with_soma** — the LLM receives the real verbalized SOMA state.
    * **without_soma** — the LLM receives a blank-slate placeholder.

    The delta between the two quantifies how much SOMA's developmental
    state influences the response.
    """

    def __init__(
        self,
        config: SOMAConfig,
        llm_model: str,
        llm_api_base: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        self.loop = InteractionLoop(
            config=config,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            **kwargs,
        )
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def empty_state_text(self) -> str:
        """Return a static blank-slate SOMA state string."""
        return (
            "=== SOMA Internal State ===\n"
            "Stage: blank-slate | Step: 0\n"
            "Prediction error: 0.000 | Novelty: 0.000\n"
            "Top activations: (none)\n"
            "Working memory: (empty)\n"
            "Recent episodes: (none)\n"
        )

    def _call_llm(self, state_text: str) -> str:
        """Call the LLM with *state_text* at temperature 0 for determinism."""
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": state_text},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": 256, "temperature": 0.0},
        }
        resp = requests.post(
            f"{self.llm_api_base}/api/chat", json=payload, timeout=60
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        inputs: list[str],
        references: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the ablation over *inputs*.

        Returns a dict with keys ``with_soma``, ``without_soma``,
        ``soma_states``, and ``delta``.  If *references* are provided
        token-F1 is computed for both conditions.
        """
        with_soma: list[str] = []
        without_soma: list[str] = []
        soma_states: list[str] = []

        for text in inputs:
            # Process through SOMA (no LLM call — we call it ourselves)
            result = self.loop.process_input(text, call_llm=False)
            state_text = result["soma_state"]
            soma_states.append(state_text)

            with_resp = self._call_llm(state_text)
            without_resp = self._call_llm(self.empty_state_text())

            with_soma.append(with_resp)
            without_soma.append(without_resp)

        # --- delta metrics ---------------------------------------------------
        avg_len_with = (
            sum(len(r) for r in with_soma) / len(with_soma) if with_soma else 0.0
        )
        avg_len_without = (
            sum(len(r) for r in without_soma) / len(without_soma)
            if without_soma
            else 0.0
        )
        delta: dict[str, Any] = {
            "avg_response_len_with": avg_len_with,
            "avg_response_len_without": avg_len_without,
            "avg_response_len_delta": avg_len_with - avg_len_without,
        }

        if references is not None:
            from benchmarks.industry.longmemeval.metrics import token_f1

            f1_with = [
                token_f1(pred, ref)
                for pred, ref in zip(with_soma, references, strict=True)
            ]
            f1_without = [
                token_f1(pred, ref)
                for pred, ref in zip(without_soma, references, strict=True)
            ]
            mean_f1_with = sum(f1_with) / len(f1_with) if f1_with else 0.0
            mean_f1_without = (
                sum(f1_without) / len(f1_without) if f1_without else 0.0
            )
            delta["token_f1_with"] = mean_f1_with
            delta["token_f1_without"] = mean_f1_without
            delta["token_f1_delta"] = mean_f1_with - mean_f1_without

        return {
            "with_soma": with_soma,
            "without_soma": without_soma,
            "soma_states": soma_states,
            "delta": delta,
        }
