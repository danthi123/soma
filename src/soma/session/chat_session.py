"""Multi-turn chat session built on SOMA + verbalizer + frozen ChatHead.

A ``ChatSession`` owns one SOMA instance whose persistent state IS the
memory mechanism between turns — we do NOT replay the full conversation
into the LLM each turn. Each ``respond(user_text)`` (T3) feeds user text
through SOMA, generates via the LLM with a verbalizer-derived soft
prompt, and (T4) feeds the assistant's response back through SOMA so
working memory evolves. Next turn sees the updated state.

Phase 5 Task 1: dataclasses + __init__ skeleton with optional system-
prompt pre-warm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

_VALID_ROLES = ("user", "assistant")


@dataclass(frozen=True)
class ChatTurn:
    """One turn in a chat history. Role is constrained at construction."""

    role: Literal["user", "assistant"]
    text: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(f"ChatTurn.role must be one of {_VALID_ROLES}, got {self.role!r}")


class ChatSession:
    """Stateful multi-turn chat over SOMA + frozen LLM."""

    def __init__(
        self,
        *,
        soma: Any,
        verbalizer: Any,
        chat_head: Any,
        tokenizer: Any,
        encoder: Any,
        system_prompt: str | None = None,
    ) -> None:
        self.soma = soma
        self.verbalizer = verbalizer
        self.chat_head = chat_head
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.system_prompt = system_prompt
        self.history: list[ChatTurn] = []

        # Pre-warm SOMA with the system prompt so the first respond() call
        # sees a state already tinted by it. The system prompt does NOT go
        # into self.history because it isn't a "turn" the user sees.
        if system_prompt:
            self._feed_text_through_soma(system_prompt)

    def _feed_text_through_soma(self, text: str) -> None:
        """Push ``text`` through SOMA per-token under no_grad/eval_mode.

        Reuses Phase 4's ``text_to_state`` (which already encodes the
        no-grad + eval_mode + per-token-step pattern). The pooled-state
        return value is discarded — we only care about side effects on
        SOMA's WM and ``Node.last_activation``. T2 will harden this with
        explicit tests.
        """
        from soma.training.verbalizer_bootstrap import text_to_state

        text_to_state(
            text=text,
            soma=self.soma,
            tokenizer=self.tokenizer,
            encoder=self.encoder,
            soma_output_dim=self.verbalizer.spec.soma_output_dim,
        )
