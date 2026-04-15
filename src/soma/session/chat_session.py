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

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

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

    def respond(
        self,
        *,
        user_text: str,
        max_new_tokens: int = 64,
        **gen_kwargs: Any,
    ) -> str:
        """Generate an assistant response to ``user_text``.

        Pipeline:
            1. Feed user_text into SOMA (per-token, no_grad, eval_mode).
            2. Pool OUTPUT activations -> verbalizer -> soft-prompt prefix.
            3. Tokenize user_text via the LLM tokenizer (NOT the SOMA one).
            4. Embed via the LLM's input embeddings, concat prefix + tokens.
            5. ChatHead.generate_text(...) -> response string.
            6. Feed the response back through SOMA so next turn sees it.
            7. Append (user, assistant) turns to ``self.history``.
        """
        import torch  # local import — keeps module-top imports lean

        from soma.io.chat_head import build_attention_mask_from_pad, build_position_ids
        from soma.io.verbalizer import SomaAggregator

        self._feed_text_through_soma(user_text)

        output_acts = self.soma._current_output_activations()
        pooled = SomaAggregator.collapse(
            output_acts,
            soma_output_dim=self.verbalizer.spec.soma_output_dim,
        )
        prefix = cast(torch.Tensor, self.verbalizer(pooled))  # (1, k, d_model)

        tok_out = self.chat_head.tokenizer(user_text, return_tensors="pt")
        input_ids = tok_out["input_ids"]
        pad_mask = tok_out.get("attention_mask")
        if pad_mask is None:
            pad_mask = torch.ones_like(input_ids)
        token_embeds = self.chat_head.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        k = self.verbalizer.spec.num_prefix_tokens
        t_tok = int(input_ids.shape[1])
        attention_mask = build_attention_mask_from_pad(num_prefix=k, pad_mask=pad_mask)
        position_ids = build_position_ids(num_prefix=k, num_tokens=t_tok, batch_size=1)

        response = cast(
            str,
            self.chat_head.generate_text(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            ),
        )

        # Close the loop: feed the assistant's response back through SOMA so
        # working memory and last_activation reflect what we just said. The
        # next respond() call will see updated state. This is the load-bearing
        # change vs. Phase 3's single-turn SOMA.chat.
        self._feed_text_through_soma(response)

        self.history.append(ChatTurn(role="user", text=user_text))
        self.history.append(ChatTurn(role="assistant", text=response))
        return response

    def save(self, *, out_dir: Path) -> None:
        """Persist a full session bundle to ``out_dir``.

        Writes:
            - SOMA brain bundle via ``soma.save_bundle`` (brain.pt,
              manifest.json, tokenizer.json, encoder.pt, verbalizer/).
            - ``chat_history.json`` sidecar with the conversation log.

        The SOMA's current weights + WM/episodic state are captured at
        save time, so loading this bundle yields a SOMA whose internal
        state reflects everything that's happened up to the save point —
        no need to replay the history through SOMA on load.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        self.soma.save_bundle(
            out_dir,
            tokenizer=self.tokenizer,
            encoder=self.encoder,
            verbalizer=self.verbalizer,
        )
        history_path = out_dir / "chat_history.json"
        history_path.write_text(
            json.dumps(
                [{"role": t.role, "text": t.text, "ts": t.ts.isoformat()} for t in self.history],
                indent=2,
            )
        )

    def load_history(self, *, out_dir: Path) -> None:
        """Load a previously-saved chat history into ``self.history``.

        Overwrites any existing history. Does NOT reload SOMA / verbalizer /
        chat_head — the caller is responsible for constructing those
        (matching the save-time configuration) and passing them to
        ``ChatSession.__init__`` before calling this method.
        """
        history_path = out_dir / "chat_history.json"
        if not history_path.exists():
            raise FileNotFoundError(f"chat_history.json missing at {history_path}")
        raw = json.loads(history_path.read_text())
        self.history = [
            ChatTurn(
                role=item["role"],
                text=item["text"],
                ts=datetime.fromisoformat(item["ts"]),
            )
            for item in raw
        ]
