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
    """Stateful multi-turn chat over SOMA + frozen LLM.

    Optional ``online_trainer`` (Phase 6 T7): if a non-None value is
    passed, each ``respond()`` call hands the ``(user_text, response)``
    pair to the trainer's ``step()`` after generation + back-loop. This
    lets the SomaVerbalizer adapt continuously during live chat; when
    left at the default ``None``, chat behavior is pure Phase 5 (the
    verbalizer stays frozen). The trainer is typed ``Any`` here — the
    concrete class lives in ``soma.training.online_verbalizer`` and
    importing it would introduce an import cycle via
    ``verbalizer_bootstrap``; same pattern as the other session
    ``Any``-typed kwargs.
    """

    def __init__(
        self,
        *,
        soma: Any,
        verbalizer: Any,
        chat_head: Any,
        tokenizer: Any,
        encoder: Any,
        system_prompt: str | None = None,
        online_trainer: Any = None,
    ) -> None:
        self.soma = soma
        self.verbalizer = verbalizer
        self.chat_head = chat_head
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.system_prompt = system_prompt
        self.online_trainer = online_trainer
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
        # Phase 7: HF tokenizers return CPU tensors regardless of where the
        # model lives. On a CUDA-loaded ChatHead, feeding CPU input_ids into
        # ``get_input_embeddings()`` raises a device-mismatch RuntimeError.
        # Move ids + pad mask onto the LLM's device before embedding / mask
        # construction so downstream tensors are all colocated. We infer the
        # device from the embedding weight (works for both HF PreTrainedModel
        # and minimal nn.Module test mocks — nn.Module has no default
        # ``.device`` property, only HF's PreTrainedModel does).
        llm_device = self.chat_head.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(llm_device)
        pad_mask = pad_mask.to(llm_device)
        token_embeds = self.chat_head.model.get_input_embeddings()(input_ids)
        # Phase 7 T3: align prefix dtype with the LLM's embedding dtype before
        # concat. The verbalizer trains in fp32 for numerical stability, but
        # on CUDA the deploy layer loads HF causal LMs in fp16 to fit consumer
        # VRAM. Without this cast, torch.cat silently promotes the whole
        # sequence to fp32 inside HF's matmul kernels — doubling VRAM and
        # defeating the fp16 load. ``.to(dtype=...)`` is autograd-safe, so
        # verbalizer gradients still flow during training. Device cast
        # matches the same concern — prefix must land on the LLM's device.
        if prefix.device != token_embeds.device or prefix.dtype != token_embeds.dtype:
            prefix = prefix.to(device=token_embeds.device, dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        k = self.verbalizer.spec.num_prefix_tokens
        t_tok = int(input_ids.shape[1])
        attention_mask = build_attention_mask_from_pad(num_prefix=k, pad_mask=pad_mask)
        position_ids = build_position_ids(num_prefix=k, num_tokens=t_tok, batch_size=1).to(
            llm_device
        )

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

        # Phase 6 T7: optional online training hook. History is appended
        # BEFORE the training step so a trainer exception can't desync the
        # conversation log from what was actually generated — the turn
        # succeeded regardless of whether the post-turn weight update does.
        if self.online_trainer is not None:
            self.online_trainer.step(user_text=user_text, response=response)

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

        # Phase 6: if an online trainer is attached, persist its state alongside.
        if self.online_trainer is not None:
            online_state_path = out_dir / "online_state.json"
            replay_entries = [
                {
                    "user_text": ex.user_text,
                    "response": ex.response,
                    "ts": ex.ts.isoformat(),
                }
                for ex in self.online_trainer.replay_buffer.entries
            ]
            state = {
                "replay_buffer": replay_entries,
                "loss_history": list(self.online_trainer._loss_history),
                "is_diverged": self.online_trainer.is_diverged,
            }
            online_state_path.write_text(json.dumps(state, indent=2))

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

    def load_online_state(self, *, out_dir: Path, online_trainer: Any) -> None:
        """Restore an OnlineVerbalizerTrainer's state from an online_state.json.

        Does NOT construct the trainer — caller must have already built a
        fresh ``OnlineVerbalizerTrainer`` (same config shape as at save
        time) and pass it in. This mirrors ``load_history``'s "caller owns
        reconstruction" contract.

        Restores:
            - replay_buffer.entries (oldest first)
            - _loss_history (preserves window order)
            - is_diverged flag
        """
        from soma.training.online_verbalizer import ChatExchange

        state_path = out_dir / "online_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"online_state.json missing at {state_path}")
        data = json.loads(state_path.read_text())

        online_trainer.replay_buffer.entries.clear()
        for item in data.get("replay_buffer", []):
            online_trainer.replay_buffer.entries.append(
                ChatExchange(
                    user_text=item["user_text"],
                    response=item["response"],
                    ts=datetime.fromisoformat(item["ts"]),
                )
            )
        online_trainer._loss_history.clear()
        for loss in data.get("loss_history", []):
            online_trainer._loss_history.append(float(loss))
        online_trainer.is_diverged = bool(data.get("is_diverged", False))
