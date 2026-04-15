"""Multi-turn chat session built on SOMA + verbalizer + frozen ChatHead.

A ``ChatSession`` owns one SOMA instance whose persistent state IS the
memory mechanism between turns — we do NOT replay the full conversation
into the LLM each turn. Each ``respond(user_text)`` (T3) feeds user text
through SOMA, generates via the LLM with a verbalizer-derived soft
prompt, and (T4) feeds the assistant's response back through SOMA so
working memory evolves. Next turn sees the updated state.

Phase 5 Task 1: dataclasses + __init__ skeleton with optional system-
prompt pre-warm.

Track D (Phase 7+): a second ``gguf_head`` backend is available for
operators who want to consume a local GGUF (e.g. from the LM Studio
cache) without re-downloading HF safetensors. GGUF mode deliberately
SKIPS the verbalizer — ``user_text`` is fed directly to
:meth:`GGUFChatHead.generate_text` as a string prompt. The SOMA graph
still runs (state evolves across turns) but its OUTPUT activations are
NOT projected into soft-prompt tokens. Online verbalizer training is
refused in GGUF mode because ``llama.cpp`` does not expose gradients.
Pick the HF backend (``chat_head``) if you want soft-prompt guidance
or training.
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

    Backend selection (Track D): exactly one of ``chat_head`` (HF causal
    LM) or ``gguf_head`` (llama.cpp-backed :class:`GGUFChatHead`) must
    be provided. The HF path runs the Phase-5 verbalizer-prefix pipeline;
    the GGUF path skips the verbalizer and feeds ``user_text`` directly
    to the llama.cpp runtime as a string prompt. SOMA state still
    evolves across turns in both modes. When ``gguf_head`` is active,
    ``verbalizer`` and ``online_trainer`` must both be ``None`` — GGUF
    cannot expose gradients and online training needs them.
    """

    def __init__(
        self,
        *,
        soma: Any,
        chat_head: Any,
        tokenizer: Any,
        encoder: Any,
        verbalizer: Any = None,
        gguf_head: Any = None,
        system_prompt: str | None = None,
        online_trainer: Any = None,
    ) -> None:
        # Backend selection: exactly one of chat_head / gguf_head. Refuse
        # both-set and neither-set up front — the downstream respond()
        # branches would otherwise fail much later with a confusing
        # AttributeError on the wrong object.
        if chat_head is None and gguf_head is None:
            raise ValueError(
                "ChatSession requires exactly one of chat_head (HF backend) "
                "or gguf_head (GGUFChatHead); neither was provided."
            )
        if chat_head is not None and gguf_head is not None:
            raise ValueError(
                "ChatSession requires exactly one of chat_head / gguf_head; "
                "both were provided. Pick one backend per session."
            )
        if gguf_head is not None and online_trainer is not None:
            raise ValueError(
                "online verbalizer training requires the HF backend; "
                "GGUFChatHead does not expose gradients."
            )
        if gguf_head is not None and verbalizer is not None:
            # The verbalizer has no consumer in GGUF mode -- refusing it
            # here keeps the "GGUF bypasses verbalizer" contract explicit.
            raise ValueError(
                "gguf_head + verbalizer is an invalid pairing: GGUF mode "
                "skips the verbalizer entirely. Pass verbalizer=None when "
                "using gguf_head."
            )
        if chat_head is not None and verbalizer is None:
            # HF-backend respond() dereferences verbalizer.spec and calls
            # verbalizer(pooled); without it the pipeline has nothing to
            # project SOMA state into soft-prompt tokens.
            raise ValueError(
                "chat_head requires a verbalizer (the soft-prompt projector); got verbalizer=None."
            )

        self.soma = soma
        self.verbalizer = verbalizer
        self.chat_head = chat_head
        self.gguf_head = gguf_head
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

        The pooling dim has to match SOMA's OUTPUT-node output_dim. When
        a verbalizer is attached we use ``verbalizer.spec.soma_output_dim``
        so any future spec-side reshape stays the single source of truth;
        in GGUF mode (no verbalizer) we fall back to
        ``soma.config.sensor_output_dim`` which today equals the OUTPUT
        output_dim (see SOMA._initialize_seed_graph).
        """
        from soma.training.verbalizer_bootstrap import text_to_state

        if self.verbalizer is not None:
            soma_output_dim = int(self.verbalizer.spec.soma_output_dim)
        else:
            soma_output_dim = int(self.soma.config.sensor_output_dim)

        text_to_state(
            text=text,
            soma=self.soma,
            tokenizer=self.tokenizer,
            encoder=self.encoder,
            soma_output_dim=soma_output_dim,
        )

    def respond(
        self,
        *,
        user_text: str,
        max_new_tokens: int = 64,
        **gen_kwargs: Any,
    ) -> str:
        """Generate an assistant response to ``user_text``.

        HF backend pipeline (``chat_head`` set):
            1. Feed user_text into SOMA (per-token, no_grad, eval_mode).
            2. Pool OUTPUT activations -> verbalizer -> soft-prompt prefix.
            3. Tokenize user_text via the LLM tokenizer (NOT the SOMA one).
            4. Embed via the LLM's input embeddings, concat prefix + tokens.
            5. ChatHead.generate_text(...) -> response string.
            6. Feed the response back through SOMA so next turn sees it.
            7. Append (user, assistant) turns to ``self.history``.

        GGUF backend pipeline (``gguf_head`` set):
            1. Feed user_text into SOMA (state still evolves per-turn).
            2. Call ``gguf_head.generate_text(prompt=user_text, ...)`` —
               verbalizer + soft-prompt prefix are deliberately SKIPPED
               because llama.cpp only accepts text prompts.
            3. Feed the response back through SOMA.
            4. Append (user, assistant) turns to ``self.history``.

        Unknown ``gen_kwargs`` (e.g. ``do_sample`` / ``min_new_tokens``)
        are forwarded verbatim. HF generate() honours them; GGUFChatHead
        silently ignores them via its ``**_ignored`` swallow.
        """
        self._feed_text_through_soma(user_text)

        if self.gguf_head is not None:
            response = self._respond_gguf(
                user_text=user_text,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
        else:
            response = self._respond_hf(
                user_text=user_text,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
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
        # NOTE: online_trainer is guaranteed None in GGUF mode (refused at
        # __init__), so this branch only fires on the HF path.
        if self.online_trainer is not None:
            self.online_trainer.step(user_text=user_text, response=response)

        return response

    def _respond_hf(
        self,
        *,
        user_text: str,
        max_new_tokens: int,
        **gen_kwargs: Any,
    ) -> str:
        """HF-backend respond core: verbalizer prefix + inputs_embeds generate.

        Split out of :meth:`respond` so the two backend branches are
        obviously-separate code paths. ``user_text`` has already been fed
        through SOMA by the caller — this method only builds the prefix,
        embeds, and calls ``chat_head.generate_text``. The response-back-
        loop and history append happen in :meth:`respond`.
        """
        import torch  # local import — keeps module-top imports lean

        from soma.io.chat_head import build_attention_mask_from_pad, build_position_ids
        from soma.io.verbalizer import SomaAggregator

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

        return cast(
            str,
            self.chat_head.generate_text(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            ),
        )

    def _respond_gguf(
        self,
        *,
        user_text: str,
        max_new_tokens: int,
        **gen_kwargs: Any,
    ) -> str:
        """GGUF-backend respond core: direct string-prompt generate.

        The SOMA graph has already been fed ``user_text`` by
        :meth:`respond`, so state evolves for the next turn even though
        we don't consume the OUTPUT activations here. ``llama.cpp`` only
        accepts text prompts; we therefore bypass the verbalizer /
        inputs_embeds path entirely.
        """
        return cast(
            str,
            self.gguf_head.generate_text(
                prompt=user_text,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            ),
        )

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
