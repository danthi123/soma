# Phase 5 — Multi-Turn Chat with Working Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Wrap Phase 3+4's single-turn `SOMA.chat()` in a stateful `ChatSession` that carries SOMA's working-memory state across turns, feeds each model response back into SOMA so subsequent turns are conditioned on the full conversation, and exposes a save/load contract for sessions that survive process restarts.

**Architecture:** `ChatSession` owns one persistent SOMA instance, one verbalizer, one ChatHead, plus a `list[ChatTurn]` history. `respond(user_text)` runs the full pipeline (feed user_text through SOMA → pool OUTPUT activations → verbalize prefix → tokenize+embed user text → concat → LLM generate → decode), then **feeds the model's response back through SOMA** so working memory and `Node.last_activation` evolve. Next turn naturally sees an updated state. Session save/load extends the Phase 2 brain-bundle directory with a `chat_history.json` sidecar.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace transformers (chat template optional), existing SOMA/verbalizer/chat_head infra.

---

## Why Phase 5 Now

Phases 1-4 deliver a single-turn pipeline: text → state → prefix → response. Phase 5 turns that into a **conversation**: the assistant remembers what the user just said, what it just answered, and what context has accumulated over many turns. Deliberately scoped:

- **WM-mediated continuity, not full transcript replay**. We do NOT prepend the entire conversation history to every LLM prompt (token-budget hostile). Instead, SOMA's persistent state IS the memory mechanism: WM slots, episodic encoding, and `Node.last_activation` all carry context forward without bloating prompts.
- **No continuous online verbalizer training**. That's Phase 6 — high-risk surface that needs careful guardrails (gradient sanity, replay buffers, divergence detection). Phase 5 keeps the verbalizer frozen during a session.
- **No GGUF / consumer packaging**. That's Phase 7. Phase 5 still runs SmolLM2-360M in fp32 on CPU/GPU.
- **One participant per session**. Multi-user federated WM is out of scope.

The value of Phase 5 in isolation: validates the WM-carries-context hypothesis end-to-end, gives us a `ChatSession` abstraction the CLI/UI can actually consume, and produces save/load semantics for "resume conversation" — the foundational user-facing artifact for the JARVIS-style vision.

---

## Key Decisions

### 1. SOMA persistence is the memory channel, not prompt history
Each `respond()` call:
1. Feeds `user_text` through SOMA (per-token sensor inputs, eval_mode, no_grad → updates WM/last_activation).
2. Pools current OUTPUT-node activations → verbalizer → soft-prompt prefix.
3. Tokenizes the user text via the LLM tokenizer (NOT the SOMA tokenizer) → token embeddings via LLM input embedder.
4. Concat prefix + user-token embeddings → `inputs_embeds`.
5. `chat_head.generate_text(...)` → response string.
6. **Feeds the response string back through SOMA** the same way (per-token, eval_mode, no_grad). This is the load-bearing change vs. Phase 3 — it lets the next turn's prefix reflect what the assistant just said.
7. Appends `(user_text, response)` to `self.history`.

Rationale: the LLM's context window stays small (just the user's current message + soft prefix), but conversational coherence comes from SOMA's accumulated state. Token-budget cost is bounded by `num_prefix_tokens + len(user_text)`, NOT by conversation length.

Trade-off accepted: the LLM doesn't see the literal prior turns, so it can't quote them verbatim. The verbalizer is responsible for compressing relevant context into the prefix. For Phase 5 with a near-zero-init or lightly-bootstrapped verbalizer, this means the LLM will not perfectly remember names/numbers across turns — but it WILL see varying prefixes that reflect the changing SOMA state. Phase 6's continuous training will tighten that loop.

### 2. `system_prompt` is optional and pre-warms SOMA
If a `ChatSession` is constructed with `system_prompt="You are a helpful assistant who..."`, the prompt is fed through SOMA once at session init (eval_mode, no_grad) so SOMA's WM/state reflects it before the first user turn. The system prompt is NOT prepended to LLM `inputs_embeds` — it lives entirely in SOMA's continuous state.

### 3. Session history is text-only, not embeddings
`ChatTurn` is a small dataclass: `role: Literal["user", "assistant"], text: str, ts: datetime`. We do NOT cache embeddings or activations per turn — replaying through SOMA from text on load is the canonical way to reproduce state. Avoids huge sidecar files.

### 4. Save/load via brain-bundle extension
A session bundle directory contains:
- `brain.pt` (the SOMA checkpoint)
- `tokenizer.json`, `encoder.pt` (existing Phase 2 sidecars)
- `verbalizer/` (Phase 2 directory format)
- `chat_history.json` — list of `{role, text, ts}` records
- `manifest.json` — extends with `"format": "soma-chat-session"` and a `"session_id"` UUID

Loading reconstructs SOMA + verbalizer + tokenizer + encoder, then constructs `ChatSession` with the loaded history (without re-feeding it through SOMA — the brain.pt already captures the post-history state).

### 5. CLI: minimal interactive REPL
`scripts/chat_repl.py` reads stdin lines, calls `session.respond(...)`, prints output. `Ctrl-D` saves the session if `--out-dir` was passed.

---

## Tasks

| # | Scope | Files |
|---|---|---|
| 1 | `ChatTurn` dataclass + `ChatSession.__init__` skeleton | `src/soma/session/chat_session.py`, test |
| 2 | `_feed_text_through_soma` helper (no-grad per-token push) | same, test |
| 3 | `respond(user_text)` single-turn flow (refactor of `SOMA.chat`) | same, test |
| 4 | Response-back-into-SOMA closing the loop | same, test |
| 5 | Optional `system_prompt` pre-warm at init | same, test |
| 6 | Multi-turn coherence — WM/state evolves across turns | same, test |
| 7 | `save(out_dir)` / `load(out_dir)` extending brain-bundle | same, test |
| 8 | CLI REPL `scripts/chat_repl.py` | new |
| 9 | SmolLM2 3-turn smoke (slow-marked) | `tests/test_session/test_chat_session_smoke.py` |
| 10 | Full regression + merge | — |

---

## Task 1: `ChatTurn` + `ChatSession.__init__`

**Files:** Create `src/soma/session/__init__.py` (empty), `src/soma/session/chat_session.py`, `tests/test_session/__init__.py`, `tests/test_session/test_chat_session.py`.

**Step 1: Failing test (`tests/test_session/test_chat_session.py`):**

```python
from datetime import datetime

import pytest
import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.system import SOMA
from soma.session.chat_session import ChatSession, ChatTurn


# Reuse Phase 4's mock pattern.
class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, **_):
        return self.lm_head(inputs_embeds)

    @torch.no_grad()
    def generate(self, *, inputs_embeds, attention_mask=None, max_new_tokens=4, **_):
        embeds = inputs_embeds
        out = []
        for _ in range(max_new_tokens):
            logits = self.forward(inputs_embeds=embeds)
            nid = logits[:, -1, :].argmax(dim=-1)
            out.append(nid)
            embeds = torch.cat([embeds, self.embed(nid).unsqueeze(1)], dim=1)
        return torch.stack(out, dim=1)


class _TinyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = [min(ord(c) % 32, 31) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids.flatten().tolist())


def _soma_cfg() -> SOMAConfig:
    return SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=128,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        seed=0,
    )


def _build_session(system_prompt: str | None = None) -> ChatSession:
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    tokenizer = train_bpe_tokenizer(iter(["hello world", "the quick brown fox"]),
                                    vocab_size=cfg.vocab_size)
    encoder = TextEncoder(tokenizer, embed_dim=cfg.text_embed_dim,
                          max_seq_len=cfg.max_input_tokens)
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim, llm_name="mock",
        llm_hidden_dim=16, num_prefix_tokens=4, proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    chat_head = ChatHead(model=_TinyCausalLM(vocab=32, d_model=16),
                         tokenizer=_TinyTokenizer())
    return ChatSession(
        soma=soma, verbalizer=verbalizer, chat_head=chat_head,
        tokenizer=tokenizer, encoder=encoder,
        system_prompt=system_prompt,
    )


def test_chat_turn_dataclass_fields():
    t = ChatTurn(role="user", text="hi")
    assert t.role == "user"
    assert t.text == "hi"
    assert isinstance(t.ts, datetime)


def test_chat_turn_role_validated():
    with pytest.raises(ValueError, match="role"):
        ChatTurn(role="system", text="hi")  # type: ignore[arg-type]


def test_session_stores_components():
    s = _build_session()
    assert s.soma is not None
    assert s.verbalizer is not None
    assert s.chat_head is not None
    assert s.tokenizer is not None
    assert s.encoder is not None
    assert s.history == []


def test_session_with_system_prompt_pre_warms_soma():
    """A session with a system_prompt should run the prompt through SOMA at
    init, so the very first respond() call sees a state already tinted by
    the system context."""
    s_with = _build_session(system_prompt="you are a helpful assistant")
    s_without = _build_session(system_prompt=None)

    # OUTPUT activations should differ (s_with has been warmed).
    acts_with = s_with.soma._current_output_activations()
    acts_without = s_without.soma._current_output_activations()
    # If both are empty (no warm), test is vacuous — guard against that:
    assert len(acts_with) > 0 or len(acts_without) > 0, (
        "no OUTPUT activations on either; system_prompt warm has nothing to verify"
    )
    # When non-empty, they should differ pointwise (in at least one node).
    if acts_with and acts_without:
        any_diff = any(
            not torch.allclose(acts_with[k], acts_without[k])
            for k in acts_with
            if k in acts_without
        )
        assert any_diff, "system_prompt warm did not change OUTPUT state"
```

**Step 2:** Run — fail (module missing).

**Step 3: Implement.** `src/soma/session/__init__.py` (empty), `tests/test_session/__init__.py` (empty), and `src/soma/session/chat_session.py`:

```python
"""Multi-turn chat session built on SOMA + verbalizer + frozen ChatHead.

A ``ChatSession`` owns one SOMA instance whose persistent state IS the
memory mechanism between turns — we do NOT replay the full conversation
into the LLM each turn. Each ``respond(user_text)`` feeds user text
through SOMA, generates via the LLM with a verbalizer-derived soft
prompt, and then feeds the assistant's response back through SOMA so
working memory evolves. Next turn sees the updated state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import torch


_VALID_ROLES = ("user", "assistant")


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    text: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"ChatTurn.role must be one of {_VALID_ROLES}, got {self.role!r}"
            )


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

        Implemented in T2 — placeholder here so __init__ can call it.
        """
        # T2 fills this in; for now stub-pass to make T1 init work.
        # (T1 tests don't exercise the system_prompt warming for content,
        # only for "did SOMA's OUTPUT state change" — so a no-op here would
        # fail T1's pre-warm test. Intentional: T1's test forces T2's
        # implementation.)
        from soma.training.verbalizer_bootstrap import text_to_state
        # Reuse the no-grad SOMA runner from Phase 4. The pooled-state
        # return is discarded here — we only care about the side-effects
        # on SOMA's WM and Node.last_activation.
        text_to_state(
            text=text,
            soma=self.soma,
            tokenizer=self.tokenizer,
            encoder=self.encoder,
            soma_output_dim=self.verbalizer.spec.soma_output_dim,
        )
```

The `_feed_text_through_soma` calls into Phase 4's `text_to_state` — proven, no-grad, eval-mode-correct. This is the right reuse: Phase 4 already encodes the "feed text through SOMA without training it" pattern.

**Step 4:** Run — 4 tests pass.

**Step 5:** Lint + mypy on touched files. Fix any issues.

**Step 6:** Commit:
```bash
git add src/soma/session/ tests/test_session/
git commit -m "feat(session): ChatSession + ChatTurn skeleton with system_prompt pre-warm"
```

---

## Task 2: `_feed_text_through_soma` consolidation

Task 1 already wired this through `text_to_state`. Task 2 makes it an explicit method of `ChatSession` (cleaner API surface) AND adds tests that prove the per-text-call side-effect (WM advances).

**Step 1: Failing tests (append to `tests/test_session/test_chat_session.py`):**

```python
def test_feed_text_advances_global_step():
    s = _build_session()
    before = s.soma.global_step
    s._feed_text_through_soma("hello world this is a test")
    after = s.soma.global_step
    assert after > before, "soma.global_step did not advance"


def test_feed_text_does_not_train_soma():
    s = _build_session()
    node = next(iter(s.soma.graph.nodes.values()))
    before = next(node.parameters()).detach().clone()
    s._feed_text_through_soma("some words to push through")
    after = next(node.parameters()).detach().clone()
    assert torch.allclose(before, after), "SOMA params drifted (eval_mode broken)"


def test_feed_text_updates_last_activation():
    s = _build_session()
    # Before: OUTPUT nodes might have last_activation=None
    s._feed_text_through_soma("warm up text")
    acts_after_first = {k: v.clone() for k, v in s.soma._current_output_activations().items()}
    s._feed_text_through_soma("different second text")
    acts_after_second = s.soma._current_output_activations()
    # At least one OUTPUT node's activation should differ between the two.
    if acts_after_first and acts_after_second:
        any_changed = any(
            not torch.allclose(acts_after_first[k], acts_after_second[k])
            for k in acts_after_first
            if k in acts_after_second
        )
        assert any_changed, "OUTPUT activations identical after distinct inputs"
```

**Step 2:** Already passes if T1's stub is wired correctly. Otherwise, formalize the method:

```python
def _feed_text_through_soma(self, text: str) -> None:
    """Push ``text`` through SOMA per-token (no_grad, eval_mode).

    Side-effects: advances ``soma.global_step``, updates WM, evolves
    ``Node.last_activation``. Does NOT train SOMA's learnable params
    (eval_mode + no_grad guarantee).
    """
    from soma.training.verbalizer_bootstrap import text_to_state
    text_to_state(
        text=text,
        soma=self.soma,
        tokenizer=self.tokenizer,
        encoder=self.encoder,
        soma_output_dim=self.verbalizer.spec.soma_output_dim,
    )
```

**Step 3:** Tests pass. Commit.

---

## Task 3: `respond(user_text) -> str` (sans response-back-loop)

Single-turn flow without yet feeding the response back. (T4 adds the loop.)

**Step 1: Failing tests:**

```python
def test_respond_returns_string():
    s = _build_session()
    out = s.respond(user_text="hello")
    assert isinstance(out, str)


def test_respond_appends_user_and_assistant_turns():
    s = _build_session()
    assert len(s.history) == 0
    s.respond(user_text="hi")
    assert len(s.history) == 2
    assert s.history[0].role == "user"
    assert s.history[0].text == "hi"
    assert s.history[1].role == "assistant"
    assert isinstance(s.history[1].text, str)


def test_respond_max_new_tokens_passes_through():
    s = _build_session()
    out = s.respond(user_text="hi", max_new_tokens=2)
    # Mock decoder: each token id → 1 char. So response length should
    # equal max_new_tokens.
    assert len(out) == 2
```

**Step 2:** Run — fail.

**Step 3: Implement** — `respond` mirrors `SOMA.chat` but uses session-bound components:

```python
def respond(self, *, user_text: str, max_new_tokens: int = 64, **gen_kwargs: Any) -> str:
    """Generate an assistant response to ``user_text``.

    Pipeline:
        1. Feed user_text into SOMA (per-token, no_grad, eval_mode).
        2. Pool OUTPUT activations → verbalizer → soft-prompt prefix.
        3. Tokenize user_text via the LLM tokenizer (NOT the SOMA one).
        4. Embed those token ids via the LLM's input embeddings.
        5. Concat prefix + token embeddings → inputs_embeds.
        6. ChatHead.generate_text(...) → response string.
        7. Append (user, assistant) turns to self.history.

    Note: the response is NOT yet fed back into SOMA — that's T4's job.
    """
    from soma.io.chat_head import build_attention_mask_from_pad, build_position_ids
    from soma.io.verbalizer import SomaAggregator

    self._feed_text_through_soma(user_text)

    output_acts = self.soma._current_output_activations()
    pooled = SomaAggregator.collapse(
        output_acts, soma_output_dim=self.verbalizer.spec.soma_output_dim,
    )
    prefix = self.verbalizer(pooled)  # (1, k, d_model)

    tok_out = self.chat_head.tokenizer(user_text, return_tensors="pt")
    input_ids = tok_out["input_ids"]
    pad_mask = tok_out.get("attention_mask")
    if pad_mask is None:
        pad_mask = torch.ones_like(input_ids)
    token_embeds = self.chat_head.model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

    k = self.verbalizer.spec.num_prefix_tokens
    T_tok = input_ids.shape[1]
    attention_mask = build_attention_mask_from_pad(num_prefix=k, pad_mask=pad_mask)
    position_ids = build_position_ids(num_prefix=k, num_tokens=T_tok, batch_size=1)

    response = self.chat_head.generate_text(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_new_tokens,
        **gen_kwargs,
    )

    self.history.append(ChatTurn(role="user", text=user_text))
    self.history.append(ChatTurn(role="assistant", text=response))
    return response
```

**Step 4:** Run — 3 tests pass.

**Step 5:** Commit.

---

## Task 4: Feed response back into SOMA (closes the loop)

After T3, `history` grows but SOMA never sees the assistant's response. T4 adds that step.

**Step 1: Failing test:**

```python
def test_respond_feeds_response_back_into_soma():
    """After respond(), SOMA's global_step should reflect TWO text-pushes:
    one for the user input, one for the assistant response.
    """
    s = _build_session()
    before = s.soma.global_step
    response = s.respond(user_text="hi", max_new_tokens=3)
    after = s.soma.global_step
    # Each token through encoder.encode → soma.step. Roughly:
    #   user "hi" ≈ 1-2 tokens; response ≈ 3 chars ≈ 1-3 tokens.
    # We just need MORE than the user-only delta from a parallel call.
    s2 = _build_session()
    before2 = s2.soma.global_step
    s2._feed_text_through_soma("hi")
    user_only_delta = s2.soma.global_step - before2
    assert (after - before) > user_only_delta, (
        "SOMA didn't see the assistant response — global_step grew only by "
        f"{after - before}, expected > {user_only_delta} (user-only baseline)"
    )
```

**Step 2:** Run — fail.

**Step 3: Modify `respond`** to add the back-feed at the end (after `chat_head.generate_text`, before the history append — order matters so the back-feed happens whether or not the history append succeeds):

```python
    response = self.chat_head.generate_text(...)

    # Close the loop: feed the assistant's response back through SOMA so
    # working memory and last_activation reflect what we just said. The
    # next respond() call will see updated state.
    self._feed_text_through_soma(response)

    self.history.append(ChatTurn(role="user", text=user_text))
    self.history.append(ChatTurn(role="assistant", text=response))
    return response
```

**Step 4:** Run — test passes.

**Step 5:** Commit.

---

## Task 5: Optional `system_prompt` pre-warm at init

Already covered in T1's tests. T5's job is to harden the contract: confirm that with `system_prompt=None` no pre-warm runs (`global_step` stays at zero post-init).

**Step 1: Failing test:**

```python
def test_session_without_system_prompt_does_not_warm_soma():
    s = _build_session(system_prompt=None)
    assert s.soma.global_step == 0, (
        f"SOMA pre-warmed without a system_prompt — global_step={s.soma.global_step}"
    )


def test_session_with_system_prompt_advances_global_step():
    s = _build_session(system_prompt="be helpful")
    assert s.soma.global_step > 0, (
        "system_prompt didn't warm SOMA — global_step still zero"
    )
```

**Step 2:** Should already pass given T1+T2's wiring. If not, fix the conditional in `__init__`.

**Step 3:** Commit (test-only).

---

## Task 6: Multi-turn coherence — state evolves across turns

The load-bearing P5 test for the WM-as-memory hypothesis.

**Step 1: Failing test:**

```python
def test_multi_turn_state_evolves():
    """After 3 turns of distinct topics, OUTPUT-node activations should
    differ from the first turn's state. Proves WM is actually carrying
    context across turns rather than being reset.
    """
    s = _build_session()

    s.respond(user_text="apples are red")
    state_after_t1 = {k: v.clone() for k, v in s.soma._current_output_activations().items()}

    s.respond(user_text="bananas are yellow")
    s.respond(user_text="grapes are purple")
    state_after_t3 = s.soma._current_output_activations()

    if state_after_t1 and state_after_t3:
        any_diff = any(
            not torch.allclose(state_after_t1[k], state_after_t3[k])
            for k in state_after_t1
            if k in state_after_t3
        )
        assert any_diff, "OUTPUT state identical after 3 distinct turns — WM frozen?"


def test_history_grows_two_per_respond_call():
    s = _build_session()
    assert len(s.history) == 0
    s.respond(user_text="a")
    assert len(s.history) == 2
    s.respond(user_text="b")
    assert len(s.history) == 4
    s.respond(user_text="c")
    assert len(s.history) == 6
    # And the order is interleaved correctly.
    assert [t.role for t in s.history] == [
        "user", "assistant", "user", "assistant", "user", "assistant",
    ]
```

**Step 2:** Should pass after T3+T4. Run — confirm.

**Step 3:** Commit (test-only).

---

## Task 7: `save(out_dir)` / `load(out_dir)` extending brain-bundle

Save: writes the SOMA bundle (existing) + verbalizer subdir (existing) + `chat_history.json` sidecar.

Load: reads the bundle + verbalizer + history, reconstructs `ChatSession` (chat_head + LLM are loaded separately by the caller — they're not part of the SOMA bundle).

**Step 1: Failing tests:**

```python
def test_save_writes_chat_history_json(tmp_path: Path):
    s = _build_session()
    s.respond(user_text="hello")
    s.respond(user_text="world")
    s.save(out_dir=tmp_path)
    history_path = tmp_path / "chat_history.json"
    assert history_path.exists()
    data = json.loads(history_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 4  # 2 turns × (user + assistant)
    assert all(item["role"] in ("user", "assistant") for item in data)


def test_save_round_trips_history(tmp_path: Path):
    s = _build_session()
    s.respond(user_text="hi")
    s.save(out_dir=tmp_path)
    # Construct a fresh session with the SAME components, then load history.
    s2 = _build_session()
    s2.load_history(out_dir=tmp_path)
    assert len(s2.history) == 2
    assert s2.history[0].text == s.history[0].text
    assert s2.history[1].text == s.history[1].text
```

We deliberately separate `save` (full bundle) from `load_history` (just the sidecar) for Phase 5. Full session reconstruction (including SOMA + verbalizer + chat_head) is the CLI's job in T8 — the session class itself only owns history persistence here. This keeps the abstraction tight.

**Step 2:** Run — fail.

**Step 3: Implement:**

```python
import json
from pathlib import Path

# ChatSession methods:

def save(self, *, out_dir: Path) -> None:
    """Save SOMA bundle + verbalizer + chat history.

    SOMA's existing ``save_bundle`` writes the brain + tokenizer + encoder
    sidecars. We add ``chat_history.json`` for the conversation log.
    Future versions may extend the bundle manifest with chat-session
    metadata; v1 keeps the sidecar separate for simplicity.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Reuse SOMA's bundle writer (Phase 2). It already handles the brain,
    # tokenizer, encoder, manifest, and verbalizer.
    self.soma.save_bundle(
        out_dir,
        tokenizer=self.tokenizer,
        encoder=self.encoder,
        verbalizer=self.verbalizer,
    )
    # Add the chat history sidecar.
    history_path = out_dir / "chat_history.json"
    history_path.write_text(
        json.dumps(
            [
                {"role": t.role, "text": t.text, "ts": t.ts.isoformat()}
                for t in self.history
            ],
            indent=2,
        )
    )


def load_history(self, *, out_dir: Path) -> None:
    """Load chat history from a session bundle (overwrites self.history).

    Does NOT reload SOMA / verbalizer / chat_head — those are the
    caller's responsibility (instantiate them, then construct the
    ChatSession with them, then call this method to restore history).
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
```

Inspect what `SOMA.save_bundle` actually wants — its signature may differ from the kwargs above. Adapt.

**Step 4:** Run — both tests pass.

**Step 5:** Commit.

---

## Task 8: CLI REPL `scripts/chat_repl.py`

```python
"""Interactive multi-turn chat REPL with SOMA + frozen LLM.

Usage:
    python scripts/chat_repl.py \\
        --soma-checkpoint artifacts/some-bundle/ \\
        --llm-name HuggingFaceTB/SmolLM2-360M-Instruct \\
        --verbalizer-checkpoint artifacts/verbalizer-bootstrap-001/verbalizer_final/ \\
        [--system-prompt "You are a helpful assistant."] \\
        [--out-dir artifacts/sessions/today/]      # save on Ctrl-D

REPL conventions:
    > prompt → user message
    Empty line → ignored
    Ctrl-D / quit / exit → save (if --out-dir) and exit
"""
```

Key implementation points:
- Reuse `_load_soma` from `train_verbalizer_bootstrap.py` (consider moving to a shared `scripts/_common.py` if both scripts grow further — but for Phase 5 just duplicate or import).
- After load, attach the verbalizer (`SomaVerbalizer.load(...)` from Phase 2).
- Build ChatHead by downloading/loading the HF model (CPU by default).
- Construct `ChatSession`.
- REPL: `while True: user = input("> "); if user in ("quit", "exit", ""): break; print(session.respond(user_text=user))`.
- On exit (or `Ctrl-D` `EOFError`): if `--out-dir` set, call `session.save(out_dir=...)`.

No unit test (CLI integration is T9's job).

**Commit:**
```bash
git add scripts/chat_repl.py
git commit -m "feat(scripts): chat_repl.py interactive multi-turn REPL"
```

---

## Task 9: SmolLM2 3-turn smoke (slow-marked)

Real model. Three turns. Verifies the whole pipeline works end-to-end with a real LLM and that history accumulates correctly.

**Files:** `tests/test_session/test_chat_session_smoke.py`

```python
"""Slow-marked smoke: ChatSession with real SmolLM2-360M, 3 turns."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow


def _hub_is_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() not in ("1", "true")


def test_three_turn_chat_with_smolm2(tmp_path: Path):
    pytest.importorskip("transformers")
    from typing import Any, cast
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soma.core.config import SOMAConfig
    from soma.io.chat_head import ChatHead
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
    from soma.session.chat_session import ChatSession
    from soma.system import SOMA

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    device = torch.device("cpu")

    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=not _hub_is_reachable(),
        )
        hf_model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32,
                local_files_only=not _hub_is_reachable(),
            ),
        ).to(device)
    except (OSError, ValueError) as exc:
        pytest.skip(f"SmolLM2 unavailable: {exc}")

    chat_head = ChatHead(model=hf_model, tokenizer=hf_tokenizer)

    cfg = SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=128,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=device)
    tokenizer = train_bpe_tokenizer(
        iter(["hello world", "the quick brown fox", "lorem ipsum dolor"]),
        vocab_size=cfg.vocab_size,
    )
    encoder = TextEncoder(tokenizer, embed_dim=cfg.text_embed_dim,
                          max_seq_len=cfg.max_input_tokens)
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name=model_name,
        llm_hidden_dim=chat_head.hidden_size,
        num_prefix_tokens=4,
    )
    verbalizer = SomaVerbalizer(spec).to(device)

    session = ChatSession(
        soma=soma, verbalizer=verbalizer, chat_head=chat_head,
        tokenizer=tokenizer, encoder=encoder,
    )

    r1 = session.respond(user_text="Hello!", max_new_tokens=10)
    r2 = session.respond(user_text="What's your favorite color?", max_new_tokens=10)
    r3 = session.respond(user_text="Tell me a joke.", max_new_tokens=10)

    print(f"\nTurn 1: {r1!r}")
    print(f"Turn 2: {r2!r}")
    print(f"Turn 3: {r3!r}")

    assert all(isinstance(r, str) and len(r) > 0 for r in (r1, r2, r3))
    assert len(session.history) == 6
    # Save round-trip.
    session.save(out_dir=tmp_path)
    assert (tmp_path / "chat_history.json").exists()
```

**Step 2:** Run manually:
```bash
pytest tests/test_session/test_chat_session_smoke.py -v -m slow
```
Expected: PASS — 3 distinct responses, each non-empty.

**Step 3:** Confirm default suite still green.

**Step 4:** Commit.

---

## Task 10: Full regression + merge

```bash
pytest tests/ -m "not slow" -q
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/soma/
```

Each fix is a fresh commit.

Merge:
```bash
git checkout main
git merge --no-ff feat/multiturn-chat -m "Merge ..."
git push origin main
git push github main
```

Phase 5 does NOT change save/load semantics for SOMA itself (the chat_history.json is a NEW sidecar; existing brains load fine). Safe to merge live.

---

## Phase 6 Preview (next plan doc)

- Continuous online verbalizer training: each `respond()` triggers a tiny gradient step on the (state, response) pair, with replay buffer + divergence guard.
- Persistent session resume across process boundaries.
- Per-user verbalizer customization.

---

## Appendix — Failure Modes

- **Response back-loop blows VRAM**: for very long responses the per-token SOMA push could be slow on CPU. Mitigation: `_feed_text_through_soma` is already O(T) per call; for Phase 5 keep `max_new_tokens` modest.
- **History grows unbounded**: `self.history` is text-only and small per turn, but a hour-long session could be 10MB+. Mitigation: optional `max_history_turns` cap in a future task (Phase 5.5 or 6).
- **System prompt warm corrupts SOMA**: the prompt is just text, processed identically to user inputs. If a system prompt produces a bad SOMA state, the user can pass `system_prompt=None`.
- **Multi-turn state diverges**: with the verbalizer near-zero-init or weakly-trained, prefixes don't compress context well, and 5+ turns may produce visibly degraded outputs. Phase 6 fixes this via continuous training.

---

*End of Phase 5 plan. 10 tasks, TDD-disciplined.*
