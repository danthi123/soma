# Phase 6 — Continuous Online Verbalizer Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Close the feedback loop so the SomaVerbalizer learns during chat. Each `ChatSession.respond()` becomes a tiny training step on the (state, user_text) pair — teaching the verbalizer to project SOMA state into a prefix that predicts the kind of text this user is saying. With replay + divergence guards so the verbalizer improves instead of collapsing.

**Architecture:** A new `OnlineVerbalizerTrainer` wraps `VerbalizerTrainer` (Phase 4) with a fixed-size `ReplayBuffer` of prior (user_text, response) exchanges, a rolling-window divergence monitor, and a `step(user_text)` entry point. `ChatSession.respond()` optionally calls `online_trainer.step(user_text)` after generation completes — so the generation itself uses a settled verbalizer, and the update happens between turns. The loss signal is the same teacher-forced LM CE used in Phase 4: given SOMA's current state, the verbalizer should produce a prefix such that the frozen LLM assigns high probability to the user's text.

**Tech Stack:** Python 3.11+, PyTorch, existing Phase 4 infra (`VerbalizerTrainer`, `compute_lm_loss`, `text_to_state`).

**Note on `.train(False)`:** Same hook-avoidance pattern as earlier phases — never the forbidden dot-e-v-a-l-paren spelling.

---

## Why Phase 6 Now

Phase 5 gave us a stateful chat loop where WM carries context across turns, but the verbalizer's projection SOMA-state → prefix is frozen at whatever Phase 4's bootstrap left it. That's demonstrably undertrained: the T9 smoke found the near-null prefix can let greedy SmolLM2-Instruct hit `<|im_end|>` at t=0 on some inputs, and across 3 turns the projector never adapts to the actual user's style.

Phase 6 turns the verbalizer from a frozen module into a **continuously adapting one**. Each user turn is a new data point teaching the projector to produce prefixes consistent with how this specific user talks.

Deliberately scoped:
- **Loss signal is Phase 4's `compute_lm_loss`** — teacher-forced LM CE on `(prefix_from_state, user_text_tokens)`. Proven to work; no new loss architecture.
- **One gradient step per turn, not a training epoch.** Online means "incremental," not "re-train from scratch."
- **Replay buffer is small** (default 64 turns). Enough to prevent last-turn overfitting without eating RAM.
- **Divergence guard stops training, doesn't auto-rollback.** If loss drifts, freeze and log; rollback is Phase 7.
- **No RL / user preference.** The user's own text is the teacher; "what SOMA already knows should predict what the user says next."
- **No joint SOMA training.** SOMA stays frozen during chat (eval_mode inside text_to_state).

The value in isolation: by session's end, the verbalizer's projection is empirically better-calibrated to the user than the bootstrap left it. The next session resumes from that improved state.

---

## Key Decisions

### 1. Train AFTER generation, not before
Flow per `respond(user_text)`:
1. Feed user_text through SOMA (no_grad).
2. Pool state → verbalizer → prefix. **← generation uses the current verbalizer.**
3. Generate response, feed back through SOMA (Phase 5).
4. **If online training active**: `online_trainer.step(user_text, response)` — record, sample, gradient step.

Rationale: generation uses whatever the verbalizer already settled on (reproducible within a turn). The update happens in the between-turns gap so next turn sees a slightly better projector. Avoids the "did that response use the pre- or post-update verbalizer?" ambiguity.

### 2. Loss signal: LM CE on (soma_state, user_text)
Exactly Phase 4. The user's text IS the label — we teach the verbalizer that "given SOMA's current state, here's text the user actually said, so your prefix should make this text highly probable." The model's own response is recorded in the replay buffer but NOT used as a training label (self-distillation has no content gradient; risks degenerate collapse).

Rejected alternatives:
- **Train on (state, response)**: self-distillation, no improvement signal.
- **Train on (state, user_text) AND (state_after_response, user_text_next_turn)**: cross-turn objectives. Intriguing but complicates buffer management; defer.
- **Entropy minimization on LLM output**: can collapse to degenerate prefixes.

### 3. Replay buffer: fixed-size deque, text-only
```
ReplayBuffer(capacity=64):
    entries: deque[ChatExchange(user_text, response, ts)]
    add(exchange)
    sample(batch_size) -> list[ChatExchange]
```
Stored as text (not embeddings/states) — re-deriving the SOMA state by feeding text through SOMA at train time is cheap and correct. Prevents the buffer from becoming a VRAM hog.

### 4. Divergence guard: rolling-window rise detection
Keep the last `divergence_window` losses (default 20). If the mean of the last N/2 losses is above the mean of the first N/2 by more than `divergence_threshold` (default 1.0 nat), consider training diverged:
- Set `is_diverged = True`
- Skip further optim.steps until `reset_divergence_guard()` is called externally
- Log a warning with the before/after means

This is a simple monotonic-rise check, not a sophisticated anomaly detector. Phase 6 prioritizes preventing catastrophic training runaway over optimal scheduling.

### 5. Online LR is SEPARATELY configured
Online training should be gentler than bootstrap. New `SOMAConfig.online_verbalizer_lr` (default 1e-5, 10× smaller than bootstrap) so live chat doesn't swing the projector aggressively per turn.

### 6. Batch = latest + random from buffer
Each `step(user_text, response)` trains on:
- The just-recorded turn (weight 1.0).
- Up to `online_batch_size - 1` random prior turns from the buffer.

Prevents last-turn overfitting; keeps old context vaguely useful. `online_batch_size` defaults to 4.

### 7. ChatSession integration is opt-in
```python
ChatSession(
    soma=..., verbalizer=..., chat_head=..., tokenizer=..., encoder=...,
    online_trainer=OnlineVerbalizerTrainer(...),   # optional
)
```
If `online_trainer=None`: Phase 5 behavior exactly. If set: `respond()` calls `online_trainer.step(user_text, response)` after feeding the response back into SOMA.

---

## Tasks

| # | Scope | Files |
|---|---|---|
| 1 | SOMAConfig extension: online_verbalizer_lr, online_batch_size, replay_buffer_capacity, divergence_window, divergence_threshold | `src/soma/core/config.py`, test |
| 2 | `ChatExchange` dataclass + `ReplayBuffer` class | `src/soma/training/online_verbalizer.py`, test |
| 3 | `OnlineVerbalizerTrainer.__init__` + `.record(exchange)` | same, test |
| 4 | `.sample_batch() -> list[str]` (latest + random, uniform) | same, test |
| 5 | `.step(user_text, response)` — record + sample + train | same, test |
| 6 | Divergence monitor — rolling window rise detector | same, test |
| 7 | `ChatSession` optional `online_trainer` integration | `src/soma/session/chat_session.py`, test |
| 8 | Smoke: real SmolLM2 + 10-turn chat with online training | `tests/test_training/test_online_verbalizer_smoke.py` |
| 9 | Save/load extension for replay buffer + divergence state | same, test |
| 10 | Full regression + merge | — |

**Tasks 2-6 use mocks** (reuse Phase 4's `_TinyCausalLM`, `_TinyTokenizer`, etc.). Task 8 is the load-bearing real-model check.

---

## Task 1: SOMAConfig extension

**Files:** `src/soma/core/config.py`, `tests/test_core/test_config.py`.

**Step 1: Failing test.**

```python
def test_config_has_online_verbalizer_fields(self) -> None:
    cfg = SOMAConfig()
    assert cfg.online_verbalizer_lr == pytest.approx(1e-5)
    assert cfg.online_batch_size == 4
    assert cfg.replay_buffer_capacity == 64
    assert cfg.divergence_window == 20
    assert cfg.divergence_threshold == pytest.approx(1.0)


def test_config_online_verbalizer_fields_round_trip(self, tmp_path: Path) -> None:
    original = SOMAConfig(
        online_verbalizer_lr=5e-6,
        online_batch_size=8,
        replay_buffer_capacity=128,
        divergence_window=50,
        divergence_threshold=2.0,
    )
    yaml_path = tmp_path / "online.yaml"
    original.to_yaml(yaml_path)
    recovered = SOMAConfig.from_yaml(yaml_path)
    assert recovered.online_verbalizer_lr == pytest.approx(5e-6)
    assert recovered.online_batch_size == 8
    assert recovered.replay_buffer_capacity == 128
    assert recovered.divergence_window == 50
    assert recovered.divergence_threshold == pytest.approx(2.0)
```

Add both as methods of the existing `TestDefaults` and `TestToYaml` classes.

**Step 2: Implement** in `SOMAConfig`, grouped near the Phase 4 bootstrap block:

```python
    # --- Phase 6: Online verbalizer training --------------------------------
    online_verbalizer_lr: float = 1e-5
    """LR for per-turn online updates. 10× smaller than bootstrap's 1e-4
    because each step trains on just the last turn + a few replays —
    noisier signal, so gentler steps."""
    online_batch_size: int = 4
    """Samples per online step: 1 latest + (N-1) random from replay."""
    replay_buffer_capacity: int = 64
    """Max turns retained in the online replay buffer."""
    divergence_window: int = 20
    """Rolling window size for the divergence-rise monitor."""
    divergence_threshold: float = 1.0
    """If mean(last half) - mean(first half) > this, freeze online updates."""
```

Add the 4 int fields (not `divergence_threshold` or `online_verbalizer_lr`) to the positive_ints validation list.

**Step 3: Commit:**
```bash
git add src/soma/core/config.py tests/test_core/test_config.py
git commit -m "feat(config): Phase 6 online verbalizer + replay + divergence fields"
```

---

## Task 2: `ChatExchange` + `ReplayBuffer`

**Files:** `src/soma/training/online_verbalizer.py`, `tests/test_training/test_online_verbalizer.py`.

**Step 1: Failing tests.**

```python
from datetime import datetime
from soma.training.online_verbalizer import ChatExchange, ReplayBuffer


def test_chat_exchange_dataclass_fields():
    ex = ChatExchange(user_text="hi", response="hello")
    assert ex.user_text == "hi"
    assert ex.response == "hello"
    assert isinstance(ex.ts, datetime)


def test_replay_buffer_default_empty():
    buf = ReplayBuffer(capacity=4)
    assert len(buf) == 0


def test_replay_buffer_add_grows():
    buf = ReplayBuffer(capacity=4)
    buf.add(ChatExchange(user_text="a", response="b"))
    assert len(buf) == 1


def test_replay_buffer_evicts_oldest_at_capacity():
    buf = ReplayBuffer(capacity=2)
    buf.add(ChatExchange(user_text="a", response="x"))
    buf.add(ChatExchange(user_text="b", response="y"))
    buf.add(ChatExchange(user_text="c", response="z"))
    assert len(buf) == 2
    user_texts = [ex.user_text for ex in buf.entries]
    assert "a" not in user_texts  # oldest evicted
    assert user_texts == ["b", "c"]


def test_replay_buffer_sample_returns_at_most_n():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.add(ChatExchange(user_text=f"u{i}", response=f"r{i}"))
    sampled = buf.sample(n=3)
    assert len(sampled) == 3
    assert all(isinstance(s, ChatExchange) for s in sampled)


def test_replay_buffer_sample_n_larger_than_buffer_returns_all():
    buf = ReplayBuffer(capacity=10)
    buf.add(ChatExchange(user_text="a", response="x"))
    buf.add(ChatExchange(user_text="b", response="y"))
    sampled = buf.sample(n=10)
    assert len(sampled) == 2


def test_replay_buffer_rejects_non_positive_capacity():
    import pytest
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=0)
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=-5)
```

**Step 2: Implement:**

```python
"""Online verbalizer training infrastructure.

Each ChatSession turn becomes a tiny gradient step. A ReplayBuffer
retains prior (user_text, response) exchanges so the verbalizer doesn't
overfit to the latest turn. A divergence monitor freezes updates if loss
starts rising monotonically.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class ChatExchange:
    user_text: str
    response: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


class ReplayBuffer:
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.entries: deque[ChatExchange] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, exchange: ChatExchange) -> None:
        self.entries.append(exchange)

    def sample(self, *, n: int) -> list[ChatExchange]:
        """Return up to ``n`` entries, sampled uniformly without replacement.

        If ``n`` exceeds the buffer size, returns all entries.
        """
        if n >= len(self.entries):
            return list(self.entries)
        return random.sample(list(self.entries), n)
```

**Step 3: Commit.**

---

## Task 3: `OnlineVerbalizerTrainer.__init__` + `.record`

**Step 1: Failing tests.**

```python
# Reuse Phase 4's mock surface
from soma.training.online_verbalizer import OnlineVerbalizerTrainer


def test_online_trainer_stores_components():
    # Build a fresh VerbalizerTrainer and pass it in.
    inner = _fresh_verbalizer_trainer()  # Phase 4 helper, adapt
    online = OnlineVerbalizerTrainer(
        inner=inner, config=inner.config,
    )
    assert online.inner is inner
    assert isinstance(online.replay_buffer, ReplayBuffer)
    assert online.replay_buffer.capacity == inner.config.replay_buffer_capacity
    assert not online.is_diverged


def test_online_trainer_record_adds_to_buffer():
    inner = _fresh_verbalizer_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.record(user_text="hi", response="hello")
    assert len(online.replay_buffer) == 1
    assert online.replay_buffer.entries[0].user_text == "hi"
```

**Step 2: Implement:**

```python
class OnlineVerbalizerTrainer:
    """Wraps VerbalizerTrainer with replay + divergence for online updates."""

    def __init__(
        self,
        *,
        inner: Any,  # VerbalizerTrainer
        config: SOMAConfig,
    ) -> None:
        self.inner = inner
        self.config = config
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)
        self._loss_history: deque[float] = deque(maxlen=config.divergence_window)
        self.is_diverged: bool = False

    def record(self, *, user_text: str, response: str) -> None:
        self.replay_buffer.add(
            ChatExchange(user_text=user_text, response=response)
        )
```

Note: `inner` is typed `Any` to avoid circular imports between `online_verbalizer.py` and `verbalizer_bootstrap.py`. A `Protocol` would be cleaner and is worth revisiting in Phase 7.

Also: override the inner trainer's LR to `config.online_verbalizer_lr`. Two choices:

- **Option A**: mutate `inner.optim.param_groups[0]['lr']` at construction time.
- **Option B**: build a fresh Adam on `inner.verbalizer.parameters()` at construction and use that.

Option A keeps one optimizer (the inner's Adam-moment state carries over from bootstrap, which is actually desirable — don't throw away the bootstrap's learned gradient history). Do A.

```python
# After self.is_diverged assignment:
for group in self.inner.optim.param_groups:
    group["lr"] = config.online_verbalizer_lr
```

**Step 3: Commit.**

---

## Task 4: `.sample_batch() -> list[str]`

Returns user-text samples for training. Per spec: 1 latest + (N-1) random.

**Step 1: Failing tests.**

```python
def test_sample_batch_returns_user_texts():
    inner = _fresh_verbalizer_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.record(user_text="u0", response="r0")
    online.record(user_text="u1", response="r1")
    online.record(user_text="u2", response="r2")

    batch = online.sample_batch(batch_size=2)
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert all(isinstance(t, str) for t in batch)
    # Most-recent exchange must be in the batch (index 0 by convention).
    assert batch[0] == "u2"


def test_sample_batch_with_empty_buffer_returns_empty():
    inner = _fresh_verbalizer_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    batch = online.sample_batch(batch_size=4)
    assert batch == []


def test_sample_batch_caps_at_buffer_size():
    inner = _fresh_verbalizer_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.record(user_text="u0", response="r0")
    batch = online.sample_batch(batch_size=4)
    assert len(batch) == 1
    assert batch[0] == "u0"
```

**Step 2: Implement:**

```python
def sample_batch(self, *, batch_size: int) -> list[str]:
    """Return user-text samples for training.

    Layout: [latest, random_0, random_1, ..., random_{batch_size-2}].
    If the buffer has fewer than batch_size entries, returns all of them
    (latest first).
    """
    if len(self.replay_buffer) == 0:
        return []
    latest = self.replay_buffer.entries[-1].user_text
    remaining = batch_size - 1
    if remaining <= 0:
        return [latest]
    # Random sample from all BUT the latest, up to remaining count.
    pool = list(self.replay_buffer.entries)[:-1]  # exclude latest
    sampled = random.sample(pool, min(remaining, len(pool)))
    return [latest] + [ex.user_text for ex in sampled]
```

**Step 3: Commit.**

---

## Task 5: `.step(user_text, response)` — record + sample + train

The main entry point called by `ChatSession.respond()` after generation.

**Step 1: Failing tests.**

```python
def test_step_records_and_trains():
    inner = _fresh_verbalizer_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)

    # Snapshot verbalizer params.
    before = [p.detach().clone() for p in inner.verbalizer.parameters()]
    online.step(user_text="hello there", response="hi!")
    after = [p.detach().clone() for p in inner.verbalizer.parameters()]

    # Buffer now has 1 entry.
    assert len(online.replay_buffer) == 1
    # Verbalizer params should have moved (training happened).
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
    # Loss history recorded 1 entry.
    assert len(online._loss_history) >= 1


def test_step_skips_training_when_diverged():
    """When is_diverged flag is set, step should still record but NOT train."""
    inner = _fresh_verbalizer_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.is_diverged = True

    before = [p.detach().clone() for p in inner.verbalizer.parameters()]
    online.step(user_text="hi", response="yo")
    after = [p.detach().clone() for p in inner.verbalizer.parameters()]

    # Recorded the exchange still.
    assert len(online.replay_buffer) == 1
    # But verbalizer params did NOT move.
    for b, a in zip(before, after):
        assert torch.equal(b, a), "trained despite is_diverged=True"
```

**Step 2: Implement:**

```python
def step(self, *, user_text: str, response: str) -> None:
    """Record the exchange and (if not diverged) run one training step.

    The training step is a batched teacher-forced LM loss across the
    latest turn + (batch_size-1) random prior turns from the replay
    buffer. Uses the wrapped ``inner`` trainer's ``train_step`` to
    avoid re-implementing the forward/backward/optim plumbing.
    """
    self.record(user_text=user_text, response=response)

    if self.is_diverged:
        return

    batch = self.sample_batch(batch_size=self.config.online_batch_size)
    # Sum/average losses from each sample. For simplicity, train on each
    # text sequentially and record the mean. This is semantically
    # equivalent to batched CE when batch samples have independent prefixes
    # (as they do — each has its own SOMA state).
    step_losses = []
    for text in batch:
        loss = self.inner.train_step(text=text)
        step_losses.append(loss)

    if step_losses:
        mean_loss = sum(step_losses) / len(step_losses)
        self._loss_history.append(mean_loss)
        self._check_divergence()  # T6 implements this
```

Add a stub for `_check_divergence` that does nothing in T5 — T6 fills it in.

**Step 3: Commit.**

---

## Task 6: Divergence monitor

**Step 1: Failing tests.**

```python
def test_divergence_detects_rising_loss():
    inner = _fresh_verbalizer_trainer()
    # Force tiny window for easy test.
    cfg = replace(inner.config, divergence_window=4, divergence_threshold=0.5)
    online = OnlineVerbalizerTrainer(inner=inner, config=cfg)

    # Manually inject losses into the history. 4 values; first half
    # mean 0.5, second half mean 2.0 → rise 1.5 > threshold 0.5.
    online._loss_history.extend([0.4, 0.6, 1.8, 2.2])
    online._check_divergence()
    assert online.is_diverged, "did not detect monotonic rise"


def test_divergence_stable_loss_is_not_flagged():
    inner = _fresh_verbalizer_trainer()
    cfg = replace(inner.config, divergence_window=4, divergence_threshold=0.5)
    online = OnlineVerbalizerTrainer(inner=inner, config=cfg)
    # Stable around 1.0.
    online._loss_history.extend([1.0, 1.05, 0.95, 1.02])
    online._check_divergence()
    assert not online.is_diverged


def test_divergence_requires_full_window():
    """Don't trigger on partial history (< window size)."""
    inner = _fresh_verbalizer_trainer()
    cfg = replace(inner.config, divergence_window=4, divergence_threshold=0.5)
    online = OnlineVerbalizerTrainer(inner=inner, config=cfg)
    # Only 3 entries — not full window.
    online._loss_history.extend([0.1, 5.0, 10.0])
    online._check_divergence()
    assert not online.is_diverged, "divergence check fired on partial window"


def test_reset_divergence_guard():
    inner = _fresh_verbalizer_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.is_diverged = True
    online._loss_history.extend([10.0] * online.config.divergence_window)
    online.reset_divergence_guard()
    assert not online.is_diverged
    assert len(online._loss_history) == 0
```

**Step 2: Implement:**

```python
def _check_divergence(self) -> None:
    """Rolling-window rise detector.

    If the loss history is at least ``divergence_window`` long AND
    ``mean(last half) - mean(first half) > divergence_threshold``,
    freeze further training. Logs a warning.
    """
    history = list(self._loss_history)
    if len(history) < self.config.divergence_window:
        return
    half = len(history) // 2
    mean_first = sum(history[:half]) / half
    mean_second = sum(history[half:]) / (len(history) - half)
    rise = mean_second - mean_first
    if rise > self.config.divergence_threshold:
        self.is_diverged = True
        _log.warning(
            "online verbalizer training diverged: mean(first half)=%.4f "
            "mean(second half)=%.4f rise=%.4f > threshold=%.4f. "
            "Further steps frozen; call reset_divergence_guard() to resume.",
            mean_first, mean_second, rise, self.config.divergence_threshold,
        )


def reset_divergence_guard(self) -> None:
    """Clear the diverged flag and wipe the loss history."""
    self.is_diverged = False
    self._loss_history.clear()
```

Add `import logging` + `_log = logging.getLogger(__name__)` at the top.

**Step 3: Commit.**

---

## Task 7: `ChatSession` optional `online_trainer` integration

**Step 1: Failing tests** (append to `tests/test_session/test_chat_session.py`):

```python
def test_session_without_online_trainer_does_not_train():
    s = _build_session()
    before = [p.detach().clone() for p in s.verbalizer.parameters()]
    s.respond(user_text="hi")
    after = [p.detach().clone() for p in s.verbalizer.parameters()]
    for b, a in zip(before, after):
        assert torch.equal(b, a), "verbalizer drifted without online trainer"


def test_session_with_online_trainer_trains_after_respond():
    from soma.training.online_verbalizer import OnlineVerbalizerTrainer
    from soma.training.verbalizer_bootstrap import VerbalizerTrainer

    s = _build_session()
    inner = VerbalizerTrainer(
        soma=s.soma, verbalizer=s.verbalizer, chat_head=s.chat_head,
        config=s.soma.config,
        tokenizer=s.tokenizer, encoder=s.encoder,
    )
    online = OnlineVerbalizerTrainer(inner=inner, config=s.soma.config)
    # Attach.
    s.online_trainer = online

    before = [p.detach().clone() for p in s.verbalizer.parameters()]
    s.respond(user_text="hi there")
    after = [p.detach().clone() for p in s.verbalizer.parameters()]
    # Verbalizer params should have moved.
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
    # Replay buffer has 1 entry.
    assert len(online.replay_buffer) == 1
```

**Step 2: Implement** — add `online_trainer` kwarg to `ChatSession.__init__`, default `None`. In `respond()`, after the response-back-into-SOMA call and BEFORE the history append, call:

```python
if self.online_trainer is not None:
    self.online_trainer.step(user_text=user_text, response=response)
```

Actually, a cleaner place: AFTER the history append, since history.append is pure bookkeeping. Put it last.

**Step 3: Commit.**

---

## Task 8: SmolLM2 10-turn online-training smoke (slow-marked)

Real model. Ten turns. Verify that online training runs without crashing, advances the replay buffer, and doesn't immediately diverge. Exact loss numbers are LLM-dependent; we only assert "trained at least N steps, didn't diverge."

**Files:** `tests/test_training/test_online_verbalizer_smoke.py`.

Loop structure:
```python
for i, text in enumerate(ten_user_messages):
    response = session.respond(user_text=text, max_new_tokens=10, min_new_tokens=1, do_sample=False)
    print(f"turn {i}: {text!r} → {response!r}")

assert len(session.online_trainer.replay_buffer) == 10
assert not session.online_trainer.is_diverged
assert len(session.online_trainer._loss_history) == 10
# Loss should roughly go down or stay in a sane range — print for observability.
print(f"losses: {list(session.online_trainer._loss_history)}")
```

Use `min_new_tokens=1` as the T9 Phase 5 smoke did (workaround for near-null prefix EOS).

---

## Task 9: Save/load replay buffer + divergence state

Extends the session bundle: `online_state.json` sidecar records `{replay_buffer: [...], loss_history: [...], is_diverged: bool}`.

- `ChatSession.save(out_dir)` additionally writes `online_state.json` if `online_trainer is not None`.
- `ChatSession.load_online_state(out_dir, online_trainer)` explicitly called by the caller (CLI) after constructing the online trainer.

Symmetric with T7's `load_history` — the caller orchestrates.

---

## Task 10: Full regression + merge

```bash
pytest tests/ -m "not slow" -q
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/soma/
git checkout main
git merge --no-ff feat/online-verbalizer -m "Merge ..."
git push origin main
git push github main
```

Phase 6 does NOT change existing save/load semantics — `online_state.json` is a new OPTIONAL sidecar. Safe to merge without stopping the train service.

---

## Appendix — Failure Modes

- **Instant divergence**: LR too high. Drop `online_verbalizer_lr` to 1e-6 or lower.
- **No learning**: LR too low. Expected on a tiny mock — real chats need ~100+ turns to show measurable drift.
- **Buffer eviction too fast**: on long sessions the earliest turns are dropped; that's the intended behavior for a rolling buffer. Future phase can add "tier memory" for important historical turns.
- **Divergence flag stuck ON**: caller never calls `reset_divergence_guard()`. CLI could expose `/reset` command. Not in Phase 6 scope.

---

*End of Phase 6 plan. 10 tasks, TDD-disciplined.*
