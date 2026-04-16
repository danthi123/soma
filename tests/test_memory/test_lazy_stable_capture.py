"""Tests for the Phase 13 lazy stable-capture path.

Before Phase 13, ``MemoryLayer.consolidate()`` ran the stable-capture
pass eagerly on every call — O(N_so_far) of SOMA forward passes per
call, dominating session cost at large N even though the output was
unused with the default ``graph_rerank_alpha=0.0``.

Phase 13 moves that pass off the write path. ``consolidate()`` now
only runs the incremental growth pass and flips
``_stable_capture_dirty`` so the next ``retrieve()`` that actually
consumes the graph signal (``alpha > 0``) refreshes the captures on
demand. A public ``stable_capture()`` method lets benchmarks and
warmup loops pay the cost eagerly when they want predictable
retrieve latency.

These tests pin every branch of the dirty-flag state machine:

- consolidate marks dirty but doesn't capture
- alpha=0 retrieve leaves the flag set
- alpha>0 retrieve triggers the capture and clears the flag
- explicit stable_capture() clears the flag
- mutations (store/forget) after consolidate set the flag
"""

from __future__ import annotations

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryLayer
from soma.system import SOMA

CORPUS = [
    "the cat sat on the mat",
    "the dog chased the ball",
    "quantum physics is fascinating",
    "photosynthesis converts sunlight into energy",
]


@pytest.fixture
def embedder() -> tuple[object, TextEncoder]:
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    return tokenizer, encoder


@pytest.fixture
def soma_stack() -> SOMA:
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    return SOMA(config)


def _attach(mem: MemoryLayer, soma: SOMA, embedder: tuple[object, TextEncoder]) -> None:
    tokenizer, encoder = embedder
    mem.attach_soma(soma, tokenizer, encoder)


def _count_recapture_calls(mem: MemoryLayer) -> list[int]:
    """Patch _recapture_activations_stable with a counter; return the
    single-element list that gets incremented on every call."""
    counter = [0]
    real = mem._recapture_activations_stable

    def _counting(dim: int) -> None:
        counter[0] += 1
        real(dim)

    mem._recapture_activations_stable = _counting  # type: ignore[method-assign]
    return counter


def test_consolidate_marks_dirty_but_doesnt_capture(embedder, soma_stack) -> None:
    """consolidate() post-growth sets dirty=True but does NOT run the
    O(N) stable-capture pass. Phase 13's core contract."""
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for t in CORPUS:
        mem.store(t)
    _attach(mem, soma_stack, embedder)
    assert mem._stable_capture_dirty is True  # stores already marked it
    counter = _count_recapture_calls(mem)

    processed = mem.consolidate()

    assert processed == len(CORPUS)
    assert mem._stable_capture_dirty is True, (
        "consolidate() must flip dirty=True post-growth"
    )
    assert counter[0] == 0, (
        f"consolidate() must NOT run stable-capture (got {counter[0]} calls)"
    )


def test_retrieve_with_alpha_zero_leaves_dirty_flag(embedder, soma_stack) -> None:
    """alpha=0 short-circuits the re-rank; stable-capture stays deferred.

    The shipping default pays zero cost on the retrieve path.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.0,
    )
    for t in CORPUS:
        mem.store(t)
    _attach(mem, soma_stack, embedder)
    mem.consolidate()
    assert mem._stable_capture_dirty is True
    counter = _count_recapture_calls(mem)

    hits = mem.retrieve("the cat", k=2)

    assert len(hits) == 2
    assert mem._stable_capture_dirty is True, (
        "alpha=0 retrieve must leave dirty flag set (no cost)"
    )
    assert counter[0] == 0, (
        f"alpha=0 retrieve must NOT run stable-capture (got {counter[0]})"
    )


def test_retrieve_with_alpha_positive_runs_stable_capture(embedder, soma_stack) -> None:
    """alpha>0 retrieve fires stable-capture once, flips dirty=False.

    A second retrieve with no intervening mutation does NOT re-run the
    capture pass.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.5,
    )
    for t in CORPUS:
        mem.store(t)
    _attach(mem, soma_stack, embedder)
    mem.consolidate()
    assert mem._stable_capture_dirty is True
    counter = _count_recapture_calls(mem)

    # First retrieve triggers the lazy capture.
    mem.retrieve("the cat", k=2)
    assert counter[0] == 1, (
        f"alpha>0 retrieve with dirty=True must run capture once, got {counter[0]}"
    )
    assert mem._stable_capture_dirty is False, (
        "stable_capture() must clear the dirty flag"
    )

    # Second retrieve without an intervening mutation — no re-run.
    mem.retrieve("the dog", k=2)
    assert counter[0] == 1, (
        f"second retrieve (clean state) must NOT re-run capture, got {counter[0]}"
    )


def test_disabled_stable_capture_flag_never_dirties(embedder, soma_stack) -> None:
    """With graph_rerank_stable_capture=False, the dirty flag stays False.

    Preserves the pre-Phase-13 opt-out for callers who explicitly
    disabled stable-capture — they get zero new bookkeeping overhead.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_stable_capture=False,
    )
    for t in CORPUS:
        mem.store(t)
    assert mem._stable_capture_dirty is False
    _attach(mem, soma_stack, embedder)
    mem.consolidate()
    assert mem._stable_capture_dirty is False


def test_explicit_stable_capture_flips_dirty(embedder, soma_stack) -> None:
    """Public stable_capture() clears the dirty flag standalone.

    Useful for benchmarks / warmup that want to amortize the cost
    before the first retrieve. A follow-on retrieve (still clean)
    does NOT re-fire the pass.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.5,
    )
    for t in CORPUS:
        mem.store(t)
    _attach(mem, soma_stack, embedder)
    mem.consolidate()
    assert mem._stable_capture_dirty is True
    counter = _count_recapture_calls(mem)

    mem.stable_capture()

    assert mem._stable_capture_dirty is False
    assert counter[0] == 1, (
        f"stable_capture() must run exactly once, got {counter[0]}"
    )

    # A subsequent retrieve at alpha>0 does NOT re-fire.
    mem.retrieve("the cat", k=2)
    assert counter[0] == 1


def test_explicit_stable_capture_is_noop_without_soma(embedder) -> None:
    """stable_capture() without a SOMA attached is a safe no-op.

    Dirty flag stays True (set by store() because the feature is
    enabled by default), but we don't crash or flip the flag — callers
    can invoke this unconditionally in warmup code without guarding.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("some text")
    assert mem._stable_capture_dirty is True
    mem.stable_capture()
    assert mem._stable_capture_dirty is True


def test_explicit_stable_capture_is_noop_when_disabled(embedder, soma_stack) -> None:
    """stable_capture() with graph_rerank_stable_capture=False is a no-op.

    Uses a recapture counter to prove the internal pass never runs
    when the feature flag is off, even with SOMA attached.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_stable_capture=False,
    )
    for t in CORPUS:
        mem.store(t)
    _attach(mem, soma_stack, embedder)
    mem.consolidate()
    counter = _count_recapture_calls(mem)

    mem.stable_capture()

    assert counter[0] == 0, (
        "stable_capture() must short-circuit when feature is disabled"
    )


def test_store_after_consolidate_sets_dirty(embedder, soma_stack) -> None:
    """Mutations after a clean stable-capture re-dirty the flag.

    Captures the full state machine: consolidate → stable_capture
    (dirty=False) → store → dirty=True again → retrieve refreshes.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.5,
    )
    for t in CORPUS:
        mem.store(t)
    _attach(mem, soma_stack, embedder)
    mem.consolidate()
    mem.stable_capture()
    assert mem._stable_capture_dirty is False

    # A store after a clean capture should re-dirty.
    mem.store("a brand-new fact")
    assert mem._stable_capture_dirty is True, (
        "store() post-consolidate must set dirty=True"
    )


def test_forget_after_consolidate_sets_dirty(embedder, soma_stack) -> None:
    """forget() also re-dirties so next retrieve sees a coherent capture."""
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.5,
    )
    ids = [mem.store(t) for t in CORPUS]
    _attach(mem, soma_stack, embedder)
    mem.consolidate()
    mem.stable_capture()
    assert mem._stable_capture_dirty is False

    removed = mem.forget(ids[0])
    assert removed is True
    assert mem._stable_capture_dirty is True, (
        "forget() must dirty the stable-capture flag"
    )


def test_store_batch_sets_dirty(embedder, soma_stack) -> None:
    """store_batch() dirties the flag once for the whole batch."""
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.5,
    )
    mem.store("seed")  # sets dirty=True
    _attach(mem, soma_stack, embedder)
    mem.consolidate()
    mem.stable_capture()
    assert mem._stable_capture_dirty is False

    mem.store_batch(list(CORPUS))
    assert mem._stable_capture_dirty is True


def test_retrieve_without_consolidate_runs_capture_when_dirty(
    embedder, soma_stack
) -> None:
    """store + attach_soma + retrieve (no explicit consolidate) still
    exercises the lazy path — dirty was set by store, retrieve clears it.

    Edge case: a user who never calls consolidate() but sets alpha>0
    should still see the stable-capture pass run once on first
    retrieve, regardless of whether any growth happened. This confirms
    the retrieve hook doesn't conflate "growth just ran" with "capture
    is stale".
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.5,
    )
    for t in CORPUS:
        mem.store(t)
    _attach(mem, soma_stack, embedder)
    counter = _count_recapture_calls(mem)

    # No consolidate() in between. store()s set dirty=True directly.
    assert mem._stable_capture_dirty is True
    mem.retrieve("the cat", k=2)
    assert counter[0] == 1
    assert mem._stable_capture_dirty is False
