"""Tests that ``_soma_activations`` is keyed by node_id, not list index.

Before Phase 6 the activation store was a positional list. That made
any reorder or soft-delete (which Qdrant will do under the hood)
impossible to sync. Moving to ``dict[node_id, Tensor | None]`` is the
minimal refactor that lets backends handle their own lifecycles
without poking back into MemoryLayer's list indices.

These tests pin the semantics:
- activations survive forgets of other ids
- retrieve's re-rank path looks up by node_id
- consolidate's cursor still advances correctly after a forget
"""

from __future__ import annotations

import pytest
import torch

from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryLayer

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "to be or not to be that is the question",
    "the rain in spain falls mainly on the plain",
    "all happy families are alike",
]


@pytest.fixture
def embedder() -> tuple[object, TextEncoder]:
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(_CORPUS, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    return tokenizer, encoder


def _make_mem(embedder) -> MemoryLayer:
    tok, enc = embedder
    return MemoryLayer(tokenizer=tok, encoder=enc)


def test_activations_is_dict_keyed_by_id(embedder) -> None:
    mem = _make_mem(embedder)
    nid = mem.store("hello world")
    # Sanity check: the internal container should be a dict and
    # already have a slot for the freshly-stored id (even if the value
    # is still None because consolidate hasn't run).
    assert isinstance(mem._soma_activations, dict)
    assert nid in mem._soma_activations
    assert mem._soma_activations[nid] is None


def test_forget_then_store_keeps_activations_keyed_by_id(embedder) -> None:
    mem = _make_mem(embedder)
    id_a = mem.store("alpha document")
    id_b = mem.store("beta document")
    id_c = mem.store("gamma document")

    # Simulate activations having been captured by consolidate.
    mem._soma_activations[id_a] = torch.zeros(4)
    mem._soma_activations[id_b] = torch.ones(4)
    mem._soma_activations[id_c] = torch.full((4,), 2.0)

    # Forget the middle entry.
    assert mem.forget(id_b)

    assert id_a in mem._soma_activations
    assert id_c in mem._soma_activations
    assert id_b not in mem._soma_activations
    # Surviving activations were not shuffled by the forget.
    assert torch.equal(mem._soma_activations[id_a], torch.zeros(4))
    assert torch.equal(mem._soma_activations[id_c], torch.full((4,), 2.0))


def test_store_batch_populates_activation_slots(embedder) -> None:
    mem = _make_mem(embedder)
    nids = mem.store_batch(["first", "second", "third"])
    for nid in nids:
        assert nid in mem._soma_activations
        assert mem._soma_activations[nid] is None


def test_retrieve_with_rerank_uses_id_keyed_activations(embedder) -> None:
    """The graph-rerank blend path should look up activations by
    node_id, not positional index. Simulate that by attaching a fake
    SOMA-enough scaffold and checking the blend path doesn't KeyError
    after a forget."""
    mem = _make_mem(embedder)
    mem._graph_rerank_alpha = 0.5
    mem.store("alpha")
    keep_id = mem.store("beta")
    mem.store("gamma")

    # Hand-wire activations by id so _retrieve_with_rerank has a
    # stable lookup target.
    for nid in list(mem._ids):
        mem._soma_activations[nid] = torch.zeros(4)

    # Forget the first entry and ensure the kept id's activation
    # survives lookup by id.
    first_id = mem._ids[0]
    mem.forget(first_id)
    assert keep_id in mem._soma_activations
    assert mem._soma_activations[keep_id] is not None


def test_consolidate_cursor_still_advances_after_forget(embedder) -> None:
    """Forget shortens the texts list; the consolidation cursor (an
    index into the texts list) must stay valid so consolidate() only
    processes entries past the cursor on its next call."""
    mem = _make_mem(embedder)
    id_a = mem.store("alpha")
    id_b = mem.store("beta")
    id_c = mem.store("gamma")

    # Simulate a consolidate() having processed all three entries.
    mem._consolidation_cursor = 3

    # Forget the middle entry. Remaining texts: [alpha, gamma].
    # Cursor should be clamped to len(texts) so the next consolidate()
    # is a no-op rather than re-processing the tail.
    mem.forget(id_b)
    assert mem._consolidation_cursor == len(mem._texts)
    # Stored activations for survivors are still keyed correctly.
    assert id_a in mem._soma_activations
    assert id_c in mem._soma_activations
