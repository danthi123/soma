"""Filter pushdown dispatch tests.

Pins the contract of `retrieve(where=...)`:
- When the backend declares ``supports_filter_pushdown=True`` and
  accepts the translated filter, MemoryLayer delegates.
- When the backend raises ``FilterPushdownUnsupported``, MemoryLayer
  falls back to its Python pre-filter path and returns the same
  results InProc would have.
- Parity across operators supported by ``_matches_where`` — callers
  must get identical hit sets regardless of which path ran.

The pushdown tests use a minimal mock backend that mirrors
InProcBackend's storage but flips ``supports_filter_pushdown=True``
so we can observe the dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from soma.memory.api import MemoryLayer
from soma.memory.backend import FilterPushdownUnsupported
from soma.memory.backends.inproc import InProcBackend


def _hash_embed_factory(dim: int):
    """Deterministic embedder so tests don't depend on sbert/torch weights."""

    import torch

    def _embed(text: str) -> torch.Tensor:
        h = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        return torch.tensor(rng.standard_normal(dim).astype(np.float32))

    return _embed


class _PushdownMockBackend:
    """Wraps an InProcBackend and advertises filter pushdown.

    Accepts a whitelist of field names the test considers ``supported``;
    any other field triggers :class:`FilterPushdownUnsupported` so the
    fallback path gets exercised.
    """

    supports_filter_pushdown = True
    name = "mock-pushdown"

    def __init__(
        self,
        *,
        dim: int,
        supported_fields: set[str],
        metadata_store: dict[str, dict[str, Any]],
    ) -> None:
        self._inner = InProcBackend(dim=dim, faiss_threshold=0)
        self._supported_fields = supported_fields
        self._meta = metadata_store  # MemoryLayer-owned, shared by ref
        self.search_calls: list[dict[str, Any]] = []

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def ntotal(self) -> int:
        return self._inner.ntotal

    def open(self) -> None:
        self._inner.open()

    def close(self) -> None:
        self._inner.close()

    def clear(self) -> None:
        self._inner.clear()

    def add(self, ids, vectors):
        self._inner.add(ids, vectors)

    def remove(self, ids):
        self._inner.remove(ids)

    def get_vectors(self, ids):
        return self._inner.get_vectors(ids)

    def search(self, query, k, *, exclude_ids=None, where=None):
        self.search_calls.append(
            {"k": k, "exclude_ids": exclude_ids, "where": where}
        )
        if where is not None:
            # Check every field in the clause is in the supported set;
            # otherwise declare pushdown unsupported so MemoryLayer
            # drops to the Python pre-filter path.
            for field_name in where:
                if field_name not in self._supported_fields:
                    raise FilterPushdownUnsupported(field=field_name)
            # For fields we "support", emulate pushdown by scoring the
            # entire store then filtering out rows whose metadata does
            # not match (look up via the shared meta store).
            from soma.memory.api import _matches_where

            raw = self._inner.search(query, k=self._inner.ntotal or 1)
            filtered = [
                (nid, score)
                for nid, score in raw
                if _matches_where(self._meta.get(nid, {}), where)
            ]
            return filtered[:k]
        return self._inner.search(query, k=k, exclude_ids=exclude_ids)

    def search_subset(self, query, candidate_ids, k):
        return self._inner.search_subset(query, candidate_ids, k=k)

    def snapshot(self, bundle_dir: Path) -> None:
        self._inner.snapshot(bundle_dir)

    def restore(self, bundle_dir: Path) -> None:
        self._inner.restore(bundle_dir)


@pytest.fixture
def corpus() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("alpha doc one", {"tag": "fiction", "year": 2020}),
        ("beta doc two", {"tag": "nonfiction", "year": 2021}),
        ("gamma doc three", {"tag": "fiction", "year": 2022}),
        ("delta doc four", {"tag": "research", "year": 2023}),
        ("epsilon doc five", {"tag": "fiction", "year": 2024}),
    ]


def _make_mem_with_pushdown(
    corpus: list[tuple[str, dict[str, Any]]],
    *,
    supported_fields: set[str],
) -> tuple[MemoryLayer, _PushdownMockBackend]:
    dim = 16
    embed = _hash_embed_factory(dim)
    # Shared metadata store; the mock backend reads it to simulate
    # server-side filter evaluation.
    meta_store: dict[str, dict[str, Any]] = {}
    backend = _PushdownMockBackend(
        dim=dim, supported_fields=supported_fields, metadata_store=meta_store
    )
    mem = MemoryLayer(
        embed_fn=embed, embed_dim=dim, backend=backend
    )
    for text, meta in corpus:
        nid = mem.store(text, metadata=dict(meta))
        meta_store[nid] = dict(meta)
    return mem, backend


def test_retrieve_with_where_delegates_to_backend_when_supported(
    corpus,
) -> None:
    mem, backend = _make_mem_with_pushdown(corpus, supported_fields={"tag"})
    hits = mem.retrieve("alpha", k=10, where={"tag": "fiction"})
    # Pushdown path was exercised exactly once.
    assert any(call["where"] == {"tag": "fiction"} for call in backend.search_calls)
    assert all(h.metadata["tag"] == "fiction" for h in hits)


def test_retrieve_falls_back_to_python_when_backend_rejects_filter(
    corpus,
) -> None:
    mem, backend = _make_mem_with_pushdown(corpus, supported_fields=set())
    hits = mem.retrieve("alpha", k=10, where={"tag": "fiction"})
    # First call attempted pushdown; backend raised; MemoryLayer
    # recovered and returned correctly-filtered results.
    assert all(h.metadata["tag"] == "fiction" for h in hits)
    # At least one search was attempted with the where dict before
    # the fallback kicked in.
    assert any(call["where"] == {"tag": "fiction"} for call in backend.search_calls)


def test_filter_parity_inproc_vs_mock_pushdown_backend(corpus) -> None:
    """Same corpus, same query, same where clause — results match."""
    dim = 16
    embed = _hash_embed_factory(dim)
    mem_inproc = MemoryLayer(embed_fn=embed, embed_dim=dim)
    for text, meta in corpus:
        mem_inproc.store(text, metadata=dict(meta))

    mem_pushdown, _ = _make_mem_with_pushdown(
        corpus, supported_fields={"tag", "year"}
    )

    for where in [
        {"tag": "fiction"},
        {"tag": {"$ne": "fiction"}},
        {"year": {"$gte": 2022}},
        {"tag": {"$in": ["fiction", "research"]}},
    ]:
        a = {h.node_id for h in mem_inproc.retrieve("doc", k=10, where=where)}
        b_results = mem_pushdown.retrieve("doc", k=10, where=where)
        # The pushdown mock uses a different id namespace — compare
        # by (text, metadata) instead of node_id.
        a_keys = {
            (h.text, tuple(sorted(h.metadata.items())))
            for h in mem_inproc.retrieve("doc", k=10, where=where)
        }
        b_keys = {
            (h.text, tuple(sorted(h.metadata.items())))
            for h in b_results
        }
        assert a_keys == b_keys, f"parity failed for where={where}"
        # ``a`` used to keep pyflakes happy; the real comparison is the
        # text+metadata tuple above because pushdown and inproc
        # MemoryLayers assign independent uuid4 node ids.
        del a


def test_inproc_backend_never_takes_pushdown_path() -> None:
    """Default InProcBackend declares supports_filter_pushdown=False
    so MemoryLayer always runs the Python pre-filter path."""
    dim = 16
    mem = MemoryLayer(embed_fn=_hash_embed_factory(dim), embed_dim=dim)
    assert mem._backend.supports_filter_pushdown is False
    mem.store("alpha", metadata={"tag": "a"})
    mem.store("beta", metadata={"tag": "b"})
    hits = mem.retrieve("anything", k=5, where={"tag": "a"})
    assert len(hits) == 1
    assert hits[0].metadata == {"tag": "a"}
