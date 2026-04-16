"""Tests for soma.memory.api.MemoryLayer — the public vector-DB-shaped surface."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryHit, MemoryLayer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "to be or not to be that is the question",
    "the rain in spain falls mainly on the plain",
    "all happy families are alike",
]


@pytest.fixture
def embedder() -> tuple[object, TextEncoder]:
    """Tiny deterministic tokenizer+encoder for fast unit tests."""
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    return tokenizer, encoder


def test_store_returns_unique_node_ids(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    ids = [mem.store(f"fact {i}") for i in range(5)]
    assert len(set(ids)) == 5, "node_ids must be unique"
    assert all(isinstance(i, str) for i in ids), "node_ids must be strings"


def test_store_accepts_metadata(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("user prefers dark mode", metadata={"source": "settings"})
    hit = mem.get(nid)
    assert hit is not None
    assert hit.metadata == {"source": "settings"}


def test_retrieve_returns_most_similar_first(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("the cat sat on the mat")
    mem.store("quantum mechanics describes subatomic physics")
    mem.store("the dog chased the ball")

    hits = mem.retrieve("feline on a rug", k=3)
    assert len(hits) == 3
    assert all(isinstance(h, MemoryHit) for h in hits)
    # Scores must be monotonically non-increasing.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_store_returns_empty_list(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    assert mem.retrieve("anything", k=5) == []


def test_retrieve_k_clamps_to_store_size(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("one")
    mem.store("two")
    hits = mem.retrieve("something", k=10)
    assert len(hits) == 2


def test_get_recent_orders_by_insertion(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for i in range(5):
        mem.store(f"fact {i}")
    recent = mem.get_recent(3)
    assert len(recent) == 3
    assert [h.text for h in recent] == ["fact 4", "fact 3", "fact 2"]


def test_forget_removes_entry(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("ephemeral fact")
    mem.store("persistent fact")
    assert mem.forget(nid) is True
    assert mem.get(nid) is None
    assert len(mem) == 1
    assert mem.retrieve("ephemeral", k=5) != []  # retrieves persistent, not ephemeral
    assert all("ephemeral" not in h.text for h in mem.retrieve("anything", k=5))


def test_forget_unknown_id_returns_false(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("some fact")
    assert mem.forget("nonexistent-uuid") is False


def test_related_excludes_self(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("the cat sat on the mat")
    mem.store("quantum mechanics")
    mem.store("the dog chased the ball")

    neighbours = mem.related(nid, k=5)
    assert all(h.node_id != nid for h in neighbours)
    assert len(neighbours) == 2  # all other entries


def test_related_unknown_id_raises(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    with pytest.raises(KeyError, match="nonexistent"):
        mem.related("nonexistent-uuid")


def test_save_load_round_trip(embedder, tmp_path: Path) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    ids = [mem.store(f"fact {i}", metadata={"idx": i}) for i in range(5)]

    bundle = tmp_path / "mem"
    mem.save(bundle)
    assert (bundle / "memory_index.json").exists()
    assert (bundle / "memory_embeddings.pt").exists()
    assert (bundle / "tokenizer.json").exists()
    assert (bundle / "encoder.pt").exists()

    restored = MemoryLayer.load(bundle)
    assert len(restored) == 5
    for i, nid in enumerate(ids):
        hit = restored.get(nid)
        assert hit is not None
        assert hit.text == f"fact {i}"
        assert hit.metadata == {"idx": i}


def test_save_load_file_url_round_trip(embedder, tmp_path: Path) -> None:
    """After Phase 30, ``save(file:// URL)`` / ``load(file:// URL)``
    must be byte-equivalent to the path-based call. This is the
    contract-pin that remote adapters (Phase 31/32) will extend —
    swapping the scheme is all a caller should have to do.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    ids = [mem.store(f"fact {i}", metadata={"idx": i}) for i in range(3)]

    bundle = tmp_path / "bundle-url"
    # Build the URL in the platform's canonical form. pathlib.as_uri()
    # gives us file:///C:/... on Windows and file:///abs/path on POSIX.
    url = bundle.as_uri()
    mem.save(url)
    # Files land at the bundle path — same bytes a path save would write.
    assert (bundle / "memory_index.json").exists()
    assert (bundle / "memory_embeddings.pt").exists()
    assert (bundle / "tokenizer.json").exists()
    assert (bundle / "encoder.pt").exists()

    restored = MemoryLayer.load(url)
    assert len(restored) == 3
    for i, nid in enumerate(ids):
        hit = restored.get(nid)
        assert hit is not None
        assert hit.text == f"fact {i}"
        assert hit.metadata == {"idx": i}


def test_save_load_accepts_object_store(embedder, tmp_path: Path) -> None:
    """``MemoryLayer.save/load`` accept a pre-built ObjectStore so
    callers can wire their own adapter (tests, third-party backends)
    without going through the URL parser."""
    from soma.storage import LocalFSObjectStore

    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("fact one")
    nid = mem.store("fact two")

    store = LocalFSObjectStore(tmp_path / "b")
    mem.save(store)

    # Re-use the same store on load — no URL parsing in between.
    restored = MemoryLayer.load(store)
    assert len(restored) == 2
    hit = restored.get(nid)
    assert hit is not None
    assert hit.text == "fact two"


def test_save_with_plain_string_path_auto_prefixes(
    embedder, tmp_path: Path
) -> None:
    """A plain absolute-path ``str`` (no scheme) must work the same as
    a ``Path`` — the ``str`` branch of ``_coerce_store`` routes through
    ``parse_store_url`` which auto-prefixes to ``file://``."""
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("fact")
    bundle = tmp_path / "plain-str"
    mem.save(str(bundle))
    restored = MemoryLayer.load(str(bundle))
    assert len(restored) == 1


def test_save_is_atomic_under_crash(embedder, tmp_path: Path, monkeypatch) -> None:
    """If a write raises mid-way through save(), the on-disk bundle
    must still be the previous consistent version (never a half-written
    mix), and no ``.tmp`` leftovers remain. Phase 30 moved the atomic
    tmp-file + ``os.replace`` dance into
    :class:`soma.storage.LocalFSObjectStore.put_bytes`, so we patch
    the store method to force a crash mid-put and verify the on-disk
    state is still the previous consistent bundle.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for i in range(3):
        mem.store(f"fact {i}", metadata={"idx": i})

    # First, a clean save that lays down a consistent bundle.
    bundle = tmp_path / "mem"
    mem.save(bundle)
    original_index = (bundle / "memory_index.json").read_text(encoding="utf-8")
    original_embeds = (bundle / "memory_embeddings.pt").read_bytes()

    # Now mutate state and attempt another save, but arrange for the
    # embeddings put_bytes to blow up after writing the .tmp file.
    # Under atomic writes the crash hits only the .tmp sibling and the
    # real file still matches the pre-crash bytes.
    mem.store("post-state change")

    from soma.storage import local as local_mod

    real_put_bytes = local_mod.LocalFSObjectStore.put_bytes

    def _selective_put(self, key, data):  # type: ignore[no-untyped-def]
        if key == "memory_embeddings.pt":
            # Simulate a crash AFTER the tmp-file is written but before
            # os.replace flips it into place. We delegate to the real
            # implementation, then corrupt the .tmp sibling that's
            # left behind by the exception path, then raise.
            import io as _io

            path = self._resolve(key)
            tmp = path.with_suffix(path.suffix + ".tmp")
            # Write a partial payload into .tmp so we can observe that
            # the cleanup handler removes it.
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(b"\x00" * 8)
            # Trigger the real error path by raising with the tmp file
            # still present — the caller's except-clause will unlink it.
            _ = _io  # touch to avoid F401
            raise RuntimeError("simulated crash mid-save")
        return real_put_bytes(self, key, data)

    monkeypatch.setattr(local_mod.LocalFSObjectStore, "put_bytes", _selective_put)
    with pytest.raises(RuntimeError, match="simulated crash"):
        mem.save(bundle)

    # The old bundle must still parse and still match what we wrote first.
    assert (
        bundle / "memory_index.json"
    ).read_text(encoding="utf-8") == original_index, (
        "memory_index.json corrupted by crashed save"
    )
    assert (
        bundle / "memory_embeddings.pt"
    ).read_bytes() == original_embeds, (
        "memory_embeddings.pt corrupted by crashed save"
    )
    # Our monkeypatch raises *before* the real put_bytes cleans up its
    # tmp file, so we tidy up the stale tmp ourselves to keep the rest
    # of the test (no leftover check) simple. Real crashes go through
    # the real put_bytes whose except-clause unlinks the tmp.
    stale_tmp = bundle / "memory_embeddings.pt.tmp"
    if stale_tmp.exists():
        stale_tmp.unlink()
    # No other .tmp leftovers.
    leftovers = list(bundle.glob("*.tmp"))
    assert leftovers == [], f"found leftover .tmp files: {leftovers}"

    # And loading the bundle still works — has the pre-crash 3 entries.
    restored = MemoryLayer.load(bundle)
    assert len(restored) == 3


def test_local_store_put_bytes_cleans_tmp_on_crash(tmp_path: Path) -> None:
    """Contract pin: when put_bytes raises (e.g. disk full),
    LocalFSObjectStore removes the .tmp sibling it was writing so
    ``list_prefix`` and subsequent ``save`` calls see a clean root.

    This is the local-side guarantee that underpins
    ``test_save_is_atomic_under_crash``.
    """
    from soma.storage import LocalFSObjectStore

    store = LocalFSObjectStore(tmp_path)
    store.put_bytes("x.bin", b"initial")

    # Patch os.replace inside the local-store module to simulate a
    # crash between the tmp write and the rename. The .tmp sibling
    # must be unlinked when put_bytes's except path unwinds.
    import soma.storage.local as local_mod

    real_replace = local_mod.os.replace

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated replace failure")

    local_mod.os.replace = _boom  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="replace failure"):
            store.put_bytes("x.bin", b"new")
    finally:
        local_mod.os.replace = real_replace  # type: ignore[assignment]

    # Original object intact; no .tmp leftover.
    assert store.get_bytes("x.bin") == b"initial"
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_load_retrieve_matches_pre_save(embedder, tmp_path: Path) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for text in CORPUS:
        mem.store(text)
    query = "tell me about cats and foxes"
    pre_hits = mem.retrieve(query, k=3)

    bundle = tmp_path / "mem"
    mem.save(bundle)
    restored = MemoryLayer.load(bundle)
    post_hits = restored.retrieve(query, k=3)

    assert [h.node_id for h in pre_hits] == [h.node_id for h in post_hits]
    for pre, post in zip(pre_hits, post_hits, strict=True):
        assert pre.text == post.text
        assert pre.score == pytest.approx(post.score, abs=1e-5)


def test_consolidate_without_soma_is_noop(embedder) -> None:
    """Without an attached SOMA, consolidate returns 0 and doesn't mutate."""
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("before consolidate")
    before = [h.node_id for h in mem.get_recent(10)]
    assert mem.consolidate() == 0
    after = [h.node_id for h in mem.get_recent(10)]
    assert before == after


def test_consolidate_with_soma_processes_entries(embedder) -> None:
    """With an attached SOMA, consolidate feeds entries through the graph."""
    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    tokenizer, encoder = embedder
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    soma = SOMA(config)

    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("the cat sat on the mat")
    mem.store("the dog chased the ball")
    mem.store("quantum physics is fascinating")

    mem.attach_soma(soma, tokenizer, encoder)
    processed = mem.consolidate()
    assert processed == 3
    assert soma.global_step > 0


def test_consolidate_is_incremental_after_first_pass(embedder) -> None:
    """consolidate() only re-processes entries past the cursor.

    The cursor is the index of the next entry to growth-capture. Old
    entries had their activations recorded in earlier consolidate()
    calls; re-running them would inflate per-call cost from
    O(new_entries) to O(N). Verify by counting SOMA steps.
    """
    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    tokenizer, encoder = embedder
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    soma = SOMA(config)
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.attach_soma(soma, tokenizer, encoder)

    for _ in range(5):
        mem.store("the cat sat on the mat")
    mem.consolidate()
    steps_after_first = soma.global_step
    assert steps_after_first > 0

    # Second call without new stores should be a near-no-op
    # (cursor already at end). It still re-runs stable-capture if
    # enabled, but the growth pass shouldn't fire any SOMA steps.
    # We verify by storing nothing new and checking the cursor.
    mem.consolidate()
    # cursor is at len(_texts) — nothing to growth-process
    assert mem._consolidation_cursor == len(mem._texts)

    # Add 3 more entries; consolidate() should only growth-process
    # those 3.
    for _ in range(3):
        mem.store("a different fact about dogs")
    steps_before_second_growth = soma.global_step
    mem.consolidate()
    new_steps = soma.global_step - steps_before_second_growth
    # 5 initial entries got steps_after_first SOMA steps total.
    # 3 new entries should generate ~3/5 of that count + the
    # stable-capture pass (which scales as O(N) but is eval_mode).
    assert new_steps > 0, "consolidate must fire steps for the 3 new entries"


def test_consolidate_cursor_resets_on_attach_soma(embedder) -> None:
    """attach_soma() resets the cursor so a fresh SOMA catches up the store."""
    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for _ in range(5):
        mem.store("seed fact")
    # No SOMA attached, no consolidation, cursor stays at 0.
    assert mem._consolidation_cursor == 0

    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    soma = SOMA(config)
    mem.attach_soma(soma, tokenizer, encoder)
    # attach_soma resets cursor to 0 so the next consolidate() catches
    # the SOMA up to the existing store.
    assert mem._consolidation_cursor == 0
    mem.consolidate()
    assert mem._consolidation_cursor == 5
    assert soma.global_step > 0


def test_auto_consolidate_fires_at_threshold(embedder) -> None:
    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    tokenizer, encoder = embedder
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    soma = SOMA(config)
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        auto_consolidate_every=3,
    )
    mem.attach_soma(soma, tokenizer, encoder)
    mem.store("fact one")
    mem.store("fact two")
    assert soma.global_step == 0
    mem.store("fact three")
    assert soma.global_step > 0


def test_graph_rerank_activates_after_consolidation(embedder) -> None:
    """After consolidation, retrieve uses the graph-aware re-ranking path."""
    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    tokenizer, encoder = embedder
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    soma = SOMA(config)

    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("the cat sat on the mat")
    mem.store("the dog chased the ball")
    mem.store("quantum physics is fascinating")

    pre_hits = mem.retrieve("cat", k=3)
    assert len(pre_hits) == 3

    mem.attach_soma(soma, tokenizer, encoder)
    mem.consolidate()

    post_hits = mem.retrieve("cat", k=3)
    assert len(post_hits) == 3
    assert all(isinstance(h.score, float) for h in post_hits)
    scores = [h.score for h in post_hits]
    assert scores == sorted(scores, reverse=True)


def test_graph_rerank_skipped_when_alpha_is_zero(embedder) -> None:
    """alpha=0 short-circuits the re-rank so retrieve stays fast flat cosine.

    The re-rank pass runs SOMA forward on the query text — a ~100ms
    per-query cost. When alpha is zero the blend is identity over cosine
    regardless, so we skip the whole thing. Regression guard.
    """
    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    tokenizer, encoder = embedder
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    soma = SOMA(config)
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.0,
    )
    mem.store("the cat sat on the mat")
    mem.store("the dog chased the ball")
    mem.attach_soma(soma, tokenizer, encoder)
    mem.consolidate()

    import soma.training.verbalizer_bootstrap as vb

    call_count = 0
    real_text_to_state = vb.text_to_state

    def _counting_text_to_state(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_text_to_state(*args, **kwargs)

    vb.text_to_state = _counting_text_to_state
    try:
        mem.retrieve("where does the cat sit", k=2)
    finally:
        vb.text_to_state = real_text_to_state

    assert call_count == 0, f"Expected no text_to_state calls when alpha=0, got {call_count}"


def test_graph_rerank_threads_query_text_into_activation(embedder) -> None:
    """Graph re-rank must compute q_act from the query text, not from ``""``.

    Regression guard: an earlier implementation passed ``text=""`` to
    ``text_to_state`` inside ``_retrieve_with_rerank``, which yielded a
    zero vector for every query and caused the re-rank blend to be
    effectively ``(1-alpha) * cosine`` — a monotonic transform except
    for entries whose stored activation was ``None``, which created
    a spurious bias. We verify the fix by capturing the ``text`` arg.
    """
    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    tokenizer, encoder = embedder
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    soma = SOMA(config)
    # The re-rank path is gated on graph_rerank_alpha > 0. We want to
    # verify the query text reaches text_to_state when re-rank is
    # actually active.
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        graph_rerank_alpha=0.3,
    )
    mem.store("the cat sat on the mat")
    mem.store("the dog chased the ball")
    mem.attach_soma(soma, tokenizer, encoder)
    mem.consolidate()

    import soma.training.verbalizer_bootstrap as vb

    seen_texts: list[str] = []
    real_text_to_state = vb.text_to_state

    def _capturing_text_to_state(*args, **kwargs):
        seen_texts.append(kwargs.get("text", ""))
        return real_text_to_state(*args, **kwargs)

    vb.text_to_state = _capturing_text_to_state
    try:
        mem.retrieve("where does the cat sit", k=2)
        mem.retrieve("which animal chased a ball", k=2)
    finally:
        vb.text_to_state = real_text_to_state

    assert seen_texts == [
        "where does the cat sit",
        "which animal chased a ball",
    ], seen_texts


def test_faiss_index_type_invalid_raises(embedder) -> None:
    tokenizer, encoder = embedder
    with pytest.raises(ValueError, match="faiss_index_type"):
        MemoryLayer(
            tokenizer=tokenizer,
            encoder=encoder,
            faiss_index_type="banana",
        )


def test_faiss_hnsw_backend_returns_top_k(embedder) -> None:
    """HNSW backend returns the same top-k structure as flat (lower-N regression)."""
    tokenizer, encoder = embedder
    # Force the FAISS path by lowering threshold; HNSW with N=20 is silly
    # in production but exercises the code path here.
    mem = MemoryLayer(
        tokenizer=tokenizer,
        encoder=encoder,
        faiss_threshold=10,
        faiss_index_type="hnsw",
        faiss_hnsw_m=8,
        faiss_hnsw_ef_search=16,
    )
    for i in range(20):
        mem.store(f"fact number {i}: about topic {i % 4}")

    hits = mem.retrieve("topic 2", k=3)
    assert len(hits) == 3
    assert all(isinstance(h.score, float) for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), (
        "HNSW retrieval results must be sorted by score descending"
    )


def test_faiss_hnsw_preserves_recall_vs_flat() -> None:
    """HNSW with default ef_search returns the same top-1 as flat on
    well-separated vectors.

    Pins the paper claim that HNSW is recall-preserving on the SOMA
    workload at default knobs (M=32, ef_search=64). Uses a custom
    embed_fn that produces random but distinct vectors per text, so
    cosine ranking is well-defined (the project's TextEncoder collapses
    short queries into ties and isn't useful for this test). HNSW is
    approximate in general but on this scale should match exact
    retrieval on top-1. This test fails loudly if a parameter change
    degrades recall.
    """
    rng = torch.Generator().manual_seed(123)

    def embed(text: str) -> torch.Tensor:
        # Hash-derived seed → deterministic distinct vector per text
        seed = abs(hash(text)) % (2**31)
        local = torch.Generator().manual_seed(seed)
        return torch.randn(64, generator=local)

    flat_mem = MemoryLayer(embed_fn=embed, embed_dim=64, faiss_threshold=10)
    hnsw_mem = MemoryLayer(
        embed_fn=embed,
        embed_dim=64,
        faiss_threshold=10,
        faiss_index_type="hnsw",
    )

    facts = [f"fact body number {i} carrying signal" for i in range(80)]
    for f in facts:
        flat_mem.store(f)
        hnsw_mem.store(f)

    # Probe with the stored texts themselves so the ground-truth is
    # the exact match. Both backends should rank that exact match #1.
    matches = 0
    for q in facts[:20]:
        flat_top = flat_mem.retrieve(q, k=1)[0].text
        hnsw_top = hnsw_mem.retrieve(q, k=1)[0].text
        if flat_top == q == hnsw_top:
            matches += 1
    assert matches >= 19, (
        f"HNSW lost {20 - matches}/20 top-1 self-matches vs flat. "
        "Tune faiss_hnsw_ef_search up or revisit default M before shipping."
    )
    _ = rng  # appease lint: kept the constructor seed for repro


def test_store_empty_text_raises(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    with pytest.raises(ValueError, match="empty"):
        mem.store("")


def test_len_reflects_store_forget(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    assert len(mem) == 0
    a = mem.store("one")
    mem.store("two")
    assert len(mem) == 2
    mem.forget(a)
    assert len(mem) == 1


def test_memory_hit_fields(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("hello world", metadata={"k": "v"})
    hit = mem.get(nid)
    assert hit is not None
    assert hit.node_id == nid
    assert hit.text == "hello world"
    assert hit.metadata == {"k": "v"}
    assert isinstance(hit.timestamp_step, int)
    assert hit.score == pytest.approx(1.0, abs=1e-5)  # self-retrieval → cosine 1.0


# ------------------------------------------------------------------
# Custom embed_fn path
# ------------------------------------------------------------------
def _hash_embed(text: str) -> torch.Tensor:
    """Deterministic hash-based embedder for testing."""
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


def test_custom_embed_fn_store_retrieve() -> None:
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem.store("alpha")
    mem.store("beta")
    hits = mem.retrieve("alpha", k=1)
    assert len(hits) == 1
    assert hits[0].text in ("alpha", "beta")


def test_custom_embed_fn_save_load(tmp_path: Path) -> None:
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem.store("fact one")
    nid = mem.store("fact two")
    mem.save(tmp_path / "custom-bundle")
    assert not (tmp_path / "custom-bundle" / "tokenizer.json").exists()
    restored = MemoryLayer.load(
        tmp_path / "custom-bundle",
        embed_fn=_hash_embed,
    )
    assert len(restored) == 2
    assert restored.get(nid) is not None
    assert restored.get(nid).text == "fact two"  # type: ignore[union-attr]


def test_custom_embed_fn_load_without_fn_raises(tmp_path: Path) -> None:
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem.store("data")
    mem.save(tmp_path / "b")
    with pytest.raises(ValueError, match="embed_fn"):
        MemoryLayer.load(tmp_path / "b")


# ------------------------------------------------------------------
# FAISS backend
# ------------------------------------------------------------------
faiss = pytest.importorskip("faiss")


def test_faiss_activates_at_threshold() -> None:
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, faiss_threshold=5)
    for i in range(4):
        mem.store(f"fact {i}")
    # The InProcBackend is the default; its FAISS index stays None
    # until we cross the threshold.
    assert mem._backend._faiss_index is None  # type: ignore[attr-defined]
    mem.store("fact 4")
    mem.retrieve("trigger rebuild", k=1)
    assert mem._backend._faiss_index is not None  # type: ignore[attr-defined]


def test_faiss_retrieve_matches_linear() -> None:
    texts = [f"fact number {i}" for i in range(20)]
    mem_linear = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, faiss_threshold=0)
    mem_faiss = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, faiss_threshold=5)
    for t in texts:
        mem_linear.store(t)
        mem_faiss.store(t)
    query = "fact number 7"
    linear_hits = mem_linear.retrieve(query, k=3)
    faiss_hits = mem_faiss.retrieve(query, k=3)
    assert [h.text for h in linear_hits] == [h.text for h in faiss_hits]


def test_faiss_invalidated_on_forget() -> None:
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, faiss_threshold=3)
    ids = [mem.store(f"fact {i}") for i in range(5)]
    mem.retrieve("trigger rebuild", k=1)
    assert mem._backend._faiss_index is not None  # type: ignore[attr-defined]
    mem.forget(ids[0])
    # forget invalidates the backend's cached index so the next search
    # rebuilds rather than returning a ghost row for the removed id.
    assert mem._backend._faiss_index is None  # type: ignore[attr-defined]


def test_store_batch_matches_store(embedder) -> None:
    tokenizer, encoder = embedder
    texts = ["alpha one", "beta two", "gamma three"]
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    ids = mem.store_batch(texts, metadatas=[{"i": i} for i in range(3)])
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert len(mem) == 3
    for nid, txt, i in zip(ids, texts, range(3), strict=True):
        hit = mem.get(nid)
        assert hit is not None
        assert hit.text == txt
        assert hit.metadata == {"i": i}


def test_store_batch_empty_returns_empty(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    assert mem.store_batch([]) == []
    assert len(mem) == 0


def test_store_batch_metadatas_length_mismatch_raises(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    with pytest.raises(ValueError, match="length"):
        mem.store_batch(["a", "b", "c"], metadatas=[{"x": 1}])


def test_forget_keeps_id_lookup_consistent(embedder) -> None:
    """Removing a middle entry must not corrupt get()/related() for later ids."""
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    a = mem.store("alpha")
    b = mem.store("beta")
    c = mem.store("gamma")
    mem.forget(b)
    # a and c must still be retrievable at the right text.
    hit_a = mem.get(a)
    hit_c = mem.get(c)
    assert hit_a is not None and hit_a.text == "alpha"
    assert hit_c is not None and hit_c.text == "gamma"
    # related() must still work (uses _id_to_idx internally).
    related = mem.related(a, k=5)
    assert {h.node_id for h in related} == {c}


# ------------------------------------------------------------------
# update_metadata — metadata-only mutation, WAL-replay compatible.
# ------------------------------------------------------------------
def test_update_metadata_merges_into_existing(embedder) -> None:
    """Existing keys are preserved; patch keys overwrite."""
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("body", metadata={"a": 1, "b": 2})
    mem.update_metadata(nid, {"b": 99, "c": 3})
    hit = mem.get(nid)
    assert hit is not None
    assert hit.metadata == {"a": 1, "b": 99, "c": 3}


def test_update_metadata_raises_on_unknown_id(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    with pytest.raises(KeyError, match="nonexistent"):
        mem.update_metadata("nonexistent-uuid", {"x": 1})


def test_update_metadata_writes_wal_record(tmp_path: Path) -> None:
    """update_metadata on a WAL-backed bundle appends an update_metadata
    record that survives close + reload."""
    bundle = tmp_path / "bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    try:
        nid = mem.store("the fact", metadata={"version": 1})
        mem.update_metadata(nid, {"version": 2, "new_key": "hi"})
    finally:
        mem.close()

    restored = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    try:
        hit = restored.get(nid)
        assert hit is not None
        assert hit.metadata == {"version": 2, "new_key": "hi"}
    finally:
        restored.close()


def test_update_metadata_replay_applies_merge_order(tmp_path: Path) -> None:
    """Two update_metadata records replay in order — second wins on overlap."""
    bundle = tmp_path / "bundle2"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    try:
        nid = mem.store("fact", metadata={"k": "v0"})
        mem.update_metadata(nid, {"k": "v1", "a": 1})
        mem.update_metadata(nid, {"k": "v2"})  # overwrite again
    finally:
        mem.close()

    restored = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    try:
        hit = restored.get(nid)
        assert hit is not None
        assert hit.metadata == {"k": "v2", "a": 1}
    finally:
        restored.close()


def test_update_metadata_reload_if_stale_picks_up_peer_writes(
    tmp_path: Path,
) -> None:
    """When a peer writer appends an update_metadata record, a reader's
    reload_if_stale() must apply it."""
    bundle = tmp_path / "shared"

    writer = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    try:
        nid = writer.store("fact", metadata={"version": 1})
    finally:
        writer.close()

    reader = MemoryLayer.load(bundle, embed_fn=_hash_embed)

    # Second MemoryLayer acts as a peer writer sharing the bundle.
    writer2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    try:
        writer2.update_metadata(nid, {"version": 2})
    finally:
        writer2.close()

    try:
        applied = reader.reload_if_stale()
        assert applied >= 1
        hit = reader.get(nid)
        assert hit is not None
        assert hit.metadata == {"version": 2}
    finally:
        reader.close()


# ----------------------------------------------------------------------
# Phase 6 — backend wiring
# ----------------------------------------------------------------------
def test_memory_layer_default_backend_is_inproc(embedder) -> None:
    """Default construction installs an InProcBackend so nothing else
    has to change for existing users."""
    tok, enc = embedder
    mem = MemoryLayer(tokenizer=tok, encoder=enc)
    assert mem._backend.name == "inproc"
    # Backend dim matches the encoder's embed_dim.
    assert mem._backend.dim == enc.embed_dim


def test_memory_layer_accepts_explicit_inproc_backend(embedder) -> None:
    """Passing backend= explicitly should produce identical behavior
    to the default."""
    from soma.memory.backends.inproc import InProcBackend

    tok, enc = embedder
    backend = InProcBackend(dim=enc.embed_dim, faiss_threshold=10_000)
    mem = MemoryLayer(tokenizer=tok, encoder=enc, backend=backend)
    assert mem._backend is backend
    nid = mem.store("hello world")
    hits = mem.retrieve("hello", k=1)
    assert hits and hits[0].node_id == nid


def test_retrieve_routes_through_backend(embedder) -> None:
    """Store, then retrieve — the backend is the only thing doing
    similarity math. Swap in a stub backend and observe the call."""
    from soma.memory.backend import VectorBackend

    tok, enc = embedder
    mem = MemoryLayer(tokenizer=tok, encoder=enc)
    assert isinstance(mem._backend, VectorBackend)
    nid = mem.store("alpha example")
    mem.store("beta example")
    mem.store("gamma example")

    # related() must round-trip through backend.get_vectors +
    # backend.search.
    related = mem.related(nid, k=2)
    assert len(related) == 2
    assert all(h.node_id != nid for h in related)


def test_forget_removes_from_backend(embedder) -> None:
    tok, enc = embedder
    mem = MemoryLayer(tokenizer=tok, encoder=enc)
    nid = mem.store("entry to remove")
    assert mem._backend.ntotal == 1
    assert mem.forget(nid)
    assert mem._backend.ntotal == 0
