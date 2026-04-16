"""End-to-end integration test for the full MemoryLayer pipeline.

store -> consolidate (with SOMA) -> graph-reranked retrieve ->
save -> load -> retrieve again -> forget -> verify.
"""

from __future__ import annotations

from pathlib import Path

import torch

from soma.core.config import SOMAConfig
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryLayer
from soma.system import SOMA

FACTS = [
    "the cat sat on the mat",
    "the dog chased the ball in the park",
    "quantum physics describes subatomic particles",
    "photosynthesis converts sunlight into energy",
    "the stock market closed higher today",
]


def _build_stack() -> tuple[object, TextEncoder, SOMA]:
    torch.manual_seed(42)
    tokenizer = train_bpe_tokenizer(FACTS, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
    )
    soma = SOMA(config)
    return tokenizer, encoder, soma


def test_full_pipeline(tmp_path: Path) -> None:
    tokenizer, encoder, soma = _build_stack()

    # 1. Create + store
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    ids = [mem.store(fact) for fact in FACTS]
    assert len(mem) == 5

    # 2. Attach SOMA + consolidate
    mem.attach_soma(soma, tokenizer, encoder)
    processed = mem.consolidate()
    assert processed == 5
    assert soma.global_step > 0

    # 3. Retrieve with graph-reranked path (SOMA attached + activations present)
    hits = mem.retrieve("cat on a rug", k=3)
    assert len(hits) == 3
    assert all(hasattr(h, "score") for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)

    # 4. Save
    bundle = tmp_path / "e2e-bundle"
    mem.save(bundle)
    assert (bundle / "memory_index.json").exists()
    assert (bundle / "memory_embeddings.pt").exists()

    # 5. Load
    restored = MemoryLayer.load(bundle)
    assert len(restored) == 5

    # 6. Retrieve from loaded bundle (no SOMA = flat cosine)
    restored_hits = restored.retrieve("cat on a rug", k=3)
    assert len(restored_hits) == 3
    # Same texts should be in top-3 (order may differ due to rerank vs flat)
    pre_texts = {h.text for h in hits}
    post_texts = {h.text for h in restored_hits}
    assert len(pre_texts & post_texts) >= 2  # at least 2 of 3 overlap

    # 7. Forget + verify
    forgotten_id = ids[0]
    assert restored.forget(forgotten_id) is True
    assert len(restored) == 4
    assert restored.get(forgotten_id) is None

    # 8. All remaining entries still retrievable
    all_hits = restored.retrieve("anything", k=10)
    assert len(all_hits) == 4


def test_pipeline_without_soma(tmp_path: Path) -> None:
    """Flat cosine path (no SOMA) also round-trips correctly."""
    torch.manual_seed(42)
    tokenizer = train_bpe_tokenizer(FACTS, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)

    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for fact in FACTS:
        mem.store(fact)

    # Consolidate without SOMA = no-op
    assert mem.consolidate() == 0

    hits = mem.retrieve("physics", k=2)
    assert len(hits) == 2

    bundle = tmp_path / "flat-bundle"
    mem.save(bundle)
    restored = MemoryLayer.load(bundle)
    assert len(restored) == 5

    restored_hits = restored.retrieve("physics", k=2)
    assert [h.text for h in hits] == [h.text for h in restored_hits]
