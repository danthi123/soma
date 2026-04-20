"""The 10-line SOMA Python API tour.

This is the first example new users should read. It covers the core
MemoryLayer loop — ``store``, ``retrieve``, ``save``, ``load`` — using
sentence-transformers for embeddings so retrieval quality is realistic.

What you'll see:
  - Memory survives process restarts (save → load round trip)
  - Retrieval ranks by semantic similarity, not lexical match
  - Metadata filters work ("where" clauses on retrieve)
  - The whole "brain" is a single directory on disk

Requires::

    pip install -e ".[sbert]"

Run::

    python examples/01_quickstart.py

Follow-on examples: ``02_persistent_chat_agent.py`` (hook SOMA into a
chat loop), ``03_multi_tenant_bundle.py`` (one server, many users).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from soma.memory import MemoryLayer


def _make_sbert_embed_fn(model_name: str = "all-MiniLM-L6-v2") -> tuple[callable, int]:
    """Build an sbert embed_fn closure plus its output dim.

    Factored out so the SAME closure can be used for create AND load —
    ``MemoryLayer.load`` needs the embed_fn explicitly when the bundle
    was saved with a non-serialisable embedder (like sbert).
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())

    def embed(text: str) -> torch.Tensor:
        return torch.tensor(model.encode(text, convert_to_numpy=True))

    return embed, dim


def main() -> None:
    # 1. Build the sbert embedder. We'll reuse the same closure for
    #    save/load round-trip below — MemoryLayer.with_sbert() is a
    #    convenience wrapper, but the explicit closure is what
    #    MemoryLayer.load() needs at restore time.
    embed_fn, embed_dim = _make_sbert_embed_fn()
    mem = MemoryLayer(embed_fn=embed_fn, embed_dim=embed_dim)

    # 2. Store a handful of facts about a user. Metadata is free-form
    #    and filterable at retrieve time.
    mem.store("user lives in Portland, OR", metadata={"user": "alex"})
    mem.store("user is vegetarian",         metadata={"user": "alex"})
    mem.store("user's dog is named Luna",   metadata={"user": "alex"})
    mem.store("user loves hiking Mount Hood", metadata={"user": "alex"})
    mem.store("user commutes by bike",      metadata={"user": "alex"})
    print(f"Stored {len(mem)} facts.")

    # 3. Retrieve by MEANING, not keyword. The query doesn't share words
    #    with the stored fact — hybrid retrieval still ranks it first.
    print("\n--- semantic retrieval ---")
    hits = mem.retrieve("dietary restrictions", k=3)
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. score={hit.score:+.3f}  {hit.text}")

    # 4. Metadata filters narrow retrieval to a subset. Handy for
    #    multi-user setups where you only want one user's facts.
    print("\n--- filtered retrieval (where user=alex) ---")
    hits = mem.retrieve("where does the user live?", k=2,
                        where={"user": "alex"})
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. score={hit.score:+.3f}  {hit.text}")

    # 5. Persist the whole brain to a directory. This is the operational
    #    posture we recommend: one bundle directory per brain, portable
    #    across machines. Includes embeddings, BM25 index, metadata,
    #    WAL, and auth keys (when applicable).
    #
    # ``ignore_cleanup_errors=True`` is a Windows workaround: the WAL
    # file stays open under the MemoryLayer until explicit close, and
    # the OS won't let TemporaryDirectory delete open files. In
    # real deployments the bundle directory outlives the process.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        bundle_path = Path(tmp) / "alex-brain"
        mem.save(bundle_path)
        print(f"\nSaved brain to {bundle_path}")
        print("Contents:", [p.name for p in bundle_path.iterdir()])

        # 6. Load it back into a fresh MemoryLayer — zero shared state
        #    with the original. This is what server restarts look like.
        #    Pass the SAME embed_fn so new stores can still be embedded.
        restored = MemoryLayer.load(bundle_path, embed_fn=embed_fn)
        print(f"Loaded brain: {len(restored)} facts restored")

        # Same query should return the same top result.
        top = restored.retrieve("dietary restrictions", k=1)[0]
        print(f"Round-trip top hit: {top.text!r}")

        # Explicit close so Windows can clean up the WAL file.
        del restored

    print("\nOK — you've seen the full SOMA memory loop in 10 lines of API.")


if __name__ == "__main__":
    main()
