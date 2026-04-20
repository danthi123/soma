"""Multi-tenant SOMA: one process, many isolated brains.

Pattern: each tenant gets their own ``MemoryLayer`` bundle. A thin
router selects the right bundle per-request and SOMA handles the
rest. No database, no per-user schema, no shared state to leak
across tenants.

What this shows:
  - Creating multiple bundles from the same process
  - Routing store/retrieve calls by tenant ID
  - Cross-tenant isolation: one tenant cannot retrieve another's
    memories even if the query term is identical
  - Per-tenant persistence — each bundle saves to its own directory

This is the Python-API view. For the HTTP equivalent (one server,
``/bundles/{tenant}/...`` routing with per-bundle JWTs), see
``docs/quickstart.md`` § "Mint a per-bundle JWT" and
``src/soma/serve.py``.

Requires::

    pip install -e ".[sbert]"

Run::

    python examples/03_multi_tenant_bundle.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from soma.memory import MemoryLayer


def _make_sbert_embed_fn(model_name: str = "all-MiniLM-L6-v2") -> tuple[callable, int]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())

    def embed(text: str) -> torch.Tensor:
        return torch.tensor(model.encode(text, convert_to_numpy=True))

    return embed, dim


class TenantRouter:
    """Dispatches SOMA calls to the right tenant bundle.

    This is ~30 lines of code because SOMA's bundle-per-brain design
    does the heavy lifting. Each bundle is its own directory, its
    own sbert closure, its own WAL. Tenants cannot see each other's
    data — isolation is filesystem-level, not query-filter-level.
    """

    def __init__(self, bundles_root: Path, embed_fn: callable, embed_dim: int) -> None:
        self.bundles_root = bundles_root
        self.embed_fn = embed_fn
        self.embed_dim = embed_dim
        self._tenants: dict[str, MemoryLayer] = {}

    def _bundle_path(self, tenant: str) -> Path:
        return self.bundles_root / tenant

    def _get_or_create(self, tenant: str) -> MemoryLayer:
        if tenant in self._tenants:
            return self._tenants[tenant]
        path = self._bundle_path(tenant)
        if path.exists() and (path / "memory_index.json").exists():
            mem = MemoryLayer.load(path, embed_fn=self.embed_fn)
        else:
            mem = MemoryLayer(embed_fn=self.embed_fn, embed_dim=self.embed_dim)
        self._tenants[tenant] = mem
        return mem

    def store(self, tenant: str, text: str, **metadata) -> str:
        return self._get_or_create(tenant).store(text, metadata=metadata or None)

    def retrieve(self, tenant: str, query: str, k: int = 3) -> list:
        return self._get_or_create(tenant).retrieve(query, k=k)

    def save_all(self) -> None:
        """Snapshot every open tenant to disk."""
        for tenant, mem in self._tenants.items():
            path = self._bundle_path(tenant)
            path.mkdir(parents=True, exist_ok=True)
            mem.save(path)


def main() -> None:
    embed_fn, embed_dim = _make_sbert_embed_fn()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        bundles_root = Path(tmp) / "bundles"
        router = TenantRouter(bundles_root, embed_fn, embed_dim)

        # 1. Seed two tenants with DIFFERENT facts that happen to share
        #    a query term ("favorite color"). This sets up the isolation
        #    test below.
        router.store("alice", "alice's favorite color is blue")
        router.store("alice", "alice is allergic to peanuts")
        router.store("alice", "alice lives in Portland, OR")

        router.store("bob", "bob's favorite color is green")
        router.store("bob", "bob is a vegetarian")
        router.store("bob", "bob lives in Austin, TX")

        print("Seeded 3 facts for each of 2 tenants.\n")

        # 2. Same query, different tenants — should return different answers.
        print("--- same query, different tenants ---")
        for tenant in ("alice", "bob"):
            hits = router.retrieve(tenant, "what is my favorite color?", k=1)
            top = hits[0] if hits else None
            print(f"{tenant:>6}> {top.text if top else '(nothing)'}"
                  f"  (score={top.score:+.3f})")

        # 3. Isolation check: ask for "peanut allergy" from bob's bundle —
        #    alice has that fact, bob doesn't. Bob's bundle must NOT return
        #    alice's allergy note.
        print("\n--- isolation check (cross-tenant leak?) ---")
        bob_hits = router.retrieve("bob", "any food allergies?", k=3)
        print(f"bob's top result for 'food allergies': "
              f"{bob_hits[0].text if bob_hits else '(nothing)'}")
        alice_texts_in_bob = [
            h for h in bob_hits if "alice" in h.text.lower()
        ]
        assert not alice_texts_in_bob, (
            "ISOLATION BROKEN: alice's data appeared in bob's bundle!"
        )
        print("bob's retrieval contains zero alice-owned facts.  OK")

        # 4. Save all tenants at once.
        router.save_all()
        print(f"\nSaved {len(router._tenants)} tenant bundles under {bundles_root}:")
        for tenant_dir in sorted(bundles_root.iterdir()):
            contents = sorted(p.name for p in tenant_dir.iterdir())
            print(f"  {tenant_dir.name}/ -> {contents}")

        # 5. Fresh restart — new router instance, same on-disk bundles.
        #    Load-on-demand: the lazy loader in _get_or_create() picks up
        #    the existing bundle the first time we touch that tenant.
        print("\n--- simulating process restart ---")
        del router  # Pretend the process died. Bundles survive on disk.
        router2 = TenantRouter(bundles_root, embed_fn, embed_dim)
        for tenant in ("alice", "bob"):
            hits = router2.retrieve(tenant, "where do I live?", k=1)
            top = hits[0] if hits else None
            print(f"{tenant:>6}> {top.text if top else '(nothing)'}  "
                  f"(score={top.score:+.3f})")

        # 6. Clean up handles so Windows can drop the WAL locks.
        del router2

    print("\nOK — two tenants, isolated bundles, survives restart.")


if __name__ == "__main__":
    main()
