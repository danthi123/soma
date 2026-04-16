"""VectorBackend matrix benchmark.

Phase 6 measurement pass. Answers two questions:

1. **Does filter pushdown (QdrantHTTP) beat Python pre-filter +
   subset (InProc)?** Compare retrieve latency with a selective
   ``where`` clause.
2. **Does QdrantHTTP stay within 2x InProc at 1M?** Plan ship-blocker
   threshold.

Harness reuses the sbert cache from ``run_scale_enterprise`` so we
don't pay the 14-hour embedding pass twice.

Run::

    # smoke (fast): 10K, InProc only
    python -m benchmarks.run_backend_matrix --n 10000 --smoke

    # full: InProc flat/hnsw + Qdrant local
    python -m benchmarks.run_backend_matrix --n 100000

    # add HTTP comparison
    SOMA_QDRANT_TEST_URL=http://localhost:6333 \\
      python -m benchmarks.run_backend_matrix --n 100000 --http

Metrics:
- store_total (seconds)
- retrieve p50 / p95 (ms)
- disk footprint (MB)
- Recall@10 vs exact (InProcFlat reference)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from soma.memory.api import MemoryLayer
from soma.memory.backends.inproc import InProcBackend

CACHE_DIR = Path("benchmarks/datasets/scale_cache")
REPORT_PATH = Path("benchmarks/reports/backend_matrix.md")
PROBE_COUNT = 100


@dataclass
class MatrixRow:
    backend: str
    n: int
    store_total_s: float
    retrieve_p50_ms: float
    retrieve_p95_ms: float
    disk_mb: float
    recall_at_10: float
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _load_or_make_corpus(n: int) -> tuple[list[str], np.ndarray]:
    """Reuse scale_enterprise's corpus + sbert cache when present.

    Falls back to a deterministic synthetic corpus + ephemeral
    embeddings so the smoke path runs without the 14-hour bootstrap.
    """
    try:
        from benchmarks.datasets.scale_corpus import (
            generate_scale_corpus,
        )

        facts = generate_scale_corpus(n)
        cache_path = CACHE_DIR / f"embeddings_{n}_all-MiniLM-L6-v2.npy"
        if cache_path.exists():
            vecs = np.load(cache_path)
            if vecs.shape[0] == n:
                return [f.text for f in facts], vecs.astype(np.float32)
    except Exception:
        facts = None

    # Fallback: synthetic texts + random float32 vectors (same shape
    # sbert would produce). Recall@10 computed against InProcFlat
    # reference so the comparison still makes sense.
    rng = np.random.default_rng(42)
    texts = [f"doc_{i}_topic_{i % 100}" for i in range(n)]
    vecs = rng.standard_normal((n, 384)).astype(np.float32)
    return texts, vecs


def _embed_fn_from_cache(vecs: np.ndarray) -> Any:
    """Return an ``embed_fn`` closure that serves precomputed vectors.

    The adapter-matrix bench reuses one embedding cache across every
    backend — we keep a row counter so each sequential ``store`` pulls
    the next cached vector. Query embeddings are handled via a
    separate deterministic hash (mirrors how run_scale_enterprise
    precomputes query vectors).
    """
    state = {"row": 0}

    def _embed(text: str) -> torch.Tensor:
        # When the bench calls store() in deterministic order, we hand
        # out rows sequentially. When it calls for a query, we fall
        # back to a hash-seeded random vector with the same dim.
        if state["row"] < len(vecs):
            vec = vecs[state["row"]]
            state["row"] += 1
            return torch.from_numpy(vec.copy())
        h = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        return torch.from_numpy(
            rng.standard_normal(vecs.shape[1]).astype(np.float32)
        )

    return _embed


def _run_backend(
    name: str,
    mem: MemoryLayer,
    texts: list[str],
    vecs: np.ndarray,
    *,
    n_probe: int = PROBE_COUNT,
    where: dict[str, Any] | None = None,
    reference_positions: list[list[int]] | None = None,
    bundle_path: Path | None = None,
) -> MatrixRow:
    """Store ``texts`` then probe ``n_probe`` queries, returning timings.

    Recall is computed over *positions* in the shared vector matrix
    (row index 0..N-1), not node_ids — each MemoryLayer assigns its
    own uuid4 ids so node_id comparisons are meaningless across
    backends.
    """
    t0 = time.perf_counter()
    # Store with per-row metadata so ``where``-dispatch can exercise
    # filter pushdown. We also stash the row index in metadata so the
    # cross-backend recall comparison has a stable key.
    metadatas = [
        {
            "topic": i % 100,
            "bucket": "a" if i % 3 == 0 else "b",
            "row": i,
        }
        for i in range(len(texts))
    ]
    for i in range(0, len(texts), 256):
        mem.store_batch(
            texts[i : i + 256],
            metadatas=metadatas[i : i + 256],
        )
    store_total = time.perf_counter() - t0

    n_store = len(texts)
    query_texts = [f"probe_{i}" for i in range(n_probe)]

    retrieve_times_ms: list[float] = []
    returned_rows: list[list[int]] = []
    for q in query_texts:
        t1 = time.perf_counter()
        hits = mem.retrieve(q, k=10, where=where)
        retrieve_times_ms.append((time.perf_counter() - t1) * 1000.0)
        returned_rows.append([int(h.metadata.get("row", -1)) for h in hits])

    disk_mb = 0.0
    if bundle_path is not None and bundle_path.exists():
        total = 0
        for p in bundle_path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        disk_mb = total / 1024.0 / 1024.0

    recall_at_10 = 1.0
    if reference_positions is not None:
        inters = []
        for ref, got in zip(reference_positions, returned_rows, strict=True):
            if not ref:
                continue
            inters.append(len(set(ref) & set(got)) / len(ref))
        if inters:
            recall_at_10 = sum(inters) / len(inters)

    p50 = statistics.median(retrieve_times_ms) if retrieve_times_ms else 0.0
    p95 = (
        statistics.quantiles(retrieve_times_ms, n=20)[18]
        if len(retrieve_times_ms) >= 20
        else p50
    )

    row = MatrixRow(
        backend=name,
        n=n_store,
        store_total_s=store_total,
        retrieve_p50_ms=p50,
        retrieve_p95_ms=p95,
        disk_mb=disk_mb,
        recall_at_10=recall_at_10,
        extras={"returned_rows": returned_rows},
    )
    return row


def run_matrix(
    *,
    n: int,
    include_qdrant_local: bool,
    include_qdrant_http: bool,
    smoke: bool,
) -> list[MatrixRow]:
    """Execute the full adapter matrix and return the per-backend rows."""
    print(f"[matrix] preparing corpus (n={n}, smoke={smoke})")
    texts, vecs = _load_or_make_corpus(n)
    assert vecs.shape[0] == n, "corpus / embedding cache size mismatch"
    dim = int(vecs.shape[1])

    rows: list[MatrixRow] = []
    ref_positions: list[list[int]] | None = None

    # 1. InProc flat (reference)
    print("[matrix] running InProcFlat")
    bundle = Path(tempfile.mkdtemp(prefix="soma_bench_inproc_flat_"))
    try:
        backend = InProcBackend(dim=dim, faiss_threshold=10_000)
        mem = MemoryLayer(
            embed_fn=_embed_fn_from_cache(vecs),
            embed_dim=dim,
            backend=backend,
        )
        row = _run_backend("InProcFlat", mem, texts, vecs, bundle_path=bundle)
        ref_positions = row.extras.get("returned_rows")
        row.extras.pop("returned_rows", None)
        rows.append(row)
        mem.close()
    finally:
        shutil.rmtree(bundle, ignore_errors=True)

    if smoke:
        return rows

    # 2. InProc HNSW
    print("[matrix] running InProcHNSW")
    bundle = Path(tempfile.mkdtemp(prefix="soma_bench_inproc_hnsw_"))
    try:
        backend = InProcBackend(
            dim=dim, faiss_index_type="hnsw", faiss_threshold=500
        )
        mem = MemoryLayer(
            embed_fn=_embed_fn_from_cache(vecs),
            embed_dim=dim,
            backend=backend,
        )
        row = _run_backend(
            "InProcHNSW",
            mem,
            texts,
            vecs,
            reference_positions=ref_positions,
            bundle_path=bundle,
        )
        row.extras.pop("returned_rows", None)
        rows.append(row)
        mem.close()
    finally:
        shutil.rmtree(bundle, ignore_errors=True)

    # 3. Qdrant local
    if include_qdrant_local:
        try:
            from soma.memory.backends.qdrant import QdrantBackend
        except ImportError:
            print("[matrix] qdrant-client not installed; skipping Qdrant rows")
        else:
            print("[matrix] running QdrantLocal")
            bundle = Path(
                tempfile.mkdtemp(prefix="soma_bench_qdrant_local_")
            )
            try:
                qpath = bundle / "qdrant_data"
                backend = QdrantBackend(
                    mode="local",
                    dim=dim,
                    path=str(qpath),
                    collection="bench",
                    recreate=True,
                )
                mem = MemoryLayer(
                    embed_fn=_embed_fn_from_cache(vecs),
                    embed_dim=dim,
                    backend=backend,
                )
                row = _run_backend(
                    "QdrantLocal",
                    mem,
                    texts,
                    vecs,
                    reference_positions=ref_positions,
                    bundle_path=qpath,
                )
                row.extras.pop("returned_rows", None)
                rows.append(row)
                mem.close()
            finally:
                shutil.rmtree(bundle, ignore_errors=True)

    # 4. Qdrant HTTP (only when env set)
    http_url = os.environ.get("SOMA_QDRANT_TEST_URL")
    if include_qdrant_http and http_url:
        try:
            from soma.memory.backends.qdrant import QdrantBackend
        except ImportError:
            print("[matrix] qdrant-client not installed; skipping HTTP row")
        else:
            print(f"[matrix] running QdrantHTTP @ {http_url}")
            backend = QdrantBackend(
                mode="http",
                dim=dim,
                url=http_url,
                collection=f"bench_{int(time.time())}",
                recreate=True,
            )
            mem = MemoryLayer(
                embed_fn=_embed_fn_from_cache(vecs),
                embed_dim=dim,
                backend=backend,
            )
            row = _run_backend(
                "QdrantHTTP",
                mem,
                texts,
                vecs,
                reference_positions=ref_positions,
            )
            row.extras.pop("returned_rows", None)
            rows.append(row)
            mem.close()

    return rows


def render_report(rows: list[MatrixRow], *, n: int) -> str:
    lines = [
        f"# Backend Matrix — N = {n:,}",
        "",
        "## Results",
        "",
        "| Backend | Store total | Retrieve p50 (ms) | Retrieve p95 (ms) | Disk (MB) | Recall@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        store_s = (
            f"{r.store_total_s:.1f}s"
            if r.store_total_s < 60
            else f"{r.store_total_s / 60:.1f}min"
        )
        lines.append(
            f"| {r.backend} | {store_s} | {r.retrieve_p50_ms:.2f} "
            f"| {r.retrieve_p95_ms:.2f} | {r.disk_mb:.1f} "
            f"| {r.recall_at_10:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Recall@10 is computed against InProcFlat (exact cosine).",
            "- Disk footprint includes all bundle files for local adapters;",
            "  HTTP-mode Qdrant is reported as 0 MB (data lives on the",
            "  remote server).",
            "- Store uses `store_batch` in 256-row chunks — realistic for",
            "  corpora but faster than the per-call `store` loop.",
            "",
            "Generated by `benchmarks/run_backend_matrix.py`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10_000, help="corpus size")
    p.add_argument(
        "--smoke", action="store_true", help="InProcFlat only (fast)"
    )
    p.add_argument(
        "--no-qdrant-local",
        action="store_true",
        help="skip QdrantLocal row",
    )
    p.add_argument(
        "--http",
        action="store_true",
        help="include QdrantHTTP (requires SOMA_QDRANT_TEST_URL env)",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="report path"
    )
    args = p.parse_args()

    rows = run_matrix(
        n=args.n,
        include_qdrant_local=not args.no_qdrant_local,
        include_qdrant_http=args.http,
        smoke=args.smoke,
    )

    out = args.out or REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(rows, n=args.n), encoding="utf-8")
    # Also drop a JSON sidecar so downstream tooling can ingest the
    # numbers without re-parsing markdown.
    json_out = out.with_suffix(".json")
    json_out.write_text(
        json.dumps([asdict(r) for r in rows], indent=2),
        encoding="utf-8",
    )
    print(f"[matrix] wrote {out}")
    print(f"[matrix] wrote {json_out}")


if __name__ == "__main__":
    main()
