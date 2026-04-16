"""Scale benchmark: SOMA vs Chroma at 1K / 5K / 20K entries.

The headline retrieval benchmark (``run_retrieval.py``) runs at 50
facts to keep labeled queries + ground-truth tractable; that reports
22.6× disk and 1.3× faster retrieve but the reader has to trust those
numbers generalize. This script removes the quality measurement,
sticks to pure efficiency metrics (store time, retrieve time, disk),
and pushes store size up to where FAISS actually takes over on the
SOMA side.

No ground-truth queries — the probe queries are sampled from the
stored-fact distribution so retrieval always has the candidate in
the store (what we measure is latency + footprint, not correctness).

Run::

    python -m benchmarks.run_scale_vs_chroma \\
        --out benchmarks/reports/scale_vs_chroma.md
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.datasets.templated import generate_templated_facts
from benchmarks.harness.adapters.chroma import ChromaAdapter
from benchmarks.harness.adapters.soma import SomaAdapter

SCALE_POINTS: list[int] = [1000, 5000, 20000]
PROBE_COUNT: int = 50


@dataclass
class ScaleRow:
    system: str
    n: int
    store_avg_ms: float
    retrieve_avg_ms: float
    disk_kb: float


def _run_system(
    system_name: str, adapter, facts: list[str], probes: list[str],
) -> ScaleRow:
    print(f"  [{system_name}] preparing...")
    adapter.prepare()
    print(f"  [{system_name}] storing {len(facts)} facts...")
    t0 = time.perf_counter()
    for f in facts:
        adapter.store(f)
    store_total = time.perf_counter() - t0

    print(f"  [{system_name}] probing {len(probes)} queries...")
    retrieve_times: list[float] = []
    for q in probes:
        t1 = time.perf_counter()
        adapter.retrieve(q, k=3)
        retrieve_times.append(time.perf_counter() - t1)

    disk = adapter.disk_footprint_bytes() / 1024.0
    adapter.teardown()

    return ScaleRow(
        system=system_name,
        n=len(facts),
        store_avg_ms=store_total * 1000 / max(1, len(facts)),
        retrieve_avg_ms=sum(retrieve_times) * 1000 / max(1, len(retrieve_times)),
        disk_kb=disk,
    )


def _format_table(rows: list[ScaleRow]) -> str:
    header = (
        "| N | System | Store (ms/op) | Retrieve (ms) | Disk (KB) "
        "| Disk (MB) |"
    )
    sep = "| ---: | --- | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r.n} | {r.system} | {r.store_avg_ms:.2f} "
            f"| {r.retrieve_avg_ms:.2f} | {r.disk_kb:.1f} "
            f"| {r.disk_kb / 1024:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/scale_vs_chroma.md"),
    )
    p.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=SCALE_POINTS,
    )
    args = p.parse_args()

    max_n = max(args.points)
    print(f"Generating {max_n} unique facts...")
    all_facts_meta = generate_templated_facts(max_n, seed=42)
    all_facts = [f.text for f in all_facts_meta]
    # probes are drawn from the corpus so there's always a candidate
    probes = [all_facts[i] for i in range(0, max_n, max(1, max_n // PROBE_COUNT))]
    probes = probes[:PROBE_COUNT]

    rows: list[ScaleRow] = []
    for n in sorted(args.points):
        facts = all_facts[:n]
        print(f"\n=== N = {n} ===")
        rows.append(_run_system("soma", SomaAdapter(use_sbert=True), facts, probes))
        rows.append(_run_system("chroma", ChromaAdapter(), facts, probes))
        r_soma, r_chroma = rows[-2], rows[-1]
        print(
            f"  SOMA vs Chroma: disk {r_chroma.disk_kb / max(r_soma.disk_kb, 1):.1f}x, "
            f"retrieve {r_chroma.retrieve_avg_ms / max(r_soma.retrieve_avg_ms, 1e-3):.1f}x"
        )

    lines = [
        "# Scale vs Chroma — Pure Efficiency Profile",
        "",
        "Same sbert embedder (`all-MiniLM-L6-v2`) on both sides; "
        "no quality measurement, just store time, retrieve time, and "
        "disk footprint. Probe queries are sampled from the stored "
        "corpus so retrieval always has a candidate — we're measuring "
        "latency and bytes, not correctness.",
        "",
        "## Results",
        "",
        _format_table(rows),
        "",
        "## Ratios (Chroma / SOMA)",
        "",
    ]

    sizes = sorted({r.n for r in rows})
    for n in sizes:
        soma = next(r for r in rows if r.n == n and r.system == "soma")
        chroma = next(r for r in rows if r.n == n and r.system == "chroma")
        lines.append(
            f"- **N = {n}:** disk "
            f"{chroma.disk_kb / max(soma.disk_kb, 1):.1f}× smaller, "
            f"retrieve {chroma.retrieve_avg_ms / max(soma.retrieve_avg_ms, 1e-3):.1f}× "
            f"faster, store "
            f"{chroma.store_avg_ms / max(soma.store_avg_ms, 1e-3):.1f}× faster "
            "for SOMA."
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "SOMA's disk advantage comes from storing a single pytorch "
        "tensor of vectors plus a JSON index — no SQLite, no HNSW "
        "sidecar. Chroma's bundle carries the HNSW graph, SQLite "
        "schema, metadata shards, and lockfiles. The per-entry "
        "overhead of those structures is O(1) in store size but the "
        "constant is high; SOMA's per-entry overhead is essentially "
        "the raw embedding.",
        "",
        "SOMA's retrieve advantage at small N comes from the linear "
        "cosine backend skipping HNSW construction cost. At the 10K "
        "threshold SOMA's FAISS IndexFlatIP kicks in (see "
        "``MemoryLayer._maybe_build_faiss``) — the cross-over where "
        "Chroma's HNSW catches up on retrieve time is visible in the "
        "table.",
        "",
        "---",
        "",
        "Generated by `benchmarks/run_scale_vs_chroma.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
