"""Benchmark runner: drives a list of systems through a dataset + queries.

Usage::

    from benchmarks.harness.runner import run_retrieval_benchmark
    from benchmarks.harness.adapters.soma import SomaAdapter
    from benchmarks.harness.adapters.chroma import ChromaAdapter

    systems = [SomaAdapter(), ChromaAdapter()]
    results = run_retrieval_benchmark(
        systems=systems,
        dataset=my_facts,
        queries=my_queries_with_labels,
        k=3,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .adapters.base import BaseMemorySystem
from .metrics.retrieval import mrr_at_k, ndcg_at_k, recall_at_k


@dataclass
class LabeledQuery:
    """A query plus its ground-truth relevant texts."""

    query: str
    relevant_texts: list[str]


@dataclass
class SystemResult:
    system: str
    num_entries: int
    store_total_s: float
    store_avg_ms: float
    retrieve_avg_ms: float
    retrieve_p99_ms: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    disk_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


def run_retrieval_benchmark(
    *,
    systems: list[BaseMemorySystem],
    dataset: list[str],
    queries: list[LabeledQuery],
    k: int = 5,
    consolidate_after_store: bool = False,
) -> list[SystemResult]:
    """Run the same dataset through every system; return per-system metrics."""
    results: list[SystemResult] = []
    for system in systems:
        print(f"[{system.name}] preparing...")
        system.prepare()

        t0 = time.perf_counter()
        for fact in dataset:
            system.store(fact)
        store_total = time.perf_counter() - t0

        if consolidate_after_store:
            t_c = time.perf_counter()
            system.consolidate()
            consolidate_time = time.perf_counter() - t_c
        else:
            consolidate_time = 0.0

        retrieve_times: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        for q in queries:
            t1 = time.perf_counter()
            hits = system.retrieve(q.query, k=k)
            retrieve_times.append(time.perf_counter() - t1)
            texts = [h.text for h in hits]
            recalls.append(recall_at_k(texts, q.relevant_texts, k))
            mrrs.append(mrr_at_k(texts, q.relevant_texts, k))
            ndcgs.append(ndcg_at_k(texts, q.relevant_texts, k))

        disk = system.disk_footprint_bytes()
        sorted_rt = sorted(retrieve_times)
        p99_idx = max(0, int(len(sorted_rt) * 0.99) - 1)
        results.append(
            SystemResult(
                system=system.name,
                num_entries=len(dataset),
                store_total_s=store_total,
                store_avg_ms=store_total * 1000 / max(1, len(dataset)),
                retrieve_avg_ms=sum(retrieve_times) * 1000 / max(1, len(retrieve_times)),
                retrieve_p99_ms=sorted_rt[p99_idx] * 1000 if sorted_rt else 0.0,
                recall_at_k=sum(recalls) / max(1, len(recalls)),
                mrr_at_k=sum(mrrs) / max(1, len(mrrs)),
                ndcg_at_k=sum(ndcgs) / max(1, len(ndcgs)),
                disk_bytes=disk,
                metadata=(
                    {"consolidate_time_s": consolidate_time}
                    if consolidate_after_store
                    else {}
                ),
            )
        )
        print(
            f"[{system.name}] Recall@{k}={results[-1].recall_at_k:.3f} "
            f"MRR@{k}={results[-1].mrr_at_k:.3f} "
            f"retrieve={results[-1].retrieve_avg_ms:.2f}ms "
            f"disk={disk / 1024:.1f}KB"
        )
        system.teardown()

    return results


def format_results_table(results: list[SystemResult], k: int = 5) -> str:
    """Produce a markdown table from benchmark results."""
    lines = [
        f"| System | Recall@{k} | MRR@{k} | NDCG@{k} | Store (ms/op) | Retrieve (ms) | Disk (KB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.system} | {r.recall_at_k:.3f} | {r.mrr_at_k:.3f} "
            f"| {r.ndcg_at_k:.3f} | {r.store_avg_ms:.2f} "
            f"| {r.retrieve_avg_ms:.2f} | {r.disk_bytes / 1024:.1f} |"
        )
    return "\n".join(lines)
