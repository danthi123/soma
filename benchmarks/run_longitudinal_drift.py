"""Longitudinal drift benchmark — does retrieval degrade as the store grows over time?

Simulates 30 days of user activity. Each day stores 5 new facts and runs
queries drawn from the full history. Measures how Recall@3 on *old*
facts (stored 10+ days ago) compares to Recall@3 on *recent* facts — and
whether old-fact recall stays flat, climbs, or decays as the store
expands and consolidation continues to run.

Why this benchmark matters for the paper:
- LoCoMo / LongMemEval are de-facto benchmarks but they need an LLM
  judge and a hand-curated long conversation. This is cheaper,
  reproducible, and isolates the "does old memory rot under new
  writes?" failure mode specifically.
- Mem0-style systems that periodically purge low-confidence entries
  should show an old-fact cliff. SOMA's plastic-graph contract says
  old facts stay retrievable as long as the embedding is intact.
- If SOMA's old-fact Recall@3 stays within noise of day-of-write
  Recall@3, that's a legitimate "memory doesn't drift" claim.

Run::

    python -m benchmarks.run_longitudinal_drift \\
        --out benchmarks/reports/longitudinal_drift.md
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.datasets.templated import TemplatedFact, generate_templated_facts
from soma.memory import MemoryLayer

FACTS_PER_DAY = 5
N_DAYS = 30
QUERIES_PER_DAY = 10
OLD_THRESHOLD_DAYS = 10  # facts older than this are "old" for the split


def _query_for_fact(fact: TemplatedFact) -> str:
    slots = fact.slots
    person = slots.get("person", "the person")
    match fact.topic:
        case "location":
            return f"Where does {person} live?"
        case "work":
            return f"What does {person} do for work?"
        case "pet":
            return f"What pet does {person} have?"
        case "hobby":
            return f"What hobbies does {person} have?"
        case "travel":
            return f"Which country did {person} visit?"
        case _:
            return fact.text


@dataclass
class StoredEntry:
    day: int
    fact: TemplatedFact


@dataclass
class DayMeasurement:
    day: int
    store_size: int
    recall_overall: float
    recall_recent: float  # facts stored in the last 3 days
    recall_old: float  # facts stored 10+ days ago
    mrr_overall: float
    retrieve_avg_ms: float


def _measure_day(
    mem: MemoryLayer,
    current_day: int,
    history: list[StoredEntry],
    rng: random.Random,
) -> DayMeasurement:
    eligible = list(history)
    if not eligible:
        return DayMeasurement(
            current_day, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    # Recency-biased query sampling: 50% recent, 30% mid, 20% old.
    by_age: dict[str, list[StoredEntry]] = {"recent": [], "mid": [], "old": []}
    for entry in eligible:
        age = current_day - entry.day
        if age <= 3:
            by_age["recent"].append(entry)
        elif age <= OLD_THRESHOLD_DAYS:
            by_age["mid"].append(entry)
        else:
            by_age["old"].append(entry)

    def _sample() -> StoredEntry:
        bucket_choice = rng.choices(
            ["recent", "mid", "old"], weights=[0.5, 0.3, 0.2], k=1,
        )[0]
        bucket = by_age[bucket_choice] or eligible
        return rng.choice(bucket)

    recall_sum_all = 0.0
    recall_sum_recent = 0.0
    recall_sum_old = 0.0
    n_recent = 0
    n_old = 0
    mrr_sum = 0.0
    retrieve_times: list[float] = []
    n_queries = min(QUERIES_PER_DAY, len(eligible))
    for _ in range(n_queries):
        entry = _sample()
        query = _query_for_fact(entry.fact)
        t0 = time.perf_counter()
        hits = mem.retrieve(query, k=3)
        retrieve_times.append(time.perf_counter() - t0)
        texts = [h.text for h in hits]
        hit_it = entry.fact.text in texts
        recall_sum_all += 1.0 if hit_it else 0.0
        if hit_it:
            mrr_sum += 1.0 / (texts.index(entry.fact.text) + 1)
        age = current_day - entry.day
        if age <= 3:
            recall_sum_recent += 1.0 if hit_it else 0.0
            n_recent += 1
        if age > OLD_THRESHOLD_DAYS:
            recall_sum_old += 1.0 if hit_it else 0.0
            n_old += 1

    return DayMeasurement(
        day=current_day,
        store_size=len(eligible),
        recall_overall=recall_sum_all / max(1, n_queries),
        recall_recent=recall_sum_recent / max(1, n_recent) if n_recent else 0.0,
        recall_old=recall_sum_old / max(1, n_old) if n_old else 0.0,
        mrr_overall=mrr_sum / max(1, n_queries),
        retrieve_avg_ms=sum(retrieve_times) * 1000 / max(1, len(retrieve_times)),
    )


def _run_simulation(
    *, use_consolidate: bool, rng_seed: int = 7,
) -> list[DayMeasurement]:
    rng = random.Random(rng_seed)
    total_facts = FACTS_PER_DAY * N_DAYS
    all_facts = generate_templated_facts(total_facts, seed=42)

    mem = MemoryLayer.with_sbert("all-MiniLM-L6-v2")

    history: list[StoredEntry] = []
    daily: list[DayMeasurement] = []
    for day in range(1, N_DAYS + 1):
        start = (day - 1) * FACTS_PER_DAY
        todays = all_facts[start:start + FACTS_PER_DAY]
        for f in todays:
            mem.store(f.text)
            history.append(StoredEntry(day=day, fact=f))
        if use_consolidate:
            mem.consolidate()
        daily.append(_measure_day(mem, day, history, rng))
    return daily


def _format_trend(rows: list[DayMeasurement]) -> str:
    header = (
        "| Day | Store size | Recall@3 overall | Recall recent (<=3d) "
        "| Recall old (>10d) | MRR@3 | Retrieve avg (ms) |"
    )
    sep = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for r in rows:
        old_str = f"{r.recall_old:.3f}" if r.day > OLD_THRESHOLD_DAYS else "—"
        lines.append(
            f"| {r.day} | {r.store_size} "
            f"| {r.recall_overall:.3f} | {r.recall_recent:.3f} "
            f"| {old_str} | {r.mrr_overall:.3f} "
            f"| {r.retrieve_avg_ms:.2f} |"
        )
    return "\n".join(lines)


def _summary(rows: list[DayMeasurement]) -> dict[str, float]:
    old_rows = [r for r in rows if r.day > OLD_THRESHOLD_DAYS]
    return {
        "overall_mean": sum(r.recall_overall for r in rows) / len(rows),
        "recent_mean": sum(r.recall_recent for r in rows) / len(rows),
        "old_mean": (
            sum(r.recall_old for r in old_rows) / max(1, len(old_rows))
            if old_rows
            else 0.0
        ),
        "final_overall": rows[-1].recall_overall,
        "final_old": rows[-1].recall_old,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/longitudinal_drift.md"),
    )
    args = p.parse_args()

    print("=== Flat cosine (no graph re-rank) ===")
    flat_rows = _run_simulation(use_consolidate=False)
    for r in flat_rows[::5]:
        print(
            f"  day={r.day} size={r.store_size} "
            f"Recall@3={r.recall_overall:.3f} old={r.recall_old:.3f}"
        )

    flat_stats = _summary(flat_rows)

    lines = [
        "# Longitudinal Drift — Does Old-Memory Recall Hold?",
        "",
        f"**Simulation:** {N_DAYS} days × {FACTS_PER_DAY} new facts/day "
        f"= {N_DAYS * FACTS_PER_DAY} facts total. Each day we run "
        f"{QUERIES_PER_DAY} recency-biased queries "
        "(50% facts from last 3 days, 30% 4–10 days old, 20% 11+ days "
        "old) and score Recall@3 against the full history.",
        "",
        "Motivation: vector-DB + periodic-purge schemes (Mem0-style) "
        "can silently drop old memories; graph-based systems can "
        "erode old embeddings if consolidation updates them. This "
        "benchmark isolates that failure mode.",
        "",
        "## Results",
        "",
        _format_trend(flat_rows),
        "",
        "## Summary",
        "",
        f"- Overall mean Recall@3 across {N_DAYS} days: "
        f"**{flat_stats['overall_mean']:.3f}**",
        f"- Mean Recall@3 on queries targeting recent (≤3d) facts: "
        f"**{flat_stats['recent_mean']:.3f}**",
        f"- Mean Recall@3 on queries targeting old (>10d) facts "
        f"(post day {OLD_THRESHOLD_DAYS + 1}): "
        f"**{flat_stats['old_mean']:.3f}**",
        f"- Final day Recall@3 overall: **{flat_stats['final_overall']:.3f}** "
        f"(old: {flat_stats['final_old']:.3f})",
        "",
        "**Interpretation:** if old-fact Recall ≈ overall Recall, "
        "SOMA's memory doesn't drift. A gap (old < overall) would "
        "signal degradation; this benchmark has a measurement "
        "against which to falsify SOMA's 'memories don't rot' claim.",
        "",
        "---",
        "",
        "Generated by `benchmarks/run_longitudinal_drift.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
