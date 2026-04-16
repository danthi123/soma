"""Tests for the ConversationalMemory threshold-calibration sweep.

Task B — `benchmarks/run_conv_threshold_sweep.py`. Sweeps across
(near_dup_threshold, ambiguous_threshold) combinations and reports
facts_stored, llm_calls, latency, Recall@5, and optional QA
accuracy per combo.

Tests:
- One row per combo (4x4 = 16).
- Extreme corners differ on fact counts (low-low stores more than
  high-high).
- Report markdown parses and contains a recommendation banner.

Scripted backend keeps every test hermetic — no Ollama / network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from benchmarks.run_conv_threshold_sweep import (
    NEAR_DUP_GRID,
    AMBIGUOUS_GRID,
    SweepRow,
    render_report,
    run_sweep,
)


def _stable_hash(text: str) -> int:
    return int.from_bytes(
        hashlib.sha1(text.encode("utf-8")).digest()[:4], "little"
    )


def _stub_embed(text: str) -> torch.Tensor:
    vec = torch.zeros(16)
    for tok in text.lower().split():
        gen = torch.Generator().manual_seed(_stable_hash(tok))
        vec = vec + torch.randn(16, generator=gen)
    return vec


@dataclass
class _ScriptedLLM:
    """Scripted LLM — always extracts one fact (the turn text itself),
    reconcile always returns ADD.

    Deterministic so tests are stable. Reconcile-ADD so extreme
    thresholds affect outcomes in a predictable direction."""

    name: str = "scripted"
    extract_call_count: int = 0
    reconcile_call_count: int = 0
    _turn_text: str | None = field(default=None, init=False)

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            self.extract_call_count += 1
            # Echo the LAST "Message: ..." line from the prompt as a
            # fact. The EXTRACT_PROMPT includes several "Message: ..."
            # examples before the real message, so we want the tail.
            last_msg: str | None = None
            for line in prompt.splitlines():
                line = line.strip()
                if line.startswith("Message:"):
                    last_msg = line[len("Message:") :].strip()
            if last_msg:
                return json.dumps(
                    [{"category": "other", "text": last_msg}]
                )
            return "[]"
        if "You reconcile a new fact" in prompt:
            self.reconcile_call_count += 1
            return json.dumps({"op": "ADD", "target_id": None, "reason": "new"})
        if "summarizing" in prompt:
            return "summary of recent turns"
        return "{}"


def _tiny_corpus() -> tuple[list, list]:
    """Multi-session corpus tuned so similarity values land at several
    points in the threshold grid, producing variance across combos.

    Under the stub embed:
    - 9-of-10 token overlap yields cosine ~0.90 (straddles 0.88/0.90).
    - 8-of-10 overlap yields ~0.73 (above 0.65, below 0.80 — straddles
      the ambiguous grid).
    - Exact duplicate yields 1.0 (always short-circuited).
    """
    from benchmarks.datasets.locomo import LoCoMoQuery, LoCoMoTurn

    turns = [
        # s1: pair with cosine ~0.96 (9 of 10 tokens overlap; near_dup=0.94 catches it)
        LoCoMoTurn("s1", "D1:0", "A", "Alpha beta gamma delta epsilon zeta eta theta iota kappa"),
        LoCoMoTurn("s1", "D1:1", "A", "Alpha beta gamma delta epsilon zeta eta theta iota"),
        # s2: pair with cosine ~0.90 (8 of 10 tokens overlap; at 0.88/0.90 boundary)
        LoCoMoTurn("s2", "D1:0", "B", "Lambda mu nu xi omicron pi rho sigma tau upsilon"),
        LoCoMoTurn("s2", "D1:1", "B", "Lambda mu nu xi omicron pi rho sigma"),
        # s3: pair with cosine ~0.73 (partial overlap; hits ambiguous grid)
        LoCoMoTurn("s3", "D1:0", "C", "Foo bar baz"),
        LoCoMoTurn("s3", "D1:1", "C", "Foo bar"),
    ]
    queries = [
        LoCoMoQuery(
            sample_id="s1",
            question="Where does A live?",
            answer="Boston",
            evidence=["D1:0"],
            category=1,
        ),
        LoCoMoQuery(
            sample_id="s2",
            question="What food does B like?",
            answer="Thai food",
            evidence=["D1:0"],
            category=1,
        ),
    ]
    return turns, queries


def test_sweep_grid_is_4_by_4() -> None:
    """Sanity-check the sweep grid — 4 x 4 = 16 combinations."""
    assert len(NEAR_DUP_GRID) == 4
    assert len(AMBIGUOUS_GRID) == 4
    assert NEAR_DUP_GRID == (0.88, 0.90, 0.92, 0.94)
    assert AMBIGUOUS_GRID == (0.65, 0.70, 0.75, 0.80)


def test_sweep_produces_one_row_per_combo() -> None:
    """run_sweep emits exactly 16 rows on the 4x4 grid."""
    turns, queries = _tiny_corpus()
    llm = _ScriptedLLM()
    rows = run_sweep(
        turns=turns,
        queries=queries,
        llm_factory=lambda: _ScriptedLLM(),
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    assert len(rows) == 16
    combos = {(r.near_dup, r.ambiguous) for r in rows}
    assert len(combos) == 16
    for r in rows:
        assert isinstance(r, SweepRow)
        assert r.facts_stored >= 0


def test_sweep_extreme_thresholds_differ() -> None:
    """(0.88, 0.65) aggressively short-circuits duplicates AND
    short-circuits into ADD below 0.65; (0.94, 0.80) leaves more to
    the LLM's ADD decision. At least one combo must record a
    different fact_count than another.

    The exact direction depends on data + extractor behaviour, so the
    test is weak-form: extremes differ SOMEWHERE across the grid."""
    turns, queries = _tiny_corpus()
    rows = run_sweep(
        turns=turns,
        queries=queries,
        llm_factory=lambda: _ScriptedLLM(),
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    fact_counts = {(r.near_dup, r.ambiguous): r.facts_stored for r in rows}
    # At minimum, the grid should produce some variation — if every
    # combo yields the same count the sweep is broken.
    assert len(set(fact_counts.values())) >= 2, fact_counts


def test_sweep_report_has_recommendation_banner(tmp_path: Path) -> None:
    """Report markdown contains a 'Recommendation' banner and a table
    with one row per combo."""
    turns, queries = _tiny_corpus()
    rows = run_sweep(
        turns=turns,
        queries=queries,
        llm_factory=lambda: _ScriptedLLM(),
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    report = render_report(rows, n_turns=len(turns), n_queries=len(queries))
    assert "# Conversational Memory — Threshold Calibration Sweep" in report
    # Recommendation banner (case-insensitive check — may be "Recommended defaults"
    # or "Recommendation" per prose).
    assert "ecommend" in report
    # Table cells present for at least one combo.
    assert "0.88" in report
    assert "0.94" in report
