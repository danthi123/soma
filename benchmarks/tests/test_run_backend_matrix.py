"""Smoke tests for the adapter-matrix bench harness.

Tiny-scale (n=200) runs just to prove the code path compiles and
produces a valid report row. Avoids heavy deps (sbert, real 1M
corpus) so CI stays fast.
"""

from __future__ import annotations

from benchmarks.run_backend_matrix import render_report, run_matrix


def test_backend_matrix_runs_inproc_only_without_qdrant(tmp_path) -> None:
    rows = run_matrix(
        n=200,
        include_qdrant_local=False,
        include_qdrant_http=False,
        smoke=False,  # also exercises InProcHNSW
    )
    # At least InProcFlat + InProcHNSW.
    names = [r.backend for r in rows]
    assert "InProcFlat" in names
    assert "InProcHNSW" in names

    report_path = tmp_path / "backend_matrix.md"
    report_path.write_text(render_report(rows, n=200), encoding="utf-8")
    content = report_path.read_text(encoding="utf-8")
    assert "Backend Matrix" in content
    assert "InProcFlat" in content


def test_backend_matrix_smoke_mode_skips_hnsw(tmp_path) -> None:
    rows = run_matrix(
        n=100,
        include_qdrant_local=False,
        include_qdrant_http=False,
        smoke=True,
    )
    names = [r.backend for r in rows]
    assert names == ["InProcFlat"]
    # Report renders without raising.
    _ = render_report(rows, n=100)


def test_backend_matrix_row_has_all_fields(tmp_path) -> None:
    rows = run_matrix(
        n=100,
        include_qdrant_local=False,
        include_qdrant_http=False,
        smoke=True,
    )
    r = rows[0]
    assert r.n == 100
    assert r.store_total_s >= 0.0
    assert r.retrieve_p50_ms >= 0.0
    # Recall@10 against self must be 1.0 (reference is itself).
    assert r.recall_at_10 == 1.0
