"""The committed TypeScript OpenAPI snapshot must match the live app's
spec. If a PR adds a route without regenerating the client, CI fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

_REPO = Path(__file__).resolve().parents[2]
_SNAPSHOT = _REPO / "clients" / "typescript" / "openapi.json"


def _normalize(spec: dict) -> dict:
    """Strip fields that vary per-process so the diff is stable."""
    spec = json.loads(json.dumps(spec))  # deep copy
    spec.pop("servers", None)
    return spec


def test_openapi_snapshot_matches_live_app():
    from soma.serve import app

    live = _normalize(app.openapi())
    snapshot = _normalize(json.loads(_SNAPSHOT.read_text(encoding="utf-8")))

    live_paths = set(live["paths"].keys())
    snap_paths = set(snapshot["paths"].keys())
    missing_in_snapshot = live_paths - snap_paths
    extra_in_snapshot = snap_paths - live_paths
    assert not missing_in_snapshot, (
        "live app has routes not in the committed snapshot:\n  "
        + "\n  ".join(sorted(missing_in_snapshot))
        + "\n\nRegenerate: python -c 'import json; from soma.serve import app; "
        "json.dump(app.openapi(), open(\"clients/typescript/openapi.json\", \"w\"), indent=2)'"
    )
    assert not extra_in_snapshot, (
        "snapshot has routes not in the live app:\n  "
        + "\n  ".join(sorted(extra_in_snapshot))
    )

    # Deep compare paths.
    assert live["paths"] == snapshot["paths"], (
        "OpenAPI paths diverge between live app and snapshot. "
        "Regenerate per the command above."
    )
