"""Tests for Phase 37: ``POST /forget`` conversational endpoint.

Phase 37 extends the existing single-id ``POST /forget`` (Phase 4) to
also accept a ConversationalMemory-style criteria body:

    {"text_matches": "gardening", "dry_run": false}

Backward compat: callers that still POST ``{"node_id": "..."}`` get
the legacy behaviour unchanged. Criteria-shape bodies route through
:meth:`ConversationalMemory.forget`, return the matching
:class:`ForgetResult` / :class:`ForgetPreview` shape, and emit one
audit record to the module-level :class:`ForgetAuditSink`.

Covered here:

- Dry-run returns the preview shape.
- Live delete returns the result shape.
- ``principal.sub`` is stamped on the audit record; when the caller's
  JWT sub differs from a ``user_id`` in the request body, the audit
  record carries both (caller + target_user_id).
- Read-only tokens get 403.
- Zero-criteria body gets 400.
- ``case_sensitive`` flows through.
- ``summary_strategy="drop"`` flows through (Task 3 integration).
- When no conversational memory is configured, the criteria branch
  returns 501 with a clear message. The legacy node_id branch still
  works.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from soma import serve as _initial_serve
from soma.auth import issue_token
from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory

_SECRET = "forget-endpoint-test-secret-32chrs"


def _stub_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(16, generator=g)


@dataclass
class _ScriptedLLM:
    extract_replies: list[str] = field(default_factory=list)
    name: str = "scripted"
    extract_calls: int = 0

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            i = self.extract_calls
            self.extract_calls += 1
            if i < len(self.extract_replies):
                return self.extract_replies[i]
            return "[]"
        if "You reconcile a new fact" in prompt:
            return json.dumps({"op": "ADD", "target_id": None, "reason": "new"})
        if "summarizing a short segment" in prompt:
            return "summary text"
        return "{}"


def _fresh_serve(
    monkeypatch: pytest.MonkeyPatch, **env: str
) -> Any:
    """Reimport soma.serve with a hermetic env snapshot.

    Clears the memory cache and both the audit-path env-vars so the
    fresh module picks up whatever the test sets.
    """
    for key in (
        "SOMA_API_KEY",
        "SOMA_JWT_SECRET",
        "SOMA_JWT_ALG",
        "SOMA_JWT_PUBLIC_KEY_PATH",
        "SOMA_FORGET_AUDIT_PATH",
        "SOMA_FORGET_AUDIT_DISABLE",
    ):
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reloaded = importlib.reload(_initial_serve)
    reloaded._mem_cache.clear()
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    reloaded._mem_cache["__default__"] = mem
    return reloaded, mem


def _inject_cm(serve_mod: Any, mem: MemoryLayer) -> ConversationalMemory:
    """Build a ConversationalMemory around ``mem`` and wire it into the server.

    The server's ``_get_conversational_memory()`` is monkey-patched to
    return this CM so the criteria branch of ``POST /forget`` has a
    target to call.
    """
    llm = _ScriptedLLM()
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        summary_every=1000,
        audit_sink=serve_mod._forget_audit,
    )
    serve_mod._get_conversational_memory = lambda name=None: cm
    return cm


# ----------------------------------------------------------------------
# 1. Dry-run returns the preview shape
# ----------------------------------------------------------------------
def test_forget_endpoint_dry_run_returns_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded, mem = _fresh_serve(monkeypatch)
    cm = _inject_cm(reloaded, mem)
    cm.add_message("user", "I love gardening")
    client = TestClient(reloaded.app)
    r = client.post("/forget", json={"text_matches": "gardening", "dry_run": True})
    assert r.status_code == 200, r.text
    body = r.json()
    # Preview shape: raw_turns / derived_facts / summaries / total_vectors.
    assert "raw_turns" in body
    assert "derived_facts" in body
    assert "summaries" in body
    assert "total_vectors" in body
    assert isinstance(body["raw_turns"], list)
    assert body["raw_turns"], "sanity: gardening turn should be previewed"


# ----------------------------------------------------------------------
# 2. Delete returns result shape
# ----------------------------------------------------------------------
def test_forget_endpoint_delete_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded, mem = _fresh_serve(monkeypatch)
    cm = _inject_cm(reloaded, mem)
    cm.add_message("user", "I love gardening")
    client = TestClient(reloaded.app)
    r = client.post("/forget", json={"text_matches": "gardening"})
    assert r.status_code == 200, r.text
    body = r.json()
    # Result shape: deleted_* + total_deleted.
    assert "deleted_turns" in body
    assert "deleted_facts" in body
    assert "deleted_summaries" in body
    assert "regenerated_summaries" in body
    assert "total_deleted" in body
    assert body["total_deleted"] >= 1


# ----------------------------------------------------------------------
# 3. Write permission required
# ----------------------------------------------------------------------
def test_forget_endpoint_requires_write_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded, mem = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    cm = _inject_cm(reloaded, mem)
    cm.add_message("user", "gardening")
    client = TestClient(reloaded.app)
    read_token = issue_token(
        sub="alice",
        bundles={"__default__": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.post(
        "/forget",
        json={"text_matches": "gardening"},
        headers={"Authorization": f"Bearer {read_token}"},
    )
    assert r.status_code == 403, r.text


# ----------------------------------------------------------------------
# 4. Zero criteria returns 400
# ----------------------------------------------------------------------
def test_forget_endpoint_zero_criteria_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded, mem = _fresh_serve(monkeypatch)
    _inject_cm(reloaded, mem)
    client = TestClient(reloaded.app)
    r = client.post("/forget", json={})
    # Empty body with no criteria and no node_id — refuse rather than
    # implicitly wipe.
    assert r.status_code == 400, r.text
    assert "criterion" in r.json().get("detail", "").lower()


# ----------------------------------------------------------------------
# 5. Audit record carries principal.sub
# ----------------------------------------------------------------------
def test_forget_endpoint_writes_audit_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit_path = tmp_path / "audit.jsonl"
    reloaded, mem = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_FORGET_AUDIT_PATH=str(audit_path),
    )
    cm = _inject_cm(reloaded, mem)
    cm.add_message("user", "gardening today")
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alice",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.post(
        "/forget",
        json={"text_matches": "gardening"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    assert audit_path.exists(), "audit sink must have written a record"
    records = [
        json.loads(line)
        for line in audit_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    rec = records[0]
    assert rec["user_id"] == "alice"
    assert rec["dry_run"] is False
    assert rec["criteria"]["text_matches"] == "gardening"


def test_forget_endpoint_records_target_user_when_different(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit_path = tmp_path / "audit.jsonl"
    reloaded, mem = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_FORGET_AUDIT_PATH=str(audit_path),
    )
    cm = _inject_cm(reloaded, mem)
    # Seed a turn owned by "alice".
    cm.add_message("user", "alice-owned turn", user_id="alice")
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="ops-root",
        bundles={"__default__": ["admin"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.post(
        "/forget",
        json={"user_id": "alice"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    records = [
        json.loads(line)
        for line in audit_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    rec = records[0]
    assert rec["user_id"] == "ops-root"
    assert rec["target_user_id"] == "alice"


# ----------------------------------------------------------------------
# 6. case_sensitive flag flows through
# ----------------------------------------------------------------------
def test_forget_endpoint_case_sensitive_flag_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded, mem = _fresh_serve(monkeypatch)
    cm = _inject_cm(reloaded, mem)
    cm.add_message("user", "GARDENING TIME")  # uppercase
    client = TestClient(reloaded.app)
    # case_sensitive=True with lowercase pattern must miss.
    r = client.post(
        "/forget",
        json={"text_matches": "gardening", "case_sensitive": True, "dry_run": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["raw_turns"] == []
    # Same pattern without case_sensitive hits.
    r2 = client.post(
        "/forget",
        json={"text_matches": "gardening", "dry_run": True},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["raw_turns"], "case-insensitive should match"


# ----------------------------------------------------------------------
# 7. summary_strategy flows through
# ----------------------------------------------------------------------
def test_forget_endpoint_summary_strategy_drop_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded, mem = _fresh_serve(monkeypatch)
    cm = _inject_cm(reloaded, mem)
    cm.add_message("user", "gardening turn 0")
    cm.add_message("user", "unrelated turn 1")
    # Manual summary: craft one covering both turns so it overlaps
    # gardening but has a survivor (turn 1). summary_strategy="drop"
    # must drop it outright without calling the LLM for regen.
    mem.store(
        "summary covering 0..1",
        metadata={
            "session_id": "s",
            "type": "summary",
            "summarized_turn_start": 0,
            "summarized_turn_end": 1,
        },
    )
    client = TestClient(reloaded.app)
    r = client.post(
        "/forget",
        json={"text_matches": "gardening", "summary_strategy": "drop"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted_summaries"], "drop-strategy must delete the summary"
    assert body["regenerated_summaries"] == []


# ----------------------------------------------------------------------
# 8. Legacy node_id path still works
# ----------------------------------------------------------------------
def test_forget_endpoint_legacy_node_id_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded, mem = _fresh_serve(monkeypatch)
    _inject_cm(reloaded, mem)
    nid = mem.store("ephemeral")
    client = TestClient(reloaded.app)
    r = client.post("/forget", json={"node_id": nid})
    assert r.status_code == 200, r.text
    # Legacy shape: {"removed": true}.
    assert r.json() == {"removed": True}


# ----------------------------------------------------------------------
# 9. 501 when conversational mode isn't configured
# ----------------------------------------------------------------------
def test_forget_endpoint_criteria_without_cm_returns_501(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded, _mem = _fresh_serve(monkeypatch)
    # Do NOT inject a CM. _get_conversational_memory() returns None.
    client = TestClient(reloaded.app)
    r = client.post("/forget", json={"text_matches": "anything"})
    assert r.status_code == 501, r.text
    assert "conversational" in r.json().get("detail", "").lower()


def test_forget_endpoint_node_id_without_cm_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy behaviour survives even when conversational mode isn't set."""
    reloaded, mem = _fresh_serve(monkeypatch)
    nid = mem.store("bar")
    client = TestClient(reloaded.app)
    r = client.post("/forget", json={"node_id": nid})
    assert r.status_code == 200, r.text
    assert r.json() == {"removed": True}
