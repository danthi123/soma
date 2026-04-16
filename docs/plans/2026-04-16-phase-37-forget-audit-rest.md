# Phase 37: Forget Audit Trail + REST Endpoint

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Close the GDPR-grade forgetting track. Adds
(1) an audit trail so every `forget()` call is logged with who-did-
what-when, (2) a `POST /forget` REST endpoint that wraps the
existing helper, (3) docs posture clarifying what GDPR guarantees
SOMA does and doesn't make.

**Architecture:**
- **Audit**: append-only JSONL at a path configured via
  `SOMA_FORGET_AUDIT_PATH` env var. Each record:
  ```json
  {
    "ts": "2026-04-16T18:22:31.415Z",
    "user_id": "alice",           // the caller (sub from JWT, or
                                 // "anonymous" if auth is off)
    "target_user_id": "bob",     // if forget was scoped to a user
    "criteria": {"text_matches": "gardening"},
    "dry_run": false,
    "result": {
      "deleted_turns": 3,
      "deleted_facts": 5,
      "deleted_summaries": 1,
      "regenerated_summaries": 1,
      "total_deleted": 9
    }
  }
  ```
- Audit is append-only. Never redacted, never rotated from SOMA
  itself (let operators run logrotate / cron). `SOMA_FORGET_AUDIT_DISABLE=1`
  turns off writes entirely (operators who log via a different
  pipeline).
- Dry-run calls are also logged — operators need visibility into
  "alice previewed forgetting bob's data at 10am" even if no delete
  followed.

- **REST endpoint**: `POST /forget` with a JSON body matching the
  helper's kwargs. Protected by `require_auth(write)` — forgetting
  data is a write operation. Request body:
  ```json
  {
    "text_matches": "gardening",
    "user_id": null,
    "subject": null,
    "dry_run": false
  }
  ```
  Response mirrors `ForgetResult` (or `ForgetPreview` on dry-run).
  On empty-criteria input, 400 Bad Request with a clear error.

- **Docs posture**: new section in `docs/auth.md` (or a new
  `docs/gdpr.md`) clarifying:
  * SOMA supports the technical right-to-erasure workflow.
  * Operators are responsible for: (a) authenticating data-subject
    requests, (b) retaining the audit trail, (c) respecting other
    legal-hold rules.
  * SOMA doesn't claim certification / compliance badges.
  * Regenerated summaries may retain information not obviously tied
    to the forgotten subject — the LLM has discretion there. If full
    certainty is required, use `summary_strategy="drop"` (add as a
    kwarg to `forget()` — opt in to always-drop rather than
    regenerate-or-drop).

**Out-of-scope (I will do centrally):** `CHANGELOG.md` final track
summary, `deferred-items.md` strikethrough (this phase finishes the
track).

---

### Task 1: Audit trail

**Files:**
- Create: `src/soma/forget_audit.py`
- Modify: `src/soma/memory/conversational.py` — call the audit sink
  on every `forget()` invocation
- Create: `tests/test_memory/test_forget_audit.py`

**API:**
```python
class ForgetAuditSink:
    def __init__(self, path: str | Path | None = None) -> None:
        # None / empty path / SOMA_FORGET_AUDIT_DISABLE=1 → no-op sink

    def emit(
        self,
        *,
        user_id: str,
        criteria: dict,
        dry_run: bool,
        result: ForgetResult | ForgetPreview,
    ) -> None:
        # Append JSONL record. Errors logged at WARNING but never
        # raised — the audit path failing shouldn't block the user's
        # forget operation.

    @classmethod
    def from_env(cls) -> "ForgetAuditSink":
        ...
```

**Tests:**
```python
def test_audit_writes_jsonl_record(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_FORGET_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    sink = ForgetAuditSink.from_env()
    sink.emit(user_id="alice", criteria={"text_matches": "x"},
              dry_run=False, result=_some_result())
    records = _read_jsonl(tmp_path / "audit.jsonl")
    assert len(records) == 1
    assert records[0]["user_id"] == "alice"

def test_audit_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_FORGET_AUDIT_PATH", str(tmp_path / "a.jsonl"))
    monkeypatch.setenv("SOMA_FORGET_AUDIT_DISABLE", "1")
    sink = ForgetAuditSink.from_env()
    sink.emit(...)
    assert not (tmp_path / "a.jsonl").exists()

def test_audit_none_path_is_noop(): ...

def test_audit_path_errors_warn_but_dont_raise(monkeypatch, caplog): ...

def test_audit_includes_dry_run_flag(): ...

def test_audit_includes_timestamp_iso8601(): ...

def test_conversational_memory_calls_audit_on_forget():
    # Integration: cm.forget(...) should emit one audit record.
```

**Step 5:** `git commit -m "feat(forget): ForgetAuditSink + JSONL audit trail"`

---

### Task 2: `POST /forget` REST endpoint

**Files:**
- Modify: `src/soma/serve.py`
- Extend: `tests/test_serve/test_jwt_auth.py` OR create
  `tests/test_serve/test_forget_endpoint.py`

**Endpoint:**
```python
from soma.memory.conversational import ForgetPreview, ForgetResult

class ForgetRequest(BaseModel):
    text_matches: str | None = None
    subject: str | None = None
    user_id: str | None = None
    case_sensitive: bool = False
    dry_run: bool = False

@app.post("/forget")
def post_forget(
    body: ForgetRequest,
    principal: Principal = Depends(require_auth(None, "write")),
) -> dict:
    cm = _get_conversational_memory()  # or however serve wires it
    try:
        outcome = cm.forget(
            text_matches=body.text_matches,
            subject=body.subject,
            user_id=body.user_id,
            case_sensitive=body.case_sensitive,
            dry_run=body.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    # Emit audit
    _forget_audit.emit(
        user_id=principal.sub,
        criteria=body.dict(exclude_defaults=True),
        dry_run=body.dry_run,
        result=outcome,
    )
    return outcome.__dict__
```

**Tests:**
```python
def test_forget_endpoint_dry_run_returns_preview(client):
    # POST /forget {"text_matches": "x", "dry_run": true}
    # → 200 with ForgetPreview shape

def test_forget_endpoint_delete_returns_result(client):
    # POST /forget {"text_matches": "x"} → 200 with ForgetResult shape

def test_forget_endpoint_requires_write_permission(client):
    # Read-only token → 403

def test_forget_endpoint_no_criteria_returns_400(client):
    # POST /forget {} → 400 + "at least one criterion" message

def test_forget_endpoint_writes_audit_record(client, tmp_path, monkeypatch):
    # End-to-end: call endpoint, verify audit file has matching record
    # including principal.sub.

def test_forget_endpoint_case_sensitive_flag_passes_through(client): ...
```

**Step 5:** `git commit -m "feat(serve): POST /forget endpoint + audit wiring"`

---

### Task 3: `summary_strategy` kwarg for conservative callers

**Files:**
- Modify: `src/soma/memory/conversational.py` — add
  `summary_strategy: Literal["regen", "drop"] = "regen"` to
  `forget()`. Phase 36's default was "regen"; `drop` forces drop
  even when survivors exist.
- Extend: test file from Phase 36 or the Phase 37 test file.

**Test:**
```python
def test_forget_summary_strategy_drop_forces_drop():
    # Partially-overlapping summary + summary_strategy="drop":
    # summary is dropped even though survivors exist.
    result = cm.forget(text_matches="x", summary_strategy="drop")
    assert result.deleted_summaries
    assert result.regenerated_summaries == []
```

**Step 5:** `git commit -m "feat(forget): summary_strategy=\"drop\" opt-in"`

---

### Task 4: Docs

**Files:**
- Create: `docs/gdpr.md` (or extend `docs/auth.md` — agent's call)
- Update `README.md` feature table: "GDPR-grade forgetting: yes"
- Update `docs/cookbook.md`: recipe showing a typical forget-flow
  (dry-run preview → confirm → delete).

**Step 5:** `git commit -m "docs(gdpr): forget workflow + compliance posture"`

---

### Final sanity

```bash
ruff check src/soma tests
pytest tests/test_memory tests/test_serve -q
```

Baseline post-Phase-36: ~+30 cumulative forget tests. Target Phase 37:
+~12 audit + 6 endpoint + 1 strategy = +~19 new, 0 regressions.

**Gotchas:**
- Audit file writes must be fsync'd (or at least flushed) per record
  — an audit record that doesn't survive a crash is worse than no
  audit at all.
- `ForgetAuditSink` should be lightweight and threadsafe: just a
  path + a lock around `open+write+flush`. No need to serialise the
  sink across processes — each worker opens its own handle and
  appends; OS-level write-append is atomic for small records on
  POSIX.
- Endpoint authentication: the principal doing the forget must be
  recorded in the audit. If the user_id in the request body differs
  from principal.sub (admin forgetting another user's data), both
  should be in the record.
- `summary_strategy="drop"` should be surfaced in the REST body too
  (add `summary_strategy: Literal["regen", "drop"] | None = None`
  to `ForgetRequest`).
- `docs/gdpr.md` must NOT claim any certification we don't have.
  Stick to capabilities ("SOMA supports the technical workflow")
  and responsibilities ("operators are responsible for X Y Z").
