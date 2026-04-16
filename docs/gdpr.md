# GDPR-grade forgetting in SOMA

This page covers the technical workflow SOMA supports for honouring
right-to-erasure ("right to be forgotten") requests under the GDPR
and comparable regimes. It is not a compliance certification, and
SOMA makes no certification claim. Operators remain responsible for
authenticating data-subject requests, retaining the audit trail,
respecting legal-hold rules, and every other process obligation.

## What SOMA guarantees

SOMA ships these primitives:

- **Dry-run inventory** — `cm.forget(..., dry_run=True)` returns a
  `ForgetPreview` listing the turn / fact / summary ids that a live
  call would touch. Safe to surface to a data-subject portal before
  any deletion lands.
- **Cascading deletion** — a live `cm.forget(...)` removes matching
  raw turns, their derived facts (linked via `metadata.source_turn_id`),
  and handles summaries whose range overlaps the matched turns. Facts
  are deleted before turns so the source-turn back-pointer never
  dangles in an intermediate state.
- **Summary cascade options** — by default (`summary_strategy="regen"`)
  summaries with partial coverage are regenerated from surviving
  turns so the long-term memory skeleton keeps accurate high-level
  history. Callers who need absolute certainty that the scrubbed
  subject can't leak into regenerated text opt into
  `summary_strategy="drop"` — every matched summary is removed
  outright and the LLM is never called for regeneration.
- **Multi-user scoping** — when entries carry `metadata.user_id`
  (Phase 12), the REST endpoint and library call both respect it.
  Admins can forget another user's data; the audit record carries
  both the caller (`user_id`) and the subject (`target_user_id`).
- **Append-only audit trail** — every `forget()` call (live or
  dry-run) emits one JSON line to the path configured by
  `SOMA_FORGET_AUDIT_PATH`.

## What SOMA does **not** guarantee

- **No compliance badge.** SOMA does not claim SOC 2, ISO 27001, or
  GDPR certification. Deploys that need a certified substrate should
  wrap SOMA in their already-certified operational envelope.
- **No identity verification.** SOMA trusts `principal.sub` from the
  JWT you issued. Verifying that the caller is the data subject (or
  authorised to act on their behalf) is the operator's job — a
  compromised token forging a user-id can delete data.
- **No retention-policy enforcement.** The audit trail grows forever;
  SOMA does not rotate or redact it. Use `logrotate` / cron / your
  SIEM pipeline.
- **No guaranteed recovery-after-forget.** Once a live forget lands,
  the data is gone from the bundle. If you need a grace period,
  snapshot the bundle before the call.
- **Embeddings derived from deleted text.** SOMA only stores vectors
  keyed by the entry id, so deleting the entry removes the vector
  too. There is no separate embedding-lake to scrub.

## Operator responsibilities

Before you turn the REST `/forget` endpoint on for end-users:

1. **Authenticate the data-subject request.** Don't trust an un-
   verified email — tie the forget call to a JWT issued only after
   the subject proved their identity out of band.
2. **Retain the audit trail.** Operators are legally exposed on the
   "we honoured the request" side; `SOMA_FORGET_AUDIT_PATH` is your
   evidence. Ship the file to your SIEM / WORM store; monitor the
   `soma.forget_audit` logger for write failures.
3. **Respect legal holds.** If a subject's data is under an active
   legal hold, do not call `cm.forget()` until the hold lifts. SOMA
   has no built-in hold primitive.
4. **Document your process.** A clean SOMA call does not replace a
   documented-and-reviewed data-subject-request procedure. The
   audit trail records what SOMA did, not why your ops team
   decided to run the call.

## REST usage

The `POST /forget` endpoint accepts both the legacy single-id shape
and the Phase 37 criteria shape. Responses mirror the library
dataclasses.

### Dry-run preview

```bash
curl -X POST http://localhost:8420/forget \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text_matches": "gardening",
    "user_id": "alice",
    "dry_run": true
  }'
```

Response (a `ForgetPreview`):

```json
{
  "raw_turns": ["t-1", "t-2"],
  "derived_facts": ["f-7"],
  "summaries": ["s-3"],
  "total_vectors": 4
}
```

### Live delete

Same body without `dry_run`:

```bash
curl -X POST http://localhost:8420/forget \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text_matches": "gardening",
    "user_id": "alice"
  }'
```

Response (a `ForgetResult`):

```json
{
  "deleted_turns": ["t-1", "t-2"],
  "deleted_facts": ["f-7"],
  "deleted_summaries": [],
  "regenerated_summaries": ["s-3-new"],
  "total_deleted": 3
}
```

Regenerated summaries are not counted in `total_deleted` because the
entry still exists, just with new text and a new id. The old id is
implicit in the swap — callers that held a reference should re-point
at the new id.

### Conservative drop (no LLM)

Pass `summary_strategy="drop"` to skip regeneration entirely:

```bash
curl -X POST http://localhost:8420/forget \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text_matches": "gardening",
    "summary_strategy": "drop"
  }'
```

Every matched summary is deleted; `regenerated_summaries` will be
`[]`. This is the recommended mode when the LLM is unavailable or
when the subject asked for a hard forget and you'd rather lose the
summary than risk a regen that still references them.

### Legacy single-entry delete

Unchanged from Phase 4 — POST `{"node_id": "<id>"}`:

```bash
curl -X POST http://localhost:8420/forget \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"node_id": "abc-123"}'
```

Returns `{"removed": true}` on success, 404 when the id is unknown.
Audit records are written for both branches.

## Audit trail

### Enable

```bash
export SOMA_FORGET_AUDIT_PATH=/var/log/soma/forget-audit.jsonl
```

Parent directories are created on first write. A sink with no path
is a no-op, so pre-Phase-37 deploys see identical behaviour until
they opt in.

### Opt out

```bash
export SOMA_FORGET_AUDIT_DISABLE=1
```

Useful when you already pipe forget events through a separate
auditing stack and don't want duplicate records. The `PATH` env-var
is still honoured, so this is a two-variable setup that lets you
toggle auditing without removing the path configuration.

### Record shape

One JSON object per line. Example (dry-run from an admin):

```json
{
  "ts": "2026-04-16T18:22:31.415Z",
  "user_id": "ops-root",
  "target_user_id": "alice",
  "criteria": {"user_id": "alice"},
  "dry_run": true,
  "result": {
    "raw_turns": 2,
    "derived_facts": 1,
    "summaries": 1,
    "total_vectors": 4
  }
}
```

Live delete example:

```json
{
  "ts": "2026-04-16T18:22:58.102Z",
  "user_id": "ops-root",
  "target_user_id": "alice",
  "criteria": {"user_id": "alice"},
  "dry_run": false,
  "result": {
    "deleted_turns": 2,
    "deleted_facts": 1,
    "deleted_summaries": 0,
    "regenerated_summaries": 1,
    "total_deleted": 3
  }
}
```

Fields:

- `ts` — ISO-8601 UTC timestamp with `Z` suffix. Milliseconds are
  included; microseconds are truncated.
- `user_id` — the caller's principal.sub (JWT claim), or
  `"anonymous"` when auth is disabled.
- `target_user_id` — the data subject when different from the
  caller; `null` when the caller scrubbed their own data.
- `criteria` — the criterion dict passed to `forget()`. Values are
  verbatim — do not put secrets in criterion strings.
- `dry_run` — `true` for previews, `false` for live deletes.
- `result` — count-only summary. Dry-run records carry the preview
  shape (`raw_turns` / `derived_facts` / `summaries` / `total_vectors`);
  live records carry the result shape (`deleted_*` / `regenerated_summaries` /
  `total_deleted`).

### Rotation and retention

SOMA does not rotate the audit file. Every operator-hostable tool
with the word "rotate" in its name will work:

- `logrotate` with `copytruncate` and daily rolling: standard.
- `cron` invoking `mv` + `HUP` when a bundle reloads: also fine; the
  sink reopens the path on the next write.
- A shipping agent (Vector / Fluent Bit) pushing lines to S3 / a SIEM:
  point the agent at the JSONL path.

Each record is ≤ 1 KB by design (counts only, not id lists) so file
growth is proportional to the number of forget events, not the
amount of data forgotten.

## Library usage (no REST)

Same contract as the endpoint — useful for scheduled jobs or
batch-ingest pipelines that don't run behind the REST server.

```python
from soma.forget_audit import ForgetAuditSink
from soma.memory.conversational import ConversationalMemory

cm = ConversationalMemory(
    memory=my_memory_layer,
    llm=my_llm_backend,
    session_id="support-desk",
    audit_sink=ForgetAuditSink.from_env(),
)

preview = cm.forget(
    text_matches="social security",
    dry_run=True,
    actor="ops-root",
)
print(f"Would forget {preview.total_vectors} entries")

result = cm.forget(
    text_matches="social security",
    user_id="alice",
    summary_strategy="drop",
    actor="ops-root",
)
print(f"Deleted {result.total_deleted} entries")
```

The `actor=` kwarg on `forget()` is what the REST endpoint uses to
stamp `principal.sub` onto the audit record. When unset, the sink
falls back to the `ConversationalMemory`'s constructor-level
`user_id`, or `"anonymous"`.

## Failure modes

- **Audit write fails.** Logged at `WARNING` on the
  `soma.forget_audit` logger; the forget call itself still lands.
  This is intentional — a broken audit pipeline must not block a
  legitimate delete. Monitor the logger and treat the warning as a
  compliance incident.
- **LLM regeneration fails.** Under the default `summary_strategy=
  "regen"`, any exception from the summary LLM call degrades to
  drop-the-summary with a WARNING log. Under `"drop"` the LLM is
  never called, so this failure mode is eliminated.
- **Concurrent forget races.** Two `forget()` calls matching the
  same entry both record the delete in their preview; only one
  actually removes the entry. Audit records both attempts — the
  losing caller sees zero deletions in its result shape.
- **Empty criteria.** Both the REST endpoint and the library raise
  (400 / `ValueError`) when no criterion is passed. Zero-criteria
  forget would implicitly match everything in the session, which is
  never what callers want.
