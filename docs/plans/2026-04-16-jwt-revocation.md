# JWT revocation — design decision (file vs Redis vs in-memory)

**Context:** Phase 4 (2026-04-16) shipped JWT auth but punted revocation. Today,
revoking a leaked token means either rotating `SOMA_JWT_SECRET` (kills every
live token) or waiting for `exp` (default 30 days). The operator gap is real
and the fix is a per-`jti` blocklist. This memo picks a backend.

**Decision:** file-backed JSONL is the default; Redis is a deferred opt-in.
In-memory is not offered as a distinct mode — it's what the file-backed store
caches when empty.

---

## Option A — File-backed JSONL (recommended default)

Append-only JSONL at `SOMA_JWT_BLOCKLIST_PATH` (default
`./data/jwt-blocklist.jsonl`). One line per revoked `jti`:

```json
{"jti": "b8f1-...", "revoked_at": 1713225600, "reason": "leaked on slack", "exp": 1715817600}
```

- Read on startup into a `set[str]` + poll `mtime` (every 30 s) for peer
  appends.
- Writes acquire a `portalocker.Lock` (already a Phase 1 WAL dep).
- `gc_expired()` rewrites the file dropping entries past their original
  `exp` (bounded growth).

**Pros:**
- Zero infra. Ships with SOMA, runs next to the WAL.
- Survives restarts. Bundle-portable.
- Matches SOMA's local-first ethos.
- Works for the 80%-case single-server deploy today.

**Cons:**
- Poll lag (up to 30 s) between processes on the same host. Acceptable —
  a 30 s revocation delay on a stolen token is orders-of-magnitude better
  than the current "wait 30 days" story.
- File lock coordination scales to tens of workers, not thousands. Fine
  for single-host uvicorn; operators running kubelet-scale need Redis.
- Unbounded file growth without `gc_expired()`. Mitigated by the GC
  subcommand + operator cron cadence.

## Option B — Redis-backed (deferred opt-in)

`SETEX jti-<jti> {exp_seconds} revoked`. New optional extra
`soma[redis-revocation]` pulling `redis>=5`.

**Pros:**
- Instant propagation across workers.
- Automatic TTL — no GC needed.
- The right answer for multi-replica / multi-node deploys.

**Cons:**
- Adds infra surface area. Operators running single-host SOMA don't need
  it; forcing a redis-py dep would cut against the local-first story.
- Needs connection-pool / reconnect / fail-open-vs-closed design
  decisions that aren't worth making before we have a concrete operator
  request.

Status: **deferred to a follow-up sprint**. The `BlocklistBackend`
Protocol designed below leaves the door open — add a
`RedisBlocklist` class and wire a second factory branch when demand
arrives.

## Option C — In-memory only

`set[str]` in the process; no persistence.

**Pros:** zero deps, fast.

**Cons:** useless across restarts; multi-worker desync.

**Not offered as a distinct mode.** This is a footgun — an operator who
picked "memory" would be surprised the revoke didn't survive a reload. The
file-backed store *already* behaves as an in-memory cache once loaded;
adding a second mode doubles the surface for no win.

---

## Decision matrix

| Criterion              | File (A) | Redis (B) | Memory (C) |
|------------------------|----------|-----------|------------|
| Zero infra             | yes      | no        | yes        |
| Survives restart       | yes      | yes       | **no**     |
| Multi-process coord    | ok (poll)| excellent | **no**     |
| Local-first aligned    | yes      | neutral   | yes        |
| Propagation latency    | ~30 s    | instant   | N/A        |
| Unbounded growth       | GC needed| TTL auto  | process-lifetime |
| Ship this sprint       | **yes**  | deferred  | no         |

File wins for SOMA's audience. Redis stays on the roadmap.

---

## Implementation surface (Part B)

- `src/soma/auth_revocation.py` — `BlocklistBackend` Protocol,
  `RevocationRecord` dataclass, `FileBlocklist` impl, `null_blocklist()`,
  `blocklist_from_env()` factory.
- `verify_token(..., blocklist=None)` — optional kwarg. If provided +
  token's `jti` is in the blocklist → raise `jwt.InvalidTokenError("token
  revoked")`.
- `serve.py` — module-level `_blocklist = blocklist_from_env()` at import
  time; `require_auth` passes it into `verify_token`; `record_auth_failure`
  gains a `revoked_token` reason.
- `cli.py` — `soma auth revoke` / `list-revoked` / `gc` subcommands.
- Docs — new "Revocation" section in `docs/auth.md`; deferred-items update.

## Explicit non-goals (this sprint)

- Redis backend (see Option B).
- Hashed-token store (remains tier-2 deferred).
- Refresh-token flow (unchanged from Phase 4 scope).
- Any change to `exp` handling or the legacy `SOMA_API_KEY` escape hatch.
