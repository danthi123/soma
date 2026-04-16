# Phase 33: Cloud-Deploy Docs + Example

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Finish the S3/GCS track by documenting how to run SOMA
against cloud object storage on scale-to-zero platforms (Cloud Run /
Lambda / App Runner / Fly Machines). This phase is docs-heavy —
minimal code. Closes the arc started in Phase 30.

**Architecture:**
- New top-level `docs/cloud.md` covering:
  1. **When to reach for cloud storage** — ephemeral local disk, multi-
     replica, warm-start latency budget.
  2. **s3:// bundle URLs** — format, credentials, endpoint overrides.
  3. **gs:// bundle URLs** — project arg, ADC chain.
  4. **Deployment recipes**:
     - AWS Lambda + S3 bundle (zipped layer, Lambda config env vars,
       IAM role).
     - Google Cloud Run + GCS bundle (container, service account).
     - Fly Machines + Cloudflare R2 (endpoint-override example since
       R2 is S3-compatible).
  5. **Caveats**: save() is a batch operation, not an incremental WAL
     stream. Operators needing high-durability writes should save()
     on a schedule or before scale-down. This is documented as a
     known tradeoff; real WAL-to-cloud is a future phase.
  6. **Cost considerations**: GET/PUT pricing, egress, storage class.

**Out-of-scope (I will do centrally):** `CHANGELOG.md` final Phase
30-33 summary, `deferred-items.md` S3/GCS strikethrough.

---

### Task 1: `docs/cloud.md`

**Files:**
- Create: `docs/cloud.md` — aim for ~200-400 lines of markdown
  covering the sections above. Concrete examples over prose.

**Content skeleton** (expand each into real sections):
```md
# Running SOMA against cloud object storage

## When this is right
(bullets: ephemeral disk, multi-replica, cost-bound)

## URL schemes
- `file:///abs/path` (default)
- `s3://bucket/prefix?endpoint=...&region=...`
- `gs://bucket/prefix?project=...`

## Recipe 1: AWS Lambda + S3
(Dockerfile, handler.py, sam/serverless template, env vars, IAM)

## Recipe 2: Cloud Run + GCS
(Dockerfile, service.yaml, `gcloud run deploy` command, ADC)

## Recipe 3: Fly Machines + Cloudflare R2
(fly.toml, endpoint override, how ADC doesn't apply)

## Caveats
- Bundle save() is batch, not incremental.
- Load latency scales with bundle size (~100 MB of vectors takes
  seconds over normal S3 read).
- Cost: PUT pricing dominates on frequent save; GET pricing
  dominates on every cold start.

## When to stay on local disk
(When your platform has persistent volumes or low enough replica
count that local-FS + a snapshot cron is sufficient.)
```

**Step 5:** `git commit -m "docs(cloud): S3/GCS bundle deployment guide"`

---

### Task 2: End-to-end example script

**Files:**
- Create: `examples/cloud_s3_demo.py` (or similar)
- Optional: add a `[project.scripts]` entry if we want
  `soma cloud-demo` (probably not — keep it as a standalone script
  to read, not run).

**Contents**: 50-80 line script that:
1. Instantiates a tiny MemoryLayer (stub embedder).
2. Adds a handful of notes.
3. Saves to `s3://` (reads bucket/prefix from env).
4. Loads from the same URL.
5. Retrieves.
6. Prints timings.

Meant as a copy-paste starting point, not a maintained tool. Has a
module docstring explaining that.

**Step 5:** `git commit -m "docs(examples): cloud S3 round-trip demo script"`

---

### Task 3: Cross-reference from existing docs

**Files:**
- Modify: `docs/backends.md` — add a "Bundle storage" section near
  the top pointing at `docs/cloud.md` for object-storage URLs.
- Modify: `README.md` — one-liner in the feature table: "bundles
  can live on S3/GCS/local disk via URL".
- Modify: `docs/cookbook.md` — new section linking to the cloud
  recipes.

**Step 5:** `git commit -m "docs: cross-reference cloud storage guide"`

---

### Final sanity

```bash
ruff check src/soma tests    # nothing should have changed in src
pytest tests -q              # nothing should have changed in tests
```

This phase is docs-only; no test delta. Phase 33 is done when the
docs build cleanly (markdown lint if available, else eyeball review)
and the example script executes end-to-end against a moto-backed
`s3://` URL.

**Gotchas:**
- Keep the deploy recipes concise — full IaC templates bloat the
  doc. Link out to AWS/GCP docs for auth setup details.
- Don't document cost numbers — they drift and mislead. Link to the
  provider's current pricing page.
- Save cadence advice should be cautious: "save after every N writes
  or every T seconds" is a real recommendation that will get used;
  make sure the chosen defaults are defensible.
