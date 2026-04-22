# Docs editorial refactor — implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> to execute this plan task-by-task (fresh subagent per task + review between).

**Goal:** Address the 9 docs-feedback findings from
`docs/plans/2026-04-21-docs-feedback-report.md` (below) without regressing
factual claims, dead links, or install flow that currently works.

**Architecture:** Edit in place, no net new top-level docs except `docs/README.md`
(hierarchy index) and `docs/rest-api.md` (API reference). Every claim / command /
path must be verified against the actual code before the task commits. The auth
story is unified by leading every deployment example with JWT and footnoting
`SOMA_API_KEY` as legacy-only — without removing the legacy path, because
`src/soma/serve.py` still supports it and existing deploys rely on it.

**Tech stack:** Markdown only for docs; Helm YAML for the two chart-metadata
touchups. Verification commands use existing `Grep`/`Read` tools; URL liveness
checks use `curl` / `pip show`.

---

## Cross-reference summary

| # | Issue | Touches | Depends on | Consumed by |
|---|-------|---------|-----------|-------------|
| 1 | Auth contradiction (deployment → legacy key) | `docs/deployment-cloud.md`, `docs/deployment-k8s.md` | — | #7, #8 |
| 2 | README install → quickstart trap | `README.md` install block | — | #3 |
| 3 | PyPI install as primary command | `README.md`, `docs/quickstart.md` | #2 | — |
| 4 | Docs hierarchy + index | CREATE `docs/README.md` | — | #5, #7 |
| 5 | REST API reference | CREATE `docs/rest-api.md` | — | #4 |
| 6 | K8s Quickstart heading | `docs/deployment-k8s.md`, `deploy/helm/soma/Chart.yaml`, `deploy/helm/soma/values.yaml` | #1 (same file) | — |
| 7 | Scope split elevated | `README.md` | #4 | #8 |
| 8 | Maturity / limitations in README | `README.md` | #1 (JWT state), #7 (substrate scope), #9 | — |
| 9 | "Six axes" count vs 5-item list | `README.md` line 5 | verify against `docs/milestones/2026-04-20-hybrid-retrieval-validated.md` | #8 |

**Batching order:** #9 → #1 → #6 → #2 → #3 → #5 → #4 → #7 → #8. Small
fact-check first; then the deployment / install fixes that unblock the
README; then the new reference pages; finally the README restructure (scope +
maturity) that pulls from everything.

---

## Pre-flight verifications already done

These were established during the brainstorming pass that produced the
feedback report. Reproduced here so each task can reference the findings:

- **`MemoryLayer.with_sbert()`** requires `sentence-transformers`
  (`src/soma/memory/api.py:467-497`). A user on the "Minimal" install
  (`pip install -e .`) who copies the Quick-start example hits an ImportError.
- **`src/soma/serve.py:318-323`** confirms 3 auth modes coexist: Open (no env
  vars set), JWT (`SOMA_JWT_SECRET` or `SOMA_JWT_PUBLIC_KEY_PATH`), Legacy
  (`SOMA_API_KEY`). Legacy still works — rewriting docs to lead with JWT
  doesn't break existing deploys.
- **`docs/milestones/2026-04-20-hybrid-retrieval-validated.md`** lists **6**
  validation axes (two separate `Cross-LLM` rows: qwen9b + Claude). README
  line 5 lists only 5. Fix: expand list OR change count to 5.
- **`deploy/helm/soma/Chart.yaml` lines 8-10, 20** still have `soma-ai/SOMA`
  URLs. `deploy/helm/soma/values.yaml:14` has `image.repository:
  ghcr.io/soma-ai/soma` — this image has never been pushed. Fix in the same
  commit as the K8s doc restructure so the chart is internally consistent.
- **PyPI**: `soma-memory==0.2.0rc4` is live (verified earlier this session).
  `pip install soma-memory` → `import soma` works end-to-end.
- **Cookbook recipe count**: 26 (verified during rc4 audit).

---

## Task 1: Reconcile the "six axes" claim (#9)

**Files:**
- Modify: `README.md:5`

**Why first:** Smallest, factual, unblocks the maturity section (#8) that
links back here.

**Step 1: Verify the milestone doc's authoritative axis list**

Run:
```
Grep pattern="\| (Cross-LLM|Cross-embedder|Cross-benchmark|Cross-judge|α)" path="docs/milestones/2026-04-20-hybrid-retrieval-validated.md"
```
Expected: 6 rows (two `Cross-LLM` rows + cross-embedder + cross-benchmark +
cross-judge + α sweep).

**Step 2: Apply the fix**

Open `README.md`. Find line 5 (the M1 callout blockquote). Replace:

```
reproducible across six axes — cross-LLM, cross-embedder, cross-benchmark, cross-judge, α-sweep.
```

with:

```
reproducible across six axes — cross-LLM (qwen9b + Claude), cross-embedder, cross-benchmark, cross-judge, α-sweep.
```

Rationale: the milestone treats the two cross-LLM responder probes
(qwen9b as responder, Claude as responder) as two independent axes. The
parenthetical makes the count honest without expanding the sentence.

**Step 3: Verify**

Run:
```
Grep pattern="six axes" path="README.md"
```
Expected: line 5 now says `six axes — cross-LLM (qwen9b + Claude)...`.

Count the axes in the new text: 1+1 cross-LLM, 1 cross-embedder,
1 cross-benchmark, 1 cross-judge, 1 α-sweep = **6**. ✓

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(README): reconcile 'six axes' claim with milestone doc"
```

---

## Task 2: Auth story — deployment-cloud.md leads with JWT (#1a)

**Files:**
- Modify: `docs/deployment-cloud.md`

**Context to carry in:**
- Legacy `SOMA_API_KEY` still works (`src/soma/serve.py:447`, "Path 3").
- Authoritative auth reference is `docs/auth.md`.
- The Helm chart still only wires `SOMA_API_KEY` (`values.yaml:66-76`) — the
  K8s section of deployment-cloud.md (§4) will keep leading with
  `SOMA_API_KEY` for that reason, but with a note. The four non-K8s
  recipes (Railway, Render, Fly, DigitalOcean/VPS) all run the bare
  container and can adopt JWT cleanly.

**Step 1: Rewrite the top-of-file intro**

Current (lines 13-18):

```
Minimum viable tier everywhere: **2 GB RAM, 1 vCPU, 1 GB persistent disk**.
...
All endpoints are documented in `src/soma/serve.py`; health is
`GET /health`, version is `GET /version`, storage is `POST /store` /
`POST /store_batch`, retrieval is `POST /retrieve`. Authentication is
off by default — set `SOMA_API_KEY` to require
`Authorization: Bearer <key>` on every non-liveness call.
```

Replace with:

```
Minimum viable tier everywhere: **2 GB RAM, 1 vCPU, 1 GB persistent disk**.
...
All endpoints are documented in [`docs/rest-api.md`](rest-api.md) (reference)
and `src/soma/serve.py` (source). Authentication is off by default; **the
recommended production setup is JWT** — generate a shared secret with
`soma auth rotate-secret`, issue per-bundle tokens with `soma auth issue`.
See [`docs/auth.md`](auth.md) for the full flow. `SOMA_API_KEY` is still
accepted as a deprecated admin escape hatch (responses carry
`X-SOMA-Deprecated: use JWT`) so existing deploys keep working.
```

**Step 2: Rewrite Railway recipe (§1, lines ~22-46) for JWT-first**

Replace:

```bash
railway variables set SOMA_API_KEY=$(openssl rand -hex 32)
```

with:

```bash
# JWT (recommended):
railway variables set SOMA_JWT_SECRET=$(soma auth rotate-secret)
# Then, per caller:
# export TOKEN=$(soma auth issue --sub alex --bundle alex:read,write --expires 30d)

# Legacy (deprecated — kept here for existing deploys):
# railway variables set SOMA_API_KEY=$(openssl rand -hex 32)
```

Replace the one-click paragraph's seeded secret wording ("seeds
`SOMA_API_KEY` as a generated secret") with "seeds `SOMA_JWT_SECRET` as a
generated secret." Verify the actual Railway template in `railway.json`
matches — if `railway.json` still seeds `SOMA_API_KEY`, ALSO modify
`railway.json` to seed `SOMA_JWT_SECRET` instead. Run a Grep first.

**Step 3: Rewrite Render recipe (§2) the same way**

Lines ~49-72. The render blueprint `render.yaml` uses
`generateValue: true` for `SOMA_API_KEY` — change the Blueprint reference
to seed `SOMA_JWT_SECRET` and note the rotation path. If `render.yaml`
still points at `SOMA_API_KEY`, modify it too (Grep first).

**Step 4: Rewrite Fly recipe (§3) the same way**

Lines ~75-96. Replace `fly secrets set SOMA_API_KEY=...` with
`fly secrets set SOMA_JWT_SECRET=$(soma auth rotate-secret)` and the
per-caller token pattern. Keep the legacy line commented-out below.

**Step 5: Leave K8s recipe (§4) on `SOMA_API_KEY` with a callout**

Lines ~100-125. The Helm chart only wires `SOMA_API_KEY` today. Add a
blockquote callout right under the `## 4. Kubernetes (Helm)` heading:

```
> **Auth status on the Helm chart:** the chart currently wires
> `SOMA_API_KEY` (legacy) only. JWT support in the chart is
> tracked but not yet shipped. Cluster-internal traffic + network
> policy is usually enough; if you need per-caller permissions,
> either switch from the chart to a raw Deployment with
> `SOMA_JWT_SECRET` in env, or wait for the chart update. The
> legacy path still works and is supported for the time being.
```

No code changes in the K8s recipe itself — it stays accurate until
the chart is updated.

**Step 6: Rewrite DigitalOcean/VPS recipe (§5, lines ~128-150)**

Replace `SOMA_API_KEY` generation in the `.env` with `SOMA_JWT_SECRET`.
Keep legacy line commented. The `soma serve --port 8420` invocation
doesn't need to change.

**Step 7: Update the 401 Troubleshooting footer (line ~175-178)**

Replace:

```
**`401 Unauthorized` from every endpoint.** `SOMA_API_KEY` is set.
Either unset it (public deploy) or send `Authorization: Bearer <key>`
on every request.
```

With:

```
**`401 Unauthorized` from every endpoint.** `SOMA_JWT_SECRET` (or legacy
`SOMA_API_KEY`) is set and your request is missing or has a wrong
`Authorization: Bearer <token>` header. Mint a fresh token with
`soma auth issue` (JWT) or check the value of `SOMA_API_KEY` in the
server env (legacy). `/health` and `/version` remain public by design.
```

**Step 8: Verify**

Run:
```
Grep pattern="SOMA_API_KEY" path="docs/deployment-cloud.md"
```
Expected: every remaining hit is either (a) in the "legacy (commented-out)"
lines, (b) in the K8s §4 callout, or (c) in the 401 troubleshooting.
No recipe is teaching `SOMA_API_KEY` as the first auth path to copy.

Run:
```
Grep pattern="SOMA_JWT_SECRET|soma auth" path="docs/deployment-cloud.md"
```
Expected: at least one hit per non-K8s recipe (§1, §2, §3, §5).

Also check `railway.json`, `render.yaml`, `fly.toml` for the seeded-value
variable name and ensure they match the rewritten doc.

**Step 9: Commit**

```bash
git add docs/deployment-cloud.md railway.json render.yaml fly.toml
git commit -m "docs(deploy): lead deployment-cloud recipes with JWT, SOMA_API_KEY legacy"
```

---

## Task 3: Auth story — deployment-k8s.md (#1b, #6 together)

**Files:**
- Modify: `docs/deployment-k8s.md`
- Modify: `deploy/helm/soma/Chart.yaml`
- Modify: `deploy/helm/soma/values.yaml`

**Why paired with #6:** Both fix the same file. Doing them in one pass
avoids a second round of reviewer-ping on the same structure.

**Step 1: Fix the stale org URLs in Chart.yaml**

Current `deploy/helm/soma/Chart.yaml` lines 8-10, 20:

```yaml
home: https://github.com/soma-ai/SOMA
sources:
  - https://github.com/soma-ai/SOMA
...
icon: https://raw.githubusercontent.com/soma-ai/SOMA/main/docs/assets/soma-icon.png
```

Replace all three with the actual repo:

```yaml
home: https://github.com/danthi123/soma
sources:
  - https://github.com/danthi123/soma
...
icon: https://raw.githubusercontent.com/danthi123/soma/main/docs/assets/soma-icon.png
```

Note: verify `docs/assets/soma-icon.png` actually exists. If not, drop the
`icon:` line entirely rather than point at a 404.

Run:
```
Glob pattern="docs/assets/*.png" path="."
```

**Step 2: Note the image.repository mismatch in values.yaml — but do not fix it in this commit**

`deploy/helm/soma/values.yaml:14` has `image.repository: ghcr.io/soma-ai/soma`.
Fixing the image registry is a separate concern (the image hasn't been
pushed anywhere). Add a YAML comment above the key calling this out:

```yaml
image:
  # NOTE: the image at `ghcr.io/soma-ai/soma` has not been published.
  # For local / self-hosted k8s, build and push your own image from the
  # in-tree Dockerfile and override this value. Publishing to a public
  # registry is tracked as a separate release task.
  repository: ghcr.io/soma-ai/soma
  ...
```

Rationale: honest status without breaking anyone's override pattern.

**Step 3: Restructure deployment-k8s.md's Quickstart (the #6 fix)**

Line 43: `## Quickstart — OCI one-liner`.

Replace that entire Quickstart section (lines 43-62) with:

```markdown
## Quickstart — install from the in-tree chart

The OCI registry `oci://ghcr.io/soma-ai/charts/soma` and the classic
repo `https://soma-ai.github.io/soma-helm` are **not yet published** (see
the registry-status callout above). The installable path today is the
chart committed at `deploy/helm/soma/`:

```bash
git clone https://github.com/danthi123/soma.git && cd soma
helm install soma deploy/helm/soma
```

Verify:

```bash
kubectl get statefulset,svc,secret -l app.kubernetes.io/name=soma
kubectl port-forward svc/soma 8420:8420
curl http://localhost:8420/health   # → {"status": "ok", ...}
```

Pull the auto-generated API key:

```bash
export SOMA_API_KEY=$(kubectl get secret soma-api \
  -o jsonpath='{.data.SOMA_API_KEY}' | base64 -d)

curl -H "Authorization: Bearer $SOMA_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8420/store \
  -d '{"text": "user lives in Portland, OR"}'
```

> **Auth note:** the Helm chart currently wires `SOMA_API_KEY` only.
> To use JWT, run SOMA as a raw Deployment (out of scope for this
> chart revision) or switch to the chart's legacy key path and rely
> on cluster-internal network policy until JWT support is added to
> the chart. The legacy key path is still supported by the server
> (`src/soma/serve.py`).

## Future state — OCI one-liner

Once the chart is published to `oci://ghcr.io/soma-ai/charts/soma`, the
install will simplify to:

```bash
helm install soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.0
```

Everything below in this doc (values reference, secret patterns,
persistence, troubleshooting) applies to either path unchanged.
```

**Step 4: Delete the redundant "Classic repo path" section**

Lines ~53-63 ("Classic repo path (GitHub Pages)"). Both paths are
aspirational; collapsing both into the "Future state" paragraph above
avoids the same lie told twice.

**Step 5: Verify**

Run:
```
Grep pattern="oci://ghcr.io/soma-ai|soma-ai.github.io" path="docs/deployment-k8s.md"
```
Expected: the only remaining hits are inside the `Registry status` callout
at the top (which is *about* those URLs) or inside the "Future state"
section (which explicitly frames them as not-yet-published).

Run:
```
Grep pattern="soma-ai/SOMA" path="deploy/helm/soma/Chart.yaml"
```
Expected: zero matches.

**Step 6: Commit**

```bash
git add docs/deployment-k8s.md deploy/helm/soma/Chart.yaml deploy/helm/soma/values.yaml
git commit -m "docs(deploy): k8s Quickstart leads with in-tree chart; fix chart URLs"
```

---

## Task 4: README install → Quickstart alignment (#2)

**Files:**
- Modify: `README.md` lines 9-46

**Step 1: Relabel + reorder the install block**

Replace the "Minimal" label with one that matches what the first code
example actually needs. `MemoryLayer.with_sbert()` in the Quick start
(line 53) requires `sentence-transformers` — so the first code block
that a copy-paster runs needs to include `[sbert]`.

Current (lines 11-16):

```markdown
```bash
# Minimal (torch + tokenizers only):
pip install -e .

# Quality retrieval (sentence-transformers):
pip install -e ".[sbert]"
```

Replace with:

```markdown
```bash
# Default (quality retrieval — what the Quick start below uses):
pip install -e ".[sbert]"

# Absolute minimum (no sbert; you must pass your own embed_fn):
pip install -e .
```

Rationale: first command matches the first runnable example. Labels say
what they are (not "minimal" in some abstract sense).

**Step 2: Verify**

Run:
```
Grep pattern="with_sbert\(\)" path="README.md"
```
Expected: line ~53 still has `MemoryLayer.with_sbert()`. This is the
claim the reordered install block now matches.

Also confirm the rest of the Install block's extras (serve, ann, metrics,
otel, qdrant, lancedb, chroma, pgvector, s3, gcs, langchain, llamaindex)
are listed in `pyproject.toml` as `[project.optional-dependencies]`
keys. Grep once to verify:

```
Grep pattern="^sbert|^ann|^serve|^metrics|^otel|^qdrant|^lancedb|^chroma|^pgvector|^s3|^gcs|^langchain|^llamaindex" path="pyproject.toml"
```
Expected: all 13 names appear.

**Step 3: Commit** (held until Task 5 also edits this file — see below)

---

## Task 5: PyPI install as primary command (#3)

**Files:**
- Modify: `README.md` lines 9-46 (same block as Task 4)
- Modify: `docs/quickstart.md:14`

**Step 1: Swap README primary command to PyPI**

Right after Task 4's edit, the install block still says
`pip install -e ".[sbert]"` first — which requires a clone. For a PyPI
user (the README is shown on pypi.org for the `soma-memory` package) the
primary command must be the one that works from that context.

Replace the whole install block (Task 4's output) with:

```markdown
```bash
# Default (quality retrieval — what the Quick start below uses):
pip install "soma-memory[sbert]"

# REST API server + JWT auth:
pip install "soma-memory[serve]"

# FAISS ANN (>10K entries):
pip install "soma-memory[ann]"

# Prometheus /metrics + OpenTelemetry tracing:
pip install "soma-memory[metrics]"
pip install "soma-memory[otel]"

# Alternative vector backends:
pip install "soma-memory[qdrant]"    # Qdrant (local file or HTTP)
pip install "soma-memory[lancedb]"   # embedded arrow-native (10M+ scale)
pip install "soma-memory[chroma]"    # drop-in for existing Chroma users
pip install "soma-memory[pgvector]"  # Postgres + pgvector

# Cloud object-store bundles:
pip install "soma-memory[s3]"        # s3:// URLs on save/load
pip install "soma-memory[gcs]"       # gs:// URLs on save/load

# Framework adapters:
pip install "soma-memory[langchain]"
pip install "soma-memory[llamaindex]"

# Everything runtime-useful:
pip install "soma-memory[sbert,ann,serve,metrics,otel,qdrant,lancedb,chroma,pgvector,s3,gcs,langchain,llamaindex]"

# Absolute minimum (no sbert; you must pass your own embed_fn):
pip install "soma-memory"
```

> **Developing on SOMA?** Clone the repo and use the editable variant
> (same set of `[extras]`): `pip install -e ".[sbert]"`, etc. See
> [`CONTRIBUTING.md`](CONTRIBUTING.md).
```

**Step 2: Update docs/quickstart.md:12-15**

Current:

```bash
pip install -e ".[sbert,serve,metrics]"
```

Replace with:

```bash
pip install "soma-memory[sbert,serve,metrics]"
```

And add one line above:

```
> Contributor? Replace with `pip install -e ".[sbert,serve,metrics]"`
> after cloning the repo.
```

**Step 3: Verify**

Run:
```
curl.exe -sS https://pypi.org/pypi/soma-memory/json | python -c "import sys, json; d=json.load(sys.stdin); print('latest:', d['info']['version']); print('yanked:', d['info'].get('yanked', False))"
```
Expected: `latest: 0.2.0rc4`, `yanked: False`. Confirms the primary
command resolves.

Also run:
```
Grep pattern="pip install -e \"?\\." path="README.md"
```
Expected: only remaining hit is in the "Developing on SOMA?" callout.

**Step 4: Commit**

```bash
git add README.md docs/quickstart.md
git commit -m "docs(install): lead with pip install soma-memory, editable variant for contributors"
```

---

## Task 6: Create docs/rest-api.md (#5)

**Files:**
- Create: `docs/rest-api.md`
- Modify: `README.md` REST API section (link to new file)

**Step 1: Enumerate the actual routes**

Run:
```
Grep pattern="^@app\.(get|post|delete)\(" path="src/soma/serve.py" output_mode="content" -n=true head_limit=80
```
Expected: the decorator-tagged routes. Collate them.

If the count is manageable (< 40), continue. If not, break into per-section
sub-enumerations.

**Step 2: Write docs/rest-api.md**

Structure (target ~300 lines):

```markdown
# SOMA REST API reference

> **Live spec:** `GET /openapi.json` on any running SOMA server returns
> the full OpenAPI 3.1 schema. The committed snapshot at
> `clients/typescript/openapi.json` is regenerated from a stub-embedder
> server on every PR — treat it as the canonical source of shapes.
>
> **Runtime:** `uvicorn soma.serve:app --port 8420` (or
> `soma serve --port 8420`).

## Base URL

`http://{host}:{port}` — port defaults to 8420. Multi-tenant routes are
prefixed with `/bundles/{name}`.

## Auth

See [`docs/auth.md`](auth.md) for the full flow. Short version:

- **Open:** no env vars set. Every route accepts unauthenticated requests.
  Not for production.
- **JWT (recommended):** `SOMA_JWT_SECRET` (HS256) or
  `SOMA_JWT_PUBLIC_KEY_PATH` (RS256). Send
  `Authorization: Bearer <JWT>`.
- **Legacy:** `SOMA_API_KEY`. Send `Authorization: Bearer <key>`.
  Deprecated (response carries `X-SOMA-Deprecated: use JWT`).

`GET /health`, `GET /version`, and `GET /metrics` (when
`soma-memory[metrics]` is installed) are always public.

## Route map

(Enumerate every route from `src/soma/serve.py`, grouped by resource.)

### Liveness / metadata

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | /health | public | liveness probe |
| GET | /version | public | package version string |
| GET | /status | read | default-bundle stats |
| GET | /metrics | public | Prometheus scrape |

### Storage

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | /store | write | store one entry |
| POST | /store_batch | write | store N entries in one call |
| POST | /retrieve | read | semantic + hybrid + rerank retrieval |
| POST | /forget | write | delete an entry |
| GET | /get/{id} | read | fetch by node id |
| GET | /recent | read | last N entries |
| GET | /related/{id} | read | graph-adjacent neighbours |
| POST | /consolidate | write | trigger consolidation cycle |
| POST | /save | write | snapshot bundle to disk |

### Multi-tenant variants

Every `/store` / `/retrieve` / `/...` route above is also available at
`/bundles/{name}/<same>` with the per-bundle permission check. See
`docs/auth.md` "Claim shape" for scoping semantics.

### Chat / LLM integration

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | /chat | write | RAGSession one-shot |

### Admin

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | /bundles | admin | list loaded bundles |

(Include only the routes that `serve.py` actually exposes — verify each.)

## Request / response shapes

For each route, give:

- Request body (JSON) with field types + examples
- Response body with field types + examples
- Auth / permission requirement
- Notable error branches (401 / 403 / 404 / 422)

Shapes authoritative source: `clients/typescript/openapi.json`. Copy from
there; don't retype from memory.

## Error model

Every 4xx response body is `{"detail": "<human-readable reason>"}`. See
the `ErrorResponse` component in the OpenAPI spec.

Standard headers:
- `X-SOMA-Deprecated: use JWT` — legacy `SOMA_API_KEY` was used.
- `Retry-After` — only when the rate limiter (`SOMA_RATE_LIMIT_RPS`) is
  tripped. Returns 429.

## Generated clients

- **TypeScript:** `npm install soma-memory` —
  [`docs/clients.md`](clients.md).
- **Other languages:** point an OpenAPI generator at `GET /openapi.json`.
```

**Step 3: Update README.md's REST API section**

Current (lines 125-139):

```markdown
## REST API + Docker

```bash
# Local:
soma serve --port 8420

# Docker:
docker compose up
```

Endpoints: `/health`, `/version`, `/status`, `/store`, ...
```

Append one sentence at the end of that section:

```markdown
Full route reference + request/response shapes: [`docs/rest-api.md`](docs/rest-api.md).
```

**Step 4: Verify**

For every row in the new doc's route map, verify:

```
Grep pattern="@app\.(get|post).*\"/{path}\"" path="src/soma/serve.py"
```

Expected: every documented route has a matching decorator. If any row
has no match in the source, either delete the row or find the actual
decorator path.

**Step 5: Commit**

```bash
git add docs/rest-api.md README.md
git commit -m "docs: add rest-api.md route reference, link from README"
```

---

## Task 7: Create docs/README.md index (#4)

**Files:**
- Create: `docs/README.md`

**Step 1: Enumerate the hierarchy**

Six sections per the reviewer's split:

1. Start here
2. Install / run
3. API reference
4. Deploy / operate
5. Deep technical / background
6. Roadmap / research / internal notes

**Step 2: Write the file**

```markdown
# SOMA documentation

> This directory mixes user-facing docs, operator-facing docs, and a lot
> of internal research and plan notes. Use this index to tell them
> apart. If you're evaluating or integrating SOMA, the "Start here" and
> "Install / run" sections are what you want; the rest is navigable but
> secondary.

## 1. Start here

- [**Positioning**](positioning.md) — what SOMA is, what it competes
  against, and what we don't claim.
- [**Comparison**](comparison.md) — SOMA vs Chroma / Mem0 / Letta /
  Zep / Pinecone.

## 2. Install / run

- [**Quickstart**](quickstart.md) — end-to-end agent flow (install →
  serve → JWT → ConversationalMemory → Grafana).
- [**Cookbook**](cookbook.md) — 26 copy-paste recipes (hybrid, rerank,
  multi-tenant, conversational, cloud bundles, schemas, context
  packing).
- [**LLM backends**](llm-backends.md) — Ollama / OpenAI / Anthropic /
  vLLM / HuggingFace.
- [**Recall improvements**](recall-improvements.md) — hybrid BM25,
  cross-encoder rerank, query expansion.

## 3. Reference

- [**REST API**](rest-api.md) — every route, shape, auth requirement.
- [**Typed schemas**](schemas.md) — built-in + custom, packer.
- [**Backends**](backends.md) — InProc / Qdrant / LanceDB / Chroma /
  pgvector tradeoffs.
- [**Auth**](auth.md) — per-bundle JWTs, RS256 split, revocation.
- [**Observability**](observability.md) — Prometheus metrics, JSON
  logs, OpenTelemetry spans, Grafana dashboards.
- [**Clients**](clients.md) — TypeScript client (`soma-memory` npm).
- [**GDPR forgetting**](gdpr.md) — `POST /forget`, audit trail,
  summary cascade.

## 4. Deploy / operate

- [**Cloud**](cloud.md) — S3/GCS bundle URLs + Lambda / Cloud Run /
  Fly recipes.
- [**Deployment (cloud PaaS)**](deployment-cloud.md) — Railway /
  Render / Fly / K8s / DigitalOcean.
- [**Deployment (Kubernetes)**](deployment-k8s.md) — Helm chart
  runbook.
- [**Demos**](demos.md) — shipped demos and when to run each.

## 5. Deep technical / background

- [**Whitepaper**](whitepaper.md) — the original research-architecture
  design (now substrate for the memory-layer product; see
  `positioning.md` for the reader-orientation note at the top).
- [**Paper draft**](../benchmarks/reports/paper-draft.md) — every
  published number wired back to its script + report.

## 6. Milestones, plans, research

- [**Milestones**](milestones/) — pinned states. Current:
  [M1 — Hybrid Retrieval Validated (2026-04-20)](milestones/2026-04-20-hybrid-retrieval-validated.md).
- [**Plans**](plans/) — phase plans (2026-04-16-phase-1 … phase-45),
  research direction plans (4a / 4b / Path A / Path B), and topic
  designs. Historical; treat as an archive of how the project got to
  where it is.
- [**Progress**](progress/) — pivot-history notes.

Non-SOMA-internal readers: everything in sections 5 and 6 is optional.
The shipping product is covered by sections 1–4.
```

**Step 3: Verify every link resolves**

For each link, `Read` the target file (or `Glob` to confirm existence):

```
Glob pattern="docs/positioning.md"
Glob pattern="docs/comparison.md"
# ... every link in the index ...
```
Expected: each target exists. If any target doesn't, either the link is
wrong (fix it) or the target should be created as part of a different
task (defer the link until it exists).

**Step 4: Commit**

```bash
git add docs/README.md
git commit -m "docs: add top-level docs/README.md index"
```

---

## Task 8: Elevate the scope-split framing to README (#7)

**Files:**
- Modify: `README.md`

**Step 1: Add a "Scope" subsection right after "How it compares"**

Insert a new H2 section after the comparison table (after line ~102) and
before the "Benchmark" subsection:

```markdown
## Scope

SOMA ships two things in the same repo; the product and the research
substrate are separable.

**The product (what `pip install soma-memory` gets you):** a local-first
agent-memory layer with hybrid BM25+cosine retrieval, multi-tenant REST,
JWT auth, pluggable vector backends, and crash-safe WAL. This lives under
`src/soma/memory/`, `src/soma/llm/`, `src/soma/cli.py`, `src/soma/serve.py`,
and `src/soma/integrations/`. It's what every benchmark number on this
page measures.

**The research substrate (ships with the same package, but is not part
of the memory-layer API):** a plastic graph / growth / pruning /
consolidation pipeline under `src/soma/core/`, `src/soma/growth/`,
`src/soma/metacognition/`, `src/soma/consolidation/`, `src/soma/io/`,
`src/soma/deploy/`. It runs end-to-end, but three serious attempts to
route its learning signal into retrieval have been null on real corpora
(see [M1 milestone, "What's ruled out"](docs/milestones/2026-04-20-hybrid-retrieval-validated.md)).
We keep it in-tree as the measurement substrate for Path A / Path B
research ([`docs/plans/2026-04-20-path-a-biophysical-representation-layer-design.md`](docs/plans/2026-04-20-path-a-biophysical-representation-layer-design.md),
[`docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md`](docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md)),
not because it currently improves the product.

If you're evaluating SOMA as a vector-DB / RAG replacement: the product
is what matters. If you're interested in the research agenda:
[`docs/milestones/2026-04-20-hybrid-retrieval-validated.md`](docs/milestones/2026-04-20-hybrid-retrieval-validated.md)
is the starting point, and `docs/plans/` has the full trail.
```

**Step 2: Remove now-redundant asterisk footnote**

The comparison table's `*` footnote on "Plastic graph substrate" (line
~100) now repeats what the new Scope section says. Drop the asterisk +
footnote to declutter; the Scope section owns the framing.

**Step 3: Verify**

Run:
```
Grep pattern="Scope" path="README.md"
```
Expected: the new section exists. Grep for `\* substrate ships` —
should return zero hits (footnote removed).

Also verify the `docs/plans/2026-04-20-path-a-...` and
`...-path-b-...` filenames are real:

```
Glob pattern="docs/plans/2026-04-20-path-a-*"
Glob pattern="docs/plans/2026-04-20-path-b-*"
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(README): add Scope section distinguishing product from research substrate"
```

---

## Task 9: Status & limitations section in README (#8)

**Files:**
- Modify: `README.md`

**Step 1: Add a "Status & known limitations" section above "License"**

```markdown
## Status & known limitations

**Release status:** `Development Status :: 4 - Beta` (`pyproject.toml`).
Latest on PyPI: `soma-memory==0.2.0rc4` (2026-04-20).

**Production-ready surface:**
- `MemoryLayer` API: store / retrieve / save / load / hybrid / rerank /
  cross-encoder / context packing / GDPR forget.
- REST API under `soma serve` (all routes listed in
  [`docs/rest-api.md`](docs/rest-api.md)).
- Per-bundle JWT auth with HS256 / RS256 + file-backed or Redis
  revocation blocklist.
- Vector backends: InProc (default), LanceDB, Qdrant, Chroma, pgvector.
- Bundle storage: local filesystem, S3, GCS (including S3-compat:
  MinIO, Cloudflare R2, DigitalOcean Spaces).

**Known limitations (today):**
- **Single-writer WAL.** Multi-process on one bundle works (peer-reload
  every 30 s), but Helm chart replicaCount is locked at 1. Horizontal
  scale lands when the external-backend path replaces the in-process
  WAL as source of truth.
- **Helm chart auth is `SOMA_API_KEY` only.** JWT support for the chart
  is tracked; for now, chart users stay on the legacy key path. See
  [`docs/auth.md`](docs/auth.md) "Deprecation" section — the server
  supports both, only the chart template hasn't been updated.
- **Helm chart OCI registry is not yet published.** Install from the
  in-tree chart: `helm install soma deploy/helm/soma`. See
  [`docs/deployment-k8s.md`](docs/deployment-k8s.md).
- **No distributed multi-node vector scale.** For >10M vectors, use
  Qdrant HTTP as the backend; SOMA orchestrates and Qdrant scales.
- **No hosted/managed offering.** SOMA is self-host only. The pivot
  doc ([`docs/plans/2026-04-15-memory-layer-pivot.md`](docs/plans/2026-04-15-memory-layer-pivot.md))
  has the rationale.

**Research substrate (not part of product positioning — see
[Scope](#scope) above):** the plastic-graph / growth / pruning pipeline
runs end-to-end but does not currently lift retrieval scores on any
tested corpus. Three closed-out attempts are documented in
[M1 milestone](docs/milestones/2026-04-20-hybrid-retrieval-validated.md)
"What's ruled out." Path A and Path B design docs lay out the next
round of experiments.

**Experimental surface (ship but not battle-tested at scale):**
- `ConversationalMemory` async extraction mode (`extraction_mode="async"`).
- Rate limiter (`SOMA_RATE_LIMIT_RPS`) — intended for dev/homelab, not
  a WAF.
- pgvector backend — passes the in-tree integration tests against
  pgvector/pgvector:pg16 but hasn't seen production traffic yet.
```

**Step 2: Verify**

Run:
```
Grep pattern="soma-memory==0.2.0rc4|Development Status :: 4" path="pyproject.toml"
```
Expected: both claims appear in `pyproject.toml`.

Run:
```
Grep pattern="replicaCount: 1" path="deploy/helm/soma/values.yaml"
```
Expected: match (the limitation about the locked replica count is real).

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs(README): add Status & known limitations section"
```

---

## Task 10: Final cross-doc audit

**Files:**
- All docs touched above.

**Step 1: Broken-link sweep**

For every internal markdown link added or changed by Tasks 1-9, resolve
the target:

```
Glob pattern="<target>"
```

For every external URL (github.com/danthi123/soma/...) added, check it
points at something real by visiting it in the browser (user step) or
by `curl -sI`.

**Step 2: Terminology consistency sweep**

Grep each of:
- `soma-memory` — must be the package name in every install example.
- `@soma-ai/client` — must appear **nowhere** (we fixed these in rc4).
- `SOMA_API_KEY` first / `SOMA_JWT_SECRET` second — the order the
  docs present auth must be JWT-first everywhere except the K8s Helm
  chart section (which has its own callout).
- `1,982` — LoCoMo question count must stay 1,982 in every
  user-facing doc (rc4 fixed this; don't regress).

Run each as a Grep, walk the hits manually.

**Step 3: Commit (only if anything was caught)**

```bash
git add <files>
git commit -m "docs: final cross-doc audit fixups"
```

If nothing was caught, skip.

---

## Task 11: Push

**Step 1: Run `git log --oneline` and confirm the commit sequence makes
sense in isolation**

Each task's commit message should be self-descriptive. A reviewer
reading `git log` from before Task 1 to after Task 10 should be able
to follow the editorial thread without reading the plan.

**Step 2: Push main**

```bash
git push origin main
```

Both remotes (github.com + git.dant123.com) push in one command per
the existing multi-remote config.

**Step 3: Sanity-check the PyPI README**

The PyPI README is rendered from the `README.md` at whatever tag last
published. Since this is a docs-only change, `soma-memory==0.2.0rc4`
stays the live version on PyPI and its rendered README stays as it
was **at the time of the rc4 tag**. The new README only shows on PyPI
once we cut a new version. Note this in the reporting-back to the
user — they may want an rc5 to propagate the README changes to the
PyPI landing page.

---

## Verification checklist (run before declaring done)

- [ ] `Grep pattern="SOMA_API_KEY" path="docs/deployment-cloud.md"` returns
  only legacy-commented-out or troubleshooting hits.
- [ ] `Grep pattern="six axes" path="README.md"` matches one line with
  an updated cross-LLM parenthetical.
- [ ] `Glob pattern="docs/rest-api.md"` returns a file.
- [ ] `Glob pattern="docs/README.md"` returns a file.
- [ ] Every link added by Tasks 1-9 has a matching `Glob` target.
- [ ] Every install command in the README uses `pip install
  "soma-memory[...]"` as primary.
- [ ] README's "Quick start" block + install block agree on the
  required extras.
- [ ] `git log --oneline` reads cleanly end-to-end.

---

## Follow-ups NOT in this plan (captured for later)

- **Actually publish the Helm chart to `ghcr.io/danthi123/charts/soma`**
  so the "Future state" section becomes reality. Chart publishing is a
  separate release workflow — not touched here.
- **Update the Helm chart to accept `SOMA_JWT_SECRET`.** The chart's
  template + values + docs all need to add the JWT env path. Separate
  plan.
- **Cut an rc5 or 0.2.0 stable** so the PyPI landing page gets the new
  README. Not this plan; decide after reviewing the docs changes.
- **Reformat `ruff format` on the whole repo** (listed in
  `CHANGELOG.md` `[Unreleased]`).
- **Node.js 24 action pins** (listed in
  `CHANGELOG.md` `[Unreleased]`).
