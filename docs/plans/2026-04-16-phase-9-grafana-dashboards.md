# Phase 9: Grafana Sample Dashboards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Ship operator-importable Grafana dashboards under `deploy/grafana/` so SOMA's Phase 3 metrics are immediately useful.

**Architecture:** Dashboard JSON files following RED (Rate/Errors/Duration) for the REST surface and USE (Utilization/Saturation/Errors) for the memory layer. Reference wshobson/agents public skill for structural patterns.

**Reference skill (fetch via WebFetch):**
- SKILL.md: `https://raw.githubusercontent.com/wshobson/agents/main/plugins/observability-monitoring/skills/grafana-dashboards/SKILL.md`
- API dashboard JSON: `https://raw.githubusercontent.com/wshobson/agents/main/plugins/observability-monitoring/skills/grafana-dashboards/assets/api-dashboard.json`
- Infra dashboard JSON: `https://raw.githubusercontent.com/wshobson/agents/main/plugins/observability-monitoring/skills/grafana-dashboards/assets/infrastructure-dashboard.json`
- DB dashboard JSON: `https://raw.githubusercontent.com/wshobson/agents/main/plugins/observability-monitoring/skills/grafana-dashboards/assets/database-dashboard.json`

Use those as structural references only — our panel queries target SOMA metrics from `docs/observability.md`.

**Tech Stack:** Grafana 10+ dashboard JSON schema; Prometheus data source; no external deps.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/observability.md`.

---

### Task 1: Explore existing metrics

**Read:** `docs/observability.md` — list every metric SOMA exposes (names + labels). Draft a mapping from metric to the panel it should drive.

No commit from this task; it's pure discovery. Confirm the full metric set before designing panels.

---

### Task 2: Main overview dashboard (RED for REST)

**Files:**
- Create: `deploy/grafana/soma-overview.json`
- Create: `tests/test_deploy/test_grafana_dashboards.py`

**Panels (eight total):**
1. Stat: total requests (5m window)
2. Stat: p95 retrieve latency
3. Stat: auth failure rate
4. Time series: request rate by route
5. Time series: p50/p95/p99 retrieve latency
6. Time series: store rate
7. Time series: error rate (5xx) by route
8. Table: slowest routes (p95 sorted)

**PromQL examples:**
```
# panel 5: p95 retrieve latency
histogram_quantile(0.95, sum by (le) (rate(soma_retrieve_seconds_bucket[5m])))
```

**Step 1:** Write test that loads the JSON, asserts parseable, asserts `schemaVersion >= 36`, asserts no panel has an empty `targets[].expr`, asserts every target uses Prometheus datasource UID placeholder `${DS_PROMETHEUS}`.

**Step 2:** Fails (no JSON yet).

**Step 3:** Build JSON. Use template variables `$bundle` and `$instance` with `label_values` queries.

**Step 4:** Test passes.

**Step 5:** `git commit -m "feat(grafana): SOMA overview dashboard (RED)"`

---

### Task 3: Auth dashboard

**Files:**
- Create: `deploy/grafana/soma-auth.json`

**Panels (five):**
1. Stat: auth failures last 5m
2. Time series: auth failures by reason (stacked)
3. Time series: successful auth rate
4. Table: per-bundle auth denial counts
5. Stat: revoked-token hits (`reason="revoked_token"`)

Extend the test file to cover this dashboard.

**Step:** `git commit -m "feat(grafana): SOMA auth dashboard"`

---

### Task 4: Bundle health dashboard (USE for memory layer)

**Files:**
- Create: `deploy/grafana/soma-bundle-health.json`

**Panels (six):**
1. Time series: WAL bytes written rate
2. Time series: consolidation duration p95 (uses `soma_compaction_seconds` from Phase 8 — gracefully null if absent)
3. Time series: bundle size over time
4. Stat: num entries per bundle
5. Time series: reload count (`soma_reload_total`)
6. Heatmap: retrieve latency distribution

Extend tests.

**Step:** `git commit -m "feat(grafana): SOMA bundle-health dashboard (USE)"`

---

### Task 5: Import README

**Files:**
- Create: `deploy/grafana/README.md` — explain import via `grafana-cli admin`, docker-compose volume mount pattern, template variable wiring.

**Step:** `git commit -m "docs(grafana): dashboard import guide"`

---

### Final sanity

```bash
pytest tests/test_deploy/test_grafana_dashboards.py -q
python -c "import json; [json.load(open(f)) for f in __import__('glob').glob('deploy/grafana/*.json')]"
ruff check tests/
```
