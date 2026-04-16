# SOMA Grafana dashboards

Drop-in Grafana dashboards for SOMA's Phase 3 Prometheus metrics. Import
any of the JSON files here into a Grafana 10+ instance that is already
scraping SOMA's `/metrics` endpoint and you get RED/USE views out of the
box — no panel authoring required.

| File                        | Methodology | What it shows                                                                                 |
| --------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
| `soma-overview.json`        | RED         | Request rate, error rate, p50/p95/p99 latency across `/retrieve`, `/store`, `/store_batch`, `/forget`, `/consolidate` plus a slowest-route table. |
| `soma-auth.json`            | —           | `soma_auth_failures_total` broken down by `reason`, success rate, and revoked-token hits.     |
| `soma-bundle-health.json`   | USE         | WAL throughput, consolidation cost, live-entry counts per bundle, peer reloads and a retrieve-latency heatmap. |

All three dashboards share the same template variables:

- `${DS_PROMETHEUS}` — the Prometheus data source bound at import time.
- `$instance` — the scrape target (`soma.example:8420`). Multi-select, defaults to all.
- `$bundle` — the SOMA bundle name. Multi-select, defaults to all. The
  auth dashboard keeps this var for cross-dashboard navigation; note
  that `soma_auth_failures_total` is not itself bundle-labelled.

> Underlying metric definitions live in [`docs/observability.md`](../../docs/observability.md).
> Every PromQL query in these dashboards maps 1:1 to a metric name documented there.

## Prerequisites

- Grafana 10.0 or later (schema version 36+).
- A Prometheus data source scraping the `/metrics` endpoint of a SOMA
  process started with `pip install "soma[metrics]"`.
- If the scraped build is older than Phase 8, the `soma-bundle-health`
  "consolidation duration p95" panel may render one empty series
  (`soma_compaction_seconds_bucket`). That is expected and has no other
  effect on the dashboard.

## Import options

### 1. Grafana web UI (one-off)

1. Open Grafana → **Dashboards** → **New** → **Import**.
2. Upload the JSON file (or paste its contents into the text area).
3. When prompted, pick your Prometheus data source for
   `DS_PROMETHEUS`. Click **Import**.
4. The dashboard lands under the folder you picked; all three use
   `soma-*` UIDs, so you can deep-link to them from alerts.

### 2. `grafana-cli admin` (scripted install against a running server)

`grafana-cli` is the stock Grafana administrative tool. The
`dashboards import` pattern for our repo:

```bash
# ship a JSON into a running Grafana (v10+). The API endpoint does
# the real work; grafana-cli handles auth via ~/.grafana/ and the
# GRAFANA_URL / GRAFANA_API_KEY env vars.
for f in deploy/grafana/soma-*.json; do
  grafana-cli --config /etc/grafana/grafana.ini admin \
    dashboard import "$f"
done
```

On older Grafana releases that ship without the `dashboard import`
subcommand, use the HTTP API directly — same JSON payload, just
wrapped in the `{"dashboard": ..., "overwrite": true}` envelope:

```bash
for f in deploy/grafana/soma-*.json; do
  jq -n --slurpfile d "$f" \
    '{dashboard: $d[0], overwrite: true, inputs: [{name:"DS_PROMETHEUS", type:"datasource", pluginId:"prometheus", value:"Prometheus"}]}' \
    | curl -sS -H "Authorization: Bearer ${GRAFANA_API_KEY}" \
           -H "Content-Type: application/json" \
           -d @- "${GRAFANA_URL}/api/dashboards/import"
done
```

Replace `"Prometheus"` with the UID of your actual Prometheus data
source (see `GET /api/datasources`).

### 3. docker-compose provisioning (cluster-friendly)

Grafana auto-provisions dashboards from
`/etc/grafana/provisioning/dashboards/`. Mount this directory
alongside a YAML provider:

```yaml
# docker-compose.yml
services:
  grafana:
    image: grafana/grafana:10.4.2
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - ./deploy/grafana:/var/lib/grafana/dashboards/soma:ro
      - ./deploy/grafana/provisioning.yaml:/etc/grafana/provisioning/dashboards/soma.yaml:ro
      - grafana-data:/var/lib/grafana
volumes:
  grafana-data: {}
```

Minimal `provisioning.yaml` (create this file yourself — it's
deployment-specific, not shipped in-tree):

```yaml
apiVersion: 1
providers:
  - name: soma
    orgId: 1
    folder: SOMA
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards/soma
      foldersFromFilesStructure: false
```

Grafana will pick up new JSON files under the mounted path on the
poll interval.

### 4. Kubernetes (ConfigMap)

```bash
kubectl create configmap soma-grafana-dashboards \
  --from-file=deploy/grafana/ \
  --namespace=monitoring
```

Then mount the ConfigMap into your Grafana pod at
`/var/lib/grafana/dashboards/soma/` and add a sidecar provider entry
the same way as the compose example above. If you use the
`kube-prometheus-stack` chart, label the ConfigMap with
`grafana_dashboard=1` so the sidecar picks it up automatically:

```bash
kubectl label configmap soma-grafana-dashboards grafana_dashboard=1 \
  --namespace=monitoring
```

## Template-variable wiring

Every dashboard declares three variables in this order:

1. **`DS_PROMETHEUS`** (`datasource`) — scans for any Prometheus data
   source and asks the operator to pick one during import. Saved as
   `${DS_PROMETHEUS}` and referenced by UID from every panel target.
2. **`instance`** (`query`) — `label_values(soma_loaded_bundles, instance)`.
   Multi-select, includes `All`, sorted ascending.
3. **`bundle`** (`query`) — `label_values(soma_entries{instance=~"$instance"}, bundle)`.
   Multi-select, includes `All`, sorted ascending. Dependent on `$instance`.

If you need a different filter dimension (e.g. slice by `backend`),
duplicate an existing variable entry in the `templating.list` array
and adjust the `label_values()` query.

## Verifying the import

A quick smoke test once a dashboard lands:

1. Open the dashboard.
2. Pick a bundle + instance that actually has traffic.
3. Watch the "Request rate by route" panel on the overview dashboard
   respond to a `curl http://soma:8420/retrieve ...` call within the
   refresh interval (default 30 s).
4. On the auth dashboard, the success-rate stat should read 1 when
   there is any successful traffic and no auth failures; it renders
   `No data` when both numerator and denominator are zero.
5. On the bundle-health dashboard, the heatmap lights up after any
   retrieve traffic arrives; the stepped "Live entries" chart should
   match `soma_entries` in a direct Prometheus query.

## Editing dashboards

Dashboards carry `schemaVersion: 39` and are ruff/JSON-lint clean. If
you make changes in the Grafana UI and want to commit them back:

1. **Dashboard settings → JSON model → Save to file** in Grafana.
2. Strip the runtime-only fields Grafana adds on export — `id`
   should be `null`, delete `iteration`, delete any `current` value
   under `templating.list[*]` except where you intentionally want a
   default, and reset `version` to `1`.
3. `python -c "import json; json.load(open('deploy/grafana/<file>.json'))"`
   then `pytest tests/test_deploy/test_grafana_dashboards.py -q`.

The test suite enforces the invariants every SOMA dashboard relies on
(`schemaVersion >= 36`, non-empty `expr`, `${DS_PROMETHEUS}` UID
everywhere, `$instance` + `$bundle` present). Keep them green.

## Troubleshooting

- **"Datasource ${DS_PROMETHEUS} was not found"** — the UID template
  placeholder never got bound at import time. Reimport via the web UI
  and pick a Prometheus data source on the dialog, or ensure your
  `grafana-cli` payload includes the `inputs[]` entry shown above.
- **Panels show `No data`** — confirm your scrape config includes the
  SOMA target and that `soma_entries` has at least one non-zero
  series in Prometheus directly. If that is empty, SOMA is not
  receiving traffic or the `[metrics]` extra is not installed.
- **Only the overview dashboard lights up** — the auth + bundle-health
  dashboards depend on metrics that the Phase 1 / Phase 4 features
  populate. Older SOMA builds will render empty panels in those
  dashboards; upgrade to a build that emits the relevant series.
- **Consolidation-duration panel is half-empty** — normal on any
  pre-Phase-8 build, where `soma_compaction_seconds_bucket` does not
  exist yet. The `soma_consolidate_seconds` target renders fine.

## Related documentation

- [`docs/observability.md`](../../docs/observability.md) — full metric reference.
- [`docs/plans/2026-04-16-phase-3-observability.md`](../../docs/plans/2026-04-16-phase-3-observability.md) — design rationale.
- [`docs/plans/2026-04-16-phase-9-grafana-dashboards.md`](../../docs/plans/2026-04-16-phase-9-grafana-dashboards.md) — these dashboards' plan.
