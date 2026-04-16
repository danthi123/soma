# Kubernetes deployment runbook

> For a higher-level tour of every shipped deploy path (Docker / Fly /
> Railway / Render), see [`docs/deployment-cloud.md`](deployment-cloud.md).

SOMA ships a first-class Helm chart at `deploy/helm/soma`, distributed
via **OCI** on `ghcr.io` (primary) and a **classic Helm repo** on
GitHub Pages (secondary). One `helm install` gets you a StatefulSet,
a PVC, a generated API-key Secret, and optional Ingress / HTTPRoute /
ServiceMonitor — all opt-in and all gated by the schema.

## Prereqs

- Kubernetes **1.28+** (chart `kubeVersion: ">=1.28.0-0"`).
- Helm **3.14+** (native OCI support, no experimental flag).
- A default `StorageClass` with dynamic provisioning, or pre-provision
  a PV and set `persistence.storageClassName: "manual"`.
- **2 GiB RAM** schedulable on one node. sbert resident footprint is
  ~400 MB on first `/retrieve`; 1 GiB nodes OOM-kill the pod mid-load.
- Optional: Prometheus Operator (`monitoring.coreos.com/v1` CRDs) for
  ServiceMonitor / PodMonitor. Optional: Gateway API v1 CRDs for the
  HTTPRoute path.

## Quickstart — OCI one-liner

```bash
helm install soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.0
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

Fresh-install latency on a pre-baked image: **~10 s** to Ready. Cold
pull on a 2 GB node: **~60 s** (covered by the 150 s `startupProbe`).

## Classic repo path (GitHub Pages)

```bash
helm repo add soma https://soma-ai.github.io/soma-helm
helm repo update
helm install soma soma/soma --version 0.1.0
```

Same chart, same versions. Pick whichever channel your fleet
standardizes on.

## Values reference (condensed)

Full shape: [`deploy/helm/soma/values.yaml`](../deploy/helm/soma/values.yaml).
Schema: [`deploy/helm/soma/values.schema.json`](../deploy/helm/soma/values.schema.json).

| Key | Default | Notes |
| --- | --- | --- |
| `replicaCount` | `1` | **LOCKED.** Schema `maximum: 1`. Unlocks when Phase 6 ships external `VectorBackend`. |
| `image.repository` | `ghcr.io/soma-ai/soma` | |
| `image.tag` | `""` (falls back to chart `appVersion`) | Pin for reproducible installs. |
| `service.port` | `8420` | Matches the Dockerfile `EXPOSE`. |
| `persistence.size` | `10Gi` | PVC via `volumeClaimTemplates`. |
| `persistence.storageClassName` | `""` (cluster default) | Use `"manual"` on minikube / bare-metal. |
| `resources.requests.memory` | `2Gi` | **Required by schema.** 1 Gi OOMs on first `/retrieve`. |
| `resources.limits.memory` | `3Gi` | No CPU limit (throttling hurts sbert). |
| `api.enabled` | `true` | Bearer-token auth on non-liveness routes. |
| `api.existingSecret` | `""` | Use a pre-created opaque Secret with key `SOMA_API_KEY`. |
| `ingress.enabled` | `false` | Mutually exclusive with `httpRoute.enabled`. |
| `httpRoute.enabled` | `false` | Needs Gateway API v1 CRDs. |
| `metrics.serviceMonitor.enabled` | `false` | Gated on `monitoring.coreos.com/v1`. |
| `metrics.podMonitor.enabled` | `false` | Fallback when ServiceMonitor isn't applicable. |
| `metrics.annotations.enabled` | `false` | Emit legacy `prometheus.io/scrape` annotations. |

## Secret management

### Auto-generate (default)

```bash
helm install soma oci://ghcr.io/soma-ai/charts/soma
```

The chart mints a 32-byte random key on first install. On every
`helm upgrade`, a `lookup` helper preserves the existing Secret value
so every caller keeps working. The Secret carries
`helm.sh/resource-policy: keep`, so `helm uninstall` leaves it behind —
reinstalling reuses the same key by design.

### Bring your own Secret

```bash
kubectl create secret generic my-soma-key \
  --from-literal=SOMA_API_KEY=$(openssl rand -hex 32)

helm install soma oci://ghcr.io/soma-ai/charts/soma \
  --set api.existingSecret=my-soma-key
```

The chart skips its own Secret template and points `envFrom` at yours.

### External secret store (ESO / Vault / CSI)

Point `api.existingSecret` at the Secret your external secret
operator projects into the namespace. Chart is agnostic about how the
Secret got there.

### Disable auth

```bash
helm install soma oci://ghcr.io/soma-ai/charts/soma \
  --set api.enabled=false
```

Every endpoint becomes open. Only for behind-a-cluster-firewall
deployments — in public clusters, set `api.enabled=true`.

## Persistence

- **RWO is fine** — StatefulSet `OrderedReady` + `RollingUpdate`
  release the volume before the replacement pod binds. No RWX
  required (until Phase 6 unlocks multi-replica).
- **Expand a PVC:** set the new size in values, run `helm upgrade`,
  then `kubectl patch pvc data-soma-0 --type merge -p
  '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'`. Requires
  a StorageClass with `allowVolumeExpansion: true`.
- **Backup:** Velero snapshots the PVC + Secret + ConfigMap on a
  schedule. SOMA bundles are plain directories — a `tar czf` of
  `/app/data` is a complete snapshot.
- **Minikube / kind / bare-metal without dynamic provisioning:**
  pre-create a PV and set `persistence.storageClassName: "manual"`.

## Observability

SOMA Phase 3 exposes `/metrics` via `prometheus-fastapi-instrumentator`.

```yaml
# values.yaml
metrics:
  serviceMonitor:
    enabled: true
    interval: 30s
    labels:
      release: prometheus    # match your Prometheus Operator selector
```

The `monitoring.coreos.com/v1` gate skips the template silently if the
Prometheus Operator isn't installed. Pair with the Grafana dashboard
pointer in [`docs/observability.md`](observability.md).

For structured JSON logs (Loki / Datadog / CloudWatch):

```yaml
env:
  SOMA_LOG_JSON: "1"
```

For OpenTelemetry spans:

```yaml
extraEnv:
  - name: SOMA_OTEL_ENABLED
    value: "1"
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://otel-collector.observability:4317"
```

## Upgrading

```bash
helm upgrade soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.1
```

What's preserved:

- **API key** — `lookup` helper reads the current Secret value.
- **PVC** — `volumeClaimTemplates` are retained by default.
- **Secret across uninstall** — `helm.sh/resource-policy: keep`.

What's rolled:

- The Pod — `OrderedReady` + `RollingUpdate` means one pod at a time.
  The `checksum/config` + `checksum/secret` annotations trigger a
  rollout on content change (not just on value overrides).

## Multi-replica roadmap

`replicaCount > 1` is **blocked** by the schema because SOMA's WAL is
single-writer. Two pods sharing a PVC would corrupt the paired
`memory_ops.wal.jsonl` + `memory_embeddings.wal.bin` sidecars. Phase 6
(external `VectorBackend`) unlocks horizontal scale; the first
adapter lands against Qdrant HTTP, and the chart will relax the
schema + ship HPA + anti-affinity / topology-spread presets.

Today's horizontal-scale pattern: **bundle sharding**. One SOMA
release per bundle, routed by a lightweight gateway. Each release
gets its own PVC and its own WAL — no shared-writer problem.

## Troubleshooting

**`OOMKilled` on first `/retrieve`.** Bump `resources.requests.memory`
to `2Gi`. sbert loads lazily on the first retrieve call and the
resident RSS jumps ~400 MB. The schema already requires a memory
request — but if you've overridden it too low, this is the symptom.

**`startupProbe` failed 30 times.** The pre-baked image should reach
Ready in ~10 s. If you see the 150 s budget exhausted, the image
layer cache is cold (fresh pull on a new node). The pod will restart
and try again; subsequent boots hit the warm layer and Ready comes
fast. If it keeps failing, pull the logs:

```bash
kubectl logs soma-0 --previous
```

**PVC stuck in `Pending`.** No default StorageClass and no manual
PV. Either pre-create a PV (`persistence.storageClassName: "manual"`)
or install a provisioner
(`kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml`
for single-node clusters).

**Rolling upgrade stuck — new pod Pending, old pod Running.** RWO +
`OrderedReady` is working as intended — the old pod must terminate
before the new one binds the PVC. If the old pod isn't terminating,
check for a stuck finalizer or a long `terminationGracePeriodSeconds`.

**`ServiceMonitor` not being scraped.** Confirm the Prometheus
Operator's selector matches the ServiceMonitor labels:

```bash
kubectl get prometheus -A -o yaml | grep -A5 serviceMonitorSelector
```

Then set `metrics.serviceMonitor.labels.release: <prometheus-release>`
to match.

**`401 Unauthorized` from every endpoint.** `SOMA_API_KEY` is set.
Either unset it (`--set api.enabled=false`) or send
`Authorization: Bearer <key>` on every request. `/health` and
`/version` remain unauthenticated by design so kubelet probes keep
working.

**Chart fails render: "Enable exactly one of ingress.enabled or
httpRoute.enabled."** Mutex guard fired — set one to `false`.

**Chart fails render: "httpRoute.enabled requires the Gateway API v1
CRD."** Install the Gateway API
(`kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.0.0/standard-install.yaml`)
or use `ingress.enabled` instead.

## See also

- Chart README: [`deploy/helm/soma/README.md`](../deploy/helm/soma/README.md)
- Plan: [`docs/plans/2026-04-16-phase-7b-helm-chart.md`](plans/2026-04-16-phase-7b-helm-chart.md)
- Observability: [`docs/observability.md`](observability.md)
- Durability (WAL): [`docs/plans/2026-04-16-phase-1-wal-autosave.md`](plans/2026-04-16-phase-1-wal-autosave.md)
