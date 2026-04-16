# SOMA Helm Chart

Local-first agent memory layer — plastic-graph replacement for
vector-DB + RAG — packaged for Kubernetes.

## TL;DR

```bash
# OCI (Helm 3.14+; primary channel):
helm install soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.0

# Classic repo (secondary channel):
helm repo add soma https://soma-ai.github.io/soma-helm
helm install soma soma/soma
```

Minimum cluster: Kubernetes 1.28+ with a default StorageClass and
at least 2 GiB RAM available on one schedulable node.

## Architecture

- **StatefulSet (`replicaCount: 1`, LOCKED).** WAL is single-writer
  until Phase 6 ships the external vector backend. Two pods would
  corrupt the paired `memory_ops.wal.jsonl` + `memory_embeddings.wal.bin`
  sidecars. `values.schema.json` pins `maximum: 1` and the chart
  re-asserts the invariant via `{{ fail }}` at render.
- **Headless + ClusterIP Services.** Headless service backs stable
  pod DNS; ClusterIP on port 8420 is what everyone else targets.
- **Secret modes:** auto-generate (default, preserved across upgrades
  via `lookup`), `existingSecret: "<name>"`, or `api.enabled: false`.
- **Ingress OR HTTPRoute,** never both (enforced by a `{{ fail }}`
  validator in `_helpers.tpl`).
- **ServiceMonitor + PodMonitor** gated on CRD presence
  (`monitoring.coreos.com/v1`).

## Common patterns

### Fresh install with generated API key

```bash
helm install soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.0

# Pull the key back out:
kubectl get secret soma-api \
  -o jsonpath='{.data.SOMA_API_KEY}' | base64 -d
```

### Bring your own Secret

```bash
kubectl create secret generic my-soma-key \
  --from-literal=SOMA_API_KEY=$(openssl rand -hex 32)

helm install soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.0 \
  --set api.existingSecret=my-soma-key
```

### Behind an Ingress (cert-manager)

```yaml
# values.yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: 10m
  hosts:
    - host: soma.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: soma-tls
      hosts:
        - soma.example.com
```

### Gateway API HTTPRoute

```yaml
# values.yaml
ingress:
  enabled: false
httpRoute:
  enabled: true
  parentRefs:
    - name: gateway
      namespace: gateway-system
  hostnames:
    - soma.example.com
```

### Prometheus Operator scraping

```yaml
# values.yaml
metrics:
  serviceMonitor:
    enabled: true
    interval: 30s
    labels:
      release: prometheus
```

## Values — partial reference

| Key | Default | Notes |
| --- | --- | --- |
| `replicaCount` | `1` | LOCKED at 1. Schema `maximum: 1`. |
| `image.repository` | `ghcr.io/soma-ai/soma` | |
| `image.tag` | `""` (falls back to chart appVersion) | |
| `service.port` | `8420` | |
| `persistence.size` | `10Gi` | PVC via `volumeClaimTemplates`. |
| `persistence.storageClassName` | `""` (cluster default) | Set to `"manual"` on bare-metal / minikube. |
| `resources.requests.memory` | `2Gi` | Required — 1 Gi OOMs on first `/retrieve`. |
| `resources.limits.memory` | `3Gi` | No CPU limit (throttling hurts sbert). |
| `api.enabled` | `true` | Bearer-token auth on non-liveness routes. |
| `api.existingSecret` | `""` | Opt into a pre-created Secret. |
| `ingress.enabled` | `false` | Mutually exclusive with `httpRoute.enabled`. |
| `httpRoute.enabled` | `false` | Needs Gateway API v1. |
| `metrics.serviceMonitor.enabled` | `false` | Gated on Prometheus Operator CRDs. |
| `metrics.podMonitor.enabled` | `false` | Fallback when ServiceMonitor isn't applicable. |
| `metrics.annotations.enabled` | `false` | Emit `prometheus.io/scrape` annotations. |

Full values: see `values.yaml`. Schema: `values.schema.json`.

## Upgrade

```bash
helm upgrade soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.1
```

`helm upgrade` is safe by default:
- The Secret has `helm.sh/resource-policy: keep` → survives uninstall,
  and the `lookup` helper preserves its value on every upgrade.
- The StatefulSet uses `podManagementPolicy: OrderedReady` +
  `updateStrategy: RollingUpdate` → only one pod restarts at a time,
  and the PVC releases RWO before the replacement pod binds.

## Uninstall

```bash
helm uninstall soma
# Secret soma-api persists because of the `helm.sh/resource-policy: keep`
# annotation. Delete it manually only if you're sure you don't want to
# reuse the API key on a future install.
kubectl delete secret soma-api
# PVCs are also retained by default. Delete them manually if desired:
kubectl delete pvc data-soma-0
```

## Roadmap

- `replicaCount > 1` + HPA template stubs when Phase 6 lands (external
  `VectorBackend`; the Qdrant adapter is the reference).
- CRD / `SomaCluster` operator for multi-tenant sharding.
- Topology-spread and anti-affinity presets tuned to common fleets.

## See also

- Runbook: [`docs/deployment-k8s.md`](../../../docs/deployment-k8s.md)
- Plan: [`docs/plans/2026-04-16-phase-7b-helm-chart.md`](../../../docs/plans/2026-04-16-phase-7b-helm-chart.md)
- Observability design: [`docs/observability.md`](../../../docs/observability.md)
