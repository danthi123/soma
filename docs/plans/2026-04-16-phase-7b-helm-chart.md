# Phase 7b — Helm chart for k8s users

> **For Claude:** Dispatch impl after Phase 3 (Observability) lands so ServiceMonitor has a real `/metrics` endpoint to scrape. Execute via TDD with `helm lint` + `helm template | kubeconform` + `ct install` on `kind` as the verification layer.

**Goal:** First-class Helm chart alongside the Docker / Fly / Railway / Render templates, so k8s operators can `helm install soma` on their own clusters. Single-replica StatefulSet (WAL is single-writer until Phase 6). Auto-generated API-key Secret. Opt-in Ingress / HTTPRoute / ServiceMonitor. OCI on ghcr.io as primary distribution + GitHub Pages as secondary.

**Architecture:**
- **StatefulSet** (not Deployment): `volumeClaimTemplates`-backed PVC at `/app/data`. Orderly rolling updates release RWO before next pod binds. Matches Qdrant/Weaviate/Milvus pattern.
- **replicaCount: 1 locked** — WAL is single-writer; `values.schema.json` pins `maximum: 1`. Relaxed when Phase 6's external `VectorBackend` lands.
- **Secret modes:** `api.generateSecret: true` (default, once on fresh install via `lookup` idempotency) / `api.existingSecret: "name"` / `api.enabled: false` (open).
- **Networking:** ClusterIP Service on 8420, opt-in Ingress (cert-manager-friendly), opt-in HTTPRoute (Gateway API v1), mutually exclusive.
- **Observability:** opt-in ServiceMonitor + PodMonitor + `prometheus.io/scrape` annotations — all gated behind `metrics.enabled` + `.Capabilities.APIVersions.Has` checks.
- **startupProbe** wide enough for sbert warmup on cold 2 GB nodes (150 s).
- **Distribution:** OCI `oci://ghcr.io/soma-ai/charts/soma` + GH Pages `https://soma-ai.github.io/soma-helm` via `helm/chart-releaser-action`.
- **CI:** `helm lint` + `kubeconform` across k8s 1.28/1.29/1.30 + `ct install` on `kind` for PRs.

**Tech stack:** Helm 3.14+ (OCI GA, no experimental flags), `helm/chart-testing-action@v2`, `helm/kind-action@v1`, `azure/setup-helm@v4`, `helm/chart-releaser-action@v1`.

---

### Task 1: Chart scaffolding

**Files:**
- Create: `deploy/helm/soma/Chart.yaml`
- Create: `deploy/helm/soma/values.yaml`
- Create: `deploy/helm/soma/values.schema.json`
- Create: `deploy/helm/soma/.helmignore`
- Create: `deploy/helm/soma/templates/_helpers.tpl`
- Create: `deploy/helm/soma/templates/NOTES.txt`
- Create: `deploy/helm/soma/README.md`

**Step 1:** `Chart.yaml`:
```yaml
apiVersion: v2
name: soma
description: SOMA — local-first agent memory layer
type: application
version: 0.1.0            # chart version
appVersion: "0.1.0"       # application version
kubeVersion: ">=1.28.0-0"
home: https://github.com/soma-ai/SOMA
sources:
  - https://github.com/soma-ai/SOMA
keywords: [memory, vector-db, rag, ai, agent]
maintainers:
  - name: Daniel Thiberge
    email: daniel.thiberge@gmail.com
```

**Step 2:** `values.yaml` per research report §9 (full shape already designed). Include explicit lock comment on `replicaCount: 1`.

**Step 3:** `values.schema.json` pins `replicaCount.maximum: 1`, all paths/ports validated, required keys enforced.

**Step 4:** `_helpers.tpl` with `soma.name`, `soma.fullname`, `soma.labels`, `soma.selectorLabels`, `soma.serviceAccountName`, `soma.apiSecretName`, plus an idempotent `soma.apiKey` that uses `lookup` to preserve an existing secret value across upgrades.

**Step 5:** `NOTES.txt`: print the port-forward command + generated API key (when auto-generated) + curl-to-/health snippet.

**Step 6:** commit `feat(helm): chart scaffolding — Chart.yaml, values, helpers, NOTES`.

### Task 2: StatefulSet + Service + probes

**Files:**
- Create: `deploy/helm/soma/templates/statefulset.yaml`
- Create: `deploy/helm/soma/templates/service.yaml`
- Create: `deploy/helm/soma/templates/serviceaccount.yaml`

**Step 1:** StatefulSet template:
- `kind: StatefulSet` + `serviceName: {{ include "soma.fullname" . }}-headless`
- `podManagementPolicy: OrderedReady`, `updateStrategy.type: RollingUpdate`
- `volumeClaimTemplates` with `accessModes: [ReadWriteOnce]`, configurable storageClass + size
- Probes: startup (150s budget), liveness, readiness — all `/health`
- Resources: request 500m/2Gi, limit 3Gi (no CPU limit — throttling hurts sbert)
- podSecurityContext + securityContext: non-root, drop-all caps, `fsGroup: 1000`
- Env from ConfigMap (non-secret) + Secret (API key / future JWT)

**Step 2:** Headless Service (`clusterIP: None`) for StatefulSet + primary ClusterIP Service on port 8420.

**Step 3:** ServiceAccount template gated on `.Values.serviceAccount.create`.

**Step 4:** commit `feat(helm): StatefulSet + Service + ServiceAccount`.

### Task 3: Secret + ConfigMap

**Files:**
- Create: `deploy/helm/soma/templates/secret-api.yaml`
- Create: `deploy/helm/soma/templates/configmap.yaml`

**Step 1:** `secret-api.yaml` gated on `.Values.api.enabled` and rendered only when `.Values.api.existingSecret` is empty:
```yaml
{{- if and .Values.api.enabled (not .Values.api.existingSecret) }}
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "soma.apiSecretName" . }}
  annotations:
    "helm.sh/resource-policy": keep     # survive uninstall to prevent key loss
stringData:
  SOMA_API_KEY: {{ include "soma.apiKey" . | quote }}
{{- end }}
```
The `soma.apiKey` helper uses `lookup "v1" "Secret" .Release.Namespace (include "soma.apiSecretName" .)` to preserve an existing value across upgrades. Only first install generates.

**Step 2:** `configmap.yaml` carries non-secret env (paths, embed model).

**Step 3:** commit `feat(helm): Secret with idempotent generate-once + ConfigMap`.

### Task 4: Ingress + HTTPRoute mutex

**Files:**
- Create: `deploy/helm/soma/templates/ingress.yaml`
- Create: `deploy/helm/soma/templates/httproute.yaml`

**Step 1:** Validator in `_helpers.tpl`:
```yaml
{{- define "soma.validateNetworking" -}}
{{- if and .Values.ingress.enabled .Values.httpRoute.enabled -}}
{{- fail "Enable exactly one of ingress.enabled or httpRoute.enabled." -}}
{{- end -}}
{{- end -}}
```
Called early in every networking template.

**Step 2:** `ingress.yaml` gated on `.Values.ingress.enabled`; supports cert-manager annotations.

**Step 3:** `httproute.yaml` gated on `.Values.httpRoute.enabled` AND `.Capabilities.APIVersions.Has "gateway.networking.k8s.io/v1"`.

**Step 4:** commit `feat(helm): Ingress + HTTPRoute with mutex validator`.

### Task 5: ServiceMonitor + PodMonitor (Phase 3 integration)

**Files:**
- Create: `deploy/helm/soma/templates/servicemonitor.yaml`
- Create: `deploy/helm/soma/templates/podmonitor.yaml`

**Step 1:** ServiceMonitor gated on `.Values.metrics.serviceMonitor.enabled` AND `.Capabilities.APIVersions.Has "monitoring.coreos.com/v1"`. Scrape `/metrics` path, port 8420.

**Step 2:** PodMonitor as fallback for same CRD family.

**Step 3:** Update Pod annotations template to emit `prometheus.io/scrape: "true"` when `metrics.annotations.enabled: true`.

**Step 4:** commit `feat(helm): ServiceMonitor + PodMonitor for Prometheus Operator`.

### Task 6: CI — lint, kubeconform, ct install on kind

**Files:**
- Create: `.github/workflows/helm.yml`
- Create: `deploy/helm/ci/default-values.yaml`
- Create: `deploy/helm/ci/ingress-enabled-values.yaml`
- Create: `deploy/helm/ci/servicemonitor-values.yaml`
- Create: `ct.yaml` at repo root

**Step 1:** `helm.yml` workflow:
```yaml
name: helm
on:
  push: { branches: [main], tags: ['chart-v*'] }
  pull_request: { paths: ['deploy/helm/**', '.github/workflows/helm.yml'] }
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: azure/setup-helm@v4
      - run: helm lint deploy/helm/soma
  kubeconform:
    runs-on: ubuntu-latest
    strategy:
      matrix: { kube-version: ['1.28', '1.29', '1.30'] }
    steps:
      - uses: actions/checkout@v4
      - uses: azure/setup-helm@v4
      - run: |
          helm template deploy/helm/soma > rendered.yaml
          curl -sSL https://github.com/yannh/kubeconform/releases/latest/download/kubeconform-linux-amd64.tar.gz | tar xz
          ./kubeconform -kubernetes-version ${{ matrix.kube-version }} -strict -schema-location default rendered.yaml
  ct-install:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: helm/kind-action@v1
      - uses: helm/chart-testing-action@v2
      - run: ct lint --config ct.yaml
      - run: ct install --config ct.yaml
```

**Step 2:** `ct.yaml` points at `deploy/helm/soma` and `deploy/helm/ci/*.yaml` for test value files.

**Step 3:** commit `ci(helm): lint + kubeconform + ct install on kind`.

### Task 7: Release pipeline — OCI + GH Pages

**Files:**
- Modify: `.github/workflows/helm.yml` — add release jobs on `chart-v*` tag
- Create: `.github/workflows/helm-release.yml` (if keeping release separate)

**Step 1:** Release job pushes to:
- `oci://ghcr.io/soma-ai/charts/soma` via `helm push deploy/helm/soma-$ver.tgz oci://ghcr.io/soma-ai/charts`
- `gh-pages` branch via `helm/chart-releaser-action@v1` (generates `index.yaml` + uploads tarball to GitHub Releases)

**Step 2:** Fail-fast if either publish step fails (no drift between channels).

**Step 3:** commit `ci(helm): OCI + GH Pages dual-channel release`.

### Task 8: Docs — k8s runbook

**Files:**
- Create: `docs/deployment-k8s.md`
- Modify: `docs/deployment-cloud.md` — add "Kubernetes" section with pointer
- Modify: `README.md` — add Helm install snippet in "Cloud deploy" section
- Modify: `CHANGELOG.md` — add `### Added — k8s` entry

**Step 1:** Runbook sections:
- Prereqs (k8s 1.28+, Helm 3.14+, optional Prometheus Operator)
- Quickstart: `helm install soma oci://ghcr.io/soma-ai/charts/soma` (one-liner + verify)
- Classic repo path: `helm repo add soma https://soma-ai.github.io/soma-helm`
- Values reference (condensed; point at chart README for full)
- Secret management (generate vs existing vs external secret store)
- Persistence tips (StorageClass, expansion, Velero backup)
- Observability (ServiceMonitor + Grafana dashboard pointer from Phase 3)
- Upgrading (idempotent secret preservation, PVC retention)
- Multi-replica roadmap note (blocked until Phase 6)
- Troubleshooting (OOM on small nodes, sbert cold-start, RWO stuck)

**Step 2:** commit `docs: Phase 7b — k8s deployment runbook + Helm install in README`.

---

## Ship-blocker thresholds

- `helm lint` clean.
- `kubeconform -strict` passes for k8s 1.28/1.29/1.30.
- `ct install` on `kind` succeeds for `default`, `ingress-enabled`, and `servicemonitor` value files.
- `NOTES.txt` renders + prints usable port-forward and `/health` curl.
- Fresh install on `kind`: pod reaches Ready within 180 s (pre-baked sbert image keeps this under budget).

## Risks

1. **Multi-replica footgun:** `replicaCount: 2` corrupts WAL. Mitigation: `values.schema.json` `maximum: 1` + `{{ fail }}` template guard. Unlocked by Phase 6.
2. **Bare-metal / minikube without dynamic provisioning:** PVC stays Pending. Documented: `persistence.storageClassName: "manual"` + pre-created PV.
3. **sbert cold-start on 1 GB:** OOM. Documented: request 2 Gi floor.
4. **RS256 PEM via `--set`:** newline handling. Documented: `--set-file jwt.privateKey=./jwt.key`.
5. **OCI + GH Pages drift:** CI release must fail if either channel fails.
6. **Helm 4 transition:** some docs carry Helm 4 previews. Stay on 3.x; document migration timing in a follow-up.

## Open questions

- **CRD/operator story:** A `SomaCluster` operator for multi-tenant sharding is attractive post-Phase 6. Out of scope for v1.
- **Horizontal scale under Phase 6:** Once the `VectorBackend` Protocol lands with a Qdrant HTTP adapter, re-evaluate HPA templates. Track in Phase 6 plan's "WAL across hosts" open question.
- **Topology spread vs anti-affinity:** ship both stubs (off by default); production users pick whichever matches their fleet.

## Related plans

- Phase 3 — `docs/plans/2026-04-16-phase-3-observability.md` (ServiceMonitor scrapes `/metrics` shipped there)
- Phase 4 — `docs/plans/2026-04-16-phase-4-jwt-auth.md` (JWT secret templating here mirrors HS256/RS256 env shape)
- Phase 6 — `docs/plans/2026-04-16-phase-6-vector-backend.md` (unblocks multi-replica and HPA)
- Phase 7 — committed at `36966aa` (Docker/Fly/Railway/Render — this is the k8s peer)
