# Cloud deployment runbooks

SOMA ships ready-to-use manifests for four common hosts. Every path
boots the same `Dockerfile` (see repo root), so the service is
identical across providers — only the control plane differs.

Minimum viable tier everywhere: **2 GB RAM, 1 vCPU, 1 GB persistent
disk**. 1 GB RAM will OOM on the first `/retrieve` once the sbert model
(~90 MB on disk, ~400 MB resident) loads. The pre-baked Docker image
already contains `all-MiniLM-L6-v2`, so cold-start is ~8 s rather than
the ~60 s a bare image pays while it pulls the snapshot from HF Hub.

All endpoints are documented in `src/soma/serve.py`; health is
`GET /health`, version is `GET /version`, storage is `POST /store` /
`POST /store_batch`, retrieval is `POST /retrieve`. Authentication is
off by default — set `SOMA_API_KEY` to require
`Authorization: Bearer <key>` on every non-liveness call.

---

## 1. Railway

Railway is the fastest path: `railway.json` pins the Dockerfile
builder, sets health-check + restart policy, and Railway auto-provisions
a public URL on first deploy.

**One-click**: click the Deploy on Railway button in the README. This
forks the repo template into your workspace, seeds `SOMA_API_KEY` as a
generated secret, and kicks off a build.

**CLI** (once the project is linked):

```bash
npm i -g @railway/cli
railway login
railway link                # or `railway init` for a fresh project
railway variables set SOMA_API_KEY=$(openssl rand -hex 32)
railway variables set SOMA_EMBED_MODEL=all-MiniLM-L6-v2
railway up                  # builds from Dockerfile, deploys
railway open                # open the generated *.up.railway.app URL
```

Railway injects `PORT` at runtime; our shell-form `CMD` honors it. For
persistence, attach a volume from the Railway dashboard and mount it at
`/app/data` so the default `SOMA_BUNDLE_PATH` survives redeploys.

---

## 2. Render

Render reads `render.yaml` as a Blueprint — one file describes the
service, its secrets, and a disk. `SOMA_API_KEY` uses
`generateValue: true` so Render mints a random secret for you on first
apply.

**One-click**: click the Deploy to Render button in the README. Render
detects the Blueprint, previews the plan, and deploys on approval.

**CLI / dashboard**:

```bash
# Dashboard: New → Blueprint → connect your fork → Apply.
# Or from a local clone with the render-cli:
brew install render
render blueprint launch     # picks up render.yaml
```

The Blueprint ships with a 1 GB `soma-data` disk mounted at
`/app/data` — bundles persist across redeploys. Bump `plan: standard`
to `plan: pro` if you need more than 2 GB RAM. Health probes hit
`/health` on every deploy; a failing probe rolls back automatically.

---

## 3. Fly.io

Fly reads `fly.toml`, provisions a volume on first `fly launch`, and
deploys a Firecracker VM per region. The config here pins `iad` as the
primary region, enables auto-start / auto-stop, and keeps one machine
warm so health probes never cold-start into timeout.

```bash
brew install flyctl          # or `curl -L https://fly.io/install.sh | sh`
fly auth login
fly launch --copy-config --no-deploy        # imports fly.toml
fly volumes create soma_data --size 1 --region iad
fly secrets set SOMA_API_KEY=$(openssl rand -hex 32)
fly deploy
fly status                   # show machine + health-check state
fly logs                     # tail application logs
```

The `[[mounts]]` block attaches `soma_data` at `/data`, and the env
block points `SOMA_BUNDLE_PATH` / `SOMA_BUNDLES_DIR` into that volume.
Scale vertically with `fly scale vm shared-cpu-1x --memory 4096`; add
regions with `fly scale count 2 --region iad,fra`.

---

## 4. Kubernetes (Helm)

SOMA ships a first-class Helm chart at `deploy/helm/soma`, distributed
via OCI on `ghcr.io` + GitHub Pages. One install gets you a
StatefulSet, a PVC, a generated API-key Secret, and optional Ingress /
HTTPRoute / ServiceMonitor.

```bash
# Helm 3.14+ — OCI is native, no experimental flag needed.
helm install soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.0

# Verify:
kubectl get statefulset,svc,secret -l app.kubernetes.io/name=soma
kubectl port-forward svc/soma 8420:8420
curl http://localhost:8420/health

# Pull the auto-generated API key:
export SOMA_API_KEY=$(kubectl get secret soma-api \
  -o jsonpath='{.data.SOMA_API_KEY}' | base64 -d)
```

`replicaCount` is locked at 1 (WAL is single-writer; Phase 6 unlocks
horizontal scale). 2 GiB RAM floor and a default StorageClass are the
only hard prereqs. Full runbook + values reference + troubleshooting:
[`docs/deployment-k8s.md`](deployment-k8s.md).

---

## 5. DigitalOcean / generic VPS

Any Docker-capable host works. The shipped `docker-compose.yml` is the
reference — copy the repo to the box and bring it up:

```bash
# On a fresh Ubuntu droplet:
curl -fsSL https://get.docker.com | sh
git clone https://github.com/soma-ai/SOMA.git && cd SOMA
export SOMA_API_KEY=$(openssl rand -hex 32)
echo "SOMA_API_KEY=$SOMA_API_KEY" > .env
docker compose up -d
docker compose logs -f soma-memory       # watch boot
curl http://localhost:8420/health        # liveness
```

For production put a reverse proxy (Caddy / Traefik / nginx) in front
for TLS. The compose file already declares a named volume `soma-data`
bound to `/app/data`, so `docker compose down` + `docker compose up -d`
preserves every bundle. Snapshots are a plain `tar czf` of the volume
root — the whole brain is a single directory.

---

## Troubleshooting

**OOM during first `/retrieve`.** sbert loads lazily on the first
retrieve call; resident RSS jumps by ~400 MB. Bump the host tier to
2 GB RAM (Render `standard`, Fly `memory_mb = 2048`, Railway default
already 2 GB). 1 GB tiers will SIGKILL the process mid-load.

**Cold start > 30 s.** The image pre-bakes `all-MiniLM-L6-v2` during
`docker build`, so a fresh container should reach `/health` in ~8 s.
If you see 60 s+, confirm the build logs include the
`SentenceTransformer('all-MiniLM-L6-v2')` warmup step — a cached
layer from an older image will skip it. Force a clean build with
`docker build --no-cache -t soma .` (or on Railway, bump a dummy env
var to invalidate their cache).

**Bundle not persisting across redeploys.** Check the mount: on
Render the disk name must match `soma-data`, on Fly run
`fly volumes list` to confirm `soma_data` is attached to the app, on
compose verify `docker volume inspect soma-data` shows a non-empty
`Mountpoint`. If `SOMA_BUNDLE_PATH` points somewhere *outside* the
mounted volume (e.g., a typo'd `/data/memory` on Render where the
mount is `/app/data`), every redeploy starts from an empty bundle.

**`401 Unauthorized` from every endpoint.** `SOMA_API_KEY` is set.
Either unset it (public deploy) or send `Authorization: Bearer <key>`
on every request. `/health` and `/version` remain unauthenticated by
design so platform probes keep working.
