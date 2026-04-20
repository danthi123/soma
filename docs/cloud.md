# Running SOMA against cloud object storage

SOMA bundles — the self-contained directories written by
`MemoryLayer.save()` and read by `MemoryLayer.load()` — can live on
S3, GCS, any S3-compatible store (Cloudflare R2, MinIO, DigitalOcean
Spaces), or any local filesystem. This page is the reference for
running against a remote object store on scale-to-zero platforms
(AWS Lambda, Cloud Run, Fly Machines, App Runner) where local disk
is ephemeral.

For PaaS-style hosts with persistent volumes (Railway, Render,
Fly Volumes, Heroku), see [deployment-cloud.md](deployment-cloud.md).
For the broader backend / vector-store story, see
[backends.md](backends.md).

---

## When to reach for cloud storage

Pick a remote bundle URL when at least one of the following is true:

- **Ephemeral local disk.** Lambda, Cloud Run, App Runner, and Fly
  Machines reset their filesystem on every cold start. A bundle on
  local disk doesn't survive — `MemoryLayer.load(...)` starts empty
  on the next request. A bundle on `s3://` or `gs://` does.
- **Multi-replica service.** Two uvicorn workers behind a load
  balancer need to read the same bundle. Pointing them at a shared
  `s3://bucket/prefix` is cheaper than running a durable volume or
  a Postgres-backed vector store.
- **Cost floor is "a bucket", not "a server".** Object storage is
  ~$0.023/GB-month and scales to zero between requests. A Postgres
  instance or Qdrant node does not.
- **Blue/green or rollback.** Saving each bundle version to a new
  prefix (`s3://bucket/bundles/2026-04-16-1842/`) gives you atomic
  rollback: flip the prefix env var, restart, done.

Stay on local disk when:

- Your platform has a persistent volume mounted and the replica
  count is low enough that "local FS + a snapshot cron" covers your
  durability story.
- Your bundle is hot-path enough that the warm-start cost of
  downloading it on every cold start is prohibitive. See
  [Warm-start latency](#warm-start-latency) below.
- You want incremental-write durability. SOMA's local-FS path has
  a WAL sidecar that fsyncs on every `store()` call; the `s3://` /
  `gs://` path does not (see [Caveats](#caveats)).

---

## URL schemes

Every `MemoryLayer.save(dest)` / `MemoryLayer.load(src)` call
accepts one of three URL shapes. Plain paths without a scheme are
treated as `file://`.

| Scheme | Backend | Extra required |
| --- | --- | --- |
| `file:///abs/path` (or plain path) | `LocalFSObjectStore` | none (default) |
| `s3://bucket/prefix` | `S3ObjectStore` | `pip install "soma-memory[s3]"` |
| `gs://bucket/prefix` | `GCSObjectStore` | `pip install "soma-memory[gcs]"` |

Keys beneath the prefix follow the bundle layout documented below.

### `file://` URLs

```python
mem.save("file:///var/data/bundle")
mem.save("/var/data/bundle")               # equivalent
mem.save(Path("/var/data/bundle"))         # equivalent
```

Windows drive letters are accepted in three forms:
`file:///C:/Users/you/bundle`, `file://C:/Users/you/bundle`, or
just `C:/Users/you/bundle`.

### `s3://` URLs

```python
mem.save("s3://my-bucket/bundles/prod")
mem.save("s3://my-bucket/bundles/prod?region=eu-west-1")
mem.save("s3://my-bucket?endpoint=https://minio.internal:9000&region=us-east-1")
```

Query parameters:

- `region=<aws-region>` — sets `region_name` on the boto3 client.
  Inherited from the default boto3 chain when absent.
- `endpoint=<url>` — sets `endpoint_url`. Use this for
  S3-compatible services: MinIO, LocalStack, Cloudflare R2,
  DigitalOcean Spaces, Backblaze B2.

Credentials follow the **boto3 default chain**: `AWS_ACCESS_KEY_ID`
+ `AWS_SECRET_ACCESS_KEY` env vars, `~/.aws/credentials`, IAM
instance profile, EKS IRSA, AWS SSO. SOMA never reads or stores
credentials itself — whatever your existing AWS tooling produces is
what the adapter uses.

### `gs://` URLs

```python
mem.save("gs://my-bucket/bundles/prod")
mem.save("gs://my-bucket/bundles/prod?project=my-project-id")
```

Query parameters:

- `project=<project-id>` — forwarded to the
  `google.cloud.storage.Client` constructor. Set it for
  cross-project service-account usage or quota attribution.

Credentials follow the **Application Default Credentials (ADC)**
chain: `GOOGLE_APPLICATION_CREDENTIALS` env var pointing at a JSON
key, the GCE/GKE/Cloud Run metadata server, or
`gcloud auth application-default login` for local dev.

---

## Bundle layout

`MemoryLayer.save()` writes four objects under the bundle prefix.
The same keys show up under every backend:

```
<prefix>/
├── memory_index.json       # metadata (ids, texts, timestamps, schema)
├── memory_embeddings.pt    # torch tensor of shape (N, dim)
├── tokenizer.json          # only if the TextEncoder path is in use
└── encoder.pt              # only if the TextEncoder path is in use
```

When the bundle is loaded against a bundle root with a WAL sidecar
(`memory_ops.wal.jsonl` + `memory_embeddings.wal.bin`), those files
are also part of the prefix.

Sizes scale with the store:

- `memory_index.json` — a few KB per thousand entries (ids, texts,
  metadata JSON).
- `memory_embeddings.pt` — `N × dim × 4 bytes`. A 100K-entry store
  at 384-dim is ~150 MB. A 1M-entry store at 384-dim is ~1.5 GB.
- `encoder.pt` — ~1-10 MB for a typical SOMA TextEncoder.

Expect **hundreds of MB for a full bundle**. Size your
warm-start latency budget accordingly.

---

## Recipe 1: AWS Lambda + S3

The canonical scale-to-zero path: one Lambda function that embeds
`MemoryLayer.load("s3://...")` and serves `/retrieve` behind API
Gateway. The bundle lives in S3; the Lambda reads it on cold start.

**Dockerfile** (Lambda container image):

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

RUN pip install "soma-memory[s3]" sentence-transformers
COPY handler.py ${LAMBDA_TASK_ROOT}/
CMD ["handler.handler"]
```

**handler.py**:

```python
import os
from soma.memory.api import MemoryLayer

_MEM = None

def _memory() -> MemoryLayer:
    global _MEM
    if _MEM is None:
        _MEM = MemoryLayer.load(
            os.environ["SOMA_BUNDLE_URL"],   # e.g. s3://my-bucket/prod
        )
    return _MEM

def handler(event, context):
    query = event.get("query", "")
    hits = _memory().retrieve(query, k=5)
    return {
        "results": [
            {"id": h.node_id, "text": h.text, "score": float(h.score)}
            for h in hits
        ],
    }
```

**Environment variables**:

- `SOMA_BUNDLE_URL=s3://my-bucket/bundles/prod` — where the bundle lives.
- `AWS_REGION=us-east-1` — boto3 picks this up automatically inside
  Lambda. Not needed in the URL.

**IAM role** (attach to the Lambda execution role):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/bundles/prod/*"
      ]
    }
  ]
}
```

Add `s3:PutObject` if the Lambda also calls `mem.save(...)`.

**Deploy**:

```bash
aws lambda create-function \
  --function-name soma-memory \
  --package-type Image \
  --code ImageUri=<account>.dkr.ecr.us-east-1.amazonaws.com/soma-memory:latest \
  --role arn:aws:iam::<account>:role/soma-memory-role \
  --environment Variables='{SOMA_BUNDLE_URL=s3://my-bucket/bundles/prod}' \
  --memory-size 2048 \
  --timeout 30
```

**Memory sizing**: set `--memory-size 2048` or higher. The
embedding matrix is RAM-resident in the default `InProcBackend`; at
100K × 384-dim that's ~150 MB, plus the sbert model (~400 MB
resident). A 512 MB Lambda will cold-start-OOM.

**Full AWS auth reference**: see the [AWS docs on Lambda IAM
roles](https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html)
and [boto3's credential resolution
order](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

---

## Recipe 2: Google Cloud Run + GCS

Cloud Run's request-scaled-to-zero container model pairs cleanly
with `gs://` bundle URLs. Bundle lives in GCS; the Cloud Run
service reads it on cold start.

**Dockerfile**:

```dockerfile
FROM python:3.11-slim

RUN pip install "soma-memory[gcs]" sentence-transformers uvicorn
COPY app.py /app/
WORKDIR /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

**app.py** (minimal FastAPI wrapper — the real `src/soma/serve.py`
app is already wired for this):

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from soma.memory.api import MemoryLayer

_MEM: MemoryLayer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MEM
    _MEM = MemoryLayer.load(os.environ["SOMA_BUNDLE_URL"])
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/retrieve")
async def retrieve(payload: dict) -> dict:
    assert _MEM is not None
    hits = _MEM.retrieve(payload["query"], k=payload.get("k", 5))
    return {"results": [h.__dict__ for h in hits]}
```

**Service account**: create a dedicated SA and grant
`roles/storage.objectViewer` on the bundle bucket (add
`roles/storage.objectAdmin` if the service also calls `mem.save()`):

```bash
gcloud iam service-accounts create soma-memory
gcloud storage buckets add-iam-policy-binding gs://my-bucket \
  --member=serviceAccount:soma-memory@<project>.iam.gserviceaccount.com \
  --role=roles/storage.objectViewer
```

**Deploy**:

```bash
gcloud run deploy soma-memory \
  --image gcr.io/<project>/soma-memory:latest \
  --region us-central1 \
  --service-account soma-memory@<project>.iam.gserviceaccount.com \
  --set-env-vars SOMA_BUNDLE_URL=gs://my-bucket/bundles/prod \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0
```

Cloud Run auto-resolves ADC against the attached service account;
`GOOGLE_APPLICATION_CREDENTIALS` does not need to be set. For local
dev run `gcloud auth application-default login` once and the same
code path works against your user credentials.

**Full GCP auth reference**: see the [ADC
docs](https://cloud.google.com/docs/authentication/application-default-credentials)
and [Cloud Run service account
setup](https://cloud.google.com/run/docs/configuring/services/service-accounts).

---

## Recipe 3: Fly Machines + Cloudflare R2

Fly Machines are scale-to-zero Firecracker VMs; Cloudflare R2 is an
S3-compatible object store with zero egress fees. Together they're
the cheapest cloud deployment path for SOMA bundles with
meaningful read traffic.

Because R2 speaks S3's protocol, you use the `s3://` scheme with an
`endpoint=` override pointing at your R2 account's S3-compatibility
URL. GCP's ADC chain does **not** apply — R2 uses access keys, same
as S3.

**fly.toml**:

```toml
app = "soma-memory"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[env]
  SOMA_BUNDLE_URL = "s3://my-r2-bucket/bundles/prod?endpoint=https://<account>.r2.cloudflarestorage.com"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  memory = "2gb"
  cpu_kind = "shared"
  cpus = 2
```

**Dockerfile** (same base as the Cloud Run recipe — drop the
`[gcs]` extra, keep `[s3]` since R2 speaks S3):

```dockerfile
FROM python:3.11-slim

RUN pip install "soma-memory[s3]" sentence-transformers uvicorn
COPY app.py /app/
WORKDIR /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Credentials** (set via `flyctl secrets`):

```bash
# Mint an R2 API token at dash.cloudflare.com → R2 → Manage API
# Tokens. The values it returns map onto boto3's env vars 1:1.
flyctl secrets set \
  AWS_ACCESS_KEY_ID=<r2-access-key> \
  AWS_SECRET_ACCESS_KEY=<r2-secret-key> \
  AWS_DEFAULT_REGION=auto
```

R2 ignores `region_name` at the protocol level but boto3 still
requires *some* region string; `auto` is the documented placeholder.

**Deploy**:

```bash
flyctl launch --no-deploy           # one-time, seeds fly.toml
flyctl secrets set ...              # as above
flyctl deploy
```

**Gotcha — R2 endpoint format**: the account-scoped URL is
`https://<account-id>.r2.cloudflarestorage.com` (no bucket in the
hostname — the bucket goes in the path component). If you see
`AccessDenied` or `InvalidArgument` on first PUT, double-check the
endpoint and that the R2 token scope includes the target bucket.

**Gotcha — R2 multipart threshold**: boto3 auto-splits uploads
over 8 MB into multipart. R2 supports multipart but the signed-URL
chunking differs slightly from AWS's; if you hit signature errors
on large bundle saves, pin `Config(s3={'multipart_threshold': 64 * 1024 * 1024})`
on a custom boto3 client and pass it to `S3ObjectStore(client=...)`
instead of using the URL-driven construction.

**Full Fly reference**: see [Fly Machines
docs](https://fly.io/docs/machines/) and [Cloudflare R2 S3-compat
docs](https://developers.cloudflare.com/r2/api/s3/api/).

---

## Warm-start latency

`MemoryLayer.load("s3://...")` downloads the entire bundle on the
first call — there is no lazy / partial-read mode. Cold-start cost
scales linearly with bundle size. Rule-of-thumb on a warm S3 read
(same-region, no throttling):

| Bundle size | S3 download | Cloud Run / Lambda cold start (add) |
| --- | --- | --- |
| 10 MB | <1 s | ~3-4 s total |
| 100 MB | 2-3 s | ~5-7 s total |
| 500 MB | 8-15 s | ~15-25 s total |
| 1 GB+ | 20-40 s | may exceed Cloud Run's 60 s startup probe |

Tactics when latency hurts:

- **Cap instance concurrency low + keep min-instances ≥ 1.** Pay
  the cold-start cost once on deploy instead of once per cold user.
- **Split by prefix.** Store hot shards at one prefix, cold archive
  at another. Boot the service against the hot prefix; fetch cold
  on demand via a separate path.
- **Accept a smaller bundle.** Vector dimensions drive bundle size
  linearly: sbert-MiniLM (384-dim) is 4× smaller than OpenAI
  embeddings (1536-dim) for the same N.
- **Move to a persistent volume.** Fly Volumes, Railway volumes,
  and Render disks all keep the bundle on local SSD between boots;
  cold-start is then O(snapshot read from disk), not O(S3 download).

---

## Caveats

### `save()` is a batch operation

`MemoryLayer.save(...)` writes the full bundle in one pass: four
object PUTs (tokenizer, encoder, embeddings, index). There is no
incremental WAL-to-cloud stream today. If the process crashes
between `store()` and the next `save()`, the un-saved records are
lost.

For the local-FS path, the WAL sidecar
(`memory_ops.wal.jsonl` + `memory_embeddings.wal.bin`) captures
every `store()` synchronously (with `durability="sync"`). The
`s3://` / `gs://` path does not yet stream the WAL to the object
store. A cloud WAL is future work.

**Recommendation**: call `mem.save(...)` on a schedule that matches
your durability budget. For a light-write agent, every 5 minutes
or every 100 writes is defensible; for a heavy-write ingest job,
every batch. Save also before scale-down events — Cloud Run
`SIGTERM`, Lambda shutdown hook, Fly Machine stop.

### Atomic writes

Per-object PUTs are atomic on both S3 and GCS — a crash mid-PUT
leaves the previous object intact. But the full bundle is four
PUTs, not one, and SOMA does **not** wrap them in a transaction.
Readers can observe a half-written bundle if they race a writer
mid-save:

- Writer uploads `tokenizer.json` at T0, `encoder.pt` at T1,
  `memory_embeddings.pt` at T2, `memory_index.json` at T3.
- A reader at T2.5 sees the new tokenizer/encoder/embeddings but
  the old index.

In practice this is rare — `save()` is seconds, not minutes, and
readers do `load()` at process start. If you need cross-object
atomicity (blue/green rollout), save to a fresh timestamped prefix
and flip a pointer once the upload is complete:

```python
mem.save("s3://bucket/bundles/2026-04-16-1842")
# ... verify ...
# Flip the env var / config pointer that your service reads from.
```

### Load is all-or-nothing

`MemoryLayer.load(...)` pulls every bundle object into memory
before returning. There is no partial-bundle mode where only the
index is read and embeddings stream on demand. A 1 GB bundle needs
≥ 1 GB of free RAM at load time (plus the normal overhead of the
live `MemoryLayer` instance).

### Local staging directory

For bundles with a WAL sidecar or a `bundle.lock`, the S3 / GCS
adapters lazily materialise a **local staging directory** — a temp
dir under `tempfile.gettempdir()` that mirrors the bundle prefix.
First access downloads every object; `close()` uploads any new or
modified files back. This is how WAL replay (which needs a real
local path) works transparently against remote storage.

Implications:

- **Disk space.** The staging dir temporarily uses the same space
  as the bundle. Lambda's `/tmp` is 512 MB (expandable to 10 GB);
  Cloud Run has 512 MB of `/tmp` by default. Set
  `TMPDIR=/mount/large` if you're pushing bundle size past the
  default `/tmp` cap.
- **Cleanup.** The staging dir is registered with `atexit`, so a
  clean shutdown uploads changes back. An unclean crash (SIGKILL,
  OOM) drops the changes — the only source of truth is what's in
  S3 / GCS.

See `src/soma/storage/s3.py` (`S3ObjectStore.local_root`) and
`src/soma/storage/gcs.py` (`GCSObjectStore.local_root`) for the
implementation.

---

## Cost considerations

Object storage pricing has three dimensions: storage, requests
(PUT / GET), and egress. For SOMA's bundle workload:

- **Storage** is usually the cheap dimension — hundreds of MB to a
  few GB is negligible (~$0.02/GB-month on S3 Standard).
- **PUT pricing** dominates on heavy `save()` cadence. A bundle
  save is four PUTs; at $0.005 per 1000 PUTs (S3 Standard), every
  1000 saves costs ~$0.02. Saving every 5 min (~12 saves/hr, ~8700
  saves/month) is a few cents. Saving every write, on the other
  hand, adds up.
- **GET pricing** dominates on cold-start-heavy services. Each
  `load()` is four GETs per bundle. A service that cold-starts
  every minute (~43 200 starts/month) at $0.0004 per 1000 GETs
  costs ~$0.07 in GETs plus the egress.
- **Egress** is the sneakiest one. Same-region traffic is free on
  AWS + GCP; cross-region is $0.01-0.09 per GB; internet egress
  is $0.08-0.12 per GB. A 500 MB bundle downloaded to a developer
  laptop costs ~$0.05 per load. Keep compute and storage in the
  same region.

**Zero-egress alternative**: Cloudflare R2 charges $0 for all
egress. For read-heavy cold-starts (e.g. Fly Machines booting in
multiple regions against one bundle), R2 is often the lowest-cost
option even accounting for slightly higher storage pricing.

**Pricing pages** (always current, unlike any numbers in this
doc):

- [AWS S3 pricing](https://aws.amazon.com/s3/pricing/)
- [GCS pricing](https://cloud.google.com/storage/pricing)
- [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/)

---

## When to stay on local disk

If your platform has a persistent volume and your replica count
is one (or small enough that a snapshot cron covers durability),
local disk is usually the right answer:

- **Fly Volumes** attached to a single Machine.
- **Railway volumes** mounted at `/app/data`.
- **Render disks** on a Starter tier or above.
- **Kubernetes PersistentVolumeClaim** on a StatefulSet.

Local disk keeps the WAL hot (fsync per write, no 4-PUT bundle
dance), keeps cold-start O(snapshot read), and avoids both
object-store request costs and the load-is-all-or-nothing RAM
requirement. The tradeoff is durability: a volume failure loses
the bundle. Pair with a periodic `mem.save("s3://backup-bucket/...")`
and you get the best of both — hot local reads, durable off-site
snapshots.

---

## Related reading

- [examples/cloud_s3_demo.py](../examples/cloud_s3_demo.py) — a
  self-contained script that spins up moto, writes a tiny bundle
  to `s3://`, reloads it, and retrieves. Copy it as a starting
  point for your own integration.
- [backends.md](backends.md) — vector-backend pluggability
  (InProc, LanceDB, Qdrant, Chroma, pgvector). Cloud object
  storage is orthogonal: any backend can save/load to any URL
  scheme.
- [deployment-cloud.md](deployment-cloud.md) — PaaS-style hosts
  (Railway, Render, Fly Volumes) that use persistent disks
  instead of object storage.
- [quickstart.md](quickstart.md) — getting a local `MemoryLayer`
  running end-to-end before moving to the cloud.
