"""REST API for SOMA MemoryLayer.

Exposes store / retrieve / get / forget / consolidate / save over HTTP
so non-Python agents can use SOMA as their memory backend. Single-
tenant by default; optional multi-tenant routing under
``/bundles/{name}`` lets one server host many independent brains.

    uvicorn soma.serve:app --host 0.0.0.0 --port 8420
    # or:
    soma serve --port 8420

Environment variables:
    SOMA_BUNDLE_PATH    — single-tenant bundle directory
                          (default: ./data/memory)
    SOMA_BUNDLES_DIR    — root dir for multi-tenant bundles under
                          /bundles/{name}/... (default: ./data/bundles)
    SOMA_EMBED_MODEL    — sentence-transformers model name
                          (default: all-MiniLM-L6-v2). Use the literal
                          value ``stub`` to return 384-d zero vectors
                          (CI / schema-gen only — no model download).
    SOMA_API_KEY        — if set, all non-health endpoints require
                          ``Authorization: Bearer <key>``. Unset = open.
    SOMA_CORS_ORIGINS   — comma-separated allow-list for the browser
                          CORS middleware (default: http://localhost:*).

Endpoints:
    GET  /health                              — liveness probe (no auth)
    GET  /version                             — package version (no auth)
    GET  /status                              — default bundle stats
    POST /store | /store_batch | /retrieve | /forget
    GET  /get/{id} | /recent | /related/{id}
    POST /consolidate | /save
    *    /bundles/{name}/<same as above>    — per-tenant variants
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from soma.log import configure_json_logging
from soma.memory import MemoryLayer

# Structured JSON logging swap (no-op unless SOMA_LOG_JSON=1). Called at
# import time so `uvicorn soma.serve:app` picks up the formatter before
# the first request flows through.
configure_json_logging()

BUNDLE_PATH = Path(os.environ.get("SOMA_BUNDLE_PATH", "./data/memory"))
BUNDLES_DIR = Path(os.environ.get("SOMA_BUNDLES_DIR", "./data/bundles"))
EMBED_MODEL = os.environ.get("SOMA_EMBED_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.environ.get("SOMA_API_KEY", "")

app = FastAPI(
    title="SOMA Memory Layer",
    description="Local-first agent memory that learns.",
    version="0.1.0",
)

# CORS middleware mounted before any routes so browser clients (the
# generated TypeScript client in particular) can talk to the server
# from localhost during dev. Operators override via SOMA_CORS_ORIGINS.
# Entries containing ``*`` are compiled into a regex allow-list (so
# the default ``http://localhost:*`` matches any dev port); literal
# origins go through the standard exact-match path.
_cors_entries = [
    o.strip()
    for o in os.environ.get("SOMA_CORS_ORIGINS", "http://localhost:*").split(",")
    if o.strip()
]
_cors_literals = [o for o in _cors_entries if "*" not in o]
_cors_patterns = [o for o in _cors_entries if "*" in o]
_cors_regex: str | None = (
    "|".join(re.escape(p).replace(r"\*", ".*") for p in _cors_patterns)
    if _cors_patterns
    else None
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_literals,
    allow_origin_regex=_cors_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# --------------------------------------------------------------------
# Optional Prometheus /metrics endpoint. Gated on prometheus-fastapi-
# instrumentator being importable (``soma[metrics]`` extra). When the
# extra isn't installed we silently skip — /metrics 404s and the rest
# of the server keeps working without the observability dep footprint.
# The instrumentator also adds per-route HTTP counters and latency
# histograms (standard FastAPI practice); our memory-layer-specific
# soma_* metrics layer on top via soma.metrics.
# --------------------------------------------------------------------
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=True,
        tags=["system"],
    )
except ImportError:  # pragma: no cover
    # soma[metrics] not installed — /metrics will 404.
    pass

# --------------------------------------------------------------------
# Optional OpenTelemetry tracing. Opt-in via SOMA_OTEL_ENABLED=1 AND
# the `soma[otel]` extra being installed. When both line up, the
# FastAPI instrumentor emits a span per HTTP request that the OTel SDK
# exports to whatever collector OTEL_EXPORTER_OTLP_ENDPOINT points at
# (standard OTel env-var setup applies — we don't override it).
# --------------------------------------------------------------------
_otel_enabled: bool = False
if os.environ.get("SOMA_OTEL_ENABLED", "").strip() in ("1", "true", "True"):
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        _otel_enabled = True
    except ImportError:  # pragma: no cover
        # soma[otel] not installed — skip spans, log loudly enough that
        # operators notice the misconfiguration.
        import warnings

        warnings.warn(
            "SOMA_OTEL_ENABLED=1 but opentelemetry-instrumentation-fastapi "
            "not installed. Install soma[otel] to enable tracing.",
            stacklevel=2,
        )

_BUNDLE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_mem_cache: dict[str, MemoryLayer] = {}
_cache_lock = threading.Lock()


# ------------------------------------------------------------------
# Error model + shared responses (advertise 401/404 in OpenAPI so
# generated clients produce typed error branches)
# ------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Uniform error body; mirrors FastAPI's HTTPException shape."""

    detail: str = Field(..., description="Human-readable error detail.")


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key."},
    404: {"model": ErrorResponse, "description": "Requested resource not found."},
}


# ------------------------------------------------------------------
# Auth — HTTPBearer security scheme so OpenAPI advertises bearer auth.
# Behaviour preserved: unset SOMA_API_KEY = open; set = require Bearer.
# ------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> None:
    """FastAPI dependency: enforce ``SOMA_API_KEY`` when set.

    Unset env var = open server (matches prior behaviour). Set env var =
    require ``Authorization: Bearer <key>``.
    """
    if not API_KEY:
        return
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or credentials.credentials != API_KEY
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ------------------------------------------------------------------
# Embedding closure — one sbert model shared across all bundles
# ------------------------------------------------------------------
_embed_fn_cache: Any = None


def _embed_fn() -> Any:
    global _embed_fn_cache  # noqa: PLW0603
    if _embed_fn_cache is not None:
        return _embed_fn_cache
    import torch

    # CI / schema-gen path: no model download, no GPU, deterministic
    # 384-d zero vector (matches all-MiniLM-L6-v2's output width so
    # bundles stay loadable if you swap back).
    if EMBED_MODEL == "stub":

        def stub_fn(_text: str) -> torch.Tensor:
            return torch.zeros(384)

        _embed_fn_cache = stub_fn
        return _embed_fn_cache

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)

    def fn(text: str) -> torch.Tensor:
        return torch.tensor(model.encode(text, convert_to_numpy=True))

    _embed_fn_cache = fn
    return _embed_fn_cache


# ------------------------------------------------------------------
# MemoryLayer cache (single-tenant default + per-bundle lookups)
# ------------------------------------------------------------------
def _path_for(name: str | None) -> Path:
    if name is None:
        return BUNDLE_PATH
    if not _BUNDLE_NAME_RE.match(name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid bundle name {name!r}; must match [A-Za-z0-9_.-]{{1,64}}",
        )
    return BUNDLES_DIR / name


def _get_mem(name: str | None = None) -> MemoryLayer:
    key = name or "__default__"
    with _cache_lock:
        if key in _mem_cache:
            mem = _mem_cache[key]
            mem.reload_if_stale()  # pick up peer-worker WAL appends
            return mem
        path = _path_for(name)
        # A loadable bundle is either a saved snapshot (memory_index.json)
        # or a WAL-only bundle (fresh store that crashed before save()).
        has_snapshot = path.exists() and (path / "memory_index.json").exists()
        has_wal = path.exists() and (path / "memory_ops.wal.jsonl").exists()
        if has_snapshot or has_wal:
            mem = MemoryLayer.load(path, embed_fn=_embed_fn())
        else:
            mem = MemoryLayer.with_sbert(EMBED_MODEL)
        _mem_cache[key] = mem
        return mem


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------
class StoreRequest(BaseModel):
    text: str = Field(
        ...,
        description="Raw text to embed and store as one memory entry.",
        examples=["alex lives in portland"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON tags for filtering at retrieval.",
        examples=[{"source": "chat", "user": "alex"}],
    )


class StoreResponse(BaseModel):
    node_id: str = Field(..., description="Opaque id of the new memory entry.")


class StoreBatchRequest(BaseModel):
    texts: list[str] = Field(..., description="Batch of texts to embed in one call.")
    metadatas: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional per-text metadata, aligned to `texts`.",
    )


class StoreBatchResponse(BaseModel):
    node_ids: list[str] = Field(..., description="Ids aligned to input `texts` order.")


class RetrieveRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural-language query; embedded then compared.",
        examples=["who lives in portland?"],
    )
    k: int = Field(
        5,
        description="Max number of hits to return (top-k by score).",
        examples=[5],
    )
    where: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filter; entries must match all keys.",
    )
    hybrid_alpha: float | None = Field(
        default=None,
        description="0..1 blend of dense vs sparse (BM25) score.",
    )
    rerank_top_n: int | None = Field(
        default=None,
        description="If set, cross-encoder rerank of this many candidates.",
    )


class HitResponse(BaseModel):
    node_id: str = Field(..., description="Opaque id of the matched memory.")
    text: str = Field(..., description="Stored text of the matched memory.")
    score: float = Field(
        ...,
        description="Relevance score; higher = closer match.",
        examples=[0.873],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata attached when the entry was stored.",
    )
    timestamp_step: int = Field(
        0, description="Monotonic step counter when stored (0 if unknown)."
    )


class RetrieveResponse(BaseModel):
    hits: list[HitResponse] = Field(..., description="Ordered matches, best first.")


class ForgetRequest(BaseModel):
    node_id: str = Field(..., description="Id of the memory entry to remove.")


class ForgetResponse(BaseModel):
    removed: bool = Field(..., description="True if the entry was deleted.")


class ConsolidateResponse(BaseModel):
    processed: int = Field(..., description="Entries processed this consolidation.")


class SaveResponse(BaseModel):
    saved_to: str = Field(..., description="Absolute bundle path written to disk.")


class StatusResponse(BaseModel):
    num_entries: int = Field(..., description="Total memories in this bundle.")
    bundle_path: str = Field(..., description="On-disk bundle directory.")
    embed_model: str = Field(..., description="Configured embedding model name.")


class HealthResponse(BaseModel):
    ok: bool = Field(..., description="Liveness flag; always true when reachable.")
    loaded_bundles: int = Field(
        ...,
        description="Count of bundles currently loaded into memory.",
        examples=[2],
    )


class VersionResponse(BaseModel):
    version: str = Field(..., description="Installed soma package version.")


def _hit(h: Any) -> HitResponse:
    return HitResponse(
        node_id=h.node_id,
        text=h.text,
        score=h.score,
        metadata=h.metadata,
        timestamp_step=h.timestamp_step,
    )


# ------------------------------------------------------------------
# Liveness / version (no auth)
# ------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    operation_id="health",
    tags=["system"],
)
def health() -> HealthResponse:
    return HealthResponse(ok=True, loaded_bundles=len(_mem_cache))


@app.get(
    "/version",
    response_model=VersionResponse,
    operation_id="version",
    tags=["system"],
)
def version_endpoint() -> VersionResponse:
    try:
        from importlib.metadata import version as _v

        return VersionResponse(version=_v("soma"))
    except Exception:
        return VersionResponse(version="unknown")


# ------------------------------------------------------------------
# Default-bundle endpoints (backward-compatible single-tenant API)
# ------------------------------------------------------------------
@app.get(
    "/status",
    response_model=StatusResponse,
    operation_id="status",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def status_ep() -> StatusResponse:
    mem = _get_mem()
    return StatusResponse(
        num_entries=len(mem),
        bundle_path=str(BUNDLE_PATH),
        embed_model=EMBED_MODEL,
    )


@app.post(
    "/store",
    response_model=StoreResponse,
    operation_id="store",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def store(req: StoreRequest) -> StoreResponse:
    mem = _get_mem()
    return StoreResponse(node_id=mem.store(req.text, metadata=req.metadata))


@app.post(
    "/store_batch",
    response_model=StoreBatchResponse,
    operation_id="store_batch",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def store_batch(req: StoreBatchRequest) -> StoreBatchResponse:
    mem = _get_mem()
    return StoreBatchResponse(
        node_ids=mem.store_batch(req.texts, metadatas=req.metadatas)
    )


@app.get(
    "/related/{node_id}",
    response_model=RetrieveResponse,
    operation_id="related",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def related(node_id: str, k: int = 5) -> RetrieveResponse:
    try:
        hits = _get_mem().related(node_id, k=max(1, k))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RetrieveResponse(hits=[_hit(h) for h in hits])


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    operation_id="retrieve",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    mem = _get_mem()
    kwargs: dict[str, Any] = {}
    if req.where is not None:
        kwargs["where"] = req.where
    if req.hybrid_alpha is not None:
        kwargs["hybrid_alpha"] = req.hybrid_alpha
    if req.rerank_top_n is not None:
        kwargs["rerank_top_n"] = req.rerank_top_n
    return RetrieveResponse(hits=[_hit(h) for h in mem.retrieve(req.query, k=req.k, **kwargs)])


@app.get(
    "/get/{node_id}",
    response_model=HitResponse,
    operation_id="get",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def get_entry(node_id: str) -> HitResponse:
    hit = _get_mem().get(node_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"node_id {node_id} not found")
    return _hit(hit)


@app.post(
    "/forget",
    response_model=ForgetResponse,
    operation_id="forget",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def forget(req: ForgetRequest) -> ForgetResponse:
    if not _get_mem().forget(req.node_id):
        raise HTTPException(status_code=404, detail=f"node_id {req.node_id} not found")
    return ForgetResponse(removed=True)


@app.post(
    "/consolidate",
    response_model=ConsolidateResponse,
    operation_id="consolidate",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def consolidate() -> ConsolidateResponse:
    return ConsolidateResponse(processed=_get_mem().consolidate())


@app.post(
    "/save",
    response_model=SaveResponse,
    operation_id="save",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def save() -> SaveResponse:
    mem = _get_mem()
    mem.save(BUNDLE_PATH)
    return SaveResponse(saved_to=str(BUNDLE_PATH))


@app.get(
    "/recent",
    response_model=RetrieveResponse,
    operation_id="recent",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def recent(n: int = 10) -> RetrieveResponse:
    hits = _get_mem().get_recent(max(1, n))
    return RetrieveResponse(hits=[_hit(h) for h in hits])


# ------------------------------------------------------------------
# Multi-tenant endpoints — /bundles/{name}/...
# ------------------------------------------------------------------
@app.get(
    "/bundles/{name}/status",
    response_model=StatusResponse,
    operation_id="bundles_status",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def status_bundle(name: str) -> StatusResponse:
    mem = _get_mem(name)
    return StatusResponse(
        num_entries=len(mem),
        bundle_path=str(_path_for(name)),
        embed_model=EMBED_MODEL,
    )


@app.post(
    "/bundles/{name}/store",
    response_model=StoreResponse,
    operation_id="bundles_store",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def store_bundle(name: str, req: StoreRequest) -> StoreResponse:
    mem = _get_mem(name)
    return StoreResponse(node_id=mem.store(req.text, metadata=req.metadata))


@app.post(
    "/bundles/{name}/store_batch",
    response_model=StoreBatchResponse,
    operation_id="bundles_store_batch",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def store_batch_bundle(name: str, req: StoreBatchRequest) -> StoreBatchResponse:
    mem = _get_mem(name)
    return StoreBatchResponse(
        node_ids=mem.store_batch(req.texts, metadatas=req.metadatas)
    )


@app.get(
    "/bundles/{name}/related/{node_id}",
    response_model=RetrieveResponse,
    operation_id="bundles_related",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def related_bundle(name: str, node_id: str, k: int = 5) -> RetrieveResponse:
    try:
        hits = _get_mem(name).related(node_id, k=max(1, k))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RetrieveResponse(hits=[_hit(h) for h in hits])


@app.post(
    "/bundles/{name}/retrieve",
    response_model=RetrieveResponse,
    operation_id="bundles_retrieve",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def retrieve_bundle(name: str, req: RetrieveRequest) -> RetrieveResponse:
    mem = _get_mem(name)
    kwargs: dict[str, Any] = {}
    if req.where is not None:
        kwargs["where"] = req.where
    if req.hybrid_alpha is not None:
        kwargs["hybrid_alpha"] = req.hybrid_alpha
    if req.rerank_top_n is not None:
        kwargs["rerank_top_n"] = req.rerank_top_n
    return RetrieveResponse(hits=[_hit(h) for h in mem.retrieve(req.query, k=req.k, **kwargs)])


@app.get(
    "/bundles/{name}/get/{node_id}",
    response_model=HitResponse,
    operation_id="bundles_get",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def get_bundle(name: str, node_id: str) -> HitResponse:
    hit = _get_mem(name).get(node_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"node_id {node_id} not found")
    return _hit(hit)


@app.post(
    "/bundles/{name}/forget",
    response_model=ForgetResponse,
    operation_id="bundles_forget",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def forget_bundle(name: str, req: ForgetRequest) -> ForgetResponse:
    if not _get_mem(name).forget(req.node_id):
        raise HTTPException(status_code=404, detail=f"node_id {req.node_id} not found")
    return ForgetResponse(removed=True)


@app.post(
    "/bundles/{name}/consolidate",
    response_model=ConsolidateResponse,
    operation_id="bundles_consolidate",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def consolidate_bundle(name: str) -> ConsolidateResponse:
    return ConsolidateResponse(processed=_get_mem(name).consolidate())


@app.post(
    "/bundles/{name}/save",
    response_model=SaveResponse,
    operation_id="bundles_save",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def save_bundle(name: str) -> SaveResponse:
    mem = _get_mem(name)
    path = _path_for(name)
    path.mkdir(parents=True, exist_ok=True)
    mem.save(path)
    return SaveResponse(saved_to=str(path))


@app.get(
    "/bundles/{name}/recent",
    response_model=RetrieveResponse,
    operation_id="bundles_recent",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
)
def recent_bundle(name: str, n: int = 10) -> RetrieveResponse:
    hits = _get_mem(name).get_recent(max(1, n))
    return RetrieveResponse(hits=[_hit(h) for h in hits])
