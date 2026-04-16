"""REST API for SOMA MemoryLayer.

Exposes store / retrieve / get / forget / consolidate / save over HTTP
so non-Python agents can use SOMA as their memory backend. Single-
tenant by default; optional multi-tenant routing under
``/bundles/{name}`` lets one server host many independent brains.

    uvicorn soma.serve:app --host 0.0.0.0 --port 8420
    # or:
    soma serve --port 8420

Environment variables:
    SOMA_BUNDLE_PATH  — single-tenant bundle directory
                        (default: ./data/memory)
    SOMA_BUNDLES_DIR  — root dir for multi-tenant bundles under
                        /bundles/{name}/... (default: ./data/bundles)
    SOMA_EMBED_MODEL  — sentence-transformers model name
                        (default: all-MiniLM-L6-v2)
    SOMA_API_KEY      — if set, all non-health endpoints require
                        ``Authorization: Bearer <key>``. Unset = open.

Endpoints:
    GET  /health                       — liveness probe (no auth)
    GET  /version                      — package version (no auth)
    GET  /status                       — default bundle stats
    POST /store | /retrieve | /forget  — default bundle ops
    GET  /get/{id} | /recent
    POST /consolidate | /save
    *    /bundles/{name}/<same as above>  — per-tenant variants
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from soma.memory import MemoryLayer

BUNDLE_PATH = Path(os.environ.get("SOMA_BUNDLE_PATH", "./data/memory"))
BUNDLES_DIR = Path(os.environ.get("SOMA_BUNDLES_DIR", "./data/bundles"))
EMBED_MODEL = os.environ.get("SOMA_EMBED_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.environ.get("SOMA_API_KEY", "")

app = FastAPI(
    title="SOMA Memory Layer",
    description="Local-first agent memory that learns.",
    version="0.1.0",
)

_BUNDLE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_mem_cache: dict[str, MemoryLayer] = {}
_cache_lock = threading.Lock()


# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------
def require_api_key(
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency: enforce ``SOMA_API_KEY`` when set.

    Unset env var = open server (matches prior behaviour). Set env var =
    require ``Authorization: Bearer <key>``.
    """
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
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
            return _mem_cache[key]
        path = _path_for(name)
        if path.exists() and (path / "memory_index.json").exists():
            mem = MemoryLayer.load(path, embed_fn=_embed_fn())
        else:
            mem = MemoryLayer.with_sbert(EMBED_MODEL)
        _mem_cache[key] = mem
        return mem


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------
class StoreRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoreResponse(BaseModel):
    node_id: str


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5


class HitResponse(BaseModel):
    node_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp_step: int = 0


class RetrieveResponse(BaseModel):
    hits: list[HitResponse]


class ForgetRequest(BaseModel):
    node_id: str


class StatusResponse(BaseModel):
    num_entries: int
    bundle_path: str
    embed_model: str


class HealthResponse(BaseModel):
    ok: bool
    loaded_bundles: int


class VersionResponse(BaseModel):
    version: str


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
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, loaded_bundles=len(_mem_cache))


@app.get("/version", response_model=VersionResponse)
def version_endpoint() -> VersionResponse:
    try:
        from importlib.metadata import version as _v

        return VersionResponse(version=_v("soma"))
    except Exception:
        return VersionResponse(version="unknown")


# ------------------------------------------------------------------
# Default-bundle endpoints (backward-compatible single-tenant API)
# ------------------------------------------------------------------
@app.get("/status", response_model=StatusResponse, dependencies=[Depends(require_api_key)])
def status_ep() -> StatusResponse:
    mem = _get_mem()
    return StatusResponse(
        num_entries=len(mem),
        bundle_path=str(BUNDLE_PATH),
        embed_model=EMBED_MODEL,
    )


@app.post("/store", response_model=StoreResponse, dependencies=[Depends(require_api_key)])
def store(req: StoreRequest) -> StoreResponse:
    mem = _get_mem()
    return StoreResponse(node_id=mem.store(req.text, metadata=req.metadata))


@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    mem = _get_mem()
    return RetrieveResponse(hits=[_hit(h) for h in mem.retrieve(req.query, k=req.k)])


@app.get("/get/{node_id}", response_model=HitResponse, dependencies=[Depends(require_api_key)])
def get_entry(node_id: str) -> HitResponse:
    hit = _get_mem().get(node_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"node_id {node_id} not found")
    return _hit(hit)


@app.post("/forget", dependencies=[Depends(require_api_key)])
def forget(req: ForgetRequest) -> dict[str, bool]:
    if not _get_mem().forget(req.node_id):
        raise HTTPException(status_code=404, detail=f"node_id {req.node_id} not found")
    return {"removed": True}


@app.post("/consolidate", dependencies=[Depends(require_api_key)])
def consolidate() -> dict[str, int]:
    return {"processed": _get_mem().consolidate()}


@app.post("/save", dependencies=[Depends(require_api_key)])
def save() -> dict[str, str]:
    mem = _get_mem()
    mem.save(BUNDLE_PATH)
    return {"saved_to": str(BUNDLE_PATH)}


@app.get("/recent", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
def recent(n: int = 10) -> RetrieveResponse:
    hits = _get_mem().get_recent(max(1, n))
    return RetrieveResponse(hits=[_hit(h) for h in hits])


# ------------------------------------------------------------------
# Multi-tenant endpoints — /bundles/{name}/...
# ------------------------------------------------------------------
@app.get(
    "/bundles/{name}/status",
    response_model=StatusResponse,
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
    dependencies=[Depends(require_api_key)],
)
def store_bundle(name: str, req: StoreRequest) -> StoreResponse:
    mem = _get_mem(name)
    return StoreResponse(node_id=mem.store(req.text, metadata=req.metadata))


@app.post(
    "/bundles/{name}/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(require_api_key)],
)
def retrieve_bundle(name: str, req: RetrieveRequest) -> RetrieveResponse:
    mem = _get_mem(name)
    return RetrieveResponse(hits=[_hit(h) for h in mem.retrieve(req.query, k=req.k)])


@app.get(
    "/bundles/{name}/get/{node_id}",
    response_model=HitResponse,
    dependencies=[Depends(require_api_key)],
)
def get_bundle(name: str, node_id: str) -> HitResponse:
    hit = _get_mem(name).get(node_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"node_id {node_id} not found")
    return _hit(hit)


@app.post(
    "/bundles/{name}/forget",
    dependencies=[Depends(require_api_key)],
)
def forget_bundle(name: str, req: ForgetRequest) -> dict[str, bool]:
    if not _get_mem(name).forget(req.node_id):
        raise HTTPException(status_code=404, detail=f"node_id {req.node_id} not found")
    return {"removed": True}


@app.post(
    "/bundles/{name}/consolidate",
    dependencies=[Depends(require_api_key)],
)
def consolidate_bundle(name: str) -> dict[str, int]:
    return {"processed": _get_mem(name).consolidate()}


@app.post(
    "/bundles/{name}/save",
    dependencies=[Depends(require_api_key)],
)
def save_bundle(name: str) -> dict[str, str]:
    mem = _get_mem(name)
    path = _path_for(name)
    path.mkdir(parents=True, exist_ok=True)
    mem.save(path)
    return {"saved_to": str(path)}


@app.get(
    "/bundles/{name}/recent",
    response_model=RetrieveResponse,
    dependencies=[Depends(require_api_key)],
)
def recent_bundle(name: str, n: int = 10) -> RetrieveResponse:
    hits = _get_mem(name).get_recent(max(1, n))
    return RetrieveResponse(hits=[_hit(h) for h in hits])
