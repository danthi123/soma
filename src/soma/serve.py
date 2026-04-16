"""REST API for SOMA MemoryLayer.

Exposes store/retrieve/get/forget/consolidate/save over HTTP so
non-Python agents can use SOMA as their memory backend.

    uvicorn soma.serve:app --host 0.0.0.0 --port 8420

Or via Docker:

    docker compose up

Environment variables:
    SOMA_BUNDLE_PATH — directory for the memory bundle (default: ./data/memory)
    SOMA_EMBED_MODEL — sentence-transformers model name (default: all-MiniLM-L6-v2)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from soma.memory import MemoryLayer

BUNDLE_PATH = Path(os.environ.get("SOMA_BUNDLE_PATH", "./data/memory"))
EMBED_MODEL = os.environ.get("SOMA_EMBED_MODEL", "all-MiniLM-L6-v2")

app = FastAPI(
    title="SOMA Memory Layer",
    description="Local-first agent memory that learns.",
    version="0.1.0",
)

_mem: MemoryLayer | None = None


def _get_mem() -> MemoryLayer:
    global _mem  # noqa: PLW0603
    if _mem is None:
        if BUNDLE_PATH.exists() and (BUNDLE_PATH / "memory_index.json").exists():
            _mem = MemoryLayer.load(BUNDLE_PATH, embed_fn=_embed_fn())
        else:
            _mem = MemoryLayer.with_sbert(EMBED_MODEL)
    return _mem


def _embed_fn() -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)

    def fn(text: str) -> torch.Tensor:
        return torch.tensor(model.encode(text, convert_to_numpy=True))

    return fn


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


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    mem = _get_mem()
    return StatusResponse(
        num_entries=len(mem),
        bundle_path=str(BUNDLE_PATH),
        embed_model=EMBED_MODEL,
    )


@app.post("/store", response_model=StoreResponse)
def store(req: StoreRequest) -> StoreResponse:
    mem = _get_mem()
    nid = mem.store(req.text, metadata=req.metadata)
    return StoreResponse(node_id=nid)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    mem = _get_mem()
    hits = mem.retrieve(req.query, k=req.k)
    return RetrieveResponse(
        hits=[
            HitResponse(
                node_id=h.node_id,
                text=h.text,
                score=h.score,
                metadata=h.metadata,
                timestamp_step=h.timestamp_step,
            )
            for h in hits
        ]
    )


@app.get("/get/{node_id}", response_model=HitResponse)
def get_entry(node_id: str) -> HitResponse:
    mem = _get_mem()
    hit = mem.get(node_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"node_id {node_id} not found")
    return HitResponse(
        node_id=hit.node_id,
        text=hit.text,
        score=hit.score,
        metadata=hit.metadata,
        timestamp_step=hit.timestamp_step,
    )


@app.post("/forget")
def forget(req: ForgetRequest) -> dict[str, bool]:
    mem = _get_mem()
    removed = mem.forget(req.node_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"node_id {req.node_id} not found")
    return {"removed": True}


@app.post("/consolidate")
def consolidate() -> dict[str, int]:
    mem = _get_mem()
    processed = mem.consolidate()
    return {"processed": processed}


@app.post("/save")
def save() -> dict[str, str]:
    mem = _get_mem()
    mem.save(BUNDLE_PATH)
    return {"saved_to": str(BUNDLE_PATH)}


@app.get("/recent", response_model=RetrieveResponse)
def recent(n: int = 10) -> RetrieveResponse:
    mem = _get_mem()
    hits = mem.get_recent(max(1, n))
    return RetrieveResponse(
        hits=[
            HitResponse(
                node_id=h.node_id,
                text=h.text,
                score=h.score,
                metadata=h.metadata,
                timestamp_step=h.timestamp_step,
            )
            for h in hits
        ]
    )
