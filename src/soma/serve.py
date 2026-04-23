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
    SOMA_API_KEY        — legacy admin escape hatch. When set, a raw
                          match against the bearer token grants full
                          access; response carries
                          ``X-SOMA-Deprecated: use JWT``.
    SOMA_JWT_SECRET     — HS256 shared secret. When set, every
                          protected endpoint verifies an
                          ``Authorization: Bearer <JWT>`` with
                          per-bundle perms (``read``/``write``/
                          ``admin``).
    SOMA_JWT_ALG        — ``HS256`` (default) or ``RS256``.
    SOMA_JWT_PUBLIC_KEY_PATH — RS256 PEM file for verification.
    SOMA_JWT_LEEWAY     — seconds of clock-skew tolerance (default 60).
    SOMA_JWT_BLOCKLIST_PATH — path to an append-only JSONL file holding
                          revoked ``jti`` values. When set, every verified
                          token is additionally checked against the
                          blocklist (poll cadence ~30s). Unset = no
                          revocation (pre-Phase-4.1 behaviour).
    SOMA_JWT_BLOCKLIST_HASHED — ``1`` to write sha256(jti) at rest instead
                          of plaintext (Phase 18). Default = off.
    SOMA_JWT_AUDIENCE   — pin the verifier to a specific ``aud`` claim
                          (Phase 18). When set, tokens without a matching
                          ``aud`` are rejected. Unset = no audience check.
    SOMA_JWT_PRIVATE_KEY_PATH — RS256 PEM file for signing. Required on
                          the refresh path only (Phase 23) — plain
                          verification still works with just the public
                          key. HS256 deployments reuse SOMA_JWT_SECRET.
    SOMA_JWT_REFRESH_TTL — Override the default refresh window. Accepts
                          ``30d | 24h | 60m`` (Phase 23). Unset = reuse
                          the original token's exp-iat window.
    SOMA_JWT_MAX_TTL    — Optional cap on any refreshed token's TTL
                          (Phase 23, same format). Unset = no cap.
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

import json
import os
import re
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from soma import __version__
from soma import metrics as _metrics
from soma.auth import Perm, Principal, parse_ttl_spec, refresh_token, verify_token
from soma.auth_revocation import blocklist_from_env
from soma.forget_audit import ForgetAuditSink
from soma.log import configure_json_logging
from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory
from soma.rate_limit import RATE_LIMITED_TOTAL, RateLimiter

# Structured JSON logging swap (no-op unless SOMA_LOG_JSON=1). Called at
# import time so `uvicorn soma.serve:app` picks up the formatter before
# the first request flows through.
configure_json_logging()

BUNDLE_PATH = Path(os.environ.get("SOMA_BUNDLE_PATH", "./data/memory"))
BUNDLES_DIR = Path(os.environ.get("SOMA_BUNDLES_DIR", "./data/bundles"))
EMBED_MODEL = os.environ.get("SOMA_EMBED_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.environ.get("SOMA_API_KEY", "")

# Phase 4 — JWT auth config. When SOMA_JWT_SECRET (HS256) or
# SOMA_JWT_PUBLIC_KEY_PATH (RS256) is set, every protected route
# requires a valid JWT whose claims carry per-bundle permissions.
# SOMA_API_KEY still works in parallel as a deprecated admin
# escape hatch — responses from that path carry ``X-SOMA-Deprecated:
# use JWT`` so callers notice.
JWT_ALG = os.environ.get("SOMA_JWT_ALG", "HS256")
JWT_SECRET = os.environ.get("SOMA_JWT_SECRET", "")
JWT_PUBLIC_KEY_PATH = os.environ.get("SOMA_JWT_PUBLIC_KEY_PATH", "")
try:
    JWT_LEEWAY = int(os.environ.get("SOMA_JWT_LEEWAY", "60"))
except ValueError:
    JWT_LEEWAY = 60
# Optional: pin the verifier to a specific `aud` claim (Phase 18). When
# set, every token must carry a matching aud or it is rejected with the
# same 401 path as any other InvalidTokenError. Unset (default) keeps
# pre-Phase-18 behaviour — the aud claim is not checked at all.
JWT_AUDIENCE = os.environ.get("SOMA_JWT_AUDIENCE", "").strip() or None
# Resolved once at import time — rotating the key file requires a
# server restart, which is the intended operator story.
_JWT_PUBLIC_KEY_PEM: bytes | None = None
if JWT_PUBLIC_KEY_PATH:
    try:
        _JWT_PUBLIC_KEY_PEM = Path(JWT_PUBLIC_KEY_PATH).read_bytes()
    except OSError:
        _JWT_PUBLIC_KEY_PEM = None

# Phase 23 — private key for the refresh path. Only the /auth/refresh
# endpoint needs signing material; every other route is verification-
# only and can run with just the public key. Read at import so an
# operator rotating the private key file knows they need to bounce
# the server.
JWT_PRIVATE_KEY_PATH = os.environ.get("SOMA_JWT_PRIVATE_KEY_PATH", "")
_JWT_PRIVATE_KEY_PEM: bytes | None = None
if JWT_PRIVATE_KEY_PATH:
    try:
        _JWT_PRIVATE_KEY_PEM = Path(JWT_PRIVATE_KEY_PATH).read_bytes()
    except OSError:
        _JWT_PRIVATE_KEY_PEM = None


def _parse_ttl_env(var: str) -> timedelta | None:
    """Read ``var`` from the process env and parse as a TTL spec.

    Returns ``None`` when the env var is unset or empty; returns
    ``None`` (and logs a warning) when the value is malformed, so the
    caller falls back to its documented default instead of crashing at
    request time.
    """
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    try:
        return parse_ttl_spec(raw)
    except ValueError:
        import warnings

        warnings.warn(
            f"{var}={raw!r} is malformed; ignoring (expected 30d | 24h | 60m)",
            stacklevel=2,
        )
        return None


# JWT revocation blocklist. Gated on ``SOMA_JWT_BLOCKLIST_PATH``; when
# unset, ``blocklist_from_env`` returns a no-op ``null_blocklist`` so
# verify_token's revoked-jti check is transparent. The blocklist is
# resolved once at import time so peer processes can share the file
# via portalocker + mtime poll (~30 s propagation).
_blocklist = blocklist_from_env()

# Phase 37 — append-only forget-event audit sink. Gated on
# ``SOMA_FORGET_AUDIT_PATH``; unset = no-op. Resolved once at import
# time so every request handler can reach the same sink without
# reparsing env vars. The sink is thread-safe (per-instance lock
# around open+write+flush) so FastAPI's thread-pool workers can share.
_forget_audit: ForgetAuditSink = ForgetAuditSink.from_env()

# Phase 26 — in-proc per-token rate limiter. Opt-in via
# SOMA_RATE_LIMIT_RPS; when unset, ``RateLimiter.from_env()`` returns
# ``None`` and every request flows through the limiter check as a no-op.
# Keyed on JWT ``jti`` by default (``per-token``); flip to
# ``per-subject`` to share a bucket across token refreshes for the same
# ``sub``. Exempt: ``/metrics`` (Prometheus scrape) and ``/health``
# (liveness probe) — a noisy tenant shouldn't throttle monitoring.
_RATE_LIMITER: RateLimiter | None = RateLimiter.from_env()
_RATE_LIMIT_SCOPE: str = os.environ.get("SOMA_RATE_LIMIT_SCOPE", "per-token").strip() or "per-token"
_RATE_LIMIT_EXEMPT_PATHS: frozenset[str] = frozenset({"/metrics", "/health"})


def _enforce_rate_limit(request: Request, principal: Principal) -> None:
    """Consume one token from the principal's bucket, or raise 429.

    Called from inside :func:`require_auth` after verification succeeds
    so the limiter always sees a valid ``Principal``. Exempts
    ``/metrics`` and ``/health`` so monitoring + k8s probes never get
    throttled. No-op when ``_RATE_LIMITER`` is ``None`` (the default).
    """
    if _RATE_LIMITER is None:
        return
    if request.url.path in _RATE_LIMIT_EXEMPT_PATHS:
        return
    # per-token scoping uses the jti so refreshes get a fresh budget;
    # per-subject shares across refreshes. Fall back to sub when jti is
    # missing (pre-Phase-18 tokens or the legacy-api-key principal).
    key = (
        principal.sub
        if _RATE_LIMIT_SCOPE == "per-subject"
        else (principal.jti or principal.sub)
    )
    allowed, retry_after = _RATE_LIMITER.check(key)
    if allowed:
        return
    RATE_LIMITED_TOTAL.labels(scope=_RATE_LIMIT_SCOPE).inc()
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="rate limit exceeded",
        headers={"Retry-After": f"{max(retry_after, 0.0):.2f}"},
    )

app = FastAPI(
    title="SOMA Memory Layer",
    description="Local-first agent memory that learns.",
    version=__version__,
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
#
# ``SOMA_METRICS_PUBLIC`` (default "1") keeps /metrics unauthenticated
# so existing Prometheus scrape configs keep working. Sensitive deploys
# opt in to auth with ``SOMA_METRICS_PUBLIC=0`` — that wraps the endpoint
# with the standard ``require_auth(None, "read")`` dependency so a bearer
# with any read perm (or the legacy SOMA_API_KEY) can still scrape.
# --------------------------------------------------------------------
METRICS_PUBLIC = os.environ.get("SOMA_METRICS_PUBLIC", "1") != "0"

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
# Auth — HTTPBearer security scheme so OpenAPI advertises bearer JWT.
# Three modes:
#   1. Open: neither SOMA_JWT_SECRET nor SOMA_JWT_PUBLIC_KEY_PATH nor
#      SOMA_API_KEY is set -> every route is open (matches prior
#      behaviour for frictionless local dev).
#   2. JWT: SOMA_JWT_SECRET (HS256) or SOMA_JWT_PUBLIC_KEY_PATH (RS256)
#      is set -> bearer token is decoded, per-bundle perms enforced.
#   3. Legacy: SOMA_API_KEY is set -> raw match against the bearer
#      token acts as an admin escape hatch. Responses carry
#      ``X-SOMA-Deprecated: use JWT`` so callers migrate.
# When both JWT and legacy envs are set, the JWT path wins for
# valid tokens; invalid JWTs fall through to the legacy check.
# ------------------------------------------------------------------
_bearer_scheme = HTTPBearer(bearerFormat="JWT", scheme_name="bearerAuth", auto_error=False)

# Principal instance used to represent a successful legacy SOMA_API_KEY
# match. Admin everywhere by convention — that's the backward-compat
# contract Phase 4 inherits. A distinct ``jti`` marker makes it easy to
# grep access logs for legacy hits.
_LEGACY_PRINCIPAL = Principal(sub="legacy-api-key", bundles={}, jti="legacy")


def record_auth_failure(reason: str) -> None:
    """Increment the ``soma_auth_failures_total{reason}`` counter.

    Reasons used: ``missing_credentials``, ``invalid_token``,
    ``expired_token``, ``insufficient_perm``. Kept as a thin helper so
    every 401/403 path funnels through one place — easier to audit and
    easier to keep the label cardinality bounded.
    """
    _metrics.AUTH_FAILURES_TOTAL.labels(reason=reason).inc()


def _auth_disabled() -> bool:
    """True when no auth env vars are set — open-mode dev server."""
    return not API_KEY and not JWT_SECRET and _JWT_PUBLIC_KEY_PEM is None


def _try_verify_jwt(token: str) -> Principal | None:
    """Decode the token; return the principal or ``None`` if no JWT
    verifier is configured.

    Raises :class:`jwt.InvalidTokenError` for any token-shaped failure
    (bad signature, expired, missing exp, alg=none, revoked jti).
    The module-level ``_blocklist`` is passed through so the revocation
    check fires on every verify path.
    """
    if JWT_ALG == "HS256" and JWT_SECRET:
        return verify_token(
            token,
            alg="HS256",
            secret=JWT_SECRET,
            leeway=JWT_LEEWAY,
            blocklist=_blocklist,
            expected_audience=JWT_AUDIENCE,
        )
    if JWT_ALG == "RS256" and _JWT_PUBLIC_KEY_PEM is not None:
        return verify_token(
            token,
            alg="RS256",
            public_key_pem=_JWT_PUBLIC_KEY_PEM,
            leeway=JWT_LEEWAY,
            blocklist=_blocklist,
            expected_audience=JWT_AUDIENCE,
        )
    return None


def require_auth(
    bundle_path_param: str | None, perm: Perm
) -> Any:
    """Build a FastAPI dependency that enforces the given perm.

    ``bundle_path_param`` names the path parameter that carries the
    bundle name (e.g., ``"name"`` for ``/bundles/{name}/store``). When
    ``None``, the route is not tenant-scoped and ``has_perm(None, perm)``
    applies — any claim that meets the bar satisfies.
    """

    async def dep(
        request: Request,
        response: Response,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
    ) -> Principal | None:
        # Path 1: open mode — no auth configured.
        if _auth_disabled():
            return None

        # Path 2: JWT, when a secret / public key is configured.
        if credentials and credentials.scheme.lower() == "bearer":
            try:
                principal = _try_verify_jwt(credentials.credentials)
            except jwt.ExpiredSignatureError:
                record_auth_failure("expired_token")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="token expired",
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                ) from None
            except jwt.InvalidTokenError as exc:
                # Revocation is a strictly different failure mode than
                # "malformed" — the token was valid, we just pulled its
                # plug. Detect via the message verify_token raises; no
                # fall-through to legacy for revoked tokens.
                if "revoked" in str(exc).lower():
                    record_auth_failure("revoked_token")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="token revoked",
                        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                    ) from None
                # Fall through to legacy check — a malformed JWT might
                # just be a legacy raw key.
                principal = None
            if principal is not None:
                bundle = (
                    request.path_params.get(bundle_path_param)
                    if bundle_path_param
                    else None
                )
                if not principal.has_perm(bundle, perm):
                    record_auth_failure("insufficient_perm")
                    detail = (
                        f"perm {perm!r} required for bundle {bundle!r}"
                        if bundle
                        else f"perm {perm!r} required"
                    )
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
                _enforce_rate_limit(request, principal)
                return principal

        # Path 3: legacy SOMA_API_KEY admin escape hatch.
        if (
            API_KEY
            and credentials is not None
            and credentials.scheme.lower() == "bearer"
            and credentials.credentials == API_KEY
        ):
            response.headers["X-SOMA-Deprecated"] = "use JWT"
            _enforce_rate_limit(request, _LEGACY_PRINCIPAL)
            return _LEGACY_PRINCIPAL

        # Anything else is a clean 401.
        reason = "missing_credentials" if credentials is None else "invalid_token"
        record_auth_failure(reason)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return dep


# Back-compat alias so any external pin (e.g., in a downstream app that
# imported `soma.serve.require_api_key` as a dependency) keeps working.
# The new code path does NOT use this — every @app route below binds
# ``require_auth(...)`` directly.
require_api_key = require_auth(None, "admin")


# --------------------------------------------------------------------
# Instrumentator wiring (deferred until after require_auth is defined
# so the SOMA_METRICS_PUBLIC=0 gate can pass ``dependencies=[...]``).
# prometheus-fastapi-instrumentator forwards **kwargs from .expose() to
# FastAPI's route registration, so the ``dependencies`` list attaches
# our bearer-check exactly like any other route.
# --------------------------------------------------------------------
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    _metrics_deps: list[Any] = (
        [] if METRICS_PUBLIC else [Depends(require_auth(None, "read"))]
    )
    Instrumentator().instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=True,
        tags=["system"],
        dependencies=_metrics_deps,
    )
except ImportError:  # pragma: no cover
    # soma[metrics] not installed — /metrics will 404.
    pass


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
        has_snapshot = path.exists() and (path / "memory_index.json").exists()
        has_wal = path.exists() and (path / "memory_ops.wal.jsonl").exists()
        if has_snapshot or has_wal:
            # Prefer the bundle's persisted sbert_model_name over the server's
            # current SOMA_EMBED_MODEL. Mismatch (e.g. bundle saved with
            # mpnet-base-v2, server booted with MiniLM) produces silent
            # retrieval garbage or a dim-mismatch crash. `load_with_sbert`
            # reads `memory_index.json` to rebuild the right embedder. Only
            # fall back to the plain env-driven path for bundles created by
            # the TextEncoder path (no sbert name persisted).
            persisted = _persisted_sbert_name(path) if has_snapshot else None
            if persisted:
                mem = MemoryLayer.load_with_sbert(path)
            else:
                mem = MemoryLayer.load(path, embed_fn=_embed_fn())
        else:
            mem = MemoryLayer.with_sbert(EMBED_MODEL)
        _mem_cache[key] = mem
        return mem


def _persisted_sbert_name(path: Path) -> str | None:
    """Peek at memory_index.json to see if the bundle recorded an sbert model
    name. Returns None if the file is missing, malformed, or lacks the key."""
    idx_path = path / "memory_index.json"
    if not idx_path.exists():
        return None
    try:
        idx = json.loads(idx_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    name = idx.get("sbert_model_name")
    return name if isinstance(name, str) and name else None


def _get_conversational_memory(
    name: str | None = None,
) -> ConversationalMemory | None:
    """Return the server's :class:`ConversationalMemory` for ``name``, if any.

    Phase 37 default: returns ``None``. SOMA's REST surface historically
    exposes only the raw :class:`MemoryLayer` endpoints; conversational
    ingest is done Python-side and the bundle is re-mounted here for
    retrieval. Until a future phase wires a fully-managed conversational
    server mode, the criteria branch of ``POST /forget`` returns 501
    when this helper yields ``None``.

    Tests inject a live CM by monkey-patching this symbol so the
    ``/forget`` route can exercise the full cascade without the server
    owning construction.
    """
    return None


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
    """Unified request body for ``POST /forget``.

    Two shapes, dispatched at the handler:

    - **By node id** — ``{"node_id": "<id>"}`` removes exactly that
      entry via :meth:`MemoryLayer.forget`. Always live on the REST
      surface.
    - **By criteria** — any of ``text_matches``, ``subject``, or
      ``user_id`` routes through :meth:`ConversationalMemory.forget`,
      returning the richer :class:`ForgetResult` / :class:`ForgetPreview`
      shape. Requires a ``ConversationalMemory`` to be wired via
      :func:`_get_conversational_memory` (returns 501 otherwise).
      Mixing ``node_id`` with criteria is not supported — the handler
      picks the node-id branch when ``node_id`` is present.

    ``summary_strategy`` controls the summary cascade on criteria-based
    forgetting: ``"drop"`` deletes every summary in the preview set
    even when survivors exist (skipping the regeneration LLM call
    entirely); the default ``"regen"`` rewrites affected summaries.
    """

    node_id: str | None = Field(
        default=None,
        description=(
            "Legacy: id of a single memory entry to remove. Mutually "
            "exclusive with the criteria fields below."
        ),
    )
    text_matches: str | None = Field(
        default=None,
        description=(
            "Substring to match against raw turn text. Case-insensitive by "
            "default; pass ``case_sensitive=true`` for an exact match."
        ),
    )
    subject: str | None = Field(
        default=None,
        description=(
            "Equality match on each fact's ``metadata.subject``. Only hits "
            "hand-stamped facts until the automatic extractor populates it."
        ),
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "Equality match on each entry's ``metadata.user_id`` "
            "(multi-user scope). When the caller's JWT sub differs, the "
            "audit record carries both."
        ),
    )
    case_sensitive: bool = Field(
        default=False,
        description="When True, ``text_matches`` becomes case-sensitive.",
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "When True, return a ForgetPreview instead of deleting. "
            "Audit-logs either way."
        ),
    )
    summary_strategy: str | None = Field(
        default=None,
        description=(
            "``'regen'`` (default) rewrites partially-covered summaries "
            "from surviving turns; ``'drop'`` deletes every matched "
            "summary without calling the LLM."
        ),
    )


class ForgetResponse(BaseModel):
    removed: bool = Field(..., description="True if the entry was deleted.")


class ConsolidateResponse(BaseModel):
    processed: int = Field(..., description="Entries processed this consolidation.")


class SaveResponse(BaseModel):
    saved_to: str = Field(..., description="Absolute bundle path written to disk.")


class SnapshotRequest(BaseModel):
    path: str = Field(
        ...,
        description=(
            "Filesystem path (relative to the server cwd, or absolute under "
            "cwd) where the bundle should be written. Paths that escape the "
            "server's cwd are rejected with a 400."
        ),
        examples=["data/snapshots/session-42"],
    )


class SnapshotResponse(BaseModel):
    saved: bool = Field(..., description="Always True when the HTTP status is 200.")
    path: str = Field(..., description="Absolute path the bundle was written to.")
    entries: int = Field(..., description="Number of memory entries in the saved bundle.")


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


def _resolve_snapshot_path(raw: str) -> Path:
    """Resolve ``raw`` to an absolute path guaranteed to sit under cwd.

    Safety belt for ``POST /snapshot``: the client supplies the target
    path, and we refuse absolute paths or ``..`` traversal that would
    let the client write outside the server's working directory. This
    does NOT replace OS-level filesystem permissions — it's the
    in-process equivalent of "no surprise writes" for the common
    operator story where the server runs inside a data directory.
    """
    if not raw or not raw.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path must be a non-empty string",
        )
    cwd = Path.cwd().resolve()
    candidate = Path(raw)
    # ``resolve(strict=False)`` follows ``..`` segments and symlinks on
    # the prefix that exists, which is exactly the containment check we
    # want. An absolute path that already resolves inside cwd is fine
    # (``/tmp/mount/cwd/snap`` works if cwd is under /tmp/mount/cwd).
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (cwd / candidate).resolve()
    )
    try:
        resolved.relative_to(cwd)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"path {raw!r} escapes the server cwd; only paths under "
                f"{cwd} are allowed"
            ),
        ) from exc
    return resolved


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
    return VersionResponse(version=__version__)


# ------------------------------------------------------------------
# Phase 23 — POST /auth/refresh
# ------------------------------------------------------------------
class RefreshResponse(BaseModel):
    """Response body for ``POST /auth/refresh``."""

    token: str = Field(..., description="Freshly-minted JWT with a new jti and exp.")
    exp: int = Field(..., description="Epoch-seconds expiry of the new token.")


@app.post(
    "/auth/refresh",
    response_model=RefreshResponse,
    operation_id="auth_refresh",
    tags=["auth"],
    responses=ERROR_RESPONSES,
)
def auth_refresh(request: Request) -> RefreshResponse:
    """Exchange a valid bearer for a fresh token with the same claims.

    The bearer in ``Authorization:`` is verified end-to-end (signature,
    exp, revocation, audience) and then re-minted via
    :func:`soma.auth.refresh_token`. The new token carries the same
    ``sub`` / ``bundles`` / ``aud`` as the original but a fresh ``jti``
    and a fresh ``exp``.

    Failure modes all return 401 (the bearer is itself the credential,
    and revoking a token is the only way to block refresh loops):

    - no/mismatched ``Authorization`` header
    - expired, revoked, or malformed token
    - server misconfigured (no signing key available)

    The old token is NOT auto-revoked. Operators wanting rotation call
    ``POST /auth/revoke`` (or ``soma auth revoke``) explicitly before
    or after refresh — intentional: revocation is orthogonal policy.
    """
    # Step 1: pull the raw bearer off the header ourselves — we can't
    # use the FastAPI dependency because Principal drops the raw token
    # string, which the refresh helper needs.
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        record_auth_failure("missing_credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    raw_token = auth_header[len("Bearer ") :].strip()
    if not raw_token:
        record_auth_failure("missing_credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="empty bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 2: resolve signing / verification material. HS256 uses the
    # shared secret for both; RS256 needs the private key on-hand to
    # sign the new token. A verification-only RS256 deployment (public
    # key but no private key) can't refresh — surface that as a 401
    # with a hint.
    if JWT_ALG == "HS256":
        if not JWT_SECRET:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="refresh unavailable: SOMA_JWT_SECRET not configured",
                headers={"WWW-Authenticate": "Bearer"},
            )
        helper_kwargs: dict[str, Any] = {"alg": "HS256", "secret": JWT_SECRET}
    elif JWT_ALG == "RS256":
        if _JWT_PUBLIC_KEY_PEM is None or _JWT_PRIVATE_KEY_PEM is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "refresh unavailable: RS256 needs both "
                    "SOMA_JWT_PUBLIC_KEY_PATH and SOMA_JWT_PRIVATE_KEY_PATH"
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )
        helper_kwargs = {
            "alg": "RS256",
            "public_key_pem": _JWT_PUBLIC_KEY_PEM,
            "private_key_pem": _JWT_PRIVATE_KEY_PEM,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"refresh unavailable: unsupported SOMA_JWT_ALG={JWT_ALG!r}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 3: mint. Env-driven TTL knobs are resolved fresh per call so
    # operators can tune them without restarting (tests monkeypatch
    # them inline for the same reason).
    try:
        new_token = refresh_token(
            raw_token,
            leeway=JWT_LEEWAY,
            blocklist=_blocklist,
            expected_audience=JWT_AUDIENCE,
            new_expires_in=_parse_ttl_env("SOMA_JWT_REFRESH_TTL"),
            max_expires_in=_parse_ttl_env("SOMA_JWT_MAX_TTL"),
            **helper_kwargs,
        )
    except jwt.ExpiredSignatureError:
        record_auth_failure("expired_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token expired",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from None
    except jwt.InvalidTokenError as exc:
        reason = "revoked_token" if "revoked" in str(exc).lower() else "invalid_token"
        record_auth_failure(reason)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"refresh failed: {exc}",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from None

    # Pull exp off the freshly-minted token for the response body.
    new_claims = jwt.decode(new_token, options={"verify_signature": False})
    return RefreshResponse(token=new_token, exp=int(new_claims.get("exp", 0)))


# ------------------------------------------------------------------
# POST /auth/revoke — HTTP surface over the BlocklistBackend primitive
# ------------------------------------------------------------------
class AuthRevokeRequest(BaseModel):
    """Body for ``POST /auth/revoke``. Supply exactly one of ``token``
    or ``jti``+``exp``.

    - ``token``: revoke by presenting the full JWT. Server decodes and
      pulls ``jti``/``exp``. Callers don't need to parse.
    - ``jti``+``exp``: revoke by identifier. Useful when the original
      token has already been rotated out of the caller's possession
      (e.g., the jti was lifted from an access log).
    """

    token: str | None = Field(None, description="Full JWT to revoke")
    jti: str | None = Field(
        None, description="Token id (if revoking without the token)"
    )
    exp: int | None = Field(
        None, description="Token expiry unix timestamp — required with jti"
    )
    reason: str | None = Field(None, description="Free-form audit note")


class AuthRevokeResponse(BaseModel):
    """Response body for ``POST /auth/revoke``."""

    revoked: str = Field(..., description="The jti that was added to the blocklist")
    exp: int = Field(..., description="Blocklist entry TTL anchor (unix ts)")
    reason: str = Field(..., description="Echoed reason or a generated default")


@app.post(
    "/auth/revoke",
    response_model=AuthRevokeResponse,
    operation_id="auth_revoke",
    tags=["auth"],
    responses=ERROR_RESPONSES,
    summary="Revoke a JWT (add its jti to the blocklist)",
    dependencies=[Depends(require_auth(None, "admin"))],
)
def auth_revoke(body: AuthRevokeRequest) -> AuthRevokeResponse:
    """Revoke a token. Either pass the full ``token`` to revoke it by
    content, or pass ``jti``+``exp`` to revoke by identifier.

    The blocklist entry's TTL is set to the token's ``exp`` so expired
    entries self-evict — no manual GC required for the common case.
    Same primitive as ``soma auth revoke`` (the CLI flow) — the two
    surfaces write identical ``RevocationRecord``s into
    ``SOMA_JWT_BLOCKLIST_PATH`` (or Redis).

    Requires ``admin`` scope on any bundle in the caller's claim; 401
    without a valid bearer, 403 with a valid non-admin bearer, 400
    when neither ``token`` nor ``jti``+``exp`` is supplied.
    """
    import time as _time

    from soma.auth_revocation import RevocationRecord, _NullBlocklist

    # If no blocklist is configured (SOMA_JWT_BLOCKLIST_PATH unset and no
    # Redis), a revoke call is a silent no-op — the record is accepted,
    # thrown on the floor, and every subsequent verify_token skips the
    # revocation gate anyway. Fail loudly so operators can't mistake
    # "route returned 200" for "token is actually revoked". Matches the
    # CLI flow at soma.cli._cmd_auth_revoke which refuses to run when
    # SOMA_JWT_BLOCKLIST_PATH is unset.
    #
    # 503 Service Unavailable (not 501 Not Implemented): the feature IS
    # implemented — it's just not wired in this deployment. Operators fix
    # by setting the env var and restarting; the service is then available.
    if isinstance(_blocklist, _NullBlocklist):
        raise HTTPException(
            status_code=503,
            detail=(
                "blocklist not configured; set SOMA_JWT_BLOCKLIST_PATH or "
                "SOMA_JWT_BLOCKLIST_REDIS_URL and restart. See docs/auth.md."
            ),
        )

    if body.token is not None:
        try:
            unsafe = jwt.decode(
                body.token,
                options={"verify_signature": False, "verify_exp": False},
            )
        except jwt.InvalidTokenError as err:
            raise HTTPException(
                status_code=400, detail=f"invalid token: {err}"
            ) from err
        jti = unsafe.get("jti")
        exp = unsafe.get("exp")
        if not isinstance(jti, str):
            raise HTTPException(
                status_code=400,
                detail="token missing jti claim; nothing to revoke",
            )
        if not isinstance(exp, int):
            raise HTTPException(
                status_code=400,
                detail="token missing exp claim; cannot compute TTL",
            )
    elif body.jti is not None and body.exp is not None:
        jti = body.jti
        exp = int(body.exp)
    else:
        raise HTTPException(
            status_code=400,
            detail="pass exactly one of (token) or (jti+exp)",
        )

    reason = body.reason or "revoked via POST /auth/revoke"
    _blocklist.add(
        RevocationRecord(
            jti=jti,
            revoked_at=int(_time.time()),
            reason=reason,
            exp=exp,
        )
    )
    return AuthRevokeResponse(revoked=jti, exp=exp, reason=reason)


# ------------------------------------------------------------------
# Default-bundle endpoints (backward-compatible single-tenant API)
# ------------------------------------------------------------------
@app.get(
    "/status",
    response_model=StatusResponse,
    operation_id="status",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth(None, "read"))],
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
    dependencies=[Depends(require_auth(None, "write"))],
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
    dependencies=[Depends(require_auth(None, "write"))],
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
    dependencies=[Depends(require_auth(None, "read"))],
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
    dependencies=[Depends(require_auth(None, "read"))],
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
    dependencies=[Depends(require_auth(None, "read"))],
)
def get_entry(node_id: str) -> HitResponse:
    hit = _get_mem().get(node_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"node_id {node_id} not found")
    return _hit(hit)


@app.post(
    "/forget",
    operation_id="forget",
    tags=["default"],
    responses=ERROR_RESPONSES,
)
def forget(
    req: ForgetRequest,
    principal: Principal | None = Depends(require_auth(None, "write")),  # noqa: B008
) -> dict[str, Any]:
    """Delete memory entries.

    Two dispatch branches:

    1. **Legacy node_id**: ``{"node_id": "<id>"}`` removes exactly one
       entry via :meth:`MemoryLayer.forget`. Returns ``{"removed":
       true}`` on success, 404 otherwise. Pre-Phase-37 contract.
    2. **Conversational criteria**: any of ``text_matches`` / ``subject``
       / ``user_id`` routes through the Conversational cascade
       (turns + derived facts + summary regenerate-or-drop) and
       returns the :class:`ForgetResult` / :class:`ForgetPreview`
       shape. Requires a ConversationalMemory to be wired through
       :func:`_get_conversational_memory`; returns 501 otherwise.

    Every invocation (both branches) emits one audit record when
    ``SOMA_FORGET_AUDIT_PATH`` is set. The caller's ``principal.sub``
    is stamped on the record; when the request body's ``user_id``
    differs, the audit line carries both.
    """
    actor = principal.sub if principal is not None else "anonymous"

    # Branch 1: legacy node_id path. Keeps the Phase 4 contract byte-
    # identical for existing clients.
    if req.node_id is not None:
        if not _get_mem().forget(req.node_id):
            raise HTTPException(
                status_code=404, detail=f"node_id {req.node_id} not found"
            )
        return {"removed": True}

    # Branch 2: conversational criteria path. Validate at least one
    # criterion so callers can't wipe a bundle by omission.
    if req.text_matches is None and req.subject is None and req.user_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "forget() requires at least one criterion "
                "(node_id=, text_matches=, subject=, or user_id=)"
            ),
        )

    cm = _get_conversational_memory()
    if cm is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Conversational forget is not configured on this server. "
                "Wire a ConversationalMemory via _get_conversational_memory "
                "or POST {\"node_id\": ...} for single-entry deletion."
            ),
        )

    # summary_strategy is opt-in; forward only when the caller set it
    # so the default stays "regen" and pre-Task-3 behaviour survives.
    kwargs: dict[str, Any] = {
        "text_matches": req.text_matches,
        "subject": req.subject,
        "user_id": req.user_id,
        "case_sensitive": req.case_sensitive,
        "dry_run": req.dry_run,
        "actor": actor,
    }
    if req.summary_strategy is not None:
        if req.summary_strategy not in ("regen", "drop"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"summary_strategy must be 'regen' or 'drop', got "
                    f"{req.summary_strategy!r}"
                ),
            )
        kwargs["summary_strategy"] = req.summary_strategy

    try:
        outcome = cm.forget(**kwargs)
    except ValueError as exc:
        # Matches the library's zero-criteria guard; shouldn't fire
        # here because we pre-validated, but forward just in case a
        # future library check fires.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Convert the dataclass to a dict for the JSON response. Pydantic
    # would also work but the dataclasses are lighter and the shape is
    # already documented on :class:`ForgetResult` / :class:`ForgetPreview`.
    return {
        k: v
        for k, v in outcome.__dict__.items()
        if not k.startswith("_")
    }


@app.post(
    "/consolidate",
    response_model=ConsolidateResponse,
    operation_id="consolidate",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth(None, "write"))],
)
def consolidate() -> ConsolidateResponse:
    return ConsolidateResponse(processed=_get_mem().consolidate())


@app.post(
    "/save",
    response_model=SaveResponse,
    operation_id="save",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth(None, "write"))],
)
def save() -> SaveResponse:
    mem = _get_mem()
    mem.save(BUNDLE_PATH)
    return SaveResponse(saved_to=str(BUNDLE_PATH))


@app.post(
    "/snapshot",
    response_model=SnapshotResponse,
    operation_id="snapshot",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth(None, "write"))],
)
def snapshot(req: SnapshotRequest) -> SnapshotResponse:
    """Write a one-shot MemoryLayer bundle to the client-supplied path.

    Use case: an ephemeral session that wants to persist at shutdown
    without reconfiguring the server-default ``SOMA_BUNDLE_PATH``. The
    target path is validated to stay under the server's cwd (see
    :func:`_resolve_snapshot_path`).
    """
    target = _resolve_snapshot_path(req.path)
    mem = _get_mem()
    target.mkdir(parents=True, exist_ok=True)
    mem.save(target)
    return SnapshotResponse(
        saved=True,
        path=str(target),
        entries=len(mem),
    )


@app.get(
    "/recent",
    response_model=RetrieveResponse,
    operation_id="recent",
    tags=["default"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth(None, "read"))],
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
    dependencies=[Depends(require_auth("name", "read"))],
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
    dependencies=[Depends(require_auth("name", "write"))],
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
    dependencies=[Depends(require_auth("name", "write"))],
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
    dependencies=[Depends(require_auth("name", "read"))],
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
    dependencies=[Depends(require_auth("name", "read"))],
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
    dependencies=[Depends(require_auth("name", "read"))],
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
    dependencies=[Depends(require_auth("name", "write"))],
)
def forget_bundle(name: str, req: ForgetRequest) -> ForgetResponse:
    # Per-tenant variant remains the legacy node_id deletion only —
    # conversational criteria routing is default-bundle for Phase 37.
    if req.node_id is None:
        raise HTTPException(
            status_code=400,
            detail="bundles/{name}/forget requires a node_id",
        )
    if not _get_mem(name).forget(req.node_id):
        raise HTTPException(status_code=404, detail=f"node_id {req.node_id} not found")
    return ForgetResponse(removed=True)


@app.post(
    "/bundles/{name}/consolidate",
    response_model=ConsolidateResponse,
    operation_id="bundles_consolidate",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth("name", "write"))],
)
def consolidate_bundle(name: str) -> ConsolidateResponse:
    return ConsolidateResponse(processed=_get_mem(name).consolidate())


@app.post(
    "/bundles/{name}/save",
    response_model=SaveResponse,
    operation_id="bundles_save",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth("name", "write"))],
)
def save_bundle(name: str) -> SaveResponse:
    mem = _get_mem(name)
    path = _path_for(name)
    path.mkdir(parents=True, exist_ok=True)
    mem.save(path)
    return SaveResponse(saved_to=str(path))


@app.post(
    "/bundles/{name}/snapshot",
    response_model=SnapshotResponse,
    operation_id="bundles_snapshot",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth("name", "write"))],
)
def snapshot_bundle(name: str, req: SnapshotRequest) -> SnapshotResponse:
    """Per-tenant variant of :func:`snapshot` — saves the named bundle
    to the caller-supplied path (same cwd-containment check)."""
    target = _resolve_snapshot_path(req.path)
    mem = _get_mem(name)
    target.mkdir(parents=True, exist_ok=True)
    mem.save(target)
    return SnapshotResponse(
        saved=True,
        path=str(target),
        entries=len(mem),
    )


@app.get(
    "/bundles/{name}/recent",
    response_model=RetrieveResponse,
    operation_id="bundles_recent",
    tags=["bundles"],
    responses=ERROR_RESPONSES,
    dependencies=[Depends(require_auth("name", "read"))],
)
def recent_bundle(name: str, n: int = 10) -> RetrieveResponse:
    hits = _get_mem(name).get_recent(max(1, n))
    return RetrieveResponse(hits=[_hit(h) for h in hits])
