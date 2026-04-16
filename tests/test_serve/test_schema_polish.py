"""Schema-polish pre-requisites for TypeScript client generation.

Phase 5 of the memory-layer pivot roadmap auto-generates a TS client
from ``GET /openapi.json``. These tests pin the seven schema gaps that
would otherwise produce a sloppy generated client: CORS, bearer auth
scheme, typed error responses, named 2xx response models, clean
operation_ids, route tags, and a ``stub`` embed path for CI that
avoids sentence-transformers.

The tests intentionally read the runtime OpenAPI document rather than
the Python source — that's the contract the TS generator consumes.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from soma import serve


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _spec() -> dict[str, Any]:
    r = TestClient(serve.app).get("/openapi.json")
    assert r.status_code == 200
    return r.json()


EXPECTED_OP_IDS: dict[tuple[str, str], str] = {
    # (method, path) -> operation_id
    ("get", "/health"): "health",
    ("get", "/version"): "version",
    ("get", "/status"): "status",
    ("post", "/store"): "store",
    ("post", "/store_batch"): "store_batch",
    ("post", "/retrieve"): "retrieve",
    ("get", "/get/{node_id}"): "get",
    ("post", "/forget"): "forget",
    ("get", "/recent"): "recent",
    ("get", "/related/{node_id}"): "related",
    ("post", "/consolidate"): "consolidate",
    ("post", "/save"): "save",
    ("get", "/bundles/{name}/status"): "bundles_status",
    ("post", "/bundles/{name}/store"): "bundles_store",
    ("post", "/bundles/{name}/store_batch"): "bundles_store_batch",
    ("post", "/bundles/{name}/retrieve"): "bundles_retrieve",
    ("get", "/bundles/{name}/get/{node_id}"): "bundles_get",
    ("post", "/bundles/{name}/forget"): "bundles_forget",
    ("get", "/bundles/{name}/recent"): "bundles_recent",
    ("get", "/bundles/{name}/related/{node_id}"): "bundles_related",
    ("post", "/bundles/{name}/consolidate"): "bundles_consolidate",
    ("post", "/bundles/{name}/save"): "bundles_save",
}

EXPECTED_TAGS: dict[str, str] = {
    "/health": "system",
    "/version": "system",
    "/status": "default",
    "/store": "default",
    "/store_batch": "default",
    "/retrieve": "default",
    "/get/{node_id}": "default",
    "/forget": "default",
    "/recent": "default",
    "/related/{node_id}": "default",
    "/consolidate": "default",
    "/save": "default",
    "/bundles/{name}/status": "bundles",
    "/bundles/{name}/store": "bundles",
    "/bundles/{name}/store_batch": "bundles",
    "/bundles/{name}/retrieve": "bundles",
    "/bundles/{name}/get/{node_id}": "bundles",
    "/bundles/{name}/forget": "bundles",
    "/bundles/{name}/recent": "bundles",
    "/bundles/{name}/related/{node_id}": "bundles",
    "/bundles/{name}/consolidate": "bundles",
    "/bundles/{name}/save": "bundles",
}


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
def test_openapi_spec_advertises_bearer_auth() -> None:
    spec = _spec()
    schemes = spec.get("components", {}).get("securitySchemes", {})
    assert schemes, "OpenAPI spec must declare at least one security scheme"
    # FastAPI emits the HTTPBearer scheme under the class name by default.
    bearer = next(
        (s for s in schemes.values() if s.get("type") == "http" and s.get("scheme") == "bearer"),
        None,
    )
    assert bearer is not None, f"no bearer security scheme in {schemes}"


def test_openapi_security_scheme_is_bearer_jwt() -> None:
    """Phase 4 — advertised scheme must carry bearerFormat=JWT under the
    named ``bearerAuth`` key so the TS generator produces a typed
    BearerJWT helper rather than a generic HTTPBasic bit.
    """
    spec = _spec()
    schemes = spec["components"]["securitySchemes"]
    assert "bearerAuth" in schemes, f"expected 'bearerAuth' key, got {list(schemes)}"
    bearer = schemes["bearerAuth"]
    assert bearer["type"] == "http"
    assert bearer["scheme"] == "bearer"
    assert bearer["bearerFormat"] == "JWT"


def test_protected_routes_reference_security_scheme() -> None:
    """Each protected operation must list ``bearerAuth`` in its
    ``security`` block; the system routes must not carry the guard.
    """
    spec = _spec()
    protected = [
        ("post", "/store"),
        ("post", "/store_batch"),
        ("post", "/retrieve"),
        ("get", "/get/{node_id}"),
        ("post", "/forget"),
        ("post", "/consolidate"),
        ("post", "/save"),
        ("get", "/recent"),
        ("get", "/related/{node_id}"),
        ("get", "/status"),
        ("post", "/bundles/{name}/store"),
        ("get", "/bundles/{name}/status"),
    ]
    for method, path in protected:
        op = spec["paths"][path][method]
        security = op.get("security") or []
        names = [list(s.keys())[0] for s in security if s]
        assert "bearerAuth" in names, (
            f"{method.upper()} {path} missing bearerAuth security requirement; "
            f"got {security!r}"
        )
    # System routes stay public — no security on /health / /version / /metrics.
    for path in ("/health", "/version"):
        op = spec["paths"][path]["get"]
        assert not op.get("security"), f"{path} should not require auth; got {op.get('security')}"


def test_error_responses_have_schemas() -> None:
    spec = _spec()
    # Pick a protected route that we registered both 401 and 404 on.
    forget = spec["paths"]["/forget"]["post"]
    responses = forget.get("responses", {})
    for code in ("401", "404"):
        assert code in responses, f"/forget missing {code} response"
        schema = responses[code]["content"]["application/json"]["schema"]
        ref = schema.get("$ref", "")
        assert ref.endswith("/ErrorResponse"), f"/forget {code} not ErrorResponse: {schema}"
    # And the schema itself exists with a `detail` field.
    err = spec["components"]["schemas"]["ErrorResponse"]
    assert "detail" in err["properties"]


def test_forget_request_schema_covers_both_shapes() -> None:
    """Phase 37: /forget accepts both legacy node_id and conversational criteria.

    The response shape depends on the branch the handler picks —
    legacy returns ``{"removed": bool}`` (unchanged), conversational
    returns the richer ``ForgetResult`` / ``ForgetPreview`` dict shape
    — so the OpenAPI response is the generic dict envelope rather
    than a single named model. The **request** schema still has a
    named model (``ForgetRequest``) that carries every field.
    """
    spec = _spec()
    forget_req_ref = (
        spec["paths"]["/forget"]["post"]["requestBody"]
        ["content"]["application/json"]["schema"]["$ref"]
    )
    assert forget_req_ref.endswith("/ForgetRequest"), forget_req_ref
    req_model = spec["components"]["schemas"]["ForgetRequest"]
    # Both the legacy node_id field and the Phase 37 criteria fields
    # are present.
    for field in ("node_id", "text_matches", "subject", "user_id", "dry_run"):
        assert field in req_model["properties"], (
            f"ForgetRequest missing {field!r}: {list(req_model['properties'])}"
        )
    # ForgetResponse is still referenced by the bundles/{name}/forget
    # variant, so the named model stays live in the component map.
    resp_model = spec["components"]["schemas"]["ForgetResponse"]
    assert "removed" in resp_model["properties"]
    assert resp_model["properties"]["removed"]["type"] == "boolean"


def test_consolidate_and_save_return_named_models() -> None:
    spec = _spec()
    cons = spec["paths"]["/consolidate"]["post"]["responses"]["200"]
    assert cons["content"]["application/json"]["schema"]["$ref"].endswith("/ConsolidateResponse")

    save = spec["paths"]["/save"]["post"]["responses"]["200"]
    assert save["content"]["application/json"]["schema"]["$ref"].endswith("/SaveResponse")

    schemas = spec["components"]["schemas"]
    assert schemas["ConsolidateResponse"]["properties"]["processed"]["type"] == "integer"
    assert schemas["SaveResponse"]["properties"]["saved_to"]["type"] == "string"


def test_operation_ids_are_clean() -> None:
    spec = _spec()
    for (method, path), expected in EXPECTED_OP_IDS.items():
        op = spec["paths"][path][method]
        assert op.get("operationId") == expected, (
            f"{method.upper()} {path}: op_id {op.get('operationId')!r} != {expected!r}"
        )


def test_tags_group_routes() -> None:
    spec = _spec()
    for path, expected_tag in EXPECTED_TAGS.items():
        ops = spec["paths"][path]
        for method, op in ops.items():
            if method not in {"get", "post"}:
                continue
            tags = op.get("tags") or []
            assert expected_tag in tags, (
                f"{method.upper()} {path} missing tag {expected_tag!r}; got {tags}"
            )


def test_cors_preflight_ok() -> None:
    client = TestClient(serve.app)
    r = client.options(
        "/store",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"


def test_stub_embed_path_returns_zeros(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reload the module under SOMA_EMBED_MODEL=stub so module-level
    # EMBED_MODEL picks up the stub branch.
    monkeypatch.setenv("SOMA_EMBED_MODEL", "stub")
    reloaded = importlib.reload(serve)
    try:
        assert reloaded.EMBED_MODEL == "stub"
        reloaded._embed_fn_cache = None  # ensure we rebuild
        fn = reloaded._embed_fn()
        vec = fn("any text")
        assert isinstance(vec, torch.Tensor)
        assert tuple(vec.shape) == (384,)
        assert torch.equal(vec, torch.zeros(384))
    finally:
        # Restore for downstream tests in the same process.
        monkeypatch.delenv("SOMA_EMBED_MODEL", raising=False)
        importlib.reload(serve)
