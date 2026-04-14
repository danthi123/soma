"""Versioned brain-bundle serialization envelope.

Every SOMA brain saved to disk goes through `wrap_payload` so that a schema
tag, SOMA version, torch/tokenizers versions, and git sha ride along with the
raw state. This lets future SOMA versions detect and migrate old brains.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import tokenizers
import torch

SCHEMA_VERSION = 1


def wrap_payload(
    payload: Mapping[str, Any],
    *,
    soma_version: str,
    git_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "format": "soma-brain",
        "schema_version": SCHEMA_VERSION,
        "soma_version": soma_version,
        "git_sha": git_sha,
        "torch_version": torch.__version__,
        "tokenizers_version": tokenizers.__version__,
        "created_at": datetime.now(UTC).isoformat(),
        "payload": dict(payload),
    }


def unwrap_payload(wrapped: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if wrapped.get("format") != "soma-brain":
        raise ValueError("Not a SOMA brain bundle (missing 'format' tag)")
    meta = {k: v for k, v in wrapped.items() if k != "payload"}
    return dict(wrapped["payload"]), meta
