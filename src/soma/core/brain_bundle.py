"""Versioned brain-bundle serialization envelope.

Every SOMA brain saved to disk goes through `wrap_payload` so that a schema
tag, SOMA version, torch/tokenizers versions, and git sha ride along with the
raw state. This lets future SOMA versions detect and migrate old brains.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
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


class MigrationError(RuntimeError):
    pass


Migrator = Callable[[dict[str, Any]], dict[str, Any]]

_MIGRATORS: dict[tuple[int, int], Migrator] = {}


def register_migrator(from_schema: int, to_schema: int) -> Callable[[Migrator], Migrator]:
    def deco(fn: Migrator) -> Migrator:
        _MIGRATORS[(from_schema, to_schema)] = fn
        return fn

    return deco


@register_migrator(0, 1)
def _migrate_0_to_1(payload: dict[str, Any]) -> dict[str, Any]:
    warnings.warn(
        "Loading pre-v1 SOMA checkpoint. Upgrading in place to schema v1. "
        "Re-save to persist the upgrade.",
        stacklevel=2,
    )
    return payload  # No structural change in 0→1; just gain the envelope.


def migrate_payload(payload: dict[str, Any], *, from_schema: int) -> tuple[dict[str, Any], int]:
    if from_schema > SCHEMA_VERSION:
        raise MigrationError(
            f"Checkpoint schema v{from_schema} is newer than this SOMA "
            f"(supports up to v{SCHEMA_VERSION}). Upgrade SOMA to load."
        )
    # Defensive shallow copy: isolate the caller's dict from mutating migrators.
    # Matches the convention in wrap_payload/unwrap_payload.
    payload = dict(payload)
    current = from_schema
    while current < SCHEMA_VERSION:
        step = _MIGRATORS.get((current, current + 1))
        if step is None:
            raise MigrationError(f"No migrator from schema v{current} to v{current + 1}")
        payload = step(payload)
        current += 1
    return payload, current


def to_cpu_state(obj: Any) -> Any:
    """Recursively move every tensor in a nested mapping/list to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(to_cpu_state(v) for v in obj)
    return obj


def peek_payload(raw: Any) -> dict[str, Any]:
    """Return the SOMA payload regardless of envelope vs legacy layout.

    If ``raw`` is a v1 brain bundle (``format="soma-brain"``), return
    its ``payload`` dict. Otherwise assume the pre-envelope legacy format
    and return the dict as-is. Used by script call sites that need to
    peek at ``state["config"]`` / ``state["global_step"]`` before handing
    the file to ``SOMA.load_state``.
    """
    if isinstance(raw, dict) and raw.get("format") == "soma-brain":
        return dict(raw.get("payload", {}))
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Expected a dict-shaped checkpoint; got {type(raw).__name__}")


# ---------------------------------------------------------------------------
# Directory-shaped brain bundle (manifest.json + brain.pt + sidecars)
# ---------------------------------------------------------------------------
def write_manifest(
    out_dir: Path,
    *,
    soma_version: str,
    vocab_size: int,
    llm_identity: str | None = None,
    interface_spec: dict[str, Any] | None = None,
) -> None:
    """Write ``manifest.json`` alongside the other bundle files.

    The manifest captures enough information for a future SOMA version
    (or sibling tool) to detect, validate, and migrate a directory-shaped
    brain bundle without having to load ``brain.pt`` first. ``vocab_size``
    is cross-checked against the tokenizer on load so swapping in a
    mismatched tokenizer fails loudly rather than silently corrupting
    embeddings. ``interface_spec`` records the text-aligned sensor /
    output node UUIDs and their dims so a future head-swap tool can find
    the I/O boundary without heuristics.
    """
    manifest: dict[str, Any] = {
        "format": "soma-brain-bundle",
        "schema_version": SCHEMA_VERSION,
        "soma_version": soma_version,
        "vocab_size": int(vocab_size),
        "llm_identity": llm_identity,
        "interface_spec": dict(interface_spec) if interface_spec else {},
        "created_at": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
        "tokenizers_version": tokenizers.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(out_dir: Path) -> dict[str, Any]:
    """Load and return the bundle manifest as a plain dict."""
    data = json.loads((out_dir / "manifest.json").read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest.json must deserialize to a dict, got {type(data).__name__}")
    return dict(data)
