"""Run every ```python``` code block in the core docs as a subprocess.

Scope: README.md, docs/quickstart.md, docs/cookbook.md, docs/backends.md.
These are the promises SOMA makes to new users. Any block that won't
run fails CI.

Opt-out: prefix a block with ``<!-- doctest: skip -->`` on the line
immediately before the opening fence if it's illustrative-only
(pseudocode, partial snippet, backend requires a running server, etc.).

Optional-dep handling: blocks that import a module guarded by an extra
(sentence_transformers, fastapi, qdrant_client, chromadb, lancedb,
psycopg, boto3) skip via ``pytest.importorskip`` — so CI without those
extras stays green, while CI with them installed exercises the real thing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_docs._extractor import DocBlock, extract_python_blocks

_REPO = Path(__file__).resolve().parents[2]

_TARGETS = [
    _REPO / "README.md",
    _REPO / "docs" / "quickstart.md",
    _REPO / "docs" / "cookbook.md",
    _REPO / "docs" / "backends.md",
]

_OPTIONAL_DEPS = {
    "sentence_transformers": "sbert",
    "fastapi": "serve",
    "uvicorn": "serve",
    "qdrant_client": "qdrant",
    "chromadb": "chroma",
    "lancedb": "lancedb",
    "psycopg": "pgvector",
    "boto3": "s3",
}


def _collect_blocks() -> list[DocBlock]:
    out: list[DocBlock] = []
    for md in _TARGETS:
        if not md.exists():
            continue
        out.extend(extract_python_blocks(md))
    return out


_BLOCKS = _collect_blocks()


@pytest.mark.parametrize("block", _BLOCKS, ids=[b.id for b in _BLOCKS])
def test_doc_block_runs(block: DocBlock, tmp_path):
    for mod, extra in _OPTIONAL_DEPS.items():
        if mod in block.source:
            pytest.importorskip(mod, reason=f"needs [{extra}] extra")

    script = tmp_path / "block.py"
    script.write_text(block.source, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{block.path.name}:{block.start_line}-{block.end_line} failed "
            f"(exit {result.returncode})\n\n"
            f"stderr:\n{result.stderr}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"block source:\n{block.source}"
        )
