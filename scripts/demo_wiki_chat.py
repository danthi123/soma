"""Thin wrapper — the real implementation lives in
:mod:`soma._cli_commands.wiki_chat`. This file exists so
``python scripts/demo_wiki_chat.py …`` still works from a clone."""
from __future__ import annotations

import sys
from pathlib import Path

# When running this file directly from a clone without `pip install -e .`,
# add `src/` to sys.path so `soma._cli_commands` resolves.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from soma._cli_commands.wiki_chat import *  # noqa: E402,F401,F403

# Explicit re-exports for private helpers. The wildcard import above
# skips underscore-prefixed names; these are kept available for existing
# tests and debugging callers that import them directly from this module.
from soma._cli_commands.wiki_chat import (  # noqa: E402,F401
    _chat,
    _extract_pdf_text,
    _heading_chain,
    _ingest,
    _iter_doc_files,
    _load_doc_text,
    _resolve_backend,
    _split_long_paragraph,
    main,
)

if __name__ == "__main__":
    main()
