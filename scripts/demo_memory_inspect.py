"""Thin wrapper — the real implementation lives in
:mod:`soma._cli_commands.memory_inspect`."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from soma._cli_commands.memory_inspect import *  # noqa: E402,F401,F403

# Explicit re-exports for private helpers. The wildcard import above
# skips underscore-prefixed names; these are kept available for existing
# tests and debugging callers that import them directly from this module.
from soma._cli_commands.memory_inspect import (  # noqa: E402,F401
    _bundle_disk_kb,
    _meta_matches,
    _parse_where,
    _truncate,
    main,
)

if __name__ == "__main__":
    main()
