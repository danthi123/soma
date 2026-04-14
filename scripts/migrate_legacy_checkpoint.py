"""One-shot migrator: pre-v1 SOMA checkpoint → v1 bundle.

Usage:
    python scripts/migrate_legacy_checkpoint.py checkpoints/current.pt
Writes the migrated bundle back over the input path (after backing up to .bak).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soma.core.brain_bundle import (  # noqa: E402
    SCHEMA_VERSION,
    migrate_payload,
    to_cpu_state,
    wrap_payload,
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/migrate_legacy_checkpoint.py <path>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"No such file: {path}")
        return 2

    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and raw.get("format") == "soma-brain":
        print(f"Already v{raw['schema_version']}, nothing to do.")
        return 0

    bak = path.with_suffix(path.suffix + ".legacy-bak")
    shutil.copy2(path, bak)
    print(f"Backed up to {bak}")

    payload, _ = migrate_payload(raw, from_schema=0)
    payload = to_cpu_state(payload)
    wrapped = wrap_payload(payload, soma_version="migrator")
    torch.save(wrapped, str(path))
    print(f"Upgraded {path} to schema v{SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
