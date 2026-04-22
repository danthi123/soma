"""CLI must not import from repo-root `scripts/`.

Once SOMA is installed via pip, only the `src/soma/` tree is on sys.path.
Any `from scripts.X import …` in production code means `soma index`,
`soma chat`, `soma stats`, `soma search` will `ModuleNotFoundError` for
every pip-installed user. This test pins the invariant.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


def test_cli_does_not_import_from_scripts():
    cli_py = SRC / "soma" / "cli.py"
    tree = ast.parse(cli_py.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "scripts" or node.module.startswith("scripts."):
                offenders.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scripts" or alias.name.startswith("scripts."):
                    offenders.append(f"line {node.lineno}: import {alias.name}")
    assert not offenders, (
        "cli.py imports from repo-root scripts/, which are not packaged. "
        "Move helpers into src/soma/_cli_commands/. Offenders:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "subcommand",
    ["index --help", "chat --help", "stats --help", "search --help", "version"],
)
def test_cli_subcommands_work_without_scripts_on_syspath(subcommand, tmp_path):
    """Invoke `soma <sub>` in a subprocess whose sys.path excludes repo root."""
    env = {
        "PATH": "",
        "PYTHONPATH": str(SRC),
        "SOMA_BUNDLE_PATH": str(tmp_path / "brain"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "soma.cli", *subcommand.split()],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined or "scripts" not in combined, (
        f"subcommand `{subcommand}` could not find the `scripts` module:\n{combined}"
    )
