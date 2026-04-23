"""Version reporting must match the installed distribution metadata.

Three surfaces must agree:
- ``soma.__version__`` (Python package attribute)
- ``soma version`` CLI
- FastAPI ``app.version`` on the live serve app

Plus: the ``bench`` extra must reference the canonical distribution name.
"""
from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest


def test_package_version_matches_distribution_metadata():
    import soma

    assert soma.__version__ == version("soma-memory")


def test_cli_version_matches_distribution_metadata():
    result = subprocess.run(
        [sys.executable, "-m", "soma.cli", "version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == version("soma-memory")
    assert result.stdout.strip() != "unknown"
    assert result.stdout.strip() != "0.1.0"


def test_fastapi_app_version_matches_distribution_metadata():
    pytest.importorskip("fastapi")
    from soma.serve import app

    assert app.version == version("soma-memory")


def test_bench_extra_uses_correct_distribution_name():
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    bench = data["project"]["optional-dependencies"]["bench"]
    for dep in bench:
        if dep.startswith("soma[") or dep.startswith("soma "):
            raise AssertionError(
                f"bench extra references 'soma[...]' but the distribution is "
                f"'soma-memory'. Offender: {dep!r}"
            )
