"""SOMA: Self-Organizing Memory Architecture."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("soma-memory")
except PackageNotFoundError:  # pragma: no cover — source checkout, no dist installed
    __version__ = "0.0.0+unknown"
