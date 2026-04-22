"""Packaged implementations of the ``soma`` CLI subcommands.

Historically these lived in ``scripts/demo_*.py`` at repo root and were
imported by :mod:`soma.cli`. That worked from a clone but broke for
anyone ``pip install``-ing the distribution — ``scripts/`` is not part
of the wheel. They were moved here (private underscore prefix — not
public API, subject to change without a deprecation cycle) so the CLI
works from a plain pip install.

The repo-root ``scripts/demo_*.py`` files remain as thin wrappers over
these modules so existing ``python scripts/demo_wiki_chat.py …``
invocations still work during development.
"""
from __future__ import annotations

__all__ = ["wiki_chat", "memory_inspect"]
