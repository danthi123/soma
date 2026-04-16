"""SOMA research code.

Anything under ``research/`` is exploratory work that lives outside the
shipping product surface. Sub-packages are named after the research
direction they belong to (e.g. ``research.graph_memory`` for direction
C — plastic graph under the memory layer). See
``docs/plans/2026-04-16-research-c-graph-memory.md`` for the master
plan and per-sub-phase scope.

Imports here intentionally stay light so importing the package is
free — actual experiments pull what they need from
``soma.*`` lazily inside their entry points.
"""
