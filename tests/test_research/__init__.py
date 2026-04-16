"""Tests for research-only code under ``research/``.

These tests cover the experiment harnesses (synthetic corpora,
evaluation metric implementations, regression checks against the
production memory layer). They never assert that the research wins —
the experiments are exploratory, and the unit tests are scoped to
"the harness does what it says" so we don't accidentally bake an
"SOMA wins" expectation into CI.
"""
