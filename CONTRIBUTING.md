# Contributing to SOMA

Thanks for your interest. SOMA is small and the code surface stays
that way on purpose — pull requests that delete as much as they add
are especially welcome.

## Setup

```bash
pip install -e ".[dev,sbert,ann,serve,langchain,llamaindex]"
pytest tests/ -v
ruff check src/ tests/
mypy src/soma/memory src/soma/llm src/soma/cli.py src/soma/serve.py
```

The fast loop is `pytest tests/test_memory tests/test_llm tests/test_cli.py tests/test_serve/`
— that's the product surface and finishes in under a minute.

## Scope

The project-facing product is the **local-first, learning agent memory
layer**: everything under `src/soma/memory/`, `src/soma/llm/`,
`src/soma/cli.py`, `src/soma/serve.py`, and `src/soma/integrations/`.
See `docs/positioning.md` for the pitch.

The graph/plasticity substrate (`src/soma/core`, `growth`, `metacognition`,
`consolidation`, `io`, `deploy`) is load-bearing for the research
agenda in `benchmarks/reports/paper-draft.md` §5 but not part of the
everyday memory-layer API. Changes there should not regress the
memory-layer public contract.

## Writing tests

New features need a test. Tests mirror the source tree under `tests/`.
The ergonomic fixtures live in `tests/conftest.py`. For anything that
would pull a model over the network, use `embed_fn=...` with a
deterministic stub embedder (`tests/test_serve/test_serve_smoke.py`
has an example).

## Style

- Python 3.11+, strict type hints, ruff-formatted, 100-col lines.
- Dataclasses for config/data; `nn.Module` for anything with learnable
  tensors.
- Prefer editing existing files over creating new ones. If a file
  grows past ~1000 lines, consider splitting by responsibility.
- Docstrings explain the *why* when non-obvious. Don't narrate the
  *what* — well-named identifiers already do that.

## Benchmarks

Every number in the README / paper-draft is tied to a committed
`benchmarks/run_*.py` + `benchmarks/reports/*.md`. If you change a
scoring path, re-run the relevant benchmark and update the report in
the same commit so the claim trail stays reproducible.

## Pull requests

- Rebase on `main` before opening.
- Single-purpose PRs — a bug fix and a feature should be two PRs.
- A one-line changelog entry under `CHANGELOG.md` `## [Unreleased]`
  is appreciated for anything user-visible.

## License

By contributing you agree that your contributions will be licensed
under the [MIT License](LICENSE).
