# Releasing SOMA

## Artifacts

SOMA ships three release artifacts on independent cadences:

| Artifact | Where | Source of version |
| -------- | ----- | ----------------- |
| `soma-memory` Python package | PyPI | `pyproject.toml` `version` field |
| `soma` Helm chart | In-tree at `deploy/helm/soma/` (OCI registry TBD) | `deploy/helm/soma/Chart.yaml` |
| `soma-memory` TypeScript client | npm (unpublished, in-tree at `clients/typescript/`) | `clients/typescript/package.json` |

Python is the canonical source. `src/soma/__init__.py:__version__`, the
FastAPI `app.version`, and the `soma version` CLI all read from
`importlib.metadata.version("soma-memory")` — whatever `pyproject.toml`
declares is what every runtime surface reports.

## Cadence

- **Python patch** (e.g. `0.2.0` → `0.2.1`): no Helm / TS bump required.
- **Python minor** (e.g. `0.2.x` → `0.3.0`): bump Helm + TS to the same
  minor unless intentionally decoupling (document the decoupling).
- **Python major** (e.g. `0.x` → `1.x`): bump all three in lockstep.
- **Release candidates** (`0.2.0rc6`): the Python package carries the rc
  suffix; Helm and TS stay on the final target minor (no rc suffix, since
  npm and Helm registries don't have first-class yank semantics for rc
  iterations the way PyPI does).

## Pre-release gates

Before tagging `v<X.Y.Z>`:

1. `pytest -q` green on `main` (or release branch).
2. `pytest tests/test_cli/test_packaging.py tests/test_cli/test_version.py tests/test_docs/ tests/test_clients/ -v` green — the drift gates.
3. `pyproject.toml`, `src/soma/__init__.py` (auto via importlib), Helm
   `Chart.yaml`, and TS `package.json` agree on the intended version.
4. `CHANGELOG.md` has an entry for the target version with today's date.
5. If any new `@app.*` routes landed since the last release, regenerate
   the TS OpenAPI snapshot:
   ```bash
   python -m uvicorn soma.serve:app --port 8420 &
   curl -s http://127.0.0.1:8420/openapi.json > clients/typescript/openapi.json
   npm --prefix clients/typescript run gen:types
   kill %1
   git add clients/typescript/openapi.json clients/typescript/src/schema.d.ts
   ```

## Release

```bash
git tag v<X.Y.Z>
git push git.dant123.com v<X.Y.Z>
git push github v<X.Y.Z>
```

The GitHub `release.yml` workflow builds and publishes to PyPI via
OIDC. Watch the run at https://github.com/danthi123/soma/actions.

## Post-release

Bump `pyproject.toml` to the next `-dev` marker (e.g. `0.2.1-dev0`) on
`main` to make it obvious the PyPI artifact and HEAD have diverged.
Helm and TS stay pinned until the next minor.
