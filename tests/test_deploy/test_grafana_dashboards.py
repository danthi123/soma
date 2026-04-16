"""Structural tests for importable Grafana dashboards under ``deploy/grafana/``.

These dashboards are shipped as a developer convenience — operators drop them
into Grafana via ``grafana-cli admin`` or a provisioning volume and get
RED/USE panels for SOMA's Phase 3 metric surface out of the box.

The tests here validate structure only; they do not stand up a real Grafana
instance. Specifically they check:

* every ``deploy/grafana/*.json`` file parses as JSON;
* ``schemaVersion`` is >= 36 (Grafana 9+ dashboard schema);
* every panel target carries a non-empty ``expr``;
* every target references the datasource UID placeholder ``${DS_PROMETHEUS}``
  so the import wizard can bind a live Prometheus data source on load;
* each dashboard declares the standard ``$bundle`` and ``$instance``
  template variables so operators can filter without editing JSON;
* per-dashboard panel-count expectations match the plan
  (``docs/plans/2026-04-16-phase-9-grafana-dashboards.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GRAFANA_DIR = Path(__file__).resolve().parents[2] / "deploy" / "grafana"

# Expected panel counts per dashboard, keyed by filename stem.
# These numbers are the contract with Phase 9's plan; shrinking a dashboard
# is a deliberate act that should update this table.
EXPECTED_PANEL_COUNTS: dict[str, int] = {
    "soma-overview": 8,
    "soma-auth": 5,
    "soma-bundle-health": 6,
}

# Template variables every dashboard must expose so operators can slice by
# bundle / scrape target without editing JSON.
REQUIRED_TEMPLATE_VARS: frozenset[str] = frozenset({"bundle", "instance"})


def _dashboards() -> list[Path]:
    if not GRAFANA_DIR.is_dir():
        return []
    return sorted(GRAFANA_DIR.glob("*.json"))


def _dashboard_params() -> list[pytest.param]:
    """Stable parametrize input even before the JSONs exist.

    pytest-parametrize chokes on an empty list at collection time; emit a
    single skipped placeholder so the collector succeeds on a fresh checkout
    and the real parametrized tests light up once JSON files land.
    """
    paths = _dashboards()
    if not paths:
        return [pytest.param(None, marks=pytest.mark.skip(reason="no dashboards yet"), id="none")]
    return [pytest.param(p, id=p.stem) for p in paths]


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _iter_panels(dash: dict) -> list[dict]:
    """Flatten row panels so tests can reason about every leaf panel."""
    out: list[dict] = []
    for panel in dash.get("panels", []):
        if panel.get("type") == "row":
            out.extend(panel.get("panels", []))
        else:
            out.append(panel)
    return out


def test_grafana_dir_exists() -> None:
    assert GRAFANA_DIR.is_dir(), f"missing directory: {GRAFANA_DIR}"


def test_dashboards_present() -> None:
    names = {p.stem for p in _dashboards()}
    missing = set(EXPECTED_PANEL_COUNTS) - names
    assert not missing, f"missing dashboard JSON(s): {sorted(missing)}"


@pytest.mark.parametrize("path", _dashboard_params())
def test_dashboard_parses(path: Path) -> None:
    dash = _load(path)
    assert isinstance(dash, dict)
    assert dash.get("title"), f"{path.name}: missing top-level title"
    assert dash.get("uid"), f"{path.name}: missing uid"


@pytest.mark.parametrize("path", _dashboard_params())
def test_schema_version(path: Path) -> None:
    dash = _load(path)
    version = dash.get("schemaVersion")
    assert isinstance(version, int), f"{path.name}: schemaVersion must be int"
    assert version >= 36, f"{path.name}: schemaVersion {version} < 36"


@pytest.mark.parametrize("path", _dashboard_params())
def test_panels_have_nonempty_expr(path: Path) -> None:
    dash = _load(path)
    panels = _iter_panels(dash)
    assert panels, f"{path.name}: no panels"
    for panel in panels:
        targets = panel.get("targets") or []
        assert targets, f"{path.name}: panel {panel.get('title')!r} has no targets"
        for idx, target in enumerate(targets):
            expr = target.get("expr")
            assert isinstance(expr, str) and expr.strip(), (
                f"{path.name}: panel {panel.get('title')!r} "
                f"target #{idx} has empty expr"
            )


@pytest.mark.parametrize("path", _dashboard_params())
def test_targets_reference_prometheus_datasource(path: Path) -> None:
    dash = _load(path)
    for panel in _iter_panels(dash):
        for target in panel.get("targets", []):
            ds = target.get("datasource")
            assert isinstance(ds, dict), (
                f"{path.name}: target on panel {panel.get('title')!r} "
                f"must declare a datasource object, got {ds!r}"
            )
            assert ds.get("type") == "prometheus", (
                f"{path.name}: target datasource.type must be 'prometheus'"
            )
            assert ds.get("uid") == "${DS_PROMETHEUS}", (
                f"{path.name}: target datasource.uid must be "
                f"'${{DS_PROMETHEUS}}', got {ds.get('uid')!r}"
            )


@pytest.mark.parametrize("path", _dashboard_params())
def test_template_variables(path: Path) -> None:
    dash = _load(path)
    templating = dash.get("templating") or {}
    names = {v.get("name") for v in templating.get("list", [])}
    missing = REQUIRED_TEMPLATE_VARS - names
    assert not missing, (
        f"{path.name}: missing required template variables {sorted(missing)}"
    )
    # Every query-type template variable must bind to the same placeholder
    # datasource so grafana-cli import prompts once, not once per variable.
    for var in templating.get("list", []):
        if var.get("type") != "query":
            continue
        ds = var.get("datasource")
        assert isinstance(ds, dict), (
            f"{path.name}: template var {var.get('name')!r} "
            f"must declare a datasource object"
        )
        assert ds.get("uid") == "${DS_PROMETHEUS}", (
            f"{path.name}: template var {var.get('name')!r} "
            f"must reference ${{DS_PROMETHEUS}}"
        )


@pytest.mark.parametrize(
    "stem,expected",
    sorted(EXPECTED_PANEL_COUNTS.items()),
)
def test_panel_count_matches_plan(stem: str, expected: int) -> None:
    path = GRAFANA_DIR / f"{stem}.json"
    if not path.exists():
        pytest.skip(f"{stem}.json not present yet")
    dash = _load(path)
    count = len(_iter_panels(dash))
    assert count == expected, (
        f"{stem}.json has {count} panels, plan requires {expected}"
    )


@pytest.mark.parametrize("path", _dashboard_params())
def test_tags_include_soma(path: Path) -> None:
    dash = _load(path)
    tags = dash.get("tags") or []
    assert "soma" in tags, f"{path.name}: tags missing 'soma'"


@pytest.mark.parametrize("path", _dashboard_params())
def test_refresh_and_time_defaults(path: Path) -> None:
    """Shipped defaults should be sane — 30s refresh, last 1h window."""
    dash = _load(path)
    assert dash.get("refresh"), f"{path.name}: missing default refresh"
    time_cfg = dash.get("time") or {}
    assert time_cfg.get("from"), f"{path.name}: missing time.from default"
    assert time_cfg.get("to"), f"{path.name}: missing time.to default"
