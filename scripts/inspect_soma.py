"""Operator-facing diagnostics CLI for saved SOMA bundles.

Reads a bundle directory or single ``.pt`` checkpoint and prints a
human-readable report covering:

- Bundle envelope (schema version, global step, config summary).
- Graph shape (node counts by type, edge density, edge age / strength).
- Working memory (occupied slots, usage / decay stats, slot age).
- Homeostasis (node gain distribution, in-range check).
- Episodic memory (occupancy, oldest entry).
- Textual I/O sidecars (tokenizer vocab, encoder shape) -- directory
  bundles only; skipped cleanly when sidecars are missing.

This is a READ-ONLY tool -- nothing in SOMA state is mutated. Use it to
spot-check a training run that looks odd (e.g., gain pinned at the
10.0 cap means homeostasis is overshooting), or to eyeball a
checkpoint before wiring it into a longer experiment.

Usage
-----

    python scripts/inspect_soma.py --bundle path/to/bundle_dir
    python scripts/inspect_soma.py --bundle path/to/single_file.pt

Exits 1 with a short stderr message on user errors (missing path,
malformed bundle). Successful runs exit 0 after printing the report
on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

# Allow running "python scripts/inspect_soma.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soma.core.brain_bundle import peek_payload  # noqa: E402
from soma.core.config import SOMAConfig  # noqa: E402
from soma.core.node import NodeType  # noqa: E402
from soma.system import SOMA  # noqa: E402


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def _header(title: str, char: str = "=") -> str:
    return f"{title}\n{char * len(title)}"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_float(x: float, places: int = 2) -> str:
    return f"{x:.{places}f}"


def _pct(num: float, denom: float) -> str:
    if denom <= 0:
        return "0.0%"
    return f"{100.0 * num / denom:.1f}%"


def _stats(values: list[float]) -> tuple[float, float, float]:
    """Return (min, mean, max) for a non-empty list; zeros for empty."""
    if not values:
        return 0.0, 0.0, 0.0
    return min(values), sum(values) / len(values), max(values)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


# ----------------------------------------------------------------------
# Bundle loading
# ----------------------------------------------------------------------
def _load_soma(bundle: Path) -> tuple[SOMA, dict[str, Any], dict[str, Any] | None]:
    """Load a SOMA from ``bundle`` (dir or file).

    Returns ``(soma, envelope_meta, manifest_or_none)``. ``envelope_meta``
    holds the top-level fields of the brain bundle envelope (schema
    version, soma version, created_at, ...); ``manifest_or_none`` is the
    parsed ``manifest.json`` when ``bundle`` is a directory, otherwise
    ``None``.
    """
    if bundle.is_dir():
        brain_pt = bundle / "brain.pt"
        if not brain_pt.exists():
            raise FileNotFoundError(f"Bundle directory missing brain.pt: {bundle}")
        raw = torch.load(str(brain_pt), map_location="cpu", weights_only=False)
        envelope = _envelope_meta(raw)
        payload = peek_payload(raw)
        cfg = SOMAConfig.from_dict(payload["config"])
        soma = SOMA(cfg, device="cpu")
        soma.load_state(str(brain_pt))
        manifest_path = bundle / "manifest.json"
        manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        return soma, envelope, manifest

    # Single-file .pt path.
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle file does not exist: {bundle}")
    raw = torch.load(str(bundle), map_location="cpu", weights_only=False)
    envelope = _envelope_meta(raw)
    payload = peek_payload(raw)
    cfg = SOMAConfig.from_dict(payload["config"])
    soma = SOMA(cfg, device="cpu")
    soma.load_state(str(bundle))
    return soma, envelope, None


def _envelope_meta(raw: Any) -> dict[str, Any]:
    """Extract the v1-envelope fields (or legacy sentinel values)."""
    if isinstance(raw, dict) and raw.get("format") == "soma-brain":
        return {k: v for k, v in raw.items() if k != "payload"}
    return {"format": "legacy", "schema_version": 0}


# ----------------------------------------------------------------------
# Section renderers
# ----------------------------------------------------------------------
def _render_envelope(bundle: Path, soma: SOMA, envelope: dict[str, Any]) -> list[str]:
    lines: list[str] = [_header("SOMA Inspection Report"), ""]
    lines.append(f"Bundle: {bundle}")
    lines.append(f"Schema: v{envelope.get('schema_version', 0)}")
    if envelope.get("format") == "legacy":
        lines.append("Format: legacy (pre-envelope)")
    soma_version = envelope.get("soma_version")
    if soma_version:
        lines.append(f"SOMA version: {soma_version}")
    created_at = envelope.get("created_at")
    if created_at:
        lines.append(f"Created: {created_at}")
    lines.append(f"Global step: {_fmt_int(soma.global_step)}")
    cfg = soma.config
    lines.append(
        f"Config: sensor_output_dim={cfg.sensor_output_dim}, "
        f"wm_slots={cfg.wm_slots}, wm_dim={cfg.wm_dim}, "
        f"episodic_capacity={_fmt_int(cfg.episodic_capacity)}, "
        f"max_nodes={_fmt_int(cfg.max_nodes)}, "
        f"max_edges_per_node={cfg.max_edges_per_node}"
    )
    return lines


def _render_graph(soma: SOMA) -> list[str]:
    g = soma.graph
    lines: list[str] = ["", _header("Graph", "-")]

    lines.append(f"Nodes: {_fmt_int(g.num_nodes)} total")
    by_type: dict[NodeType, int] = {t: 0 for t in NodeType}
    for n in g.all_nodes():
        by_type[n.node_type] += 1
    for t in (NodeType.SENSOR, NodeType.ASSOCIATOR, NodeType.INTEGRATOR, NodeType.OUTPUT):
        lines.append(f"  {t.name:11s} {_fmt_int(by_type[t])}")

    edges = g.all_edges()
    lines.append(f"Edges: {_fmt_int(len(edges))}")
    if edges:
        edges_per_node: list[int] = []
        for n in g.all_nodes():
            edges_per_node.append(len(g.get_outgoing_edges(n.id)))
        mn, mean, mx = _stats([float(v) for v in edges_per_node])
        lines.append(
            f"  per-source min={_fmt_int(int(mn))} "
            f"mean={_fmt_float(mean)} max={_fmt_int(int(mx))} "
            f"(config cap {soma.config.max_edges_per_node})"
        )

        grace = soma.config.pruning_grace_period
        young = sum(1 for e in edges if (soma.global_step - e.creation_step) < grace)
        mature = len(edges) - young
        lines.append(f"  by age: young (<{_fmt_int(grace)} steps) {_fmt_int(young)}")
        lines.append(f"          mature                    {_fmt_int(mature)}")

        weak_cut = soma.config.edge_strength_threshold * 100.0  # e.g., 0.1
        weights_abs = [abs(float(e.weight.detach().item())) for e in edges]
        weak = sum(1 for w in weights_abs if w < weak_cut)
        normal = len(edges) - weak
        lines.append(
            f"  by |weight|: weak (<{_fmt_float(weak_cut)}) {_fmt_int(weak)} "
            f"/ normal {_fmt_int(normal)}"
        )
        w_min, w_mean, w_max = _stats(weights_abs)
        lines.append(
            f"  |weight| min={_fmt_float(w_min, 4)} "
            f"mean={_fmt_float(w_mean, 4)} max={_fmt_float(w_max, 4)}"
        )

        strengths = [float(e.strength) for e in edges]
        s_min, s_mean, s_max = _stats(strengths)
        lines.append(
            f"  strength min={_fmt_float(s_min, 4)} "
            f"mean={_fmt_float(s_mean, 4)} max={_fmt_float(s_max, 4)}"
        )
    else:
        lines.append("  (no edges)")

    # Boundary summary.
    sensor_mods = list(g.sensor_nodes.keys())
    output_mods = list(g.output_nodes.keys())
    lines.append(f"I/O boundary: sensors={sensor_mods}, outputs={output_mods}")
    return lines


def _render_working_memory(soma: SOMA) -> list[str]:
    wm = soma.working_memory
    lines: list[str] = ["", _header(f"Working Memory ({wm.num_slots} slots)", "-")]

    usage = wm.usage.detach().cpu()
    age = wm.age.detach().cpu()
    occupied = int((usage >= wm.fade_threshold).sum().item())
    lines.append(f"Occupied: {_fmt_int(occupied)} / {_fmt_int(wm.num_slots)}")
    lines.append(
        f"Usage: mean={_fmt_float(float(usage.mean().item()))}, "
        f"min={_fmt_float(float(usage.min().item()))}, "
        f"max={_fmt_float(float(usage.max().item()))}"
    )
    lines.append(f"Decay rate (per-step): {_fmt_float(wm.decay_rate)}")

    occupied_mask = usage >= wm.fade_threshold
    if bool(occupied_mask.any().item()):
        occ_age = age[occupied_mask].tolist()
        med = _median([float(v) for v in occ_age])
        mx = max(occ_age)
        lines.append(
            f"Slot age (steps since write, occupied): "
            f"median={_fmt_int(int(med))} max={_fmt_int(int(mx))}"
        )
    else:
        lines.append("Slot age: (no occupied slots)")

    # Quick sanity flag: any stuck slot (full usage AND very old) would
    # usually mean writes aren't happening / decay is off.
    return lines


def _render_homeostasis(soma: SOMA) -> list[str]:
    g = soma.graph
    cfg = soma.config
    lines: list[str] = ["", _header("Homeostasis", "-")]

    gains = [float(n.gain) for n in g.all_nodes()]
    if gains:
        g_min, g_mean, g_max = _stats(gains)
        in_range = cfg.gain_min <= g_min and g_max <= cfg.gain_max
        flag = "in-range" if in_range else "OUT OF RANGE"
        lines.append(
            f"Gain: min={_fmt_float(g_min)} "
            f"mean={_fmt_float(g_mean)} "
            f"max={_fmt_float(g_max)} "
            f"({flag} [{cfg.gain_min}, {cfg.gain_max}])"
        )
        pinned_hi = sum(1 for v in gains if v >= cfg.gain_max * 0.99)
        pinned_lo = sum(1 for v in gains if v <= cfg.gain_min * 1.01)
        if pinned_hi or pinned_lo:
            lines.append(
                f"Gain pinned: {_fmt_int(pinned_hi)} at/near max, {_fmt_int(pinned_lo)} at/near min"
            )
    else:
        lines.append("Gain: (no nodes)")

    # EMA activations across node types -- useful to see whether signal
    # is flowing through the middle of the graph and landing on OUTPUT.
    for nt in (NodeType.SENSOR, NodeType.ASSOCIATOR, NodeType.OUTPUT):
        emas = [float(n.activation_ema) for n in g.all_nodes() if n.node_type is nt]
        if not emas:
            continue
        e_min, e_mean, e_max = _stats(emas)
        lines.append(
            f"Activation EMA ({nt.name:11s}): "
            f"min={_fmt_float(e_min, 4)} "
            f"mean={_fmt_float(e_mean, 4)} "
            f"max={_fmt_float(e_max, 4)}"
        )

    # Regulator state from the live module (reflects the last update
    # that happened pre-save, since these scalars round-trip through
    # state_dict).
    reg = soma.homeostasis
    lines.append(
        f"Regulator: loss_ema={_fmt_float(reg.loss_ema, 4)}, "
        f"lr_multiplier={_fmt_float(reg.global_lr_multiplier, 3)}, "
        f"allow_synaptogenesis={reg.allow_synaptogenesis}, "
        f"allow_neurogenesis={reg.allow_neurogenesis}"
    )
    return lines


def _render_episodic(soma: SOMA) -> list[str]:
    em = soma.episodic_memory
    lines: list[str] = ["", _header("Episodic Memory", "-")]
    n_valid = em.num_valid
    lines.append(
        f"Entries: {_fmt_int(n_valid)} / {_fmt_int(em.capacity)} "
        f"({_pct(n_valid, em.capacity)} full)"
    )
    if n_valid > 0:
        ts = em.timestamps.detach().cpu()
        valid = em.valid.detach().cpu()
        valid_ts = ts[valid]
        oldest = int(valid_ts.min().item())
        newest = int(valid_ts.max().item())
        lines.append(f"Oldest entry: step {_fmt_int(oldest)}")
        lines.append(f"Newest entry: step {_fmt_int(newest)}")
        surprise = em.surprise.detach().cpu()[valid]
        if surprise.numel() > 0:
            lines.append(
                f"Surprise: mean={_fmt_float(float(surprise.mean().item()), 4)} "
                f"max={_fmt_float(float(surprise.max().item()), 4)}"
            )
    return lines


def _render_sidecars(bundle: Path, manifest: dict[str, Any] | None) -> list[str]:
    """Textual I/O section -- directory bundles only.

    Quietly returns an empty list for single-file checkpoints or when no
    sidecars are present, so the report skips the heading entirely
    rather than printing an empty section.
    """
    if not bundle.is_dir():
        return []

    tokenizer_path = bundle / "tokenizer.json"
    encoder_path = bundle / "encoder.pt"
    verbalizer_dir = bundle / "verbalizer"

    if (
        not any([tokenizer_path.exists(), encoder_path.exists(), verbalizer_dir.is_dir()])
        and manifest is None
    ):
        return []

    lines: list[str] = ["", _header("Textual I/O sidecars", "-")]

    if manifest is not None:
        lines.append(
            f"Manifest: vocab_size={_fmt_int(int(manifest.get('vocab_size', 0)))}, "
            f"llm_identity={manifest.get('llm_identity')!r}"
        )

    if tokenizer_path.exists():
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tokenizer_path))
            lines.append(f"Tokenizer: vocab_size={_fmt_int(tok.get_vocab_size())}")
        except Exception as exc:  # noqa: BLE001 -- best-effort diagnostic
            lines.append(f"Tokenizer: (failed to load: {exc})")
    else:
        lines.append("Tokenizer: (missing)")

    if encoder_path.exists():
        try:
            blob = torch.load(str(encoder_path), map_location="cpu", weights_only=False)
            embed_dim = int(blob.get("embed_dim", 0))
            max_seq_len = int(blob.get("max_seq_len", 0))
            lines.append(f"Encoder: embed_dim={embed_dim}, max_seq_len={max_seq_len}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Encoder: (failed to load: {exc})")
    else:
        lines.append("Encoder: (missing)")

    if verbalizer_dir.is_dir():
        lines.append("Verbalizer: present")
    else:
        lines.append("Verbalizer: (missing)")

    return lines


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def _build_report(bundle: Path) -> str:
    soma, envelope, manifest = _load_soma(bundle)

    sections: list[list[str]] = [
        _render_envelope(bundle, soma, envelope),
        _render_graph(soma),
        _render_working_memory(soma),
        _render_homeostasis(soma),
        _render_episodic(soma),
        _render_sidecars(bundle, manifest),
    ]
    joined = "\n".join(line for section in sections for line in section)
    return joined + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print diagnostics for a saved SOMA bundle (read-only).",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to a SOMA bundle directory or a single brain .pt file.",
    )
    args = parser.parse_args(argv)

    bundle: Path = args.bundle
    if not bundle.exists():
        print(
            f"Bundle not found at {bundle}. Pass --bundle pointing at a directory or .pt file.",
            file=sys.stderr,
        )
        return 1

    try:
        report = _build_report(bundle)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as exc:
        # Malformed checkpoint -- keep it friendly, no traceback.
        print(f"Failed to load SOMA bundle: {exc}", file=sys.stderr)
        return 1

    print(report, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
