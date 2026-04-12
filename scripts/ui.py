"""Launch the SOMA Control Center GUI.

Usage::

    python scripts/ui.py
    python scripts/ui.py --config configs/default.yaml

The GUI wraps every piece of the SOMA system (training, monitoring,
checkpointing, interactive chat) in a single window so the user never
needs to touch the CLI for routine tasks.
"""

from __future__ import annotations

import argparse
import logging
import sys

from soma.ui.app import AppOptions, run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the SOMA Control Center UI.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path. If omitted, SOMAConfig defaults are used.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="SOMA Control Center",
        help="Window title.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="Initial window width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=960,
        help="Initial window height in pixels.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args(argv)
    options = AppOptions(
        config_path=args.config,
        window_title=args.title,
        window_width=args.width,
        window_height=args.height,
    )
    run(options)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry
    sys.exit(main())
