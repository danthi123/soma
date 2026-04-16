"""``soma`` command-line entry point.

Wraps the most common one-shot operations into a single binary so users
don't need to remember script paths::

    soma index   --wiki path/to/docs --bundle my-brain/
    soma chat    --bundle my-brain/        # auto-picks LLM backend
    soma stats   --bundle my-brain/
    soma search  --bundle my-brain/ --query "where does the user live?"
    soma serve   --port 8420               # REST API
    soma version

Each subcommand is a thin wrapper around the corresponding
``scripts/demo_*.py`` or ``soma.serve``. The intent is "minimal manual
effort to get going" — the CLI doesn't try to hide power but it does
cover the 95% case in one verb.

Backend selection follows :func:`soma.llm.backend_from_env`: pass
``--backend ollama|openai|anthropic|openai-compat|hf`` or set
``SOMA_LLM_BACKEND``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _cmd_index(args: argparse.Namespace) -> int:
    from scripts.demo_wiki_chat import _ingest

    if not args.wiki.is_dir():
        print(f"error: --wiki {args.wiki} is not a directory", file=sys.stderr)
        return 2
    _ingest(args.wiki, args.bundle, include_pdf=not args.no_pdf)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from scripts.demo_wiki_chat import _chat

    if not args.bundle.exists():
        print(f"error: bundle {args.bundle} not found", file=sys.stderr)
        return 2
    _chat(args.bundle, backend_name=args.backend, k=args.k, dry_run=args.dry_run)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from scripts.demo_memory_inspect import cmd_stats
    from soma.memory import MemoryLayer

    if not args.bundle.exists():
        print(f"error: bundle {args.bundle} not found", file=sys.stderr)
        return 2
    mem = MemoryLayer.load(args.bundle)
    cmd_stats(mem, args.bundle)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from scripts.demo_memory_inspect import cmd_search
    from soma.memory import MemoryLayer

    if not args.bundle.exists():
        print(f"error: bundle {args.bundle} not found", file=sys.stderr)
        return 2
    mem = MemoryLayer.load(args.bundle)
    cmd_search(mem, args.query, args.k)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "error: serve needs uvicorn. Install: pip install 'soma[serve]'",
            file=sys.stderr,
        )
        return 1
    uvicorn.run(
        "soma.serve.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    try:
        from importlib.metadata import version

        print(version("soma"))
    except Exception:
        print("unknown")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soma",
        description="SOMA — local-first agent memory layer (CLI entry point)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bundle_arg = argparse.ArgumentParser(add_help=False)
    bundle_arg.add_argument("--bundle", type=Path, required=True)

    p_index = sub.add_parser(
        "index", parents=[bundle_arg], help="Ingest a wiki/PDF folder into a bundle"
    )
    p_index.add_argument("--wiki", type=Path, required=True)
    p_index.add_argument("--no-pdf", action="store_true", help="Skip PDFs")
    p_index.set_defaults(func=_cmd_index)

    p_chat = sub.add_parser("chat", parents=[bundle_arg], help="REPL chat against a bundle")
    p_chat.add_argument(
        "--backend",
        choices=("auto", "ollama", "openai", "anthropic", "openai-compat", "hf", "dry-run"),
        default="auto",
    )
    p_chat.add_argument("--k", type=int, default=5)
    p_chat.add_argument("--dry-run", action="store_true")
    p_chat.set_defaults(func=_cmd_chat)

    p_stats = sub.add_parser("stats", parents=[bundle_arg], help="Bundle stats")
    p_stats.set_defaults(func=_cmd_stats)

    p_search = sub.add_parser("search", parents=[bundle_arg], help="Vector search (no LLM)")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--k", type=int, default=5)
    p_search.set_defaults(func=_cmd_search)

    p_serve = sub.add_parser("serve", help="Start REST API server (uvicorn)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8420)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    p_version = sub.add_parser("version", help="Print installed soma version")
    p_version.set_defaults(func=_cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
