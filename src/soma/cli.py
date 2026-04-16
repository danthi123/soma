"""``soma`` command-line entry point.

Wraps the most common one-shot operations into a single binary so users
don't need to remember script paths::

    soma index   --wiki path/to/docs --bundle my-brain/
    soma chat    --bundle my-brain/        # auto-picks LLM backend
    soma stats   --bundle my-brain/
    soma search  --bundle my-brain/ --query "where does the user live?"
    soma forget  --bundle my-brain/ --node-id <uuid>
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
import json
import os
import re
import sys
from datetime import timedelta
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


def _cmd_forget(args: argparse.Namespace) -> int:
    from soma.memory import MemoryLayer

    if not args.bundle.exists():
        print(f"error: bundle {args.bundle} not found", file=sys.stderr)
        return 2
    mem = MemoryLayer.load(args.bundle)
    if not mem.forget(args.node_id):
        # Try unique-prefix match for convenience.
        matches = [nid for nid in mem._ids if nid.startswith(args.node_id)]
        if len(matches) == 1:
            mem.forget(matches[0])
            print(f"forgot {matches[0]} (prefix match)")
        elif len(matches) > 1:
            print(
                f"error: prefix {args.node_id!r} matches {len(matches)} entries; "
                "use a longer prefix or the full id",
                file=sys.stderr,
            )
            return 2
        else:
            print(f"error: node_id {args.node_id!r} not found", file=sys.stderr)
            return 2
    else:
        print(f"forgot {args.node_id}")
    mem.save(args.bundle)
    print(f"saved {args.bundle}")
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
        "soma.serve:app",
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


# ------------------------------------------------------------------
# `soma auth` — JWT issue / verify / rotate-secret (Phase 4)
# ------------------------------------------------------------------
_EXPIRES_RE = re.compile(r"^(\d+)([dhm])$")


def _parse_expires(spec: str) -> timedelta:
    """Parse ``30d|7d|24h|60m`` shorthand into :class:`timedelta`.

    Raises ``ValueError`` on any malformed spec so the CLI can surface
    a clean error rather than ``argparse``'s generic message.
    """
    m = _EXPIRES_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"invalid --expires {spec!r}; expected NUMBER + unit (d|h|m), e.g. 30d"
        )
    n, unit = int(m.group(1)), m.group(2)
    if n <= 0:
        raise ValueError(f"--expires must be positive; got {spec!r}")
    if unit == "d":
        return timedelta(days=n)
    if unit == "h":
        return timedelta(hours=n)
    return timedelta(minutes=n)


def _parse_bundle_spec(spec: str) -> tuple[str, list[str]]:
    """Parse ``NAME:PERMS`` into ``(name, [perm, ...])``.

    Perms is a comma-separated list drawn from ``read|write|admin``.
    Raises ``ValueError`` on empty name, unknown perm, or missing
    colon. CLI layer converts the exception into an exit=2.
    """
    if ":" not in spec:
        raise ValueError(f"--bundle {spec!r}: expected NAME:PERMS (e.g. alex:read,write)")
    name, perms_csv = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--bundle {spec!r}: bundle name must be non-empty")
    perms = [p.strip() for p in perms_csv.split(",") if p.strip()]
    if not perms:
        raise ValueError(f"--bundle {spec!r}: at least one perm required")
    for p in perms:
        if p not in ("read", "write", "admin"):
            raise ValueError(
                f"--bundle {spec!r}: unknown perm {p!r}; must be read | write | admin"
            )
    return name, perms


def _cmd_auth_issue(args: argparse.Namespace) -> int:
    from soma.auth import issue_token

    alg = os.environ.get("SOMA_JWT_ALG", "HS256")
    secret = os.environ.get("SOMA_JWT_SECRET", "")
    private_key_path = os.environ.get("SOMA_JWT_PRIVATE_KEY_PATH", "")

    if alg == "HS256":
        if not secret:
            print(
                "error: SOMA_JWT_SECRET is required to issue HS256 tokens. "
                "Run `soma auth rotate-secret` to generate one.",
                file=sys.stderr,
            )
            return 2
        signing_kwargs: dict[str, object] = {"secret": secret}
    elif alg == "RS256":
        if not private_key_path:
            print(
                "error: SOMA_JWT_PRIVATE_KEY_PATH is required to issue RS256 tokens.",
                file=sys.stderr,
            )
            return 2
        try:
            pem = Path(private_key_path).read_bytes()
        except OSError as exc:
            print(f"error: cannot read {private_key_path}: {exc}", file=sys.stderr)
            return 2
        signing_kwargs = {"private_key_pem": pem}
    else:
        print(f"error: unsupported SOMA_JWT_ALG={alg!r}", file=sys.stderr)
        return 2

    try:
        expires_in = _parse_expires(args.expires)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    bundles: dict[str, list[str]] = {}
    for raw in args.bundle or []:
        try:
            name, perms = _parse_bundle_spec(raw)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        bundles[name] = perms

    token = issue_token(
        sub=args.sub,
        bundles=bundles,  # type: ignore[arg-type]
        expires_in=expires_in,
        alg=alg,
        **signing_kwargs,  # type: ignore[arg-type]
    )
    print(token)
    return 0


def _cmd_auth_verify(args: argparse.Namespace) -> int:
    from soma.auth import verify_token

    alg = os.environ.get("SOMA_JWT_ALG", "HS256")
    secret = os.environ.get("SOMA_JWT_SECRET", "")
    public_key_path = os.environ.get("SOMA_JWT_PUBLIC_KEY_PATH", "")

    verify_kwargs: dict[str, object] = {"alg": alg}
    if alg == "HS256":
        if not secret:
            print("error: SOMA_JWT_SECRET is required to verify HS256 tokens.", file=sys.stderr)
            return 2
        verify_kwargs["secret"] = secret
    elif alg == "RS256":
        if not public_key_path:
            print(
                "error: SOMA_JWT_PUBLIC_KEY_PATH is required to verify RS256 tokens.",
                file=sys.stderr,
            )
            return 2
        try:
            pem = Path(public_key_path).read_bytes()
        except OSError as exc:
            print(f"error: cannot read {public_key_path}: {exc}", file=sys.stderr)
            return 2
        verify_kwargs["public_key_pem"] = pem
    else:
        print(f"error: unsupported SOMA_JWT_ALG={alg!r}", file=sys.stderr)
        return 2

    try:
        principal = verify_token(args.token, **verify_kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 — CLI surface, want the message
        print(f"error: token verification failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "sub": principal.sub,
                "bundles": principal.bundles,
                "jti": principal.jti,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_auth_rotate_secret(_: argparse.Namespace) -> int:
    from soma.auth import generate_secret

    print(generate_secret())
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

    p_forget = sub.add_parser(
        "forget", parents=[bundle_arg], help="Delete an entry by node_id (or unique prefix)"
    )
    p_forget.add_argument("--node-id", required=True)
    p_forget.set_defaults(func=_cmd_forget)

    p_serve = sub.add_parser("serve", help="Start REST API server (uvicorn)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8420)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    p_version = sub.add_parser("version", help="Print installed soma version")
    p_version.set_defaults(func=_cmd_version)

    # --- `soma auth` ------------------------------------------------
    p_auth = sub.add_parser(
        "auth",
        help="JWT issuance / verification for the REST server",
        description=(
            "Mint and inspect JWTs scoped to per-bundle read/write/admin "
            "perms. Requires SOMA_JWT_SECRET (HS256) or "
            "SOMA_JWT_PRIVATE_KEY_PATH + SOMA_JWT_PUBLIC_KEY_PATH (RS256)."
        ),
    )
    auth_sub = p_auth.add_subparsers(dest="auth_cmd", required=True)

    p_auth_issue = auth_sub.add_parser("issue", help="Mint a signed JWT")
    p_auth_issue.add_argument(
        "--sub",
        required=True,
        help="Token subject / caller id (goes into the JWT `sub` claim)",
    )
    p_auth_issue.add_argument(
        "--bundle",
        action="append",
        metavar="NAME:PERMS",
        help=(
            "Per-bundle grant. Repeatable. PERMS is a comma-separated "
            "subset of read,write,admin. Example: --bundle alex:read,write"
        ),
    )
    p_auth_issue.add_argument(
        "--expires",
        required=True,
        metavar="SPEC",
        help="Token TTL: 30d | 7d | 24h | 60m",
    )
    p_auth_issue.set_defaults(func=_cmd_auth_issue)

    p_auth_verify = auth_sub.add_parser("verify", help="Decode and validate a JWT")
    p_auth_verify.add_argument("--token", required=True, help="JWT string to verify")
    p_auth_verify.set_defaults(func=_cmd_auth_verify)

    p_auth_rotate = auth_sub.add_parser(
        "rotate-secret",
        help="Generate a fresh HS256 shared secret (stdout)",
    )
    p_auth_rotate.set_defaults(func=_cmd_auth_rotate_secret)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
