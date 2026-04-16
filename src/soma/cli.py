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
from typing import Any

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


# ------------------------------------------------------------------
# `soma auth revoke` / `list-revoked` / `gc` — blocklist operations
# ------------------------------------------------------------------
def _require_blocklist_path() -> str | None:
    """Return ``SOMA_JWT_BLOCKLIST_PATH`` or ``None`` with a stderr hint.

    Central helper so each subcommand emits the same migration-clear
    error when the env var is unset. Keeps CLI callers from guessing
    the name or misspelling the default path.
    """
    path = os.environ.get("SOMA_JWT_BLOCKLIST_PATH", "").strip()
    if not path:
        print(
            "error: SOMA_JWT_BLOCKLIST_PATH is required. Set it to a writable "
            "path (e.g. SOMA_JWT_BLOCKLIST_PATH=./data/jwt-blocklist.jsonl) and "
            "restart any running `soma serve` so it picks up the file.",
            file=sys.stderr,
        )
        return None
    return path


def _cmd_auth_revoke(args: argparse.Namespace) -> int:
    """Add a revocation to the blocklist file.

    Two input modes:
    - ``--token <jwt>``: decode the token, pull ``jti`` + ``exp`` off the
      claims, append the record. Requires a verifier secret.
    - ``--jti <j> --exp <ts>``: the caller already has the jti (e.g.,
      lifted from an access log) and the token itself is lost. Both
      args required in this mode.
    """
    import time

    from soma.auth_revocation import FileBlocklist, RevocationRecord

    path = _require_blocklist_path()
    if path is None:
        return 2

    reason = args.reason or "revoked via soma auth revoke"
    now = int(time.time())

    if args.token:
        # Token path — need a verifier secret to pull jti/exp safely.
        from soma.auth import verify_token

        alg = os.environ.get("SOMA_JWT_ALG", "HS256")
        secret = os.environ.get("SOMA_JWT_SECRET", "")
        public_key_path = os.environ.get("SOMA_JWT_PUBLIC_KEY_PATH", "")

        verify_kwargs: dict[str, object] = {"alg": alg}
        if alg == "HS256":
            if not secret:
                print(
                    "error: SOMA_JWT_SECRET is required to revoke by --token.",
                    file=sys.stderr,
                )
                return 2
            verify_kwargs["secret"] = secret
        elif alg == "RS256":
            if not public_key_path:
                print(
                    "error: SOMA_JWT_PUBLIC_KEY_PATH is required to revoke by --token.",
                    file=sys.stderr,
                )
                return 2
            try:
                verify_kwargs["public_key_pem"] = Path(public_key_path).read_bytes()
            except OSError as exc:
                print(f"error: cannot read {public_key_path}: {exc}", file=sys.stderr)
                return 2
        else:
            print(f"error: unsupported SOMA_JWT_ALG={alg!r}", file=sys.stderr)
            return 2

        # Pull the jti/exp straight off the signed claims so an operator
        # can't accidentally revoke a forged id. Leeway stays at 60s —
        # a just-expired token is still worth blocking, but past that
        # the verifier will refuse it anyway.
        try:
            principal = verify_token(args.token, **verify_kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 — CLI surface
            print(f"error: token verification failed: {exc}", file=sys.stderr)
            return 2

        if not principal.jti:
            print("error: token has no jti claim; nothing to revoke.", file=sys.stderr)
            return 2

        # exp comes off the unverified-but-signature-checked payload.
        import jwt as _jwt

        claims = _jwt.decode(args.token, options={"verify_signature": False})
        exp = int(claims.get("exp", 0))
        jti = principal.jti
    else:
        if not args.jti or args.exp is None:
            print(
                "error: supply either --token OR (--jti AND --exp).",
                file=sys.stderr,
            )
            return 2
        jti = args.jti
        exp = int(args.exp)

    FileBlocklist(Path(path)).add(
        RevocationRecord(jti=jti, revoked_at=now, reason=reason, exp=exp)
    )
    print(json.dumps({"revoked": jti, "exp": exp, "reason": reason}))
    return 0


def _cmd_auth_list_revoked(_: argparse.Namespace) -> int:
    """Emit one JSON object per live revocation entry, newline-separated.

    Past-exp entries are skipped — the reader treats them as "not
    revoked" anyway. Operators who want full history can ``cat`` the
    file directly.
    """
    path = _require_blocklist_path()
    if path is None:
        return 2

    p = Path(path)
    if not p.exists():
        return 0  # nothing to list

    import time

    now = int(time.time())
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(rec.get("exp", 0)) <= now:
            continue
        print(json.dumps(rec, sort_keys=True))
    return 0


# ------------------------------------------------------------------
# `soma bundle` — introspection + lifecycle management (Phase 10)
# ------------------------------------------------------------------
def _format_bytes(n: int) -> str:
    """Render a byte count as a short human-readable string.

    Mirrors GNU ``du -h`` style for the WAL column in ``bundle list``.
    Uses base-1024 because operators are used to seeing MB/GB for WAL
    file sizes; keeps precision to one decimal for the sub-GB range.
    """
    if n <= 0:
        return "—"
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(n)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _format_rows(
    infos: list[Any],
) -> list[tuple[list[str], str | None]]:
    """Turn a list of ``BundleInfo`` into ``(cells, footnote)`` pairs.

    ``footnote`` is ``None`` for healthy rows and the truncated
    corrupt reason for broken ones. :func:`_print_table` renders it
    below the row as a ``  ↳`` continuation so the main grid stays
    aligned.
    """
    rows: list[tuple[list[str], str | None]] = []
    for info in infos:
        path_str = str(info.path)
        if info.corrupt:
            entries = "CORRUPT"
            embed_dim = "—"
            backend = "—"
        else:
            entries = f"{info.entries:,}"
            embed_dim = str(info.embed_dim) if info.embed_dim else "—"
            backend = info.backend or "—"
        last_mod = info.last_modified.strftime("%Y-%m-%d %H:%M")
        wal = _format_bytes(info.wal_bytes)
        cells = [path_str, entries, embed_dim, backend, last_mod, wal]
        footnote = info.corrupt_reason if info.corrupt and info.corrupt_reason else None
        rows.append((cells, footnote))
    return rows


def _print_table(
    header: list[str],
    rows: list[tuple[list[str], str | None]],
) -> None:
    """Print ``rows`` with ``header`` as aligned columns on stdout.

    Each row is ``(cells, footnote)``; when footnote is non-empty, a
    ``  ↳ <reason>`` continuation line is printed under the row. This
    keeps the main grid's alignment tidy even when one bundle has a
    200-char corrupt reason attached.
    """
    widths = [len(h) for h in header]
    for cells, _ in rows:
        for i, cell in enumerate(cells):
            if i < len(widths) and len(cell) > widths[i]:
                widths[i] = len(cell)
    fmt_parts = []
    for i, w in enumerate(widths):
        # Right-align numeric-ish columns (entries, dim, wal); left-
        # align the rest. Cheap heuristic: by column index.
        if i in (1, 2, 5):
            fmt_parts.append(f"{{:>{w}}}")
        else:
            fmt_parts.append(f"{{:<{w}}}")
    fmt = "  ".join(fmt_parts)
    print(fmt.format(*header))
    for cells, footnote in rows:
        padded = cells + [""] * (len(header) - len(cells))
        print(fmt.format(*padded[: len(header)]))
        if footnote:
            print(f"  ↳ {footnote}")


def _cmd_bundle_list(args: argparse.Namespace) -> int:
    """Scan ``args.root`` up to depth 3 and print a bundle summary table.

    Exits 2 if the root doesn't exist or isn't a directory. An empty
    root returns 0 after printing just the header — makes ``watch -n 5
    soma bundle list`` pleasant on a fresh data dir.
    """
    from soma.bundle import list_bundles

    root = args.root
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    infos = list_bundles(root)
    header = ["PATH", "ENTRIES", "DIM", "BACKEND", "LAST-MODIFIED", "WAL"]
    rows = _format_rows(infos)
    _print_table(header, rows)
    return 0


def _cmd_bundle_info(args: argparse.Namespace) -> int:
    """Print a detailed view of a single bundle.

    Loads one :class:`BundleInfo` and additionally computes the total
    bytes on disk via a recursive walk. Exits 2 on a missing path, a
    non-bundle directory, or a corrupt bundle (reason printed to
    stderr).
    """
    from soma.bundle import is_bundle_dir, load_info

    path = args.path
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    if not is_bundle_dir(path):
        print(f"error: {path} is not a SOMA bundle", file=sys.stderr)
        return 2

    info = load_info(path)
    if info.corrupt:
        print(f"error: bundle is corrupt: {info.corrupt_reason}", file=sys.stderr)
        return 2

    total_bytes = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total_bytes += p.stat().st_size
        except OSError:
            continue

    print(f"path:          {info.path}")
    print(f"entries:       {info.entries:,}")
    print(f"embed_dim:     {info.embed_dim}")
    print(f"backend:       {info.backend}")
    print(f"last_modified: {info.last_modified.isoformat(timespec='seconds')}")
    if info.snapshot_ts:
        print(f"snapshot_ts:   {info.snapshot_ts}")
    print(f"wal_bytes:     {info.wal_bytes:,}  ({_format_bytes(info.wal_bytes)})")
    print(f"disk_bytes:    {total_bytes:,}  ({_format_bytes(total_bytes)})")
    return 0


def _cmd_bundle_delete(args: argparse.Namespace) -> int:
    """Remove a bundle directory, with a confirmation prompt by default.

    Safety: refuses (exit 2) if :func:`soma.bundle.is_bundle_dir` returns
    False. That check is what stops ``soma bundle delete ~`` from
    nuking a user's home dir even when ``--yes`` is set. With
    ``--yes`` we still run the check; we only skip the interactive
    y/N prompt.

    A ``N`` / empty / unrecognised reply at the prompt aborts with
    a message and exit 0 — the operator made a conscious choice, not
    an error we should report non-zero for.
    """
    import shutil

    from soma.bundle import is_bundle_dir, load_info

    path = args.path
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    if not is_bundle_dir(path):
        print(
            f"error: refusing to delete {path} — not a SOMA bundle",
            file=sys.stderr,
        )
        return 2

    info = load_info(path)
    entry_str = "CORRUPT" if info.corrupt else f"{info.entries:,}"
    if not args.yes:
        prompt = f"delete {path} with {entry_str} entries? [y/N] "
        try:
            reply = input(prompt)
        except EOFError:
            reply = ""
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 0

    try:
        shutil.rmtree(path)
    except OSError as exc:
        print(f"error: failed to delete {path}: {exc}", file=sys.stderr)
        return 2
    print(f"deleted {path}")
    return 0


def _cmd_auth_gc(_: argparse.Namespace) -> int:
    """Drop past-exp entries from the blocklist file.

    Prints the count removed (0 if nothing to do). Safe to cron —
    take the sibling ``.lock`` file + rewrite atomically via temp +
    replace.
    """
    from soma.auth_revocation import FileBlocklist

    path = _require_blocklist_path()
    if path is None:
        return 2

    removed = FileBlocklist(Path(path)).gc_expired()
    print(f"removed {removed} expired revocation(s)")
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

    # --- revocation subcommands (require SOMA_JWT_BLOCKLIST_PATH) ---
    p_auth_revoke = auth_sub.add_parser(
        "revoke",
        help="Revoke a JWT by its jti (append to the blocklist file)",
        description=(
            "Add a revocation record to SOMA_JWT_BLOCKLIST_PATH. Supply "
            "either --token (the full JWT; jti + exp are read from its "
            "claims) OR --jti with --exp (raw id + epoch seconds). "
            "--reason is free-form and capped at 256 chars."
        ),
    )
    p_auth_revoke.add_argument(
        "--token",
        help="Full JWT to revoke; jti + exp are pulled from signed claims",
    )
    p_auth_revoke.add_argument(
        "--jti",
        help="Raw jti value to revoke (use with --exp)",
    )
    p_auth_revoke.add_argument(
        "--exp",
        type=int,
        help="Epoch-seconds expiry for --jti mode (needed for GC)",
    )
    p_auth_revoke.add_argument(
        "--reason",
        default="",
        help="Free-form operator note; capped at 256 chars on disk",
    )
    p_auth_revoke.set_defaults(func=_cmd_auth_revoke)

    p_auth_list = auth_sub.add_parser(
        "list-revoked",
        help="Print one JSON object per live revocation (past-exp skipped)",
    )
    p_auth_list.set_defaults(func=_cmd_auth_list_revoked)

    p_auth_gc = auth_sub.add_parser(
        "gc",
        help="Rewrite the blocklist dropping entries whose exp is in the past",
    )
    p_auth_gc.set_defaults(func=_cmd_auth_gc)

    # --- `soma bundle` -------------------------------------------------
    p_bundle = sub.add_parser(
        "bundle",
        help="List, inspect, and delete on-disk SOMA bundles",
        description=(
            "Manage bundle directories on disk. `list` scans a root "
            "up to 3 levels deep; `info` shows a single bundle in "
            "detail; `delete` removes a bundle (with confirmation)."
        ),
    )
    bundle_sub = p_bundle.add_subparsers(dest="bundle_cmd", required=True)

    p_bundle_list = bundle_sub.add_parser(
        "list",
        help="Scan a directory and print one row per bundle",
    )
    p_bundle_list.add_argument(
        "root",
        nargs="?",
        default=Path("."),
        type=Path,
        help="Directory to scan (default: .)",
    )
    p_bundle_list.set_defaults(func=_cmd_bundle_list)

    p_bundle_info = bundle_sub.add_parser(
        "info",
        help="Detailed view of a single bundle",
    )
    p_bundle_info.add_argument("path", type=Path, help="Bundle directory")
    p_bundle_info.set_defaults(func=_cmd_bundle_info)

    p_bundle_delete = bundle_sub.add_parser(
        "delete",
        help="Remove a bundle directory (prompts unless --yes)",
    )
    p_bundle_delete.add_argument("path", type=Path, help="Bundle directory")
    p_bundle_delete.add_argument(
        "--yes",
        action="store_true",
        help="Skip the y/N prompt (still refuses non-bundle paths)",
    )
    p_bundle_delete.set_defaults(func=_cmd_bundle_delete)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
