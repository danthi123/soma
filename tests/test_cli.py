"""Tests for the soma CLI entry point."""

from __future__ import annotations

import base64
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from soma.cli import build_parser, main


def test_parser_has_all_subcommands() -> None:
    p = build_parser()
    sub_actions = [a for a in p._actions if a.dest == "cmd"]
    assert sub_actions
    choices = sub_actions[0].choices
    assert {
        "index",
        "chat",
        "stats",
        "search",
        "forget",
        "serve",
        "version",
        "auth",
    } <= set(choices)


def test_parser_index_requires_wiki_and_bundle() -> None:
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["index"])
    with pytest.raises(SystemExit):
        p.parse_args(["index", "--bundle", "x"])
    args = p.parse_args(["index", "--bundle", "b", "--wiki", "w"])
    assert args.cmd == "index"
    assert args.bundle == Path("b")
    assert args.wiki == Path("w")
    assert args.no_pdf is False


def test_parser_chat_defaults_to_auto_backend() -> None:
    args = build_parser().parse_args(["chat", "--bundle", "b"])
    assert args.backend == "auto"
    assert args.k == 5
    assert args.dry_run is False


def test_parser_chat_rejects_unknown_backend() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["chat", "--bundle", "b", "--backend", "claude"])


def test_parser_search_requires_query() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["search", "--bundle", "b"])


def test_version_prints_something(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out  # version string or "unknown"


def test_index_returns_2_for_missing_wiki(tmp_path: Path) -> None:
    err_buf = io.StringIO()
    with redirect_stderr(err_buf):
        rc = main(
            [
                "index",
                "--bundle",
                str(tmp_path / "out"),
                "--wiki",
                str(tmp_path / "missing"),
            ]
        )
    assert rc == 2
    assert "is not a directory" in err_buf.getvalue()


def test_chat_returns_2_for_missing_bundle(tmp_path: Path) -> None:
    err_buf = io.StringIO()
    with redirect_stderr(err_buf):
        rc = main(["chat", "--bundle", str(tmp_path / "missing")])
    assert rc == 2
    assert "not found" in err_buf.getvalue()


def test_stats_runs_against_real_bundle(tmp_path: Path) -> None:
    """End-to-end: build a tiny bundle with TextEncoder so save/load round-
    trips without needing the embed_fn rebound externally."""
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.memory import MemoryLayer

    tok = train_bpe_tokenizer(["alpha beta gamma"], vocab_size=32)
    enc = TextEncoder(tok, embed_dim=8, max_seq_len=16)
    mem = MemoryLayer(tokenizer=tok, encoder=enc)
    mem.store("alpha", metadata={"source": "x"})
    mem.store("beta", metadata={"source": "y"})
    bundle = tmp_path / "b"
    mem.save(bundle)

    out_buf = io.StringIO()
    with redirect_stdout(out_buf):
        rc = main(["stats", "--bundle", str(bundle)])
    assert rc == 0
    out = out_buf.getvalue()
    assert "entries: 2" in out
    assert "embed_dim: 8" in out


def test_forget_deletes_entry_by_full_id(tmp_path: Path) -> None:
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.memory import MemoryLayer

    tok = train_bpe_tokenizer(["alpha beta gamma"], vocab_size=32)
    enc = TextEncoder(tok, embed_dim=8, max_seq_len=16)
    mem = MemoryLayer(tokenizer=tok, encoder=enc)
    keep = mem.store("alpha")
    drop = mem.store("beta")
    bundle = tmp_path / "b"
    mem.save(bundle)

    rc = main(["forget", "--bundle", str(bundle), "--node-id", drop])
    assert rc == 0

    reloaded = MemoryLayer.load(bundle)
    assert keep in reloaded
    assert drop not in reloaded


def test_forget_accepts_unique_prefix(tmp_path: Path) -> None:
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.memory import MemoryLayer

    tok = train_bpe_tokenizer(["alpha beta"], vocab_size=32)
    enc = TextEncoder(tok, embed_dim=8, max_seq_len=16)
    mem = MemoryLayer(tokenizer=tok, encoder=enc)
    target = mem.store("alpha")
    bundle = tmp_path / "b"
    mem.save(bundle)

    rc = main(["forget", "--bundle", str(bundle), "--node-id", target[:6]])
    assert rc == 0
    assert target not in MemoryLayer.load(bundle)


def test_forget_rejects_unknown_id(tmp_path: Path) -> None:
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.memory import MemoryLayer

    tok = train_bpe_tokenizer(["alpha"], vocab_size=32)
    enc = TextEncoder(tok, embed_dim=8, max_seq_len=16)
    mem = MemoryLayer(tokenizer=tok, encoder=enc)
    mem.store("alpha")
    bundle = tmp_path / "b"
    mem.save(bundle)

    err_buf = io.StringIO()
    with redirect_stderr(err_buf):
        rc = main(["forget", "--bundle", str(bundle), "--node-id", "definitely-not-real"])
    assert rc == 2
    assert "not found" in err_buf.getvalue()


# ------------------------------------------------------------------
# Phase 4 — `soma auth` subcommands
# ------------------------------------------------------------------
_TEST_SECRET = "cli-test-secret-not-used-elsewhere"


def test_cli_auth_issue_prints_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SOMA_JWT_SECRET", _TEST_SECRET)
    rc = main(
        [
            "auth",
            "issue",
            "--sub",
            "alex",
            "--bundle",
            "alex:read,write",
            "--expires",
            "30d",
        ]
    )
    out = capsys.readouterr().out.strip()
    assert rc == 0
    # Token is three dot-separated base64url segments.
    parts = out.split(".")
    assert len(parts) == 3, f"not a JWT: {out!r}"

    # Round-trip via the same library to verify claims landed right.
    from soma.auth import verify_token

    principal = verify_token(out, secret=_TEST_SECRET)
    assert principal.sub == "alex"
    assert principal.bundles == {"alex": ["read", "write"]}


def test_cli_auth_issue_refuses_without_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SOMA_JWT_SECRET", raising=False)
    monkeypatch.delenv("SOMA_JWT_PRIVATE_KEY_PATH", raising=False)
    rc = main(["auth", "issue", "--sub", "x", "--expires", "60m"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "SOMA_JWT_SECRET" in err


def test_cli_auth_verify_prints_claims(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from datetime import timedelta

    from soma.auth import issue_token

    token = issue_token(
        sub="verified-user",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=_TEST_SECRET,
    )
    monkeypatch.setenv("SOMA_JWT_SECRET", _TEST_SECRET)
    rc = main(["auth", "verify", "--token", token])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["sub"] == "verified-user"
    assert data["bundles"] == {"alex": ["read"]}


def test_cli_auth_verify_bad_token_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SOMA_JWT_SECRET", _TEST_SECRET)
    rc = main(["auth", "verify", "--token", "not.a.real.jwt"])
    err = capsys.readouterr().err
    assert rc == 2
    assert err  # non-empty error message


def test_cli_auth_rotate_secret_prints_high_entropy_secret(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["auth", "rotate-secret"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert len(out) >= 43
    decoded = base64.urlsafe_b64decode(out + "=" * (-len(out) % 4))
    assert len(decoded) >= 32


def test_cli_auth_issue_supports_multiple_bundles(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SOMA_JWT_SECRET", _TEST_SECRET)
    rc = main(
        [
            "auth",
            "issue",
            "--sub",
            "multi",
            "--bundle",
            "alex:read,write",
            "--bundle",
            "bobbi:read",
            "--expires",
            "24h",
        ]
    )
    assert rc == 0
    token = capsys.readouterr().out.strip()

    from soma.auth import verify_token

    principal = verify_token(token, secret=_TEST_SECRET)
    assert principal.bundles == {"alex": ["read", "write"], "bobbi": ["read"]}


def test_cli_auth_issue_empty_bundles_scope_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Default scope is empty — token with no bundles map is still valid
    # (admin elsewhere via the legacy API key path, or rejected cleanly).
    monkeypatch.setenv("SOMA_JWT_SECRET", _TEST_SECRET)
    rc = main(["auth", "issue", "--sub", "empty", "--expires", "7d"])
    assert rc == 0
    token = capsys.readouterr().out.strip()

    from soma.auth import verify_token

    principal = verify_token(token, secret=_TEST_SECRET)
    assert principal.sub == "empty"
    assert principal.bundles == {}


def test_cli_auth_issue_expires_parse_variants(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SOMA_JWT_SECRET", _TEST_SECRET)
    for spec in ("30d", "7d", "24h", "60m"):
        rc = main(["auth", "issue", "--sub", "x", "--expires", spec])
        capsys.readouterr()  # drain between runs
        assert rc == 0, f"failed to parse --expires {spec!r}"


def test_cli_auth_issue_rejects_bad_expires(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SOMA_JWT_SECRET", _TEST_SECRET)
    rc = main(["auth", "issue", "--sub", "x", "--expires", "plenty"])
    err = capsys.readouterr().err
    assert rc != 0
    assert err


# ------------------------------------------------------------------
# `soma auth revoke` / `list-revoked` / `gc`
# ------------------------------------------------------------------
def test_cli_auth_revoke_adds_to_blocklist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`soma auth revoke --token <t> --reason ...` lands a record on disk."""
    from datetime import timedelta

    from soma.auth import issue_token, verify_token

    bl_path = tmp_path / "bl.jsonl"
    monkeypatch.setenv("SOMA_JWT_SECRET", _TEST_SECRET)
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(bl_path))

    token = issue_token(
        sub="alex",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=_TEST_SECRET,
    )
    rc = main(
        [
            "auth",
            "revoke",
            "--token",
            token,
            "--reason",
            "leaked in test",
        ]
    )
    assert rc == 0
    # Drain CLI stdout so subsequent capsys calls see clean output.
    capsys.readouterr()

    # Record is on disk, keyed by the token's jti.
    principal = verify_token(token, secret=_TEST_SECRET)
    assert principal.jti is not None
    contents = bl_path.read_text(encoding="utf-8")
    assert principal.jti in contents
    assert "leaked in test" in contents


def test_cli_auth_revoke_accepts_jti_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`soma auth revoke --jti <j> --exp <ts> --reason ...` when the token is
    already discarded (operator only has the jti from a log)."""
    import time

    bl_path = tmp_path / "bl.jsonl"
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(bl_path))

    future = int(time.time()) + 3600
    rc = main(
        [
            "auth",
            "revoke",
            "--jti",
            "manual-jti-xyz",
            "--exp",
            str(future),
            "--reason",
            "raw id only",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    contents = bl_path.read_text(encoding="utf-8")
    assert "manual-jti-xyz" in contents
    assert str(future) in contents


def test_cli_auth_list_revoked_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`soma auth list-revoked` prints one JSON per line for each entry."""
    import time

    bl_path = tmp_path / "bl.jsonl"
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(bl_path))

    now = int(time.time())
    for i in range(2):
        rc = main(
            [
                "auth",
                "revoke",
                "--jti",
                f"jti-{i}",
                "--exp",
                str(now + 3600),
                "--reason",
                f"r{i}",
            ]
        )
        assert rc == 0
        capsys.readouterr()  # drain

    rc = main(["auth", "list-revoked"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    parsed = [json.loads(line) for line in out]
    jtis = {row["jti"] for row in parsed}
    assert jtis == {"jti-0", "jti-1"}


def test_cli_auth_gc_removes_expired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`soma auth gc` drops past-exp entries and prints the count."""
    import time

    bl_path = tmp_path / "bl.jsonl"
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(bl_path))

    now = int(time.time())
    # One live, one dead.
    main(["auth", "revoke", "--jti", "live", "--exp", str(now + 3600), "--reason", "ok"])
    capsys.readouterr()
    main(["auth", "revoke", "--jti", "dead", "--exp", str(now - 3600), "--reason", "old"])
    capsys.readouterr()

    rc = main(["auth", "gc"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    # Output mentions the removed count.
    assert "1" in out

    # The surviving file only has the live entry.
    surviving = [
        json.loads(line)
        for line in bl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(surviving) == 1
    assert surviving[0]["jti"] == "live"


def test_cli_auth_revoke_requires_blocklist_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without SOMA_JWT_BLOCKLIST_PATH, revoke errors out with a clear hint."""
    import time

    monkeypatch.delenv("SOMA_JWT_BLOCKLIST_PATH", raising=False)
    rc = main(
        [
            "auth",
            "revoke",
            "--jti",
            "x",
            "--exp",
            str(int(time.time()) + 60),
            "--reason",
            "no path set",
        ]
    )
    err = capsys.readouterr().err
    assert rc != 0
    assert "SOMA_JWT_BLOCKLIST_PATH" in err
