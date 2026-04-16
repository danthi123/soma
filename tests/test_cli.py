"""Tests for the soma CLI entry point."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from soma.cli import build_parser, main


def test_parser_has_all_subcommands() -> None:
    p = build_parser()
    sub_actions = [a for a in p._actions if a.dest == "cmd"]
    assert sub_actions
    choices = sub_actions[0].choices
    assert {"index", "chat", "stats", "search", "serve", "version"} <= set(choices)


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
