"""Tests for the wiki-chat demo's chunker and ingester."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.demo_wiki_chat import (
    MAX_CHUNK_CHARS,
    _iter_doc_files,
    _load_doc_text,
    chunk_markdown,
)


def test_chunker_splits_paragraphs() -> None:
    text = "Para one line.\n\nPara two lines.\nSecond line of two."
    chunks = chunk_markdown(text, path="x.md")
    assert len(chunks) == 2
    assert chunks[0].text == "Para one line."
    assert chunks[1].text == "Para two lines.\nSecond line of two."


def test_chunker_tracks_heading_chain() -> None:
    text = """# Top
intro para under top

## Sub A
under sub A

### Deep
deep content

## Sub B
under sub B
"""
    chunks = chunk_markdown(text, path="doc.md")
    headings = [c.heading for c in chunks]
    assert headings == [
        "Top",
        "Top > Sub A",
        "Top > Sub A > Deep",
        "Top > Sub B",
    ]


def test_chunker_pops_to_correct_level() -> None:
    """Going from H3 to H2 should drop the H3 from the chain."""
    text = """# A
para a

## B
para b

### C
para c

## D
para d
"""
    chunks = chunk_markdown(text, path="doc.md")
    headings = [c.heading for c in chunks]
    assert headings == ["A", "A > B", "A > B > C", "A > D"]


def test_chunker_skips_empty_paragraphs() -> None:
    text = "\n\n\n\nonly paragraph\n\n\n\n"
    chunks = chunk_markdown(text, path="x.md")
    assert len(chunks) == 1
    assert chunks[0].text == "only paragraph"


def test_chunker_no_heading_means_empty_string() -> None:
    text = "no heading here, just text"
    chunks = chunk_markdown(text, path="x.md")
    assert len(chunks) == 1
    assert chunks[0].heading == ""


def test_chunker_handles_empty_document() -> None:
    assert chunk_markdown("", path="x.md") == []
    assert chunk_markdown("\n\n\n", path="x.md") == []
    assert chunk_markdown("# Heading only\n", path="x.md") == []


def test_chunker_splits_oversize_paragraph_with_overlap() -> None:
    big = "x" * (MAX_CHUNK_CHARS * 2 + 50)
    chunks = chunk_markdown(big, path="x.md")
    # Multiple windows produced, none exceeds MAX_CHUNK_CHARS.
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS


def test_chunker_carries_path_metadata() -> None:
    chunks = chunk_markdown("# H\n\npara", path="folder/file.md")
    assert all(c.path == "folder/file.md" for c in chunks)


def test_iter_doc_files_finds_md_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "sub" / "c.txt").write_text("c", encoding="utf-8")
    found = list(_iter_doc_files(tmp_path, include_pdf=False))
    rels = sorted(p.relative_to(tmp_path).as_posix() for p in found)
    assert rels == ["a.md", "sub/b.md"]


def test_iter_doc_files_includes_pdf_when_requested(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 stub")
    found_with = list(_iter_doc_files(tmp_path, include_pdf=True))
    found_without = list(_iter_doc_files(tmp_path, include_pdf=False))
    assert len(found_with) == 2
    assert len(found_without) == 1


def test_load_doc_text_returns_none_on_non_utf8(tmp_path: Path) -> None:
    p = tmp_path / "bin.md"
    p.write_bytes(b"\xff\xfe\x00\x00invalid")
    assert _load_doc_text(p) is None


def test_load_doc_text_returns_text_on_md(tmp_path: Path) -> None:
    p = tmp_path / "ok.md"
    p.write_text("# hello\n\npara", encoding="utf-8")
    assert _load_doc_text(p) == "# hello\n\npara"


def test_load_doc_text_returns_none_for_corrupt_pdf(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not actually a pdf at all")
    assert _load_doc_text(p) is None
