"""Extract ```python``` fenced code blocks from markdown files.

Blocks are skipped if preceded by an HTML comment ``<!-- doctest: skip -->``.
Blocks are captured with their file path, line range, and contents so
pytest failures point at the doc, not at the extracted source.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocBlock:
    path: Path
    start_line: int
    end_line: int
    source: str

    @property
    def id(self) -> str:
        return f"{self.path.name}:{self.start_line}-{self.end_line}"


_FENCE = "```"
_SKIP_MARKER = "<!-- doctest: skip -->"


def extract_python_blocks(md_path: Path) -> list[DocBlock]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    blocks: list[DocBlock] = []
    in_block = False
    start_line = -1
    body: list[str] = []
    prev_non_blank = ""
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_block:
            if stripped.startswith(_FENCE + "python"):
                in_block = True
                skip = prev_non_blank.strip() == _SKIP_MARKER
                start_line = -1 if skip else idx
                body = []
                continue
            if stripped:
                prev_non_blank = line
        else:
            if stripped == _FENCE:
                if start_line > 0:
                    blocks.append(
                        DocBlock(
                            path=md_path,
                            start_line=start_line,
                            end_line=idx,
                            source="\n".join(body),
                        )
                    )
                in_block = False
                start_line = -1
                body = []
                prev_non_blank = line
                continue
            body.append(line)
    return blocks
