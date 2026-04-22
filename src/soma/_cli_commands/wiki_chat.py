"""Wiki-chat demo: feed SOMA a folder of markdown (and PDFs), chat
about the contents.

End-to-end proof that ``MemoryLayer.with_sbert()`` + a local LLM gives
you a drop-in "chat with your wiki" experience at vector-DB parity. The
pipeline:

  1. **Ingest**: walk ``--wiki-dir`` for ``.md`` (and optionally
     ``.pdf``) files, split each into heading-aware chunks,
     ``mem.store(chunk, metadata={...})``, save bundle.
  2. **Chat**: reload bundle, run a REPL that retrieves top-k chunks
     per question and feeds them to an LLM with citation markers.

PDF ingestion uses ``pypdf`` if installed; otherwise PDFs are skipped
with a warning. Markdown is always supported (no extra deps).

Historical note: this module used to live at
``scripts/demo_wiki_chat.py``. It was moved here so ``pip install
soma-memory`` users who run ``soma index`` / ``soma chat`` don't hit
``ModuleNotFoundError: No module named 'scripts'`` — ``scripts/`` is
not part of the wheel. The repo-root file still exists as a thin
wrapper for ``python scripts/demo_wiki_chat.py …`` use from a clone.

Usage from a clone (still works via the scripts/ wrapper)::

    # Index a wiki directory (one-time, save the brain):
    python scripts/demo_wiki_chat.py --wiki-dir path/to/wiki --bundle my-brain/ --index

    # Chat against it (LLM via deploy tier=auto):
    python scripts/demo_wiki_chat.py --bundle my-brain/ --chat

    # Dry-run: skip the LLM, show the top-k retrieved chunks per query:
    python scripts/demo_wiki_chat.py --bundle my-brain/ --chat --dry-run

    # Index + chat in one shot:
    python scripts/demo_wiki_chat.py --wiki-dir path/to/wiki --bundle my-brain/ --index --chat

The chunker is intentionally simple (paragraph-level with heading
context) so it works on any well-formed markdown without external deps.
Swap in a richer splitter (LangChain ``RecursiveCharacterTextSplitter``,
LlamaIndex ``MarkdownNodeParser``, etc.) if you have one — the store
side doesn't care what produced the chunks.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from soma.llm import (
    DryRunBackend,
    LLMBackend,
    RAGSession,
    backend_from_env,
)
from soma.memory import MemoryLayer

MAX_CHUNK_CHARS = 1000  # ~250 tokens; above this, split with overlap
CHUNK_OVERLAP_CHARS = 120  # continuity between adjacent windows
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class WikiChunk:
    text: str
    path: str  # relative to wiki root
    heading: str  # "#" joined chain, e.g. "Architecture > Graph"


def _iter_doc_files(root: Path, *, include_pdf: bool) -> Iterator[Path]:
    patterns = ["*.md"]
    if include_pdf:
        patterns.append("*.pdf")
    seen: set[Path] = set()
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def _extract_pdf_text(path: Path) -> str | None:
    """Return PDF text joined with blank lines between pages; None on failure."""
    try:
        import pypdf
    except ImportError:
        return None
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception:
        return None
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _heading_chain(active: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in active) if active else ""


def _split_long_paragraph(text: str) -> list[str]:
    """Split text longer than ``MAX_CHUNK_CHARS`` into overlapping windows."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    windows: list[str] = []
    step = MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS
    for start in range(0, len(text), step):
        windows.append(text[start : start + MAX_CHUNK_CHARS])
        if start + MAX_CHUNK_CHARS >= len(text):
            break
    return windows


def chunk_markdown(text: str, *, path: str) -> list[WikiChunk]:
    """Split a markdown document into heading-aware paragraph chunks.

    - Tracks heading nesting so each chunk carries its section context.
    - Splits on blank-line paragraph boundaries.
    - Oversize paragraphs (>``MAX_CHUNK_CHARS``) get sliding-window split
      with ``CHUNK_OVERLAP_CHARS`` overlap for continuity.
    - Skips empty paragraphs and heading-only lines.
    """
    chunks: list[WikiChunk] = []
    active: list[tuple[int, str]] = []  # (level, title) stack
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        para = "\n".join(buffer).strip()
        buffer.clear()
        if not para:
            return
        heading = _heading_chain(active)
        for piece in _split_long_paragraph(para):
            chunks.append(WikiChunk(text=piece, path=path, heading=heading))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = HEADING_RE.match(line)
        if m is not None:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop any deeper-or-equal headings off the stack.
            while active and active[-1][0] >= level:
                active.pop()
            active.append((level, title))
            continue
        if line.strip() == "":
            flush()
            continue
        buffer.append(line)
    flush()
    return chunks


def _load_doc_text(path: Path) -> str | None:
    """Return the text body of a supported file, or None to skip."""
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _ingest(
    wiki_dir: Path,
    bundle_path: Path,
    *,
    include_pdf: bool = True,
    verbose: bool = True,
) -> tuple[int, int]:
    """Walk ``wiki_dir``, chunk every supported file, store into a fresh
    MemoryLayer, persist to ``bundle_path``. Returns (file_count, chunk_count)."""
    mem = MemoryLayer.with_sbert()
    file_count = 0
    chunk_count = 0
    for doc_path in _iter_doc_files(wiki_dir, include_pdf=include_pdf):
        rel = doc_path.relative_to(wiki_dir).as_posix()
        text = _load_doc_text(doc_path)
        if text is None:
            if verbose:
                reason = (
                    "pypdf not installed or unreadable PDF"
                    if doc_path.suffix.lower() == ".pdf"
                    else "non-utf8"
                )
                print(f"  [skip {reason}] {rel}")
            continue
        chunks = chunk_markdown(text, path=rel)
        if not chunks:
            continue
        file_count += 1
        for ch in chunks:
            mem.store(
                ch.text,
                metadata={"path": ch.path, "heading": ch.heading},
            )
            chunk_count += 1
        if verbose:
            print(f"  [indexed] {rel}: {len(chunks)} chunks")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    mem.save(bundle_path)
    if verbose:
        print(
            f"\nIndexed {file_count} files / {chunk_count} chunks "
            f"-> {bundle_path}\n"
        )
    return file_count, chunk_count


def _resolve_backend(backend_name: str, *, dry_run: bool) -> LLMBackend:
    if dry_run or backend_name == "dry-run":
        return DryRunBackend()
    if backend_name == "auto":
        return backend_from_env()
    return backend_from_env(prefer=backend_name)


def _chat(
    bundle_path: Path, *, backend_name: str, k: int, dry_run: bool
) -> None:
    from soma.cli import _chat_reply
    from soma.llm.rag import RAGAnswer

    print(f"Loading memory bundle from {bundle_path} ...")
    mem = MemoryLayer.load(bundle_path)
    print(f"  Loaded {len(mem)} chunks.\n")

    backend = _resolve_backend(backend_name, dry_run=dry_run)
    print(f"  LLM backend: {backend.name}\n")

    session = RAGSession(memory=mem, llm=backend, k=k)
    streaming = hasattr(backend, "stream_generate")
    print("Type a question, or 'quit' to exit. Ctrl-C also works.\n")
    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                break
            if streaming:
                hits = session._retrieve(question)
                prompt = session._build_prompt(question, hits)
                print("Assistant: ", end="", flush=True)
                try:
                    text = _chat_reply(
                        backend, prompt, max_tokens=session.max_tokens
                    )
                except KeyboardInterrupt:
                    continue
                answer = RAGAnswer(text=text, hits=hits, backend_name=backend.name)
                print()
            else:
                answer = session.ask(question)
                print(f"Assistant: {answer.text}\n")
            if answer.hits:
                print("  Sources:")
                for line in answer.cite_lines():
                    print(f"    {line}")
                print()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wiki-dir", type=Path, help="Root of markdown wiki (with --index)")
    p.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/demo-wiki-brain"),
        help="Where to save/load the memory bundle",
    )
    p.add_argument("--index", action="store_true", help="Ingest wiki into bundle")
    p.add_argument("--chat", action="store_true", help="Start chat REPL against bundle")
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "ollama", "openai", "anthropic", "openai-compat", "hf", "dry-run"),
        help=(
            "LLM backend (default: auto — picks ollama if running, else "
            "OpenAI/Anthropic if API key is set, else local HF model). "
            "Ignored with --dry-run."
        ),
    )
    p.add_argument("--k", type=int, default=5, help="Retrieval depth per query")
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDFs during ingestion (markdown only).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM; print top retrieved chunk per query instead.",
    )
    args = p.parse_args()

    if not (args.index or args.chat):
        p.error("pass --index, --chat, or both")

    if args.index:
        if args.wiki_dir is None:
            p.error("--index requires --wiki-dir")
        if not args.wiki_dir.is_dir():
            p.error(f"--wiki-dir {args.wiki_dir} is not a directory")
        print(f"=== Indexing {args.wiki_dir} -> {args.bundle} ===\n")
        _ingest(args.wiki_dir, args.bundle, include_pdf=not args.no_pdf)

    if args.chat:
        if not args.bundle.exists():
            p.error(f"--chat needs an existing bundle; {args.bundle} not found")
        print(f"=== Chat against {args.bundle} ===\n")
        _chat(args.bundle, backend_name=args.backend, k=args.k, dry_run=args.dry_run)
