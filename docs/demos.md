# SOMA Demos

Ready-to-run scripts that exercise the most common agent-memory use
cases. Every demo uses only the public `MemoryLayer` API — copy the
script, point it at your data, swap the embedder/LLM if you want.

| Script | Use case | Needs LLM? |
| --- | --- | :---: |
| [`demo_memory_layer.py`](#1-the-tour-pure-api) | Pure API tour: store / retrieve / save / load / forget | no |
| [`demo_wiki_chat.py`](#2-chat-with-a-wiki-or-pdf-folder) | Index a folder of `.md` (and `.pdf`) and chat about it | yes (or `--dry-run`) |
| [`demo_chat_persistent.py`](#3-persistent-conversation) | Save chat history, reload, recall earlier facts | yes (or `--dry-run`) |
| [`demo_memory_inspect.py`](#4-inspect-and-search-a-bundle) | Browse / search / filter / dump any saved bundle | no |
| [`demo_related_browser.py`](#5-graph-walk-via-related) | Walk the entry-to-entry graph from any starting node | no |
| [`demo_web_ui.py`](#6-browser-based-chat-ui-gradio) | Gradio chat UI over any bundle | yes (or `--dry-run`) |

All demos work CPU-only. The LLM-backed ones load a small local model
(`soma.deploy.chat_head_factory`, tier `auto` picks Qwen2.5 / Phi-3 /
whatever fits your hardware). Use `--dry-run` to skip the LLM and just
see the retrieved chunks.

Bundle locations are arbitrary directories; we use `artifacts/<demo>/`
in the examples below but anything writeable works. Bundles are
portable — copy or rsync the directory and `MemoryLayer.load(path)`
elsewhere.

---

## 1. The tour (pure API)

```bash
python scripts/demo_memory_layer.py
```

Prints a guided walkthrough: build a layer, store five facts, retrieve
some, list the K most recent, fetch by id, forget one, save the
bundle, reload it. Useful as a copy-paste reference for what
`MemoryLayer` actually does without any LLM in the loop.

## 2. Chat with a wiki (or PDF folder)

Index a directory of markdown / PDFs into a fresh memory bundle, then
ask questions about its contents.

```bash
# 1. Ingest. Walks recursively, splits into heading-aware chunks.
python scripts/demo_wiki_chat.py \
  --wiki-dir path/to/your/wiki \
  --bundle artifacts/wiki-brain \
  --index

# 2. Chat (loads a local LLM via deploy tier=auto):
python scripts/demo_wiki_chat.py --bundle artifacts/wiki-brain --chat

# Or dry-run to see retrieved chunks without loading an LLM:
python scripts/demo_wiki_chat.py --bundle artifacts/wiki-brain --chat --dry-run

# Markdown-only (skip PDFs even if pypdf is installed):
python scripts/demo_wiki_chat.py --wiki-dir w/ --bundle b/ --index --no-pdf
```

The chunker is paragraph-level with heading-chain metadata
(`{"path": "design.md", "heading": "Architecture > Graph"}`).
Oversize paragraphs (>1000 chars) get split into overlapping windows.
Replace it with anything richer (LangChain
`RecursiveCharacterTextSplitter`, LlamaIndex `MarkdownNodeParser`)
without touching the store side.

## 3. Persistent conversation

Two-phase demo proving the product story: chat → save brain → reload
in a fresh process → the model remembers what you told it earlier.

```bash
# Full LLM:
python scripts/demo_chat_persistent.py --bundle artifacts/persistent-chat

# Dry-run (echoes the retrieved context as the "reply"):
python scripts/demo_chat_persistent.py --bundle artifacts/persistent-chat --dry-run
```

Phase 1 seeds five user facts ("Alex lives in Portland", etc.), runs
two short LLM turns that reference them, saves the bundle. Phase 2
reloads from disk and asks recall questions — the LLM grounds its
answer in the persisted memory.

## 4. Inspect and search a bundle

Read-only CLI for any saved bundle. Useful right after ingest to
sanity-check what got stored, or as a standalone vector-search tool
without spinning up an LLM.

```bash
# Quick stats: count, embed dim, on-disk size, metadata-key histogram.
python scripts/demo_memory_inspect.py stats --bundle artifacts/wiki-brain

# Most recent N entries (newest first).
python scripts/demo_memory_inspect.py recent --bundle artifacts/wiki-brain --n 20

# Filter by metadata. Multiple --where = AND.
python scripts/demo_memory_inspect.py filter --bundle artifacts/wiki-brain \
  --where path=architecture.md

# Vector search — top-k by cosine, no LLM.
python scripts/demo_memory_inspect.py search --bundle artifacts/wiki-brain \
  --query "how does consolidation work?" --k 5

# Dump everything as JSONL on stdout (no embeddings, just text + metadata).
python scripts/demo_memory_inspect.py dump --bundle artifacts/wiki-brain > store.jsonl
```

## 5. Graph walk via `.related()`

SOMA exposes entry-to-entry similarity (`mem.related(node_id)`) on top
of the query-to-entry retrieve path. This demo lets you start at one
entry and traverse its neighborhood interactively — useful for
finding tangential context an LLM might miss when it only sees a
query's top-k.

```bash
# Seed by query (top-1 cosine becomes the cursor):
python scripts/demo_related_browser.py --bundle artifacts/wiki-brain \
  --seed "consolidation cycle"

# Or seed by node id (full or unique prefix):
python scripts/demo_related_browser.py --bundle artifacts/wiki-brain \
  --seed-id 3a4b5c
```

At each step the cursor entry is shown with its k nearest neighbors.
Pick a neighbor by number to move the cursor, `0` to quit.

## 6. Browser-based chat UI (Gradio)

One-file web UI for non-Python users — a chat box over any bundle,
with retrieved sources displayed inline. Uses the same
`backend_from_env` path as the CLI so any LLM works.

```bash
pip install gradio

# Auto-pick LLM + simple chat:
python scripts/demo_web_ui.py --bundle my-brain/

# With recall boosters:
python scripts/demo_web_ui.py --bundle my-brain/ \
  --hybrid-alpha 0.3 --rerank-top-n 20

# No LLM — just inspect what gets retrieved:
python scripts/demo_web_ui.py --bundle my-brain/ --dry-run
```

Opens at `http://127.0.0.1:7860`. Pass `--share` for a public Gradio
share link (useful for quick demos).

---

## Patterns the demos cover

Every common agent-memory pattern shows up at least once across the
five demos:

- **Ingest from disk** → `demo_wiki_chat.py` (markdown + PDF)
- **Conversational persistence** → `demo_chat_persistent.py`
- **Vector search** → `demo_memory_inspect.py search`
- **Metadata filtering** → `demo_memory_inspect.py filter`
- **Recency view** → `demo_memory_inspect.py recent`
- **Bundle stats / sizing** → `demo_memory_inspect.py stats`
- **Bulk export** → `demo_memory_inspect.py dump`
- **Entry-to-entry similarity** → `demo_related_browser.py`
- **Save / load** → all of them
- **LLM-grounded answer with citations** → `demo_wiki_chat.py --chat`

For deployment patterns (REST API, Docker), see
[`README.md` § REST API](../README.md) and `docker-compose.yml`.

For benchmarks comparing SOMA against Chroma at scale, see
[`benchmarks/reports/`](../benchmarks/reports/).
