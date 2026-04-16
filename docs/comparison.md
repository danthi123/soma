# SOMA vs the Memory-Layer Field

A practical, opinionated comparison for picking an agent memory layer
in 2026. Numbers are from `benchmarks/reports/` where measured;
qualitative claims link to the source product's docs.

> **TL;DR**: pick **SOMA** if you want a single-file, local-first,
> vector-DB-quality memory layer with the lowest store/disk overhead at
> scale. Pick **Mem0/Letta/Zep** if you want managed multi-tenant with
> built-in summarization. Pick **Chroma/Qdrant/LanceDB** if you only
> need vector search and don't care about graph/working-memory or
> conversational ergonomics.

## Quick decision matrix

| If you want… | Use |
| --- | --- |
| Local-first, one-directory "brain", lowest overhead | **SOMA** |
| Hosted multi-tenant memory + summarization | Mem0 |
| Long-running stateful agent with paged memory | Letta (formerly MemGPT) |
| Temporal knowledge graph + summarization | Zep |
| Pure vector search (no agent ergonomics) | Chroma / Qdrant / LanceDB |
| Cloud-managed vector DB (no infra) | Pinecone / Weaviate Cloud |

## Feature matrix

| Capability | SOMA | Chroma | Mem0 | Letta | Zep | Pinecone |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Vector retrieval | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Local-first / offline | ✅ | ✅ | ⚠️¹ | ✅ | ⚠️¹ | ❌ |
| Zero deps install (pip + 1 file) | ✅ | ⚠️² | ⚠️² | ⚠️² | ❌ | ❌ |
| Single-directory portable brain | ✅ | ⚠️³ | ❌ | ⚠️ | ❌ | ❌ |
| Working-memory window | ✅ | ❌ | ⚠️ | ✅ | ❌ | ❌ |
| Episodic-memory store | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Plastic-graph substrate (in-place) | ✅⁴ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Built-in LLM-driven summarization | ❌⁵ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Temporal/recency awareness | ✅ | ⚠️ | ✅ | ✅ | ✅ | ⚠️ |
| LLM-agnostic (swap models) | ✅ | ✅ | ⚠️⁶ | ⚠️⁶ | ⚠️⁶ | ✅ |
| Plug-and-play LLM backends | ✅⁷ | ❌ | ✅ | ✅ | ✅ | ❌ |
| LangChain / LlamaIndex retrievers | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| REST API | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Open source, MIT/Apache | ✅ MIT | ✅ Apache | ✅ Apache | ✅ Apache | ⚠️ AGPL | ❌ closed |

¹ Open-source self-host available; defaults to managed cloud.
² Pulls SQLite + bindings; SOMA's bundle is just JSON + a torch tensor.
³ Chroma is a directory but with SQLite + WAL + lockfiles.
⁴ Substrate ships and runs; currently doesn't move retrieval scores
  on synthetic small corpora — see paper-draft §4.1 for the honest
  ablation. Open research direction.
⁵ Out of scope by design — bring your own LLM, see
  `docs/llm-backends.md`. Summarization is one prompt away.
⁶ Tied to specific LLM providers; check current docs.
⁷ Ships 5 backends (Ollama / OpenAI / Anthropic / OpenAI-compatible /
  HuggingFace) with one-line auto-detect; see `docs/llm-backends.md`.

## Measured: SOMA vs Chroma at scale

From `benchmarks/reports/`. Same sbert embedder
(`all-MiniLM-L6-v2`, 384-d) on both sides — only index/storage
mechanics differ.

### Full-pipeline (1K / 5K / 20K, includes embed cost)

`scale_vs_chroma.md`:

| N | SOMA-flat | SOMA-hnsw | Chroma | SOMA win |
| ---: | ---: | ---: | ---: | --- |
| Store ms/op | 8–9 | 8–9 | 26–29 | **3.2–3.6× faster** |
| Disk @ 50 facts | 85 KB | — | 1920 KB | **22.6× smaller** |
| Disk @ 20K | 32.7 MB | 32.7 MB | 45.2 MB | **1.4× smaller** |
| Retrieve ms (HNSW backend) | — | 8.85 | 10.7 | **1.18–1.21× faster** |

### Index-only (5K / 20K / 100K, embed amortized)

`scale_enterprise_*.md`:

| N | SOMA-flat store | Chroma store | SOMA win | SOMA-hnsw retrieve | Chroma retrieve | SOMA win |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 5K | <0.1s | 83.1s | ~1000× | 5.81 ms | 8.91 ms | 1.53× |
| 20K | 0.1s | 5.3 min | ~3180× | 6.34 ms | 8.44 ms | 1.33× |
| 100K | 0.4s | 23.6 min | ~3535× | 5.58 ms | 28.56 ms | **5.12×** |

### Quality (real conversational data)

LoCoMo (Maharana 2024 — 10 long convs, 5,882 turns, 1,986 questions
with gold-evidence annotations):

| System | R@1 | R@5 | R@10 | Retrieve ms |
| --- | :---: | :---: | :---: | :---: |
| SOMA-flat | **0.098** | **0.238** | **0.285** | 17.04 |
| SOMA-hnsw | 0.094 | 0.231 | 0.277 | **9.52** |
| Chroma | 0.096 | 0.235 | 0.281 | 11.87 |

Same sbert embedder → quality parity by construction. SOMA wins on
mechanics (store time, retrieve time, disk).

## Honest comparison vs Mem0 / Letta / Zep

We haven't yet run head-to-head benchmarks against Mem0 / Letta / Zep
because each requires either a managed account or LLM API access for
their summarization paths (deferred until those integrations are
written; see `paper-draft.md` §5). The qualitative tradeoffs:

- **Mem0** runs an LLM at every store call to summarize and decide
  what's worth remembering. SOMA stores everything cheaply and lets
  retrieval pick what matters at query time. Different bet: Mem0 is
  *opinionated* about what to remember; SOMA is *neutral* and lets
  the embedder + your LLM decide at retrieve time.
- **Letta (MemGPT)** is a paged-memory architecture with explicit
  recall/archival operations driven by the LLM. SOMA is a flat
  vector store with optional graph metadata. Letta is a better fit
  if your agent needs to *manage* its own memory deliberately; SOMA
  is a better fit if you want memory to "just work" behind a
  retrieval call.
- **Zep** is a managed service centered on temporal knowledge
  graphs. SOMA's plastic graph is a substrate not yet doing
  retrieval-altering work. If "show me what changed about user X
  last week" is your core query, Zep's temporal indexing is more
  mature. If "find context relevant to this question" is your core
  query, SOMA's vector path is faster and cheaper.

Concrete benchmarks against these systems are pre-scoped in
`paper-draft.md` §5 and will land once an LLM-judge harness is wired.

## When NOT to pick SOMA

Be honest:

- **You want managed multi-tenant out of the box** — SOMA is a Python
  library + REST server you self-host. No hosted plan exists yet.
- **You want LLM-driven summarization built-in** — SOMA leaves that to
  the caller's LLM. If you want "store these chat turns and have the
  system decide what to keep," Mem0/Letta/Zep have that today.
- **You only need vector search and don't care about API ergonomics** —
  Chroma/Qdrant/LanceDB are mature, well-documented, and cheap.
  SOMA's win at small N is real (22× disk, 3× store) but if you're
  storing <1K vectors that's pennies regardless.
- **You need exotic vector indices** — SOMA ships flat + HNSW. If
  you need IVF / PQ / OPQ (compressed indexes for billion-scale),
  use FAISS or Milvus directly.

## Migration

- **From Chroma**: see `scripts/migrate_chroma.py`. Reads any Chroma
  PersistentClient bundle, copies docs+metadata+embeddings into a
  fresh MemoryLayer bundle. Tested on the 50-fact retrieval benchmark
  set; quality preserved.
- **From plain RAG (langchain + chroma)**: drop in
  `soma.integrations.langchain.SomaRetriever` — same `BaseRetriever`
  interface, swap the constructor.
- **From Mem0 / Letta / Zep**: no automatic migration today. The
  scripted path is "export their memory as JSONL, loop
  `mem.store(text, metadata=...)` over each row."

## Reproducibility

Every number above comes from a script + report committed to this
repo:

```bash
pip install -e '.[dev]' sentence-transformers chromadb
python -m benchmarks.run_retrieval                # quality
python -m benchmarks.run_scale_vs_chroma          # 1K/5K/20K full pipeline
python -m benchmarks.run_scale_enterprise --n 100000  # index-only @ enterprise
python -m benchmarks.run_locomo                   # real conversations
```

Reports land in `benchmarks/reports/`. Numbers should reproduce
within floating-point reordering on a CPU-only laptop in ~30 min.
