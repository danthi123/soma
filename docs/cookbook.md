# SOMA Cookbook — Recipes for Common Patterns

Drop-in code snippets for the patterns most agent-memory callers
need. All recipes use only the public API; copy them and adapt.

## 1. Ingest a wiki and chat

```python
from pathlib import Path
from soma.llm    import OllamaBackend, RAGSession
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()
for md in Path("wiki/").rglob("*.md"):
    for para in md.read_text(encoding="utf-8").split("\n\n"):
        if para.strip():
            mem.store(para.strip(), metadata={"path": str(md)})
mem.save("brain/")

chat = RAGSession(memory=mem, llm=OllamaBackend(model="llama3.2"))
print(chat.ask("how do we deploy?").text)
```

(For real use, prefer `soma index --wiki ... --bundle ...` from the
CLI — it does heading-aware chunking with overlap.)

## 2. Persistent chat session

```python
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()  # or .load("brain/") to resume

# Each turn, store both sides so future turns can recall them.
def turn(user_msg: str, llm) -> str:
    hits = mem.retrieve(user_msg, k=5)
    ctx  = "\n".join(f"- {h.text}" for h in hits)
    reply = llm.generate(
        f"Context from prior turns:\n{ctx}\n\nUser: {user_msg}\nAssistant:"
    )
    mem.store(user_msg, metadata={"role": "user"})
    mem.store(reply,    metadata={"role": "assistant"})
    return reply

mem.save("brain/")  # at end of session
```

## 3. Per-user memory (multi-tenancy by metadata)

```python
mem = MemoryLayer.with_sbert()  # one shared store

mem.store("Alex prefers vegetarian recipes",  metadata={"user_id": "alex"})
mem.store("Bobbi prefers seafood",            metadata={"user_id": "bobbi"})

# In-index pre-filter (same semantics as Chroma's `where`):
alex_hits = mem.retrieve("food preferences?", k=5, where={"user_id": "alex"})

# Multi-field = AND; value-list via $in; numeric comparisons:
hits = mem.retrieve(
    "q", k=5,
    where={"user_id": {"$in": ["alex", "bobbi"]}, "priority": {"$gte": 3}},
)
```

If you'd rather isolate physically, use one bundle per user:

```python
alex_mem  = MemoryLayer.load("brains/alex/")
bobbi_mem = MemoryLayer.load("brains/bobbi/")
```

## 4. Soft-forget by metadata (GDPR / right-to-be-forgotten)

```python
to_forget = [
    h.node_id for h in mem.get_recent(len(mem))  # full scan
    if h.metadata.get("user_id") == "alex"
]
for node_id in to_forget:
    mem.forget(node_id)
mem.save("brain/")
```

## 5. Importance-weighted store (mark some entries critical)

```python
# Pre-store, weight by source confidence (only metadata; retrieval
# still ranks by cosine — use this on the read side):
mem.store(text, metadata={"source": "policy-doc", "priority": "high"})
mem.store(other_text, metadata={"source": "user-chat", "priority": "low"})

# At retrieve time, re-rank by prior:
def boosted(query: str, k: int = 5):
    hits = mem.retrieve(query, k=k * 3)  # over-fetch
    boost = {"high": 1.2, "low": 0.9}
    for h in hits:
        h_score = h.score * boost.get(h.metadata.get("priority", ""), 1.0)
        h.metadata["_boosted_score"] = h_score
    return sorted(
        hits, key=lambda h: h.metadata["_boosted_score"], reverse=True
    )[:k]
```

## 6. Hybrid retrieval — vector + recency

```python
def vector_plus_recent(query: str, k: int = 5, recency_weight: float = 0.3):
    semantic = mem.retrieve(query, k=k * 2)
    recent   = mem.get_recent(k * 2)
    seen, out = set(), []
    for h in semantic:
        out.append(h)
        seen.add(h.node_id)
        if h.score < recency_weight:
            break
    for h in recent:
        if h.node_id not in seen:
            out.append(h)
        if len(out) >= k:
            break
    return out[:k]
```

## 7. Streaming ingest from JSONL

```python
import json
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()
with open("export.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        mem.store(rec["text"], metadata=rec.get("metadata", {}))
        if mem._step % 1000 == 0:  # snapshot every 1K
            mem.save("brain/")
mem.save("brain/")
```

## 8. Use any OpenAI-compatible local server (vLLM / LM Studio / llama.cpp)

```python
from soma.llm import OpenAICompatibleBackend, RAGSession

backend = OpenAICompatibleBackend(
    model="qwen2.5-7b-instruct",
    base_url="http://localhost:8000/v1",  # vLLM
)
chat = RAGSession(memory=mem, llm=backend)
```

## 9. Custom prompt template

```python
from soma.llm import RAGSession

PROMPT = """\
You are a code-aware assistant. The user's repo has been indexed
into memory. Use only the snippets below; cite [n] inline. If the
snippets don't cover the question, say so and suggest what to grep.

Snippets:
{context}

Question: {question}

Reply (cite like [1]):"""

chat = RAGSession(memory=mem, llm=backend, prompt_template=PROMPT)
```

## 10. Migrate from Chroma

```bash
python scripts/migrate_chroma.py \
  --chroma-dir path/to/chroma-store/ \
  --soma-dir   path/to/new-soma-bundle/
```

The script reads docs + metadata + embeddings from Chroma's
PersistentClient, writes them into a fresh MemoryLayer bundle. Same
embedder = same retrieval results, just a different store.

## 11. Run as a background REST API

```bash
soma serve --port 8420
# then
curl -X POST http://localhost:8420/store \
  -H 'Content-Type: application/json' \
  -d '{"text": "user lives in Portland", "metadata": {"src":"chat"}}'
curl -X POST http://localhost:8420/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "where does the user live?", "k": 3}'
```

## 12. Hybrid lexical + vector retrieval

When queries hinge on specific terminology (proper names, domain
jargon, numeric IDs) that sbert's sub-word tokenizer smears into a
broader semantic neighbourhood, add BM25 alongside cosine:

```python
# alpha=0 → pure BM25; alpha=1 → pure cosine; 0.3–0.5 is typical.
hits = mem.retrieve("reading list for Kubernetes RBAC", k=5, hybrid_alpha=0.3)
```

BM25 index is built lazily and invalidated when new entries are
stored. No extra deps.

## 13. Cross-encoder re-ranking

Over-fetch cosine candidates, re-rank them with a small
cross-encoder (~5–10 ms/candidate on CPU). Usually +5–15% Recall@5
on real queries:

```python
from soma.memory.rerank import CrossEncoderReranker

mem.attach_reranker(CrossEncoderReranker())  # lazy model load
hits = mem.retrieve("...", k=5, rerank_top_n=20)

# Combine with hybrid for best of both:
hits = mem.retrieve("...", k=5, hybrid_alpha=0.3, rerank_top_n=20)
```

The pre-rerank score is kept in `h.metadata["_pre_rerank_score"]` so
you can compare.

## 14. Multi-tenant REST server

One server, many brains. Tenant routes under `/bundles/{name}`:

```bash
# Start server (optional API key for bearer auth):
SOMA_API_KEY=change-me SOMA_BUNDLES_DIR=./data/bundles soma serve --port 8420

# Per-tenant store:
curl -X POST http://localhost:8420/bundles/alex/store \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{"text": "alex prefers vegetarian"}'

curl -X POST http://localhost:8420/bundles/bobbi/store \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{"text": "bobbi prefers seafood"}'

# Health check (no auth):
curl http://localhost:8420/health
```

Each `/bundles/{name}` path maps to `$SOMA_BUNDLES_DIR/{name}/` on
disk; bundles are loaded lazily and cached in memory.

## 15. Inspect a bundle without loading the LLM

```bash
soma stats  --bundle brain/
soma search --bundle brain/ --query "deployment notes" --k 5
python scripts/demo_memory_inspect.py recent --bundle brain/ --n 50
python scripts/demo_memory_inspect.py dump --bundle brain/ > snapshot.jsonl
```

---

Missing a recipe you want? Open an issue with the use case — most
agent-memory patterns are 5–20 lines of glue around `store` /
`retrieve` / `related` / `get_recent` / `forget`.
