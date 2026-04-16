# SOMA Quickstart — 60 Seconds to Value

```bash
pip install -e '.[sbert]'        # core + sentence-transformers
```

## Index your wiki

```bash
soma index --wiki path/to/your/wiki --bundle my-brain/
```

That walked the directory, split every `.md` (and `.pdf` if `pypdf`
is installed) into heading-aware chunks, embedded each, and saved a
portable bundle. Time: ~1 min for a 1000-file wiki on a laptop.

## Chat about it

```bash
soma chat --bundle my-brain/
```

`soma chat` auto-picks an LLM:

- Have **Ollama** running (`ollama serve` + `ollama pull llama3.2`)? It uses that.
- `OPENAI_API_KEY` set? It uses GPT.
- `ANTHROPIC_API_KEY` set? It uses Claude.
- None of the above? It loads a small local HuggingFace model.

Override with `--backend ollama|openai|anthropic|openai-compat|hf`.
Add `--dry-run` to skip the LLM and just see the retrieved chunks.

## Inspect what got stored

```bash
soma stats  --bundle my-brain/                  # entry count, disk size
soma search --bundle my-brain/ --query "..."    # vector search, no LLM
```

## In Python

```python
from soma.memory import MemoryLayer
from soma.llm   import RAGSession, backend_from_env

mem  = MemoryLayer.load("my-brain/")
chat = RAGSession(memory=mem, llm=backend_from_env())

answer = chat.ask("where does the user live?")
print(answer.text)              # the LLM's reply
for line in answer.cite_lines():
    print(line)                 # [1] alex.md (Bio)  score=0.81
```

## What's the bundle?

A directory. Move it, copy it, back it up, share it across machines —
the entire "brain" is `tokenizer.json` + `encoder.pt` (or sbert
metadata) + `memory_index.json` + `memory_embeddings.pt`. No SQLite,
no daemon, no cloud.

## Next steps

- **More demos**: [`docs/demos.md`](demos.md) — wiki/PDF chat,
  persistent conversation, memory inspection, related-entry browser.
- **Different LLM?** [`docs/llm-backends.md`](llm-backends.md) covers
  Ollama, OpenAI, Anthropic, vLLM/LM Studio, local HuggingFace.
- **Why SOMA over X?** [`docs/comparison.md`](comparison.md) —
  Mem0 / Letta / Zep / Chroma / Pinecone tradeoffs.
- **REST API**: `soma serve --port 8420`, then `curl localhost:8420/...`.
- **Benchmarks** vs Chroma at 1K → 100K:
  [`benchmarks/reports/paper-draft.md`](../benchmarks/reports/paper-draft.md).
