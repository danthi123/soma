# LLM Backends — Plug Any Model into SOMA

SOMA's `MemoryLayer` is LLM-agnostic by design — `store(text)` /
`retrieve(query)` return text and metadata, and what you do with them
is up to you. The vast majority of users want the obvious next step:
take the retrieved chunks, send them to an LLM with a prompt, get a
grounded answer back. The `soma.llm` package ships that wiring so you
don't have to write the same glue every time.

## TL;DR

```python
from soma.llm import RAGSession, backend_from_env
from soma.memory import MemoryLayer

mem  = MemoryLayer.load("brain/")
chat = RAGSession(memory=mem, llm=backend_from_env())
print(chat.ask("where does the user live?").text)
```

`backend_from_env()` picks an LLM based on environment:
1. `SOMA_LLM_BACKEND` env if set (`ollama` / `openai` / `anthropic` /
   `openai-compat` / `hf`)
2. `OPENAI_API_KEY` set → OpenAI
3. `ANTHROPIC_API_KEY` set → Anthropic
4. Ollama reachable at `localhost:11434` → Ollama
5. Fallback → local HuggingFace via `soma.deploy.chat_head_factory`

## Backends

| Backend | What it talks to | Extra dep | Auth |
| --- | --- | --- | --- |
| `OllamaBackend` | Local Ollama server (`localhost:11434`) | none | none |
| `OpenAIBackend` | OpenAI cloud API | `openai` | `OPENAI_API_KEY` |
| `AnthropicBackend` | Anthropic Claude API | `anthropic` | `ANTHROPIC_API_KEY` |
| `OpenAICompatibleBackend` | vLLM / LM Studio / LiteLLM / llama.cpp / etc. | `openai` | varies |
| `HuggingFaceBackend` | Local HF model via `soma.deploy.chat_head_factory` | `transformers`, `accelerate` | none |
| `DryRunBackend` | No LLM — echoes context (testing) | none | none |

All optional SDKs are lazy-imported so installing SOMA never pulls
them in unless you actually use that backend.

### Ollama (recommended for local-first)

Install [Ollama](https://ollama.com), pull a model, point SOMA at it.
No API key, no network egress.

```bash
ollama pull llama3.2
```

```python
from soma.llm import OllamaBackend, RAGSession
chat = RAGSession(memory=mem, llm=OllamaBackend(model="llama3.2"))
chat.ask("...").text
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
pip install openai
```

```python
from soma.llm import OpenAIBackend, RAGSession
chat = RAGSession(memory=mem, llm=OpenAIBackend(model="gpt-4o-mini"))
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic
```

```python
from soma.llm import AnthropicBackend, RAGSession
chat = RAGSession(memory=mem, llm=AnthropicBackend(model="claude-haiku-4-5-20251001"))
```

### Generic OpenAI-compatible (vLLM, LM Studio, LiteLLM, llama.cpp …)

Anything that speaks OpenAI's `/v1/chat/completions`. The SDK is the
official `openai` package — only the base URL changes.

```python
from soma.llm import OpenAICompatibleBackend, RAGSession
chat = RAGSession(
    memory=mem,
    llm=OpenAICompatibleBackend(
        model="qwen2.5-7b",
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    ),
)
```

### Local HuggingFace (default fallback)

When no other backend is available, SOMA falls back to a local
HuggingFace model loaded by `soma.deploy.chat_head_factory`. Tier
`auto` lets the deploy module pick a model that fits your
VRAM/RAM. Needs `pip install soma[dev-chat]`.

```python
from soma.llm import HuggingFaceBackend, RAGSession
chat = RAGSession(memory=mem, llm=HuggingFaceBackend(tier="auto"))
```

## Writing your own backend

The `LLMBackend` protocol is one method:

```python
class MyBackend:
    name = "mybackend:1.0"
    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        return "..."  # your call to whatever model
```

Anything that satisfies the protocol is plug-and-play with
`RAGSession`. No registration, no config — just hand it in.

## Customising the prompt

`RAGSession` defaults to a citation-friendly template. Override
either or both:

```python
chat = RAGSession(
    memory=mem,
    llm=backend,
    prompt_template=(
        "Answer the user's question using the context. "
        "Reply in 2 sentences, cite sources [1], [2].\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)
```

For richer formatting (XML tags, system messages, role-tagged turns),
pass `format_context_fn=...` or wrap `RAGSession` with your own glue.

## REST API

If you'd rather have an HTTP front-end, `soma serve` boots a FastAPI
app exposing `/store`, `/retrieve`, and friends. You can run any LLM
on top of that — the LLM never needs to know SOMA exists, just call
the retrieve endpoint to fetch context. See
[`README.md` § REST API](../README.md).

## CLI shortcut

The `soma` console script picks a backend automatically:

```bash
soma index --wiki path/to/docs --bundle my-brain/
soma chat  --bundle my-brain/                 # auto-picks LLM backend
soma chat  --bundle my-brain/ --backend ollama
soma chat  --bundle my-brain/ --dry-run       # no LLM, echo context
```
