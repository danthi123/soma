# Phase 28: `soma chat` Async Streaming

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Today `soma chat` blocks on the full LLM response before
printing anything — on long answers the REPL feels dead for seconds.
Stream tokens as they arrive so typing feels live. Ships a better
demo-path without changing the non-interactive CLI behaviour.

**Architecture:**
- Detect streaming capability on the configured `LLMBackend`. The
  existing `LLMBackend` Protocol has a synchronous `complete(prompt)
  -> str`. Adding a streaming capability without breaking existing
  backends: introduce `stream_complete(prompt) -> Iterator[str]` as
  an optional Protocol method (via `hasattr` check at call time).
- `soma chat` REPL loop: if the backend has `stream_complete`, pipe
  chunks to stdout with `flush=True` as they arrive. Otherwise fall
  back to `complete()` + full-response print — existing behaviour
  preserved exactly.
- Streaming adapters ship for the backends that natively support it:
  * OpenAI backend — `client.chat.completions.create(stream=True)`.
  * Ollama backend — `/api/generate` with `stream=True`.
  * Anthropic backend — `client.messages.stream(...)`.
  * LM Studio backend — OpenAI-compatible, reuses the OpenAI path.
  * Stub/Dry backends — no streaming (falls back to `complete`).

**Out-of-scope:**
- Streaming in the `/chat` REST endpoint. That's a different
  protocol decision (SSE vs WebSocket vs polling) and a separate
  phase.
- Streaming through `ConversationalMemory.add_message`. Extraction
  still runs on the full assistant response after the stream ends.
  Token-by-token extraction is a non-goal.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/cookbook.md` polish, `deferred-items.md` strikethrough.

---

### Task 1: `stream_complete` capability + REPL loop

**Files:**
- Modify: `src/soma/cli.py` (look for the existing `soma chat` command
  handler — likely a `_cmd_chat` or similar).
- Modify: `src/soma/llm.py` OR `src/soma/memory/llm_backend.py`
  (wherever the `LLMBackend` Protocol lives) — document the optional
  `stream_complete` method.
- Extend: `tests/test_cli.py`

**REPL pseudo-code:**
```python
def _chat_reply(llm, prompt: str) -> str:
    stream = getattr(llm, "stream_complete", None)
    if stream is None:
        reply = llm.complete(prompt)
        print(reply)
        return reply
    chunks: list[str] = []
    for chunk in stream(prompt):
        chunks.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(chunks)
```

**Step 1: Failing tests.**
```python
def test_chat_prefers_stream_complete_when_available(capsys):
    class FakeStreamingLLM:
        def complete(self, prompt): return "SHOULD_NOT_CALL"
        def stream_complete(self, prompt):
            yield "hello "
            yield "world"
    # Drive the REPL one-shot; assert stdout == "hello world\n"
    # and complete() was NEVER called.

def test_chat_falls_back_to_complete_when_no_stream(capsys):
    class SyncLLM:
        def complete(self, prompt): return "one-shot reply"
    # Drive the REPL; assert stdout == "one-shot reply\n".

def test_chat_reply_returns_joined_text_for_memory_storage():
    # The full response needs to be captured for ConversationalMemory
    # to do its post-stream extraction. Test that the return value
    # round-trips into the stored turn.
```

**Step 5:** `git commit -m "feat(cli): soma chat streams when backend supports stream_complete"`

---

### Task 2: Streaming adapters for real backends

**Files:**
- Modify: wherever the OpenAI / Ollama / Anthropic / LM Studio
  backends live (search for `class OpenAIBackend` etc). Probably
  `src/soma/llm.py` or a submodule.

Add `stream_complete` to each backend that supports it. Implementation
notes:
- **OpenAI / LM Studio**: `client.chat.completions.create(stream=True)`
  yields chunks with `delta.content`. Skip empty deltas.
- **Ollama**: POST `/api/generate` with `"stream": true`. Parse
  newline-delimited JSON; yield `response` field per chunk.
- **Anthropic**: use `client.messages.stream(...)` context manager;
  iterate over `text_stream`.

Tests: for each backend, mock the streaming response source (fake
HTTP or a stubbed client object) and assert the yielded chunks
concatenate to the expected full text. Don't hit real APIs.

**Step 5:** `git commit -m "feat(llm): stream_complete for OpenAI, Ollama, Anthropic, LM Studio"`

---

### Task 3: Cookbook recipe

**Files:**
- Modify: `docs/cookbook.md` — short note on when streaming kicks in
  (automatic when backend supports it) and how to plug a custom
  streaming backend.

**Step 5:** `git commit -m "docs(cookbook): soma chat streaming recipe"`

---

### Final sanity

```bash
ruff check src/soma tests
pytest tests/test_cli.py -q
```

Baseline post-Phase-23: `tests/test_cli.py` ≈ 35 tests. Target +~6 new
Phase 28 tests (3 REPL + 3 backend-streaming — one per non-dry
backend), 0 regressions.

**Gotchas:**
- stdout flushing on Windows terminals can be surprising — tests
  that assert `capsys.readouterr().out` might succeed even if the
  human-visible behaviour is buffered. Add `flush=True` explicitly
  on each write.
- KeyboardInterrupt during a stream should leave the REPL in a
  usable state — add a `try/except KeyboardInterrupt` around the
  stream loop, print a newline, return the partial text. Don't
  store a partial turn; raise to the outer loop.
- Extraction still runs on the FULL response after the stream ends
  (one LLM call for streaming display, one for extraction — same
  as today). Don't try to tee the stream into two consumers.
