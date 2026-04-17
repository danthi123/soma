# Phase 43: Built-in Domain Schemas + Context Packer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Ship the three highest-value built-in domains (agent, conv,
knowledge) on top of the Phase 42 schema framework, plus the real
context-packer implementation. After this phase, agent developers can
`from soma.schemas.builtin.agent import TaskState, ToolCall` and get
validated store/retrieve + automatic context packing.

**Depends on:** Phase 42 (schema framework).

**Architecture:**
- `src/soma/schemas/builtin/agent.py` — TaskState, ToolCall, Observation, Decision
- `src/soma/schemas/builtin/conv.py` — Fact, Preference, Contradiction (extends existing ConversationalMemory types)
- `src/soma/schemas/builtin/knowledge.py` — Note, Connection, Question, Insight
- `src/soma/schemas/packing.py` — real `pack_context()` with configurable mixing strategy

**Out-of-scope:** Phases 44 covers remaining domains (code, research, collab, customer, creative) + full developer documentation.

---

### Tasks (3):
1. Agent domain schemas + tests
2. Conv + knowledge domain schemas + tests
3. Context packer implementation + tests + integration with MemoryLayer

Each task = one commit. ~1 week total.
