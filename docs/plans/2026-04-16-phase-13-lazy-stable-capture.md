# Phase 13 — Lazy Stable-Capture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Remove the O(N²/K) stable-capture cost from `consolidate()` by deferring the re-capture to the first `retrieve()` call after a growth pass. Closes Task #173.

**Background:** See `docs/plans/2026-04-16-lazy-stable-capture.md` for the full design exploration. Current `consolidate()` runs stable-capture eagerly over every stored entry post-growth, which is O(N_so_far) per call and dominates session cost at large N. Today the output is also unused (alpha=0 default), but we want the path fast before graph-rerank reactivation.

**Architecture:**
1. Introduce `_stable_capture_dirty: bool` flag on `MemoryLayer` (True whenever a growth pass has just finished but stable-capture hasn't run).
2. `consolidate()` sets the flag and skips the stable-capture loop.
3. First `retrieve()` after growth (when `alpha > 0` AND `_stable_capture_dirty`) triggers stable-capture before search.
4. Alpha=0 fast-path skips stable-capture entirely — today's behaviour for the default.
5. Add explicit `stable_capture()` public method for callers who want to pay the cost eagerly.

**Tech Stack:** no new deps; pure refactor of `src/soma/memory/api.py`.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`.

---

### Task 1: Introduce `_stable_capture_dirty` state

**Files:**
- Modify: `src/soma/memory/api.py`
- Create: `tests/test_memory/test_lazy_stable_capture.py`

**Step 1: Write failing tests.**
```python
def test_consolidate_marks_dirty_but_doesnt_capture():
    # SOMA attached + graph_rerank_stable_capture=True.
    # After consolidate(): mem._stable_capture_dirty == True.
    # Stored activations have NOT been re-captured under the new
    # weights (assert the recorded capture counter didn't increment).

def test_retrieve_with_alpha_zero_leaves_dirty_flag():
    # alpha=0 retrieve must not pay the stable-capture cost.
    # _stable_capture_dirty remains True after retrieve.

def test_retrieve_with_alpha_positive_runs_stable_capture():
    # alpha=0.5 retrieve(): stable-capture fires before search.
    # _stable_capture_dirty flips to False.
    # Second retrieve() does NOT re-run capture.
```

**Step 2-4: TDD.**

Implementation notes for the subagent:
- Bisect `consolidate()` into `_growth_pass()` + optional `_stable_capture_pass()`; hoist the latter's guts into a `stable_capture()` public method.
- Move the `graph_rerank_stable_capture=True` check from `consolidate()` time to the `retrieve()` path.
- Retrieve's `_maybe_stable_capture()` helper: `if self._graph_rerank_alpha > 0 and self._stable_capture_dirty: self.stable_capture()`.
- Any direct store/forget/clear call should also set the dirty flag so intervening mutations re-trigger the capture on next retrieve.

**Step 5:** `git commit -m "feat(memory): lazy stable-capture deferred to first retrieve"`

---

### Task 2: Explicit `stable_capture()` public API

**Files:**
- Modify: `src/soma/memory/api.py` (add `stable_capture` to `__all__`, public method)
- Extend tests with a `test_explicit_stable_capture_flips_dirty`

**Rationale:** benchmarks and users who want to time stable-capture separately need a way to trigger it eagerly. Also lets pre-retrieve warmup on cold starts.

**Step 5:** `git commit -m "feat(memory): MemoryLayer.stable_capture() public API"`

---

### Task 3: Benchmark update

**Files:**
- Modify: `benchmarks/harness/adapters/soma.py` — add an `eager_stable_capture=True` parameter to the adapter so existing plasticity benchmarks pin the pre-Phase-13 behaviour; new benchmarks can opt into the lazy path by leaving it False.
- Confirm `benchmarks/run_plasticity_scale.py` still runs green (no signal regression — the numbers should match since alpha=0 means stable-capture is unused).

**Step 5:** `git commit -m "bench(soma): eager_stable_capture flag for adapter"`

---

### Final sanity

```bash
pytest tests/test_memory benchmarks/tests -q
ruff check src/soma/memory/api.py benchmarks/harness/adapters/soma.py tests/test_memory
```

**Numeric regression check:** run the graph-ablation benchmark to confirm no result drift from the lazy path. Since `graph_rerank_alpha=0.0` is the shipped default, there should be zero user-visible difference — this is purely a cost reduction.

**Risk — call site audit:** hunt for every external caller of `consolidate()` that implicitly relies on stable-capture being done before return. `grep -rn "consolidate()" src tests benchmarks` and audit each hit.
