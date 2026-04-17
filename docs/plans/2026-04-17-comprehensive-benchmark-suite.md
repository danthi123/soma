# Comprehensive SOMA Benchmark Suite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a publication-grade benchmark suite that quantifies SOMA's real-world value using industry-standard benchmarks for one-to-one comparison with competitors (Mem0, Zep, Letta, A-MEM), plus internal scaling experiments for the paper.

**Architecture:** Five benchmark tracks. Track A: adopt industry-standard agent-memory benchmarks (LoCoMo, LongMemEval, AMA-Bench). Track B: custom realistic agentic tasks for areas not covered by existing benchmarks. Track C: full-scale CL experiment at publication quality. Track D: node-count scaling sweep. Track E: final clean-room runs on dedicated Linux for paper-quality numbers.

**Tech Stack:** Python 3.11+, PyTorch, Ollama (OpenAI-compat), HuggingFace datasets, existing SOMA benchmark + CL harness.

**Competitors to compare against:** Mem0, Zep, Letta, A-MEM (Zettelkasten-style), raw context window (baseline).

---

## Track A: Industry-Standard Agent Memory Benchmarks

These are established benchmarks with published leaderboards and competitor numbers. Using them enables direct, apples-to-apples comparison.

### A1: LongMemEval (ICLR 2025)

**Source:** https://github.com/xiaowu0162/LongMemEval
**Dataset:** 500 manually curated questions testing 5 core memory abilities:
  - Information extraction
  - Multi-session reasoning
  - Temporal reasoning
  - Knowledge updates
  - Abstention (knowing when NOT to answer)
**Format:** HuggingFace dataset (longmemeval_oracle.json, longmemeval_s_cleaned.json)
**Metrics:** Token-level F1, exact match, ROUGE-1/2/L, LLM-as-judge accuracy
**Why:** Most directly comparable to SOMA's value proposition. Tests whether SOMA's typed retrieval outperforms raw context windows on the exact scenarios we claim to improve.

### A2: LoCoMo (Snap Research)

**Source:** https://snap-research.github.io/locomo/
**Dataset:** 1500-2000 QA pairs over long-term conversations (weeks/months)
**Categories:** Single-hop, multi-hop, temporal, open-domain, adversarial reasoning
**Metrics:** Precision, recall, F1 (SQuAD 2.0-style), ROUGE, FactScore
**Why:** Tests long-term conversational memory at scale -- SOMA's typed schema storage should excel at temporal and multi-hop reasoning where context windows fail.

### A3: AMA-Bench (2026)

**Source:** https://ama-bench.github.io/
**Dataset:** Real-world agentic trajectories + synthetic scaling trajectories
**Focus:** Agent-environment interactions (not just dialogue), needle-in-haystack for agent memory
**Metrics:** LLM-as-judge (binary yes/no), needle retrieval accuracy
**Why:** Most directly tests agentic (tool-based) memory -- the exact use case for SOMA's `ToolCall` and `Observation` schemas. Unlike LoCoMo/LongMemEval which are conversational, AMA-Bench uses machine-generated tool trajectories.

### A4: Letta Context-Bench

**Source:** https://github.com/letta-ai/letta (context-bench module)
**Focus:** File operations, entity tracing, multi-step information retrieval
**Metrics:** Task completion accuracy, retrieval precision
**Why:** Letta is our most direct competitor. Running their benchmark with SOMA as the memory backend produces a direct comparison.

---

## Track B: Custom Realistic Agentic Tasks

The existing 5 tasks are synthetic and short. These 3 new tasks test multi-tool orchestration, long-horizon workflows, and retrieval relevance over extended conversations -- the scenarios where SOMA's memory layer should provide measurable advantage over context-window-only baselines.

### Task 1: Multi-tool codebase debugging (60-120 turns)

A realistic debugging workflow: agent receives a bug report, must use file/search/test/edit tools to diagnose and fix a multi-file issue across a simulated codebase.

**Files:**
- Create: `benchmarks/agentic/tasks/codebase_debug.py`
- Test: `tests/test_agentic/test_codebase_debug.py`

**Scenario design:**
- Simulated 8-file Python project (in-memory, no real FS)
- Bug spans 3 files (import chain, config propagation, logic error)
- 6 tools: `search_codebase(query)`, `read_file(path)`, `write_file(path, content)`, `run_tests()`, `list_files()`, `grep(pattern)`
- Agent must discover the root cause by tracing through files, then fix all 3
- Scoring: files_correctly_fixed / 3, tests_passing, total_steps, backtrack_count
- 3 scenario variants (indexed by seed % 3), each with different file layouts and bug types

**Why this tests SOMA:** The agent must remember which files it already read, what patterns it searched for, and what hypotheses it formed. A context-window-only agent forgets early explorations as the conversation grows past the window. SOMA stores `ToolCall` and `Observation` records that survive truncation.

**Step 1: Write the failing tests**

```python
# tests/test_agentic/test_codebase_debug.py
"""Tests for the codebase debugging task."""
from benchmarks.agentic.tasks.codebase_debug import CodebaseDebugTask

class TestCodebaseDebug:
    def test_setup_returns_bug_report(self):
        task = CodebaseDebugTask(seed=0)
        obs = task.setup()
        assert "bug" in obs.lower() or "error" in obs.lower()
        assert not task.is_complete()

    def test_list_files(self):
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action('{"tool": "list_files", "arguments": {}}')
        assert ".py" in result

    def test_read_file(self):
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action('{"tool": "list_files", "arguments": {}}')
        # extract a filename from the listing
        task.execute_action('{"tool": "read_file", "arguments": {"path": "main.py"}}')
        # should not error

    def test_search_codebase(self):
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "search_codebase", "arguments": {"query": "import"}}'
        )
        assert "main.py" in result or "match" in result.lower()

    def test_run_tests_initially_fails(self):
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action('{"tool": "run_tests", "arguments": {}}')
        assert "FAIL" in result

    def test_scoring_defaults(self):
        task = CodebaseDebugTask(seed=0)
        task.setup()
        score = task.score()
        assert score.accuracy == 0.0
        assert not score.completion

    def test_grep_tool(self):
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "grep", "arguments": {"pattern": "def "}}'
        )
        assert "match" in result.lower() or ":" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agentic/test_codebase_debug.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement CodebaseDebugTask**

```python
# benchmarks/agentic/tasks/codebase_debug.py
"""Multi-tool codebase debugging task (60-120 turns).

The agent receives a bug report and must use 6 tools (search, read,
write, test, list, grep) to find and fix a 3-file bug in a simulated
8-file project. Tests SOMA's ability to retain exploration history
across long conversations.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult

# --- Scenario definitions ------------------------------------------------
# Each scenario: 8 files, 3 buggy, specific fixes required.

_SCENARIOS = [
    {
        "bug_report": (
            "Bug report: The /api/users endpoint returns 500 when "
            "called with a valid auth token. Logs show "
            "'TypeError: process_request() got unexpected keyword "
            "argument max_retries'. Started after yesterday's config "
            "change. Please investigate and fix."
        ),
        "files": {
            "config.py": {
                "content": (
                    "# Application configuration\n"
                    "DATABASE_URL = 'postgresql://localhost/app'\n"
                    "MAX_RETRIES = '3'  # BUG: should be int, not str\n"
                    "TIMEOUT = 30\n"
                    "DEBUG = False\n"
                    "API_VERSION = 'v2'\n"
                ),
                "fixed": (
                    "# Application configuration\n"
                    "DATABASE_URL = 'postgresql://localhost/app'\n"
                    "MAX_RETRIES = 3\n"
                    "TIMEOUT = 30\n"
                    "DEBUG = False\n"
                    "API_VERSION = 'v2'\n"
                ),
                "buggy": True,
                "hint": "MAX_RETRIES is a string, should be int",
            },
            "middleware.py": {
                "content": (
                    "from config import MAX_RETRIES, TIMEOUT\n\n"
                    "def process_request(request, max_retries=None):\n"
                    "    retries = max_retries or MAX_RETRIES\n"
                    "    for i in range(retries):  # BUG: range() needs int\n"
                    "        try:\n"
                    "            return handle(request)\n"
                    "        except TimeoutError:\n"
                    "            if i == retries - 1:\n"
                    "                raise\n"
                ),
                "fixed": (
                    "from config import MAX_RETRIES, TIMEOUT\n\n"
                    "def process_request(request, max_retries=None):\n"
                    "    retries = max_retries if max_retries is not None else MAX_RETRIES\n"
                    "    for i in range(int(retries)):\n"
                    "        try:\n"
                    "            return handle(request)\n"
                    "        except TimeoutError:\n"
                    "            if i == retries - 1:\n"
                    "                raise\n"
                ),
                "buggy": True,
                "hint": "range() crashes on string; also or-default masks 0",
            },
            "routes.py": {
                "content": (
                    "from middleware import process_request\n"
                    "from auth import verify_token\n\n"
                    "def handle_users(request):\n"
                    "    token = request.get('token')\n"
                    "    if not verify_token(token):\n"
                    "        return {'status': 401}\n"
                    "    # BUG: passes max_retries as kwarg but config is str\n"
                    "    return process_request(request, max_retries=MAX_RETRIES)\n"
                ),
                "fixed": (
                    "from middleware import process_request\n"
                    "from auth import verify_token\n"
                    "from config import MAX_RETRIES\n\n"
                    "def handle_users(request):\n"
                    "    token = request.get('token')\n"
                    "    if not verify_token(token):\n"
                    "        return {'status': 401}\n"
                    "    return process_request(request, max_retries=MAX_RETRIES)\n"
                ),
                "buggy": True,
                "hint": "Missing config import; MAX_RETRIES not in scope",
            },
            "auth.py": {
                "content": (
                    "import hashlib\n\n"
                    "VALID_TOKENS = {'abc123', 'def456'}\n\n"
                    "def verify_token(token):\n"
                    "    return token in VALID_TOKENS\n"
                ),
                "fixed": None,
                "buggy": False,
            },
            "models.py": {
                "content": (
                    "class User:\n"
                    "    def __init__(self, name, email):\n"
                    "        self.name = name\n"
                    "        self.email = email\n"
                ),
                "fixed": None,
                "buggy": False,
            },
            "database.py": {
                "content": (
                    "from config import DATABASE_URL\n\n"
                    "def get_connection():\n"
                    "    return connect(DATABASE_URL)\n"
                ),
                "fixed": None,
                "buggy": False,
            },
            "utils.py": {
                "content": (
                    "import logging\n\n"
                    "logger = logging.getLogger(__name__)\n\n"
                    "def log_request(request):\n"
                    "    logger.info(f'Request: {request}')\n"
                ),
                "fixed": None,
                "buggy": False,
            },
            "main.py": {
                "content": (
                    "from routes import handle_users\n\n"
                    "def app(request):\n"
                    "    path = request.get('path', '/')\n"
                    "    if path == '/api/users':\n"
                    "        return handle_users(request)\n"
                    "    return {'status': 404}\n"
                ),
                "fixed": None,
                "buggy": False,
            },
        },
        "test_output_before": (
            "Running tests...\n"
            "FAIL test_users_endpoint: TypeError: 'str' object cannot "
            "be interpreted as an integer (in middleware.py:5)\n"
            "FAIL test_retry_logic: range() argument must be int\n"
            "PASS test_auth_valid\n"
            "PASS test_auth_invalid\n"
            "PASS test_404\n"
            "3 passed, 2 failed"
        ),
        "test_output_after": (
            "Running tests...\n"
            "PASS test_users_endpoint\n"
            "PASS test_retry_logic\n"
            "PASS test_auth_valid\n"
            "PASS test_auth_invalid\n"
            "PASS test_404\n"
            "5 passed, 0 failed"
        ),
    },
    # Scenario 2 and 3 follow similar structure with different bugs
    # (omitted for brevity -- implement with different file layouts)
]


@dataclass
class CodebaseDebugTask:
    """Multi-tool codebase debugging benchmark."""

    seed: int = 0
    _step: int = 0
    _files: dict[str, dict] = field(default_factory=dict)
    _current_files: dict[str, str] = field(default_factory=dict)
    _fixes_applied: dict[str, bool] = field(default_factory=dict)
    _done: bool = False
    _max_steps: int = 120
    _scenario: dict = field(default_factory=dict)

    TOOLS: list[dict] = field(default_factory=lambda: [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files in the project.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file (overwrites).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "New file content"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run the project test suite.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": "Search all files for a text query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search text"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search files with a regex pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern"},
                    },
                    "required": ["pattern"],
                },
            },
        },
    ])

    def setup(self) -> str:
        idx = self.seed % len(_SCENARIOS)
        self._scenario = _SCENARIOS[idx]
        self._files = self._scenario["files"]
        self._current_files = {
            name: info["content"] for name, info in self._files.items()
        }
        self._fixes_applied = {
            name: False for name, info in self._files.items() if info["buggy"]
        }
        self._step = 0
        self._done = False
        return self._scenario["bug_report"]

    def execute_action(self, action: str) -> str:
        self._step += 1
        if self._step >= self._max_steps:
            self._done = True
            return "Maximum steps reached."

        tool_name, args = self._parse_tool_call(action)

        if tool_name == "list_files":
            return "Project files:\n" + "\n".join(
                f"  {name}" for name in sorted(self._current_files)
            )

        if tool_name == "read_file":
            path = args.get("path", "")
            if path not in self._current_files:
                return f"Error: File '{path}' not found."
            return f"--- {path} ---\n{self._current_files[path]}"

        if tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            if path not in self._current_files:
                return f"Error: File '{path}' not found."
            self._current_files[path] = content
            # Check if this fix is correct
            if path in self._fixes_applied:
                expected = self._files[path]["fixed"]
                if expected and self._normalize(content) == self._normalize(expected):
                    self._fixes_applied[path] = True
            return f"File '{path}' updated ({len(content)} chars)."

        if tool_name == "run_tests":
            if all(self._fixes_applied.values()):
                self._done = True
                return self._scenario["test_output_after"]
            return self._scenario["test_output_before"]

        if tool_name == "search_codebase":
            query = args.get("query", "")
            results = []
            for name, content in self._current_files.items():
                if query.lower() in content.lower():
                    # Find matching lines
                    for i, line in enumerate(content.split("\n"), 1):
                        if query.lower() in line.lower():
                            results.append(f"{name}:{i}: {line.strip()}")
            if not results:
                return f"No matches for '{query}'."
            return f"Found {len(results)} matches:\n" + "\n".join(results[:20])

        if tool_name == "grep":
            pattern = args.get("pattern", "")
            results = []
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                return f"Error: Invalid regex pattern '{pattern}'."
            for name, content in self._current_files.items():
                for i, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        results.append(f"{name}:{i}: {line.strip()}")
            if not results:
                return f"No matches for pattern '{pattern}'."
            return f"Found {len(results)} matches:\n" + "\n".join(results[:20])

        return f"Error: Unknown tool '{tool_name}'. Available: list_files, read_file, write_file, run_tests, search_codebase, grep"

    def is_complete(self) -> bool:
        return self._done

    def score(self) -> TaskResult:
        fixed = sum(1 for v in self._fixes_applied.values() if v)
        total = len(self._fixes_applied)
        accuracy = (fixed / total * 100) if total > 0 else 0.0
        return TaskResult(
            completion=all(self._fixes_applied.values()),
            accuracy=accuracy,
            steps=self._step,
            extra={
                "files_fixed": fixed,
                "total_buggy": total,
                "fixes": dict(self._fixes_applied),
            },
        )

    @staticmethod
    def _normalize(content: str) -> str:
        """Normalize whitespace for comparison."""
        return "\n".join(line.rstrip() for line in content.strip().split("\n"))

    @staticmethod
    def _parse_tool_call(action: str) -> tuple[str, dict]:
        try:
            parsed = json.loads(action)
            if isinstance(parsed, dict):
                name = parsed.get("tool") or parsed.get("name", "")
                args = parsed.get("arguments") or parsed.get("args", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return name, args
        except (json.JSONDecodeError, TypeError):
            pass
        return "", {}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agentic/test_codebase_debug.py -v`
Expected: PASS

**Step 5: Register the task**

Modify: `benchmarks/agentic/tasks/__init__.py`
Add `"codebase_debug": CodebaseDebugTask` to `TASK_REGISTRY`.

**Step 6: Commit**

```bash
git add benchmarks/agentic/tasks/codebase_debug.py tests/test_agentic/test_codebase_debug.py benchmarks/agentic/tasks/__init__.py
git commit -m "feat(benchmarks): codebase_debug task — multi-tool debugging over 60-120 turns"
```

---

### Task 2: Customer support with ticket history (40-80 turns)

Agent is a support rep who must resolve a customer issue by referencing 5 prior tickets (stored via tool calls at the start). Tests whether SOMA's retrieval surfaces the right past ticket when the customer describes a related problem.

**Files:**
- Create: `benchmarks/agentic/tasks/customer_support.py`
- Test: `tests/test_agentic/test_customer_support.py`

**Scenario design:**
- 5 historical tickets loaded at setup (via `load_ticket` tool responses)
- Customer describes a new issue that relates to 2-3 prior tickets
- Agent must use `search_tickets(query)`, `get_ticket(id)`, `reply_to_customer(message)`, `escalate(reason)`, `resolve(solution)`
- Scoring: correct_ticket_references / expected_references, resolution_quality (keyword match), steps
- Key test: after 30+ turns of troubleshooting, does the agent still reference the correct early tickets?

**Why this tests SOMA:** The 5 historical tickets are loaded early in the conversation. After 30+ turns of back-and-forth troubleshooting, a baseline agent's context window has likely truncated those tickets. SOMA's `Observation` records retain them as retrievable typed entries.

**Step 1-6:** Same TDD flow as Task 1 (tests, implementation, register, commit).

---

### Task 3: Research assistant with iterative refinement (50-100 turns)

Agent must answer a multi-part research question by searching a simulated knowledge base, synthesizing findings, and iteratively refining answers based on contradictory evidence.

**Files:**
- Create: `benchmarks/agentic/tasks/research_assistant.py`
- Test: `tests/test_agentic/test_research_assistant.py`

**Scenario design:**
- Knowledge base: 30 "papers" with abstracts + key findings (some contradictory)
- Agent uses: `search_papers(query)`, `read_abstract(paper_id)`, `read_full(paper_id)`, `submit_finding(claim, evidence[])`, `revise_finding(finding_id, new_claim, new_evidence[])`
- 5 research questions asked sequentially, each building on prior findings
- Later papers may contradict earlier ones -- agent must revise
- Scoring: findings_correct / 5, evidence_quality (cited correct papers), revision_count (fewer = better retrieval), total_steps

**Why this tests SOMA:** The agent must remember which papers it already read, what claims they support, and update its mental model when contradictory evidence appears. SOMA stores `ToolCall` records (which papers were read) and `Observation` records (what they said), enabling targeted retrieval when contradictions surface.

**Step 1-6:** Same TDD flow (tests, implementation, register, commit).

---

### Task 4: Register all new tasks + run full benchmark matrix

**Files:**
- Modify: `benchmarks/agentic/tasks/__init__.py`

**Step 1:** Verify all 8 tasks are in TASK_REGISTRY (5 existing + 3 new).

**Step 2:** Run dry-run to confirm:
```bash
python -m benchmarks.agentic.run_agentic --dry-run --tasks codebase_debug,customer_support,research_assistant --models small-high --seeds 1
```

**Step 3:** Run smoke test with ultralight model:
```bash
python -m benchmarks.agentic.run_agentic --models ultralight --tasks codebase_debug,customer_support,research_assistant --seeds 1 --max-steps 30
```

**Step 4: Commit**

---

## Track C: Publication-Quality CL Benchmark

Run B3 at full scale with statistical rigor for publication.

### Task 5: Full-scale B3 run (10 tasks, 5 epochs, 3 seeds)

**No new code needed** -- use existing `run_b3.py` with proper parameters.

**Step 1:** Run seed 42 (plastic-only first, already in flight):
```bash
PYTHONUNBUFFERED=1 python -m research.cl.run_b3 \
  --tasks 10 --epochs 5 --train-samples 2000 --test-samples 1000 \
  --seed 42 --ablation soma-plastic
```

**Step 2:** Run seeds 42, 123, 7 for all 4 ablations on GPU:
```bash
for seed in 42 123 7; do
  PYTHONUNBUFFERED=1 python -m research.cl.run_b3 \
    --tasks 10 --epochs 5 --train-samples 2000 --test-samples 1000 \
    --seed $seed
done
```

**Step 3:** Aggregate results across seeds into publication report.

**Files:**
- Create: `research/cl/reports/b3_full_results.json` (3-seed averaged)
- Create: `research/cl/reports/b3_full_results.md`

**Step 4: Commit**

```bash
git add research/cl/reports/b3_full_*
git commit -m "feat(research-b): B3 full CL results (10 tasks, 5 epochs, 3 seeds)"
```

---

### Task 6: Add multi-seed aggregation to B3 runner

**Files:**
- Modify: `research/cl/run_b3.py`

Add `--seeds` flag (comma-separated, e.g., `--seeds 42,123,7`) that runs all ablations for each seed and outputs mean +/- std for ACC, BWT, FWT.

**Step 1: Write the failing test**

```python
# Test that multi-seed averaging works
def test_multi_seed_averaging():
    results = {
        "soma-plastic": [
            {"acc": 0.75, "bwt": -0.02},
            {"acc": 0.77, "bwt": -0.01},
        ]
    }
    avg = _average_seed_results(results["soma-plastic"])
    assert 0.75 < avg["acc_mean"] < 0.77
    assert avg["acc_std"] > 0
```

**Step 2-5:** Implement `_average_seed_results()`, wire `--seeds` flag, commit.

---

## Track D: Node-Count Scaling Sweep

### Task 7: CL node-count sweep

Run the B3 `soma-frozen` ablation (fastest, same results as plastic) with integrator counts {8, 16, 32, 64, 128}.

**Files:**
- Create: `research/cl/run_node_sweep.py`
- Create: `research/cl/reports/node_sweep_cl.json`
- Create: `research/cl/reports/node_sweep_cl.md`

**Step 1: Write the sweep orchestrator**

```python
# research/cl/run_node_sweep.py
"""Node-count sweep: frozen SOMA with {8, 16, 32, 64, 128} integrators."""
import json, time, subprocess, sys
from pathlib import Path

INTEGRATOR_COUNTS = [8, 16, 32, 64, 128]
REPORTS_DIR = Path("research/cl/reports")

def main():
    results = {}
    for n_int in INTEGRATOR_COUNTS:
        print(f"\n{'='*60}")
        print(f"Running frozen SOMA with {n_int} integrators ({n_int*2} associators)")
        print(f"{'='*60}")
        
        t0 = time.perf_counter()
        # Shell out to run_b3 with --ablation soma-frozen --integrators N
        cmd = [
            sys.executable, "-m", "research.cl.run_b3",
            "--tasks", "5", "--epochs", "3",
            "--train-samples", "1000", "--test-samples", "500",
            "--seed", "42",
            "--ablation", "soma-frozen",
            "--integrators", str(n_int),
        ]
        subprocess.run(cmd, check=True)
        wall = time.perf_counter() - t0
        
        # Read the JSON results (run_b3 overwrites the same file)
        with open(REPORTS_DIR / "b3_soma_cl.json") as f:
            data = json.load(f)
        
        frozen = data["soma-frozen"]
        results[f"int_{n_int}"] = {
            "integrators": n_int,
            "associators": n_int * 2,
            "total_nodes": n_int + n_int * 2 + 6,  # + sensor/output nodes
            "acc": frozen["acc"],
            "bwt": frozen["bwt"],
            "fwt": frozen["fwt"],
            "wall_clock_s": round(wall, 1),
        }
        
        print(f"  ACC={frozen['acc']:.4f}  BWT={frozen['bwt']:.4f}  "
              f"wall={wall:.1f}s")

    # Save results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "node_sweep_cl.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print(f"\n{'='*60}")
    print("NODE-COUNT SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Integrators':>12s} {'Nodes':>8s} {'ACC':>8s} "
          f"{'BWT':>8s} {'Wall(s)':>8s}")
    for key, r in results.items():
        print(f"{r['integrators']:>12d} {r['total_nodes']:>8d} "
              f"{r['acc']:>8.4f} {r['bwt']:>8.4f} {r['wall_clock_s']:>8.1f}")

if __name__ == "__main__":
    main()
```

**Step 2:** Run the sweep:
```bash
PYTHONUNBUFFERED=1 python -m research.cl.run_node_sweep
```

**Step 3:** Write report interpreting results (does ACC scale with node count? Does wall-clock scale linearly? Does this confirm D4's O(output_dim) theory?)

**Step 4: Commit**

---

### Task 8: Associative memory node-count sweep

Run D2's attractor-mode protocol with integrator counts {8, 16, 32, 64, 128}.

**Files:**
- Create: `research/associative/run_node_sweep.py`
- Create: `research/associative/reports/node_sweep_assoc.json`

**Step 1:** Write sweep orchestrator (similar pattern to Task 7 but using `research/associative/run_d2.py`'s protocol).

**Step 2:** Run the sweep.

**Step 3:** Write combined scaling report.

**Files:**
- Create: `research/reports/node_count_scaling.md`

**Step 4: Commit**

---

## Track E: Full Agentic Benchmark Matrix

### Task 9: Run baseline benchmark across all model tiers

With the tool-wrapping bug fixed, run all 12 models on all 8 tasks (5 existing + 3 new).

**Step 1:** Run small tier (4 models x 8 tasks x 1 seed):
```bash
python -m benchmarks.agentic.run_agentic \
  --models ultralight,small-low,small-high,small-gemma \
  --seeds 1 --max-steps 60 --agent baseline \
  --out-md benchmarks/agentic/reports/baseline_small_v2.md
```

**Step 2:** Run mid tier (3 models x 8 tasks):
```bash
python -m benchmarks.agentic.run_agentic \
  --models sota-mid,moe-small,reasoning \
  --seeds 1 --max-steps 60 --agent baseline \
  --out-md benchmarks/agentic/reports/baseline_mid.md
```

**Step 3:** Run SOTA tier (4 models x 8 tasks):
```bash
python -m benchmarks.agentic.run_agentic \
  --models sota-max,dense-large,moe-large,sota-alt \
  --seeds 1 --max-steps 60 --agent baseline \
  --out-md benchmarks/agentic/reports/baseline_sota.md
```

**Step 4:** Run SomaAgent on the same matrix for comparison:
```bash
# Repeat steps 1-3 with --agent soma
```

**Step 5:** Merge all results into one comparison report.

**Files:**
- Create: `benchmarks/agentic/reports/full_comparison.md`

**Step 6: Commit**

---

## Track F: Clean-Room Final Runs

All publication-quality numbers should be gathered on a clean environment to avoid contention artifacts from Windows background services, LM Studio, Discord, browser, etc.

### Setup

**Step 1:** Boot from a temporary Linux medium (USB live, or Unraid VM with GPU passthrough).

**Step 2:** Install minimal environment:
```bash
# Python 3.11+, PyTorch with CUDA, Ollama
pip install -e ".[dev]"
ollama pull <all 12 model tags>
```

**Step 3:** Re-run all benchmark tracks:
- Track A: Industry benchmarks (LoCoMo, LongMemEval, AMA-Bench)
- Track C: Full B3 CL (10 tasks, 5 epochs, 3 seeds)
- Track D: Node-count sweep
- Track E: Full agentic matrix (12 models x 8 tasks x 3 seeds)

**Step 4:** Compare results against Windows runs to quantify contention impact.

**Step 5:** Use clean-room numbers for all paper claims. Windows numbers stay as development validation.

---

## Execution order

Dependencies:
- Track A (industry benchmarks) is the highest priority -- enables competitor comparison
- Track B tasks 1-3 are independent (can parallelize)
- Track B task 4 depends on 1-3
- Track C task 5 is already partially running
- Track C task 6 is independent
- Track D tasks 7-8 are independent
- Track E task 9 depends on Track B task 4 (needs all tasks registered)
- Track F depends on all other tracks (final step)

**Recommended batch order:**
1. **Batch 1:** Track A (adopt LoCoMo/LongMemEval) + Track B tasks 1-3 (custom tasks)
2. **Batch 2:** Track B task 4 (register all tasks) + Track C task 6 (multi-seed runner)
3. **Batch 3:** Track C task 5 (full B3) + Track D (node sweeps)
4. **Batch 4:** Track E (full agentic matrix with all tasks + models)
5. **Batch 5:** Track F (clean-room re-runs for paper)

## Industry-standard comparisons to publish

| Benchmark | What we show | Competitors |
|---|---|---|
| LongMemEval | SOMA vs raw context window | Mem0, Zep, Letta |
| LoCoMo | SOMA multi-hop reasoning | Published baselines |
| AMA-Bench | SOMA agentic trajectories | A-MEM, Letta |
| Permuted-MNIST CL | SOMA BWT vs EWC/A-GEM | Published baselines |
| Node-count scaling | SOMA capacity = O(output_dim) | Theoretical prediction |

## Commit pattern

One commit per task. Co-Authored-By trailer as usual.

## Total scope

- Track A: ~2-3 days (download datasets, write evaluation adapters, run)
- Track B: ~3-4 hours implementation (3 custom tasks + tests)
- Track C: ~4-6 hours GPU time (unattended), ~1 hour code
- Track D: ~2-3 hours GPU/CPU time (unattended), ~30 min code
- Track E: ~8-12 hours Ollama inference time (unattended), no new code
- Track F: ~1-2 days (Linux setup + re-runs)

Sources:
- [LongMemEval (ICLR 2025)](https://github.com/xiaowu0162/LongMemEval)
- [LoCoMo](https://snap-research.github.io/locomo/)
- [AMA-Bench](https://ama-bench.github.io/)
- [Letta Leaderboard](https://www.letta.com/blog/letta-leaderboard)
- [A-MEM](https://arxiv.org/abs/2502.12110)
- [Agent Memory Paper List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
