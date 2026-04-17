"""Task 5: Multi-step planning with dependencies.

The agent must fix a bug across 5 interconnected files.  Each file
has a clue pointing to the next.  The fix requires changes in a
specific order.

Tools: read_file, write_file, run_tests
Scoring: did tests pass? Steps to completion.  Backtrack count.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult

# ---------------------------------------------------------------------------
# Bug scenario
# ---------------------------------------------------------------------------

_FILE_TEMPLATES: list[list[dict[str, str]]] = [
    # Scenario 0: config value propagation bug
    [
        {
            "path": "config.py",
            "original": (
                '# Configuration\n'
                'DATABASE_URL = "postgres://localhost:5432/mydb"\n'
                'MAX_CONNECTIONS = 10\n'
                'TIMEOUT = 30  # BUG: should be 60 for production\n'
                '# Note: connection pool settings are in pool.py\n'
            ),
            "fixed": (
                '# Configuration\n'
                'DATABASE_URL = "postgres://localhost:5432/mydb"\n'
                'MAX_CONNECTIONS = 10\n'
                'TIMEOUT = 60  # production timeout\n'
                '# Note: connection pool settings are in pool.py\n'
            ),
            "clue": "pool.py",
        },
        {
            "path": "pool.py",
            "original": (
                'from config import TIMEOUT, MAX_CONNECTIONS\n\n'
                'class ConnectionPool:\n'
                '    def __init__(self):\n'
                '        self.timeout = TIMEOUT\n'
                '        self.max_conn = MAX_CONNECTIONS\n'
                '        self.retry_delay = TIMEOUT // 10  # BUG: should use TIMEOUT // 6\n'
                '        # Retry logic is in retry.py\n'
            ),
            "fixed": (
                'from config import TIMEOUT, MAX_CONNECTIONS\n\n'
                'class ConnectionPool:\n'
                '    def __init__(self):\n'
                '        self.timeout = TIMEOUT\n'
                '        self.max_conn = MAX_CONNECTIONS\n'
                '        self.retry_delay = TIMEOUT // 6\n'
                '        # Retry logic is in retry.py\n'
            ),
            "clue": "retry.py",
        },
        {
            "path": "retry.py",
            "original": (
                'from pool import ConnectionPool\n\n'
                'def retry_connection(pool: ConnectionPool):\n'
                '    for attempt in range(3):  # BUG: should be range(5)\n'
                '        try:\n'
                '            return pool.connect()\n'
                '        except TimeoutError:\n'
                '            time.sleep(pool.retry_delay)\n'
                '    # Error handling is in handler.py\n'
                '    raise ConnectionError("All retries failed")\n'
            ),
            "fixed": (
                'from pool import ConnectionPool\n\n'
                'def retry_connection(pool: ConnectionPool):\n'
                '    for attempt in range(5):\n'
                '        try:\n'
                '            return pool.connect()\n'
                '        except TimeoutError:\n'
                '            time.sleep(pool.retry_delay)\n'
                '    # Error handling is in handler.py\n'
                '    raise ConnectionError("All retries failed")\n'
            ),
            "clue": "handler.py",
        },
        {
            "path": "handler.py",
            "original": (
                'from retry import retry_connection\n\n'
                'def handle_request(pool):\n'
                '    try:\n'
                '        conn = retry_connection(pool)\n'
                '        return conn.execute("SELECT 1")\n'
                '    except ConnectionError as e:\n'
                '        log_error(e)  # BUG: should also return error response\n'
                '        # Logging is in logger.py\n'
            ),
            "fixed": (
                'from retry import retry_connection\n\n'
                'def handle_request(pool):\n'
                '    try:\n'
                '        conn = retry_connection(pool)\n'
                '        return conn.execute("SELECT 1")\n'
                '    except ConnectionError as e:\n'
                '        log_error(e)\n'
                '        return {"error": str(e)}  # return error response\n'
                '        # Logging is in logger.py\n'
            ),
            "clue": "logger.py",
        },
        {
            "path": "logger.py",
            "original": (
                'import logging\n\n'
                'logger = logging.getLogger(__name__)\n'
                'logger.setLevel(logging.DEBUG)  # BUG: should be logging.WARNING in production\n\n'
                'def log_error(error):\n'
                '    logger.error(f"Connection error: {error}")\n'
            ),
            "fixed": (
                'import logging\n\n'
                'logger = logging.getLogger(__name__)\n'
                'logger.setLevel(logging.WARNING)  # production level\n\n'
                'def log_error(error):\n'
                '    logger.error(f"Connection error: {error}")\n'
            ),
            "clue": None,
        },
    ],
]


@dataclass
class MultiStepPlanTask:
    """Multi-step planning / bug fix benchmark."""

    seed: int = 0
    _step: int = 0
    _files: dict[str, str] = field(default_factory=dict)
    _scenario: list[dict[str, str]] = field(default_factory=list)
    _fixes_applied: set[str] = field(default_factory=set)
    _backtracks: int = 0
    _tests_run: int = 0
    _tests_passed: bool = False
    _done: bool = False
    _max_steps: int = 50
    _rng: random.Random = field(default_factory=random.Random)

    TOOLS: list[dict] = field(default_factory=lambda: [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file (overwrites existing).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run the test suite. Returns pass/fail with error details.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ])

    def setup(self) -> str:
        self._rng = random.Random(self.seed)
        scenario_idx = self.seed % len(_FILE_TEMPLATES)
        self._scenario = _FILE_TEMPLATES[scenario_idx]
        self._files = {f["path"]: f["original"] for f in self._scenario}
        self._fixes_applied = set()
        self._backtracks = 0
        self._tests_run = 0
        self._tests_passed = False
        self._step = 0
        self._done = False

        return (
            "There is a bug causing connection timeouts in production. "
            "The codebase has 5 files: config.py, pool.py, retry.py, "
            "handler.py, logger.py.\n\n"
            "Each file may contain bugs. Fix all bugs and run_tests to verify.\n"
            "Start by reading config.py to understand the configuration."
        )

    def execute_action(self, action: str) -> str:
        self._step += 1
        if self._step >= self._max_steps:
            self._done = True
            return "Maximum steps reached."

        tool_name, args = self._parse_tool_call(action)

        if tool_name == "read_file":
            return self._handle_read(args)
        elif tool_name == "write_file":
            return self._handle_write(args)
        elif tool_name == "run_tests":
            return self._handle_run_tests()

        return "Use read_file, write_file, or run_tests."

    def _handle_read(self, args: dict) -> str:
        path = args.get("path", "")
        if path not in self._files:
            return f"File not found: {path}. Available: {', '.join(self._files.keys())}"
        return self._files[path]

    def _handle_write(self, args: dict) -> str:
        path = args.get("path", "")
        content = args.get("content", "")
        if path not in self._files:
            return f"File not found: {path}. Available: {', '.join(self._files.keys())}"

        self._files[path] = content

        # Check if this is a correct fix
        for f in self._scenario:
            if f["path"] == path:
                if self._is_fix_correct(content, f["fixed"]):
                    self._fixes_applied.add(path)
                elif path in self._fixes_applied:
                    # Was correct, now reverted = backtrack
                    self._fixes_applied.discard(path)
                    self._backtracks += 1
                break

        return f"File '{path}' updated."

    def _is_fix_correct(self, actual: str, expected: str) -> bool:
        """Check if the fix contains the key correction.

        We normalize whitespace and check that the critical changed lines
        are present.
        """
        actual_lines = {line.strip() for line in actual.strip().split("\n") if line.strip()}
        expected_lines = {line.strip() for line in expected.strip().split("\n") if line.strip()}
        # The fix is correct if all expected lines appear in the actual content
        return expected_lines.issubset(actual_lines)

    def _handle_run_tests(self) -> str:
        self._tests_run += 1
        # Check which fixes are missing
        missing = []
        for f in self._scenario:
            if f["path"] not in self._fixes_applied:
                missing.append(f["path"])

        if not missing:
            self._tests_passed = True
            self._done = True
            return "All tests PASSED! All bugs have been fixed."

        # Give a hint about the first failing file
        first_fail = missing[0]
        return (
            f"Tests FAILED.\n"
            f"Error: test_connection_handling failed\n"
            f"  AssertionError in {first_fail}: unexpected behavior detected.\n"
            f"  {len(missing)} file(s) still have bugs."
        )

    def _parse_tool_call(self, action: str) -> tuple[str, dict]:
        try:
            data = json.loads(action)
            if "tool" in data:
                return data["tool"], data.get("arguments", data.get("args", {}))
            if "name" in data:
                return data["name"], data.get("arguments", data.get("args", {}))
        except (json.JSONDecodeError, TypeError):
            pass
        lines = action.strip().split("\n")
        tool_name = ""
        args_str = ""
        for line in lines:
            if line.upper().startswith("TOOL:"):
                tool_name = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ARGS:"):
                args_str = line.split(":", 1)[1].strip()
        if tool_name:
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}
            return tool_name, args
        return "", {}

    def is_complete(self) -> bool:
        return self._done or self._step >= self._max_steps

    def score(self) -> TaskResult:
        total_files = len(self._scenario)
        fixed = len(self._fixes_applied)
        accuracy = fixed / total_files * 100 if total_files > 0 else 0.0

        return TaskResult(
            completion=self._tests_passed,
            accuracy=accuracy,
            steps=self._step,
            extra={
                "files_fixed": fixed,
                "total_files": total_files,
                "backtracks": self._backtracks,
                "tests_run": self._tests_run,
                "tests_passed": self._tests_passed,
            },
        )
