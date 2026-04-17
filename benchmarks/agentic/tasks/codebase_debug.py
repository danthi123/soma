"""Multi-tool codebase debugging task (60-120 turns).

The agent receives a bug report and must use 6 tools (search, read,
write, test, list, grep) to find and fix a 3-file bug in a simulated
8-file project.  Tests SOMA's ability to retain exploration history
across long conversations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Each scenario: 8 files, 3 buggy, specific fixes required.

_SCENARIOS: list[dict] = [
    # Scenario 0: config type + import chain bug
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
                    "    retries = max_retries if max_retries is not None"
                    " else MAX_RETRIES\n"
                    "    for i in range(int(retries)):\n"
                    "        try:\n"
                    "            return handle(request)\n"
                    "        except TimeoutError:\n"
                    "            if i == retries - 1:\n"
                    "                raise\n"
                ),
                "buggy": True,
            },
            "routes.py": {
                "content": (
                    "from middleware import process_request\n"
                    "from auth import verify_token\n\n"
                    "def handle_users(request):\n"
                    "    token = request.get('token')\n"
                    "    if not verify_token(token):\n"
                    "        return {'status': 401}\n"
                    "    # BUG: passes max_retries as kwarg but"
                    " config is str\n"
                    "    return process_request(request,"
                    " max_retries=MAX_RETRIES)\n"
                ),
                "fixed": (
                    "from middleware import process_request\n"
                    "from auth import verify_token\n"
                    "from config import MAX_RETRIES\n\n"
                    "def handle_users(request):\n"
                    "    token = request.get('token')\n"
                    "    if not verify_token(token):\n"
                    "        return {'status': 401}\n"
                    "    return process_request(request,"
                    " max_retries=MAX_RETRIES)\n"
                ),
                "buggy": True,
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
    # Scenario 1: caching + serialization bug
    {
        "bug_report": (
            "Bug report: The data export endpoint returns empty JSON "
            "for all requests. Logs show 'KeyError: format' in "
            "serializer.py. The cache appears to serve stale data. "
            "Please investigate and fix."
        ),
        "files": {
            "serializer.py": {
                "content": (
                    "import json as _json\n\n"
                    "def serialize(data, format='json'):\n"
                    "    # BUG: format shadows builtin, and default"
                    " should be checked\n"
                    "    if format == 'json':\n"
                    "        return _json.dumps(data)\n"
                    "    elif format == 'csv':\n"
                    "        return ','.join(str(v) for v in data)\n"
                    "    raise ValueError(f'Unknown format: {format}')\n"
                ),
                "fixed": (
                    "import json as _json\n\n"
                    "def serialize(data, fmt='json'):\n"
                    "    if fmt == 'json':\n"
                    "        return _json.dumps(data)\n"
                    "    elif fmt == 'csv':\n"
                    "        return ','.join(str(v) for v in data)\n"
                    "    raise ValueError(f'Unknown format: {fmt}')\n"
                ),
                "buggy": True,
            },
            "cache.py": {
                "content": (
                    "from collections import OrderedDict\n\n"
                    "_cache = OrderedDict()\n"
                    "MAX_SIZE = 100\n\n"
                    "def get(key):\n"
                    "    return _cache.get(key)\n\n"
                    "def put(key, value):\n"
                    "    _cache[key] = value\n"
                    "    # BUG: never evicts -- should pop oldest"
                    " when full\n"
                ),
                "fixed": (
                    "from collections import OrderedDict\n\n"
                    "_cache = OrderedDict()\n"
                    "MAX_SIZE = 100\n\n"
                    "def get(key):\n"
                    "    return _cache.get(key)\n\n"
                    "def put(key, value):\n"
                    "    _cache[key] = value\n"
                    "    if len(_cache) > MAX_SIZE:\n"
                    "        _cache.popitem(last=False)\n"
                ),
                "buggy": True,
            },
            "export.py": {
                "content": (
                    "from serializer import serialize\n"
                    "from cache import get, put\n\n"
                    "def export_data(data, format='json'):\n"
                    "    key = f'export_{format}_{id(data)}'\n"
                    "    cached = get(key)\n"
                    "    if cached is not None:\n"
                    "        return cached\n"
                    "    # BUG: passes 'format' kwarg but serializer"
                    " param renamed\n"
                    "    result = serialize(data, format=format)\n"
                    "    put(key, result)\n"
                    "    return result\n"
                ),
                "fixed": (
                    "from serializer import serialize\n"
                    "from cache import get, put\n\n"
                    "def export_data(data, fmt='json'):\n"
                    "    key = f'export_{fmt}_{id(data)}'\n"
                    "    cached = get(key)\n"
                    "    if cached is not None:\n"
                    "        return cached\n"
                    "    result = serialize(data, fmt=fmt)\n"
                    "    put(key, result)\n"
                    "    return result\n"
                ),
                "buggy": True,
            },
            "api.py": {
                "content": (
                    "from export import export_data\n\n"
                    "def handle_export(request):\n"
                    "    data = request.get('data', [])\n"
                    "    fmt = request.get('format', 'json')\n"
                    "    return export_data(data, format=fmt)\n"
                ),
                "fixed": None,
                "buggy": False,
            },
            "models.py": {
                "content": (
                    "class DataRecord:\n"
                    "    def __init__(self, key, value):\n"
                    "        self.key = key\n"
                    "        self.value = value\n"
                ),
                "fixed": None,
                "buggy": False,
            },
            "validators.py": {
                "content": (
                    "def validate_format(fmt):\n"
                    "    return fmt in ('json', 'csv')\n"
                ),
                "fixed": None,
                "buggy": False,
            },
            "config.py": {
                "content": (
                    "EXPORT_DIR = '/tmp/exports'\n"
                    "DEFAULT_FORMAT = 'json'\n"
                    "CACHE_ENABLED = True\n"
                ),
                "fixed": None,
                "buggy": False,
            },
            "main.py": {
                "content": (
                    "from api import handle_export\n\n"
                    "def app(request):\n"
                    "    path = request.get('path', '/')\n"
                    "    if path == '/export':\n"
                    "        return handle_export(request)\n"
                    "    return {'status': 404}\n"
                ),
                "fixed": None,
                "buggy": False,
            },
        },
        "test_output_before": (
            "Running tests...\n"
            "FAIL test_json_export: TypeError: serialize() got an "
            "unexpected keyword argument 'format'\n"
            "FAIL test_cache_eviction: cache size exceeds MAX_SIZE\n"
            "PASS test_csv_validate\n"
            "PASS test_record_create\n"
            "PASS test_404\n"
            "3 passed, 2 failed"
        ),
        "test_output_after": (
            "Running tests...\n"
            "PASS test_json_export\n"
            "PASS test_cache_eviction\n"
            "PASS test_csv_validate\n"
            "PASS test_record_create\n"
            "PASS test_404\n"
            "5 passed, 0 failed"
        ),
    },
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
    _tool_errors: int = 0

    TOOLS: list[dict] = field(
        default_factory=lambda: [
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
                            "path": {
                                "type": "string",
                                "description": "File path",
                            },
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
                            "path": {
                                "type": "string",
                                "description": "File path",
                            },
                            "content": {
                                "type": "string",
                                "description": "New file content",
                            },
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
                            "query": {
                                "type": "string",
                                "description": "Search text",
                            },
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
                            "pattern": {
                                "type": "string",
                                "description": "Regex pattern",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
        ]
    )

    def setup(self) -> str:
        idx = self.seed % len(_SCENARIOS)
        self._scenario = _SCENARIOS[idx]
        self._files = self._scenario["files"]
        self._current_files = {
            name: info["content"] for name, info in self._files.items()
        }
        self._fixes_applied = {
            name: False
            for name, info in self._files.items()
            if info["buggy"]
        }
        self._step = 0
        self._done = False
        self._tool_errors = 0
        return self._scenario["bug_report"]

    def execute_action(self, action: str) -> str:
        self._step += 1
        if self._step >= self._max_steps:
            self._done = True
            return "Maximum steps reached."

        tool_name, args = _parse_tool_call(action)

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
                if expected and _normalize(content) == _normalize(expected):
                    self._fixes_applied[path] = True
                else:
                    self._fixes_applied[path] = False
            return f"File '{path}' updated ({len(content)} chars)."

        if tool_name == "run_tests":
            if all(self._fixes_applied.values()):
                self._done = True
                return self._scenario["test_output_after"]
            return self._scenario["test_output_before"]

        if tool_name == "search_codebase":
            query = args.get("query", "")
            results: list[str] = []
            for name, content in self._current_files.items():
                if query.lower() in content.lower():
                    for i, line in enumerate(content.split("\n"), 1):
                        if query.lower() in line.lower():
                            results.append(
                                f"{name}:{i}: {line.strip()}"
                            )
            if not results:
                return f"No matches for '{query}'."
            return (
                f"Found {len(results)} matches:\n"
                + "\n".join(results[:20])
            )

        if tool_name == "grep":
            pattern = args.get("pattern", "")
            results_g: list[str] = []
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                return f"Error: Invalid regex pattern '{pattern}'."
            for name, content in self._current_files.items():
                for i, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        results_g.append(
                            f"{name}:{i}: {line.strip()}"
                        )
            if not results_g:
                return f"No matches for pattern '{pattern}'."
            return (
                f"Found {len(results_g)} matches:\n"
                + "\n".join(results_g[:20])
            )

        self._tool_errors += 1
        return (
            f"Error: Unknown tool '{tool_name}'. Available: "
            "list_files, read_file, write_file, run_tests, "
            "search_codebase, grep"
        )

    def is_complete(self) -> bool:
        return self._done or self._step >= self._max_steps

    def score(self) -> TaskResult:
        fixed = sum(1 for v in self._fixes_applied.values() if v)
        total = len(self._fixes_applied)
        accuracy = (fixed / total * 100) if total > 0 else 0.0
        return TaskResult(
            completion=all(self._fixes_applied.values()),
            accuracy=accuracy,
            steps=self._step,
            tool_errors=self._tool_errors,
            extra={
                "files_fixed": fixed,
                "total_buggy": total,
                "fixes": dict(self._fixes_applied),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(content: str) -> str:
    """Normalize whitespace for comparison."""
    return "\n".join(
        line.rstrip() for line in content.strip().split("\n")
    )


def _parse_tool_call(action: str) -> tuple[str, dict]:
    """Parse a tool call from the agent's response."""
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
