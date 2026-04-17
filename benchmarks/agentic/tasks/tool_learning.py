"""Task 2: Tool-call learning.

The agent must query a simulated ``search_database(query)`` tool 10 times.
The tool has quirky rules:
  - queries must be lowercase
  - max 3 words
  - results are paginated (``has_more=true`` with ``cursor``)

On format violations the tool returns an error message explaining the rule.
Scoring: success rate, total calls, format-error count.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult

# ---------------------------------------------------------------------------
# Search items (simulated database)
# ---------------------------------------------------------------------------
_ITEM_POOLS: list[list[dict[str, str]]] = [
    [
        {"id": "1", "title": "quantum entanglement basics", "category": "physics"},
        {"id": "2", "title": "neural network architectures", "category": "ml"},
        {"id": "3", "title": "rust memory safety", "category": "programming"},
        {"id": "4", "title": "sourdough starter guide", "category": "cooking"},
        {"id": "5", "title": "kubernetes pod scheduling", "category": "devops"},
        {"id": "6", "title": "graph theory fundamentals", "category": "math"},
        {"id": "7", "title": "espresso extraction tips", "category": "cooking"},
        {"id": "8", "title": "tcp congestion control", "category": "networking"},
        {"id": "9", "title": "bayesian inference intro", "category": "statistics"},
        {"id": "10", "title": "guitar chord progressions", "category": "music"},
    ],
]

_SEARCH_TARGETS: list[list[str]] = [
    [
        "quantum",
        "neural network",
        "rust memory",
        "sourdough",
        "kubernetes",
        "graph theory",
        "espresso",
        "tcp",
        "bayesian",
        "guitar",
    ],
]


@dataclass
class ToolLearningTask:
    """Tool-call learning with a quirky search API."""

    seed: int = 0
    _step: int = 0
    _successful_searches: int = 0
    _total_calls: int = 0
    _format_errors: int = 0
    _targets_found: int = 0
    _target_queries: list[str] = field(default_factory=list)
    _items: list[dict[str, str]] = field(default_factory=list)
    _cursors: dict[str, int] = field(default_factory=dict)
    _done: bool = False
    _max_steps: int = 60  # safety limit
    _rng: random.Random = field(default_factory=random.Random)

    # Tool definitions for the agent
    TOOLS: list[dict] = field(default_factory=lambda: [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the database. Query must be lowercase, max 3 words.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (lowercase, max 3 words)",
                        },
                        "cursor": {
                            "type": "string",
                            "description": "Pagination cursor from previous results",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_results",
                "description": "Submit collected search results when done.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of found item titles",
                        }
                    },
                    "required": ["results"],
                },
            },
        },
    ])

    def setup(self) -> str:
        self._rng = random.Random(self.seed)
        pool_idx = self.seed % len(_ITEM_POOLS)
        self._items = _ITEM_POOLS[pool_idx]
        self._target_queries = _SEARCH_TARGETS[pool_idx].copy()
        self._step = 0
        self._successful_searches = 0
        self._total_calls = 0
        self._format_errors = 0
        self._targets_found = 0
        self._cursors = {}
        self._done = False

        targets_list = "\n".join(
            f"  {i + 1}. {t}" for i, t in enumerate(self._target_queries)
        )
        return (
            "You have access to a search_database tool. "
            "Search for the following 10 items and collect results.\n"
            "IMPORTANT: The search API has rules -- queries must be lowercase "
            "and at most 3 words. Results may be paginated (check has_more).\n"
            "When you have found all items, call submit_results.\n\n"
            f"Items to find:\n{targets_list}\n\n"
            "Use the search_database tool to begin."
        )

    def execute_action(self, action: str) -> str:
        """Parse the agent's tool call and return the result."""
        self._step += 1

        if self._step >= self._max_steps:
            self._done = True
            return "Maximum steps reached. Task ending."

        # Parse the action as a tool call
        tool_name, args = self._parse_tool_call(action)

        if tool_name == "submit_results":
            self._done = True
            # Count how many targets were found
            submitted = args.get("results", [])
            for target in self._target_queries:
                for item in submitted:
                    item_str = str(item) if not isinstance(item, str) else item
                    if target.lower() in item_str.lower():
                        self._targets_found += 1
                        break
            found = self._targets_found
            total = len(self._target_queries)
            return f"Results submitted. Found {found}/{total} items."

        if tool_name == "search_database":
            return self._handle_search(args)

        # Unknown tool or unparseable action
        self._format_errors += 1
        self._total_calls += 1
        return (
            "Error: Could not parse tool call. "
            "Please call search_database(query) or submit_results(results). "
            f"Received: {action[:200]}"
        )

    def _handle_search(self, args: dict) -> str:
        self._total_calls += 1
        query = args.get("query", "")
        cursor = args.get("cursor")

        # Validate: must be lowercase
        if query != query.lower():
            self._format_errors += 1
            return json.dumps({
                "error": "QUERY_FORMAT_ERROR",
                "message": "Query must be entirely lowercase. "
                f"Got: '{query}'. Please retry with lowercase.",
            })

        # Validate: max 3 words
        words = query.split()
        if len(words) > 3:
            self._format_errors += 1
            return json.dumps({
                "error": "QUERY_FORMAT_ERROR",
                "message": f"Query must be at most 3 words. Got {len(words)} words. "
                "Please shorten your query.",
            })

        # Search
        results = [
            item for item in self._items
            if query.lower() in item["title"].lower()
        ]

        # Pagination: return 2 items per page
        page_size = 2
        start = 0
        if cursor and cursor in self._cursors:
            start = self._cursors[cursor]

        page = results[start : start + page_size]
        has_more = (start + page_size) < len(results)

        response: dict = {"results": page, "total": len(results)}
        if has_more:
            next_cursor = f"cursor_{query}_{start + page_size}"
            self._cursors[next_cursor] = start + page_size
            response["has_more"] = True
            response["cursor"] = next_cursor
        else:
            response["has_more"] = False

        self._successful_searches += 1
        return json.dumps(response)

    def _parse_tool_call(self, action: str) -> tuple[str, dict]:
        """Parse a tool call from the agent's response.

        Accepts either raw JSON or the format:
        TOOL: tool_name
        ARGS: {json}
        """
        try:
            data = json.loads(action)
            if "tool" in data:
                return data["tool"], data.get("arguments", data.get("args", {}))
            if "name" in data:
                return data["name"], data.get("arguments", data.get("args", {}))
        except (json.JSONDecodeError, TypeError):
            pass

        # Try TOOL:/ARGS: format
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
        success_rate = (
            self._successful_searches / self._total_calls * 100
            if self._total_calls > 0
            else 0.0
        )
        return TaskResult(
            completion=self._targets_found >= len(self._target_queries),
            accuracy=success_rate,
            steps=self._step,
            tool_errors=self._format_errors,
            extra={
                "successful_searches": self._successful_searches,
                "total_calls": self._total_calls,
                "targets_found": self._targets_found,
                "targets_total": len(self._target_queries),
            },
        )
