"""Task 4: Session resume.

A 10-step project setup task.  After step 5 the agent's context is
cleared (simulating a restart).  The agent is told "continue from
where you left off" and must use ``check_status()`` to figure out
what was already done.

Tools: create_file, run_command, check_status
Scoring: correct-continuation rate, redundant-step count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult


# ---------------------------------------------------------------------------
# Project steps (the "ground truth" plan)
# ---------------------------------------------------------------------------
def _step(
    sid: str, action: str, args: str, description: str
) -> dict[str, str]:
    return {
        "id": sid,
        "action": action,
        "args": args,
        "description": description,
    }


_PROJECT_STEPS: list[dict[str, str]] = [
    _step("1", "create_file",
          '{"name": "README.md", "content": "# MyProject"}',
          "Create README.md"),
    _step("2", "create_file",
          '{"name": "setup.py", "content": "setup()"}',
          "Create setup.py"),
    _step("3", "run_command",
          '{"cmd": "git init"}',
          "Initialize git repository"),
    _step("4", "create_file",
          '{"name": ".gitignore", "content": "__pycache__/"}',
          "Create .gitignore"),
    _step("5", "create_file",
          '{"name": "src/__init__.py", "content": ""}',
          "Create src package"),
    _step("6", "create_file",
          '{"name": "tests/__init__.py", "content": ""}',
          "Create tests package"),
    _step("7", "create_file",
          '{"name": "requirements.txt", "content": "pytest"}',
          "Create requirements.txt"),
    _step("8", "run_command",
          '{"cmd": "pip install -r requirements.txt"}',
          "Install dependencies"),
    _step("9", "create_file",
          '{"name": "ci.yml", "content": "name: CI"}',
          "Create CI configuration"),
    _step("10", "run_command",
          '{"cmd": "git add . && git commit -m init"}',
          "Initial git commit"),
]


@dataclass
class SessionResumeTask:
    """Session resume benchmark."""

    seed: int = 0
    _step: int = 0
    _completed_step_ids: set[str] = field(default_factory=set)
    _created_files: dict[str, str] = field(default_factory=dict)
    _commands_run: list[str] = field(default_factory=list)
    _session_break_at: int = 5  # clear context after step 5
    _session_broken: bool = False
    _redundant_steps: int = 0
    _correct_continuations: int = 0
    _done: bool = False
    _max_steps: int = 40

    TOOLS: list[dict] = field(default_factory=lambda: [
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Create a file with given content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["name", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                    },
                    "required": ["cmd"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_status",
                "description": "Check what has been done so far in the project setup.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ])

    def setup(self) -> str:
        self._step = 0
        self._completed_step_ids = set()
        self._created_files = {}
        self._commands_run = []
        self._session_broken = False
        self._redundant_steps = 0
        self._correct_continuations = 0
        self._done = False

        steps_desc = "\n".join(
            f"  {s['id']}. {s['description']}" for s in _PROJECT_STEPS
        )
        return (
            "Set up a new Python project by completing these 10 steps in order:\n\n"
            f"{steps_desc}\n\n"
            "Use the available tools (create_file, run_command, check_status) "
            "to complete each step."
        )

    def execute_action(self, action: str) -> str:
        self._step += 1
        if self._step >= self._max_steps:
            self._done = True
            return "Maximum steps reached."

        tool_name, args = self._parse_tool_call(action)

        # Check for session break
        if (
            not self._session_broken
            and len(self._completed_step_ids) >= self._session_break_at
        ):
            self._session_broken = True
            return (
                "[SESSION RESTART]\n\n"
                "Your previous session ended. Continue setting up the project "
                "from where you left off. Use check_status() to see what has "
                "already been completed."
            )

        if tool_name == "check_status":
            return self._handle_check_status()
        elif tool_name == "create_file":
            return self._handle_create_file(args)
        elif tool_name == "run_command":
            return self._handle_run_command(args)

        return "Unknown tool. Use create_file, run_command, or check_status."

    def _handle_check_status(self) -> str:
        if not self._completed_step_ids:
            return json.dumps({
                "completed_steps": [],
                "files_created": [],
                "commands_run": [],
                "remaining": [s["description"] for s in _PROJECT_STEPS],
            })

        completed = [
            s["description"]
            for s in _PROJECT_STEPS
            if s["id"] in self._completed_step_ids
        ]
        remaining = [
            s["description"]
            for s in _PROJECT_STEPS
            if s["id"] not in self._completed_step_ids
        ]
        return json.dumps({
            "completed_steps": completed,
            "files_created": list(self._created_files.keys()),
            "commands_run": self._commands_run,
            "remaining": remaining,
        })

    def _handle_create_file(self, args: dict) -> str:
        name = args.get("name", "")
        content = args.get("content", "")

        # Check if this is a redundant step
        if name in self._created_files:
            self._redundant_steps += 1
            return f"File '{name}' already exists (redundant step)."

        self._created_files[name] = content

        # Match against project steps
        for ps in _PROJECT_STEPS:
            if ps["action"] == "create_file" and ps["id"] not in self._completed_step_ids:
                expected_args = json.loads(ps["args"])
                if expected_args["name"] == name:
                    self._completed_step_ids.add(ps["id"])
                    # Post-session-break correct continuations
                    if self._session_broken:
                        self._correct_continuations += 1
                    break

        if len(self._completed_step_ids) >= len(_PROJECT_STEPS):
            self._done = True
            return f"File '{name}' created. All steps complete!"

        return f"File '{name}' created successfully."

    def _handle_run_command(self, args: dict) -> str:
        cmd = args.get("cmd", "")

        # Check if redundant
        if cmd in self._commands_run:
            self._redundant_steps += 1
            return f"Command already run: {cmd} (redundant step)."

        self._commands_run.append(cmd)

        # Match against project steps
        for ps in _PROJECT_STEPS:
            if ps["action"] == "run_command" and ps["id"] not in self._completed_step_ids:
                expected_args = json.loads(ps["args"])
                exp_cmd = expected_args["cmd"]
                prefix = exp_cmd.split("&&")[0].strip()
                if exp_cmd == cmd or cmd.startswith(prefix):
                    self._completed_step_ids.add(ps["id"])
                    if self._session_broken:
                        self._correct_continuations += 1
                    break

        if len(self._completed_step_ids) >= len(_PROJECT_STEPS):
            self._done = True
            return f"Command executed: {cmd}. All steps complete!"

        return f"Command executed: {cmd}"

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
        total_steps = len(_PROJECT_STEPS)
        completed = len(self._completed_step_ids)
        post_break_steps = total_steps - self._session_break_at
        continuation_acc = (
            self._correct_continuations / post_break_steps * 100
            if post_break_steps > 0
            else 0.0
        )
        overall_acc = completed / total_steps * 100 if total_steps > 0 else 0.0

        return TaskResult(
            completion=completed == total_steps,
            accuracy=overall_acc,
            steps=self._step,
            extra={
                "completed_steps": completed,
                "total_steps": total_steps,
                "redundant_steps": self._redundant_steps,
                "correct_continuations": self._correct_continuations,
                "continuation_accuracy": continuation_acc,
                "session_broken": self._session_broken,
            },
        )
