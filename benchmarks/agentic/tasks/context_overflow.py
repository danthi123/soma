"""Task 3: Context-window overflow.

The agent processes 50 documents via a ``read_document(id)`` tool.
Each document is ~500 tokens with one embedded key fact.  Then 10
synthesis questions require combining facts from multiple documents.

Scoring: per-document fact-extraction accuracy + synthesis accuracy.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult

# ---------------------------------------------------------------------------
# Document / fact generation
# ---------------------------------------------------------------------------
_TOPICS = [
    "renewable energy", "ocean currents", "compiler design",
    "ancient Rome", "fungi biology", "urban planning",
    "cryptography", "glaciology", "behavioral economics",
    "marine biology", "astrophysics", "linguistics",
    "robotics", "archaeology", "meteorology",
    "genetics", "philosophy", "volcanology",
    "music theory", "agricultural science",
]

_FILLER_SENTENCES = [
    "This area of study has seen significant advances in recent years.",
    "Researchers continue to investigate the underlying mechanisms.",
    "The implications extend across multiple disciplines.",
    "Historical context provides important perspective on current findings.",
    "Methodological improvements have enabled more precise measurements.",
    "Cross-disciplinary collaboration has accelerated progress.",
    "Computational models have become increasingly sophisticated.",
    "Field observations complement laboratory experiments.",
    "Public interest in this topic has grown substantially.",
    "Standardized frameworks facilitate comparison across studies.",
]


def _generate_documents(
    rng: random.Random, count: int = 50
) -> tuple[list[dict[str, str]], dict[int, str]]:
    """Generate ``count`` documents with embedded key facts.

    Returns (documents, facts_by_doc_id).
    """
    topics = rng.sample(_TOPICS * 3, k=count)
    docs: list[dict[str, str]] = []
    facts: dict[int, str] = {}

    for i in range(count):
        topic = topics[i]
        val = rng.randint(100, 999)
        key_fact = (
            f"The key finding in document {i} about {topic}"
            f" is VALUE_{i}_{val}."
        )
        # Build ~500 tokens of filler around the fact
        filler_lines = rng.choices(_FILLER_SENTENCES, k=15)
        insert_pos = rng.randint(4, 12)
        filler_lines.insert(insert_pos, key_fact)
        body = (
            f"Document {i}: {topic.title()}\n\n"
            + " ".join(filler_lines)
        )
        docs.append({"id": str(i), "topic": topic, "body": body})
        facts[i] = key_fact

    return docs, facts


def _generate_synthesis_questions(
    rng: random.Random, facts: dict[int, str], count: int = 10
) -> list[dict]:
    """Generate synthesis questions that reference 2-3 documents each."""
    doc_ids = list(facts.keys())
    questions = []
    for q_idx in range(count):
        refs = rng.sample(doc_ids, k=min(3, len(doc_ids)))
        q = {
            "question": (
                f"Synthesis Q{q_idx + 1}: What are the key findings from "
                f"documents {', '.join(str(r) for r in refs)}?"
            ),
            "referenced_docs": refs,
            "expected_facts": [facts[r] for r in refs],
        }
        questions.append(q)
    return questions


@dataclass
class ContextOverflowTask:
    """Context-window overflow benchmark."""

    seed: int = 0
    _step: int = 0
    _phase: str = "reading"  # "reading" | "synthesis" | "done"
    _docs: list[dict[str, str]] = field(default_factory=list)
    _facts: dict[int, str] = field(default_factory=dict)
    _synthesis_qs: list[dict] = field(default_factory=list)
    _current_doc_idx: int = 0
    _synthesis_idx: int = 0
    _extraction_correct: int = 0
    _synthesis_correct: int = 0
    _synthesis_answers: list[str] = field(default_factory=list)
    _done: bool = False
    _max_steps: int = 200
    _rng: random.Random = field(default_factory=random.Random)

    TOOLS: list[dict] = field(default_factory=lambda: [
        {
            "type": "function",
            "function": {
                "name": "read_document",
                "description": "Read a document by ID (0-49).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Document ID"},
                    },
                    "required": ["id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_fact",
                "description": "Submit the key fact extracted from a document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "fact": {"type": "string"},
                    },
                    "required": ["doc_id", "fact"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "answer_synthesis",
                "description": "Answer a synthesis question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question_id": {"type": "integer"},
                        "answer": {"type": "string"},
                    },
                    "required": ["question_id", "answer"],
                },
            },
        },
    ])

    def setup(self) -> str:
        self._rng = random.Random(self.seed)
        self._docs, self._facts = _generate_documents(self._rng)
        self._synthesis_qs = _generate_synthesis_questions(self._rng, self._facts)
        self._step = 0
        self._phase = "reading"
        self._current_doc_idx = 0
        self._synthesis_idx = 0
        self._extraction_correct = 0
        self._synthesis_correct = 0
        self._synthesis_answers = []
        self._done = False

        return (
            "You will process 50 documents one at a time. For each document:\n"
            "1. Call read_document(id) to read it\n"
            "2. Call submit_fact(doc_id, fact) with the KEY finding\n\n"
            "After all 50 documents, you will answer 10 synthesis questions.\n\n"
            "Start by reading document 0."
        )

    def execute_action(self, action: str) -> str:
        self._step += 1
        if self._step >= self._max_steps:
            self._done = True
            return "Maximum steps reached."

        tool_name, args = self._parse_tool_call(action)

        if tool_name == "read_document":
            return self._handle_read(args)
        elif tool_name == "submit_fact":
            return self._handle_submit_fact(args)
        elif tool_name == "answer_synthesis":
            return self._handle_synthesis(args)

        return (
            "Please use one of the available tools: "
            "read_document, submit_fact, or answer_synthesis."
        )

    def _handle_read(self, args: dict) -> str:
        doc_id_str = args.get("id", "0")
        try:
            doc_id = int(doc_id_str)
        except (ValueError, TypeError):
            return f"Invalid document ID: {doc_id_str}"
        if doc_id < 0 or doc_id >= len(self._docs):
            return f"Document ID must be 0-{len(self._docs) - 1}."
        return self._docs[doc_id]["body"]

    def _handle_submit_fact(self, args: dict) -> str:
        doc_id_str = args.get("doc_id", "0")
        fact = args.get("fact", "")
        try:
            doc_id = int(doc_id_str)
        except (ValueError, TypeError):
            return f"Invalid doc_id: {doc_id_str}"

        expected = self._facts.get(doc_id, "")
        # Check if the submitted fact contains the key value
        key_value = expected.split("is ")[-1].rstrip(".")
        if key_value.lower() in fact.lower():
            self._extraction_correct += 1
            result = "Correct!"
        else:
            result = "Noted."

        self._current_doc_idx = max(self._current_doc_idx, doc_id + 1)

        if self._current_doc_idx >= len(self._docs) and self._phase == "reading":
            self._phase = "synthesis"
            return (
                f"{result}\n\nAll documents processed. "
                "Now answer synthesis questions.\n\n"
                f"{self._synthesis_qs[0]['question']}"
            )

        if self._phase == "reading":
            return f"{result} Now read document {self._current_doc_idx}."
        return result

    def _handle_synthesis(self, args: dict) -> str:
        q_id = args.get("question_id", self._synthesis_idx)
        answer = args.get("answer", "")
        self._synthesis_answers.append(answer)

        if isinstance(q_id, str):
            try:
                q_id = int(q_id)
            except ValueError:
                q_id = self._synthesis_idx

        # Score: check if any expected facts appear in the answer
        if q_id < len(self._synthesis_qs):
            q = self._synthesis_qs[q_id]
            matches = sum(
                1 for ef in q["expected_facts"]
                if ef.split("is ")[-1].rstrip(".").lower() in answer.lower()
            )
            if matches > 0:
                self._synthesis_correct += 1

        self._synthesis_idx += 1
        if self._synthesis_idx >= len(self._synthesis_qs):
            self._done = True
            return "All synthesis questions answered. Task complete."

        return (
            f"Answer recorded. Next question:\n\n"
            f"{self._synthesis_qs[self._synthesis_idx]['question']}"
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
        doc_count = len(self._docs)
        synth_count = len(self._synthesis_qs)
        extract_acc = self._extraction_correct / doc_count * 100 if doc_count else 0
        synth_acc = self._synthesis_correct / synth_count * 100 if synth_count else 0
        overall = (extract_acc + synth_acc) / 2

        return TaskResult(
            completion=self._done and self._synthesis_idx >= synth_count,
            accuracy=overall,
            steps=self._step,
            extra={
                "extraction_accuracy": extract_acc,
                "synthesis_accuracy": synth_acc,
                "extraction_correct": self._extraction_correct,
                "synthesis_correct": self._synthesis_correct,
                "docs_processed": self._current_doc_idx,
            },
        )
