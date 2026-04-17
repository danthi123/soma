"""Load and iterate over LongMemEval dataset files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"

ORACLE_FILE = "longmemeval_oracle.json"
SMALL_FILE = "longmemeval_s_cleaned.json"
MEDIUM_FILE = "longmemeval_m_cleaned.json"


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    has_answer: bool = False


@dataclass(frozen=True)
class LongMemEvalItem:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_dates: list[str]
    haystack_session_ids: list[str]
    haystack_sessions: list[list[Turn]]
    answer_session_ids: list[str]

    @property
    def num_sessions(self) -> int:
        return len(self.haystack_sessions)

    @property
    def total_turns(self) -> int:
        return sum(len(s) for s in self.haystack_sessions)


def _parse_turns(raw_session: list[dict[str, Any]]) -> list[Turn]:
    turns: list[Turn] = []
    for t in raw_session:
        turns.append(
            Turn(
                role=t["role"],
                content=t["content"],
                has_answer=bool(t.get("has_answer", False)),
            )
        )
    return turns


def load_dataset(
    variant: str = "oracle",
    data_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[LongMemEvalItem]:
    """Load a LongMemEval dataset variant.

    Parameters
    ----------
    variant:
        One of ``"oracle"``, ``"small"``, ``"medium"``.
    data_dir:
        Override directory containing the JSON files.
    limit:
        Cap the number of items returned (for smoke tests).
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    filenames = {
        "oracle": ORACLE_FILE,
        "small": SMALL_FILE,
        "medium": MEDIUM_FILE,
    }
    if variant not in filenames:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(filenames)}")
    path = root / filenames[variant]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Download it with:\n"
            f"  curl -sL https://huggingface.co/datasets/xiaowu0162/"
            f"longmemeval-cleaned/resolve/main/{filenames[variant]} -o {path}"
        )
    with open(path, encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    items: list[LongMemEvalItem] = []
    for entry in raw:
        sessions = [_parse_turns(s) for s in entry["haystack_sessions"]]
        items.append(
            LongMemEvalItem(
                question_id=entry["question_id"],
                question_type=entry["question_type"],
                question=entry["question"],
                answer=entry["answer"],
                question_date=entry.get("question_date", ""),
                haystack_dates=entry.get("haystack_dates", []),
                haystack_session_ids=entry.get("haystack_session_ids", []),
                haystack_sessions=sessions,
                answer_session_ids=entry.get("answer_session_ids", []),
            )
        )
    if limit is not None:
        items = items[:limit]
    return items
