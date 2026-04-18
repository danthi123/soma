"""Load and iterate over the LoCoMo (Long Conversation Memory) dataset.

LoCoMo contains 10 long conversations (19-32 sessions each, spanning months)
with ~2000 QA pairs across 5 categories:
  1 = single-hop factual
  2 = temporal reasoning
  3 = multi-hop reasoning
  4 = open-domain knowledge
  5 = adversarial / unanswerable

Data source: https://github.com/Backboard-io/Backboard-Locomo-Benchmark
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent
DATASET_FILE = "locomo_dataset.json"

CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str
    dia_id: str


@dataclass(frozen=True)
class Session:
    index: int
    date_time: str
    turns: list[Turn]


@dataclass(frozen=True)
class QAPair:
    question: str
    answer: str
    evidence: list[str]
    category: int

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"unknown_{self.category}")


@dataclass(frozen=True)
class LoCoMoConversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa_pairs: list[QAPair]
    event_summary: Any = field(default=None, repr=False)
    observation: Any = field(default=None, repr=False)
    session_summary: Any = field(default=None, repr=False)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def total_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    @property
    def num_qa(self) -> int:
        return len(self.qa_pairs)


def _parse_conversation(raw: dict[str, Any]) -> LoCoMoConversation:
    conv = raw["conversation"]

    # Extract sessions (keys like session_1, session_2, ...)
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    sessions = []
    for sk in session_keys:
        idx = int(sk.split("_")[1])
        date_key = f"{sk}_date_time"
        date_time = conv.get(date_key, "")
        turns = [
            Turn(speaker=t["speaker"], text=t["text"], dia_id=t["dia_id"])
            for t in conv[sk]
        ]
        sessions.append(Session(index=idx, date_time=date_time, turns=turns))

    qa_pairs = [
        QAPair(
            question=qa["question"],
            answer=qa.get("answer", qa.get("adversarial_answer", "")),
            evidence=qa["evidence"],
            category=qa["category"],
        )
        for qa in raw["qa"]
    ]

    return LoCoMoConversation(
        sample_id=raw["sample_id"],
        speaker_a=conv.get("speaker_a", ""),
        speaker_b=conv.get("speaker_b", ""),
        sessions=sessions,
        qa_pairs=qa_pairs,
        event_summary=raw.get("event_summary"),
        observation=raw.get("observation"),
        session_summary=raw.get("session_summary"),
    )


def load_dataset(
    data_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[LoCoMoConversation]:
    """Load the LoCoMo dataset.

    Parameters
    ----------
    data_dir:
        Override directory containing ``locomo_dataset.json``.
    limit:
        Cap the number of conversations returned.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    path = root / DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Download it with:\n"
            f"  curl -sL https://raw.githubusercontent.com/Backboard-io/"
            f"Backboard-Locomo-Benchmark/main/locomo_dataset.json -o {path}"
        )

    with open(path, encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    items = [_parse_conversation(entry) for entry in raw]
    if limit is not None:
        items = items[:limit]
    return items
