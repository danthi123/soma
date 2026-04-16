"""LoCoMo dataset loader (Snap Research, Maharana et al. 2024).

LoCoMo (LOng COnversation MEMory) is the de-facto agent-memory
benchmark: 10 long conversations (~588 turns each) between two
personas, with 1,986 question-answer pairs total. Each QA has an
``evidence`` field listing the dialogue turn IDs (``D1:3`` =
session 1, turn 3) that contain the answer.

The full LoCoMo paper evaluates QA accuracy via a GPT-4 judge;
we deliberately stop at retrieval (does the system fetch the
evidence turns?) so the benchmark stays runnable without external
API access. Retrieval Recall@k is the cleanest signal for the
memory layer's job — generation quality is on the LLM, not the
memory.

Source: https://github.com/snap-research/locomo (locomo10.json).
File is committed in-tree (~2.8 MB) so the benchmark is hermetic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).parent / "locomo10.json"

CATEGORY_NAMES: dict[int, str] = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}


@dataclass(frozen=True)
class LoCoMoTurn:
    sample_id: str
    dia_id: str  # "D1:3" = session 1, turn 3
    speaker: str
    text: str


@dataclass(frozen=True)
class LoCoMoQuery:
    sample_id: str
    question: str
    answer: str
    evidence: list[str]  # dia_id list of supporting turns
    category: int  # 1..5 mapped via CATEGORY_NAMES


def load_locomo() -> tuple[list[LoCoMoTurn], list[LoCoMoQuery]]:
    """Return (all_turns, all_queries) flattened across the 10 samples."""
    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    turns: list[LoCoMoTurn] = []
    queries: list[LoCoMoQuery] = []
    for sample in raw:
        sample_id = sample["sample_id"]
        conv = sample["conversation"]
        for k, v in conv.items():
            if k.startswith("session_") and not k.endswith("date_time") and isinstance(v, list):
                for turn in v:
                    turns.append(LoCoMoTurn(
                        sample_id=sample_id,
                        dia_id=turn["dia_id"],
                        speaker=turn["speaker"],
                        text=turn["text"],
                    ))
        for q in sample["qa"]:
            ev = q.get("evidence")
            if not ev:
                # Some adversarial questions have no evidence (the answer
                # is "I don't know"). Skip — there's no retrieval target.
                continue
            ev_list = ev if isinstance(ev, list) else [ev]
            queries.append(LoCoMoQuery(
                sample_id=sample_id,
                question=q["question"],
                answer=str(q.get("answer", "")),
                evidence=[str(e) for e in ev_list],
                category=int(q.get("category", 0)),
            ))
    return turns, queries


def turns_for_sample(turns: list[LoCoMoTurn], sample_id: str) -> list[LoCoMoTurn]:
    return [t for t in turns if t.sample_id == sample_id]


def queries_for_sample(queries: list[LoCoMoQuery], sample_id: str) -> list[LoCoMoQuery]:
    return [q for q in queries if q.sample_id == sample_id]
