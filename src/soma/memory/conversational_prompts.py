"""Prompt templates for :class:`soma.memory.conversational.ConversationalMemory`.

Four string templates:

- :data:`EXTRACT_PROMPT` — turns a user message into a JSON list of
  atomic facts. Closed-vocab category, JSON-only output, empty-list
  example as a small-model anchor.
- :data:`RECONCILE_PROMPT` — given a new fact and the top-k existing
  candidates, decide one of ADD / UPDATE / SUPERSEDE / NOOP.
- :data:`SUMMARY_PROMPT` — roll N recent turns into a short paragraph,
  with an explicit "do not invent" clause to curb drift.
- :data:`RESUMMARY_PROMPT` — every Mth summary, re-derive from raw
  turns only (no previous-summary dependency) to break the chained-
  summarization drift loop. See Phase 17 plan.

Interpolation uses :meth:`str.format` — no templating engine. Callers
pass keyword arguments (``message``, ``new_fact`` + ``candidates``,
``turns``) and get a ready-to-send string back.

Exact wording is pinned by ``tests/test_memory/test_conversational_prompts.py``
so a refactor can't silently regress extraction or reconcile quality.
"""

from __future__ import annotations

EXTRACT_PROMPT = (
    """You extract atomic facts from a single user message.

Return ONLY valid JSON — a list of objects. No prose, no code fences.
Each object has exactly two keys:
  - "category": one of ["identity", "location", "preference", """
    """"relationship", "goal", "other"]
  - "text": a single short declarative sentence about the user.

If the message contains no facts worth remembering (greetings,
filler, questions with no new info), return an empty list: [].

Examples:
  Message: "Hey"
  Output: []

  Message: "I'm Alex and I live in Boston"
  Output: [{{"category": "identity", "text": "User's name is Alex"}}, """
    """{{"category": "location", "text": "User lives in Boston"}}]

  Message: "I love Thai food but hate cilantro"
  Output: [{{"category": "preference", "text": "User loves Thai food"}}, """
    """{{"category": "preference", "text": "User hates cilantro"}}]

Now extract facts from this message:
  Message: {message}
  Output:"""
)


BATCH_EXTRACT_PROMPT = (
    """You extract atomic facts from a batch of {n} conversation turns.

Return ONLY valid JSON — a list of objects. No prose, no code fences.
Each object has exactly four keys:
  - "turn_index": integer 0..{max_idx} — which turn produced this fact.
  - "category": one of ["identity", "location", "preference", """
    """"relationship", "goal", "other"]
  - "text": a single short declarative sentence about the user.

If a turn contains no facts worth remembering (greetings, filler,
questions with no new info), emit no entry for it. If the entire
batch is factless, return an empty list: [].

Example:
  Turn 0 (user): Hey
  Turn 1 (user): I'm Alex and I live in Boston
  Output: [{{"turn_index": 1, "category": "identity", """
    """"text": "User's name is Alex"}}, {{"turn_index": 1, """
    """"category": "location", "text": "User lives in Boston"}}]

Now extract facts from this batch:
{turns}
Output:"""
)


RECONCILE_PROMPT = (
    """You reconcile a new fact against existing stored facts.

Return ONLY a JSON object (no prose, no code fences) with these keys:
  - "op": one of "ADD", "UPDATE", "SUPERSEDE", "NOOP"
  - "target_id": the id of an existing fact """
    """(required for UPDATE / SUPERSEDE; null otherwise)
  - "reason": one short sentence explaining the decision

Op meanings:
  - ADD: new information, no existing fact is related enough to merge.
  - UPDATE: the new fact refines an existing one """
    """(same topic, more detail). Use target_id.
  - SUPERSEDE: the new fact contradicts or replaces an older one """
    """(user moved, changed preference). Use target_id.
  - NOOP: the new fact is already covered by existing facts; store nothing.

New fact: {new_fact}

Existing candidates:
{candidates}

Output:"""
)


SUMMARY_PROMPT = """You are summarizing a short segment of a conversation for long-term memory.

Write one paragraph (3-5 sentences) capturing only what was said.
Stay faithful to the turns — do not invent details, do not speculate,
do not add framing the speakers did not express.

Conversation turns:
{turns}

Summary:"""


RESUMMARY_PROMPT = """Summarise the following conversation turns into \
a short, factual summary (3-5 sentences). Focus on stable information \
about the participants (names, locations, preferences, goals) and \
on decisions / commitments that were made. Ignore small-talk unless \
it reveals stable facts. Do not reference any prior summary — this \
summary is being re-derived from the raw turns below.

Turns:
{turns}

Summary:"""
