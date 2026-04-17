"""Task 1: Deep fact recall.

100-turn simulated conversation.  Turns 1-10 establish 5 facts about
a fictional user.  Turns 11-95 are semantically-unrelated filler.
Turns 96-100 ask about the original facts.

The agent's only tool is ``ask_user(question)`` which returns a scripted
answer.  Scoring is based on accuracy of recalled facts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult

# ---------------------------------------------------------------------------
# Fact sets (keyed by seed for reproducibility)
# ---------------------------------------------------------------------------
_FACT_POOLS: list[dict[str, str]] = [
    {
        "name": "Alice Chen",
        "city": "Portland",
        "food": "pad thai",
        "job": "marine biologist",
        "pet": "a bearded dragon named Spike",
    },
    {
        "name": "Marcus Rivera",
        "city": "Lisbon",
        "food": "ceviche",
        "job": "sound engineer",
        "pet": "two parakeets named Salt and Pepper",
    },
    {
        "name": "Yuki Tanaka",
        "city": "Vancouver",
        "food": "poutine",
        "job": "data journalist",
        "pet": "a rescue greyhound named Dash",
    },
]

# Filler topics -- deliberately unrelated to any personal facts
_FILLER_TOPICS: list[str] = [
    "What do you think about the latest developments in quantum computing?",
    "Can you explain how tides work?",
    "Tell me about the history of chess.",
    "What are the main differences between TCP and UDP?",
    "How does sourdough bread fermentation work?",
    "What is the Banach-Tarski paradox?",
    "Explain the water cycle in simple terms.",
    "What are the tallest buildings in the world?",
    "How do noise-canceling headphones work?",
    "Tell me about the Voyager space probes.",
    "What is the difference between a virus and a bacterium?",
    "How does a combustion engine work?",
    "Explain the concept of supply and demand.",
    "What are prime numbers and why do they matter?",
    "Tell me about the history of the Olympic Games.",
    "How does GPS determine your location?",
    "What are the different types of clouds?",
    "Explain how a vaccine works.",
    "What is the Fibonacci sequence?",
    "How do airplanes generate lift?",
]


@dataclass
class FactRecallTask:
    """Deep fact recall over a 100-turn conversation."""

    seed: int = 0
    _turn: int = 0
    _max_turns: int = 100
    _facts: dict[str, str] = field(default_factory=dict)
    _recall_questions: list[dict[str, str]] = field(default_factory=list)
    _recall_answers: list[str] = field(default_factory=list)
    _correct: int = 0
    _total_questions: int = 0
    _done: bool = False
    _rng: random.Random = field(default_factory=random.Random)

    # -- Conversation script built at setup time --
    _script: list[dict[str, str]] = field(default_factory=list)

    def setup(self) -> str:
        self._rng = random.Random(self.seed)
        self._facts = _FACT_POOLS[self.seed % len(_FACT_POOLS)].copy()
        self._turn = 0
        self._correct = 0
        self._recall_answers = []
        self._done = False

        # Build the conversation script
        self._script = []

        # Turns 1-10: establish facts (2 turns per fact)
        fact_items = list(self._facts.items())
        for _i, (key, value) in enumerate(fact_items):
            # User volunteers the fact
            self._script.append(
                {
                    "role": "user_message",
                    "content": self._fact_introduction(key, value),
                }
            )
            # Filler acknowledgment turn
            self._script.append(
                {
                    "role": "user_message",
                    "content": "Thanks for noting that! Anyway, moving on...",
                }
            )

        # Turns 11-95: filler
        fillers = self._rng.sample(
            _FILLER_TOPICS * 5, k=85
        )  # oversample, take 85
        for f in fillers:
            self._script.append({"role": "user_message", "content": f})

        # Turns 96-100: recall questions
        self._recall_questions = [
            {"key": "name", "question": "What is my name?"},
            {"key": "city", "question": "What city do I live in?"},
            {"key": "food", "question": "What is my favorite food?"},
            {"key": "job", "question": "What do I do for a living?"},
            {"key": "pet", "question": "Tell me about my pet."},
        ]
        for q in self._recall_questions:
            self._script.append({"role": "recall_question", "content": q["question"]})

        self._total_questions = len(self._recall_questions)

        # Return initial observation (system prompt + first user message)
        return self._format_observation(
            "You are a helpful assistant chatting with a user. "
            "Pay attention to personal details the user shares.\n\n"
            f"User: {self._script[0]['content']}"
        )

    def execute_action(self, action: str) -> str:
        """Process agent response and return next observation."""
        self._turn += 1

        # Check if we're in the recall phase
        recall_start = len(self._script) - len(self._recall_questions)
        if self._turn > recall_start:
            # Agent just answered a recall question -- score it
            q_idx = self._turn - recall_start - 1
            if q_idx < len(self._recall_questions):
                key = self._recall_questions[q_idx]["key"]
                expected = self._facts[key]
                self._recall_answers.append(action)
                if expected.lower() in action.lower():
                    self._correct += 1

        # Advance to next script entry
        if self._turn < len(self._script):
            entry = self._script[self._turn]
            return self._format_observation(f"User: {entry['content']}")

        self._done = True
        return "Conversation complete."

    def is_complete(self) -> bool:
        return self._done or self._turn >= len(self._script)

    def score(self) -> TaskResult:
        accuracy = (
            (self._correct / self._total_questions * 100)
            if self._total_questions > 0
            else 0.0
        )
        return TaskResult(
            completion=self._correct == self._total_questions,
            accuracy=accuracy,
            steps=self._turn,
            extra={
                "correct": self._correct,
                "total": self._total_questions,
                "answers": self._recall_answers,
            },
        )

    # -- Helpers --
    @staticmethod
    def _fact_introduction(key: str, value: str) -> str:
        templates = {
            "name": f"Hi! My name is {value}.",
            "city": f"I live in {value}, by the way.",
            "food": f"I absolutely love {value} -- it's my favorite food!",
            "job": f"I work as a {value}.",
            "pet": f"I have {value}.",
        }
        return templates.get(key, f"My {key} is {value}.")

    @staticmethod
    def _format_observation(content: str) -> str:
        return content
