"""Built-in domain schemas shipped with SOMA.

Importing this package auto-registers all built-in schemas in the
global registry. Individual modules can also be imported directly::

    from soma.schemas.builtin.agent import TaskState, ToolCall
    from soma.schemas.builtin.conv import Fact, Preference
    from soma.schemas.builtin.knowledge import Note, Connection
    from soma.schemas.builtin.code import CodeDecision, Incident
    from soma.schemas.builtin.research import Hypothesis, Experiment
"""

from soma.schemas.builtin.agent import Decision, Observation, TaskState, ToolCall
from soma.schemas.builtin.code import CodeDecision, DependencyNote, Incident, Pattern
from soma.schemas.builtin.conv import Contradiction, Fact, Preference
from soma.schemas.builtin.knowledge import Connection, Insight, Note, Question
from soma.schemas.builtin.research import Experiment, Hypothesis, Literature, Result

__all__ = [
    # agent
    "TaskState",
    "ToolCall",
    "Observation",
    "Decision",
    # conv
    "Fact",
    "Preference",
    "Contradiction",
    # knowledge
    "Note",
    "Connection",
    "Question",
    "Insight",
    # code
    "CodeDecision",
    "Pattern",
    "Incident",
    "DependencyNote",
    # research
    "Hypothesis",
    "Experiment",
    "Result",
    "Literature",
]
