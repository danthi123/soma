"""Built-in domain schemas shipped with SOMA.

Importing this package auto-registers all built-in schemas in the
global registry. Individual modules can also be imported directly::

    from soma.schemas.builtin.agent import TaskState, ToolCall
    from soma.schemas.builtin.conv import Fact, Preference
    from soma.schemas.builtin.knowledge import Note, Connection
    from soma.schemas.builtin.code import CodeDecision, Incident
    from soma.schemas.builtin.research import Hypothesis, Experiment
    from soma.schemas.builtin.collab import ActionItem, CollabDecision
    from soma.schemas.builtin.customer import Profile, Sentiment
    from soma.schemas.builtin.creative import Character, PlotThread
"""

from soma.schemas.builtin.agent import Decision, Observation, TaskState, ToolCall
from soma.schemas.builtin.code import CodeDecision, DependencyNote, Incident, Pattern
from soma.schemas.builtin.collab import (
    ActionItem,
    CollabDecision,
    FollowUp,
    StakeholderPosition,
)
from soma.schemas.builtin.conv import Contradiction, Fact, Preference
from soma.schemas.builtin.creative import Character, Continuity, PlotThread, WorldDetail
from soma.schemas.builtin.customer import (
    CustomerIssue,
    CustomerPreference,
    Profile,
    Sentiment,
)
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
    # collab
    "ActionItem",
    "CollabDecision",
    "FollowUp",
    "StakeholderPosition",
    # customer
    "Profile",
    "CustomerIssue",
    "Sentiment",
    "CustomerPreference",
    # creative
    "Character",
    "WorldDetail",
    "Continuity",
    "PlotThread",
]
