"""Synthetic dataset generator with topic clusters + ground-truth queries.

Produces a reproducible dataset where each fact is tagged to a topic
cluster, and each query has one or more ground-truth relevant facts.
Topic clusters enable novel benchmarks (longitudinal drift, skewed
usage) on top of the basic retrieval benchmark.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from benchmarks.harness.runner import LabeledQuery


@dataclass(frozen=True)
class TopicFact:
    topic: str
    fact: str


TOPICS: dict[str, list[str]] = {
    "person_alex_location": [
        "Alex lives in Portland, Oregon",
        "Alex moved to Portland in 2018",
        "Alex's neighborhood is in Southeast Portland",
        "Alex's zip code is 97214",
        "Alex's address is on SE Division Street",
    ],
    "person_alex_diet": [
        "Alex is vegetarian",
        "Alex is allergic to shellfish",
        "Alex doesn't drink alcohol",
        "Alex prefers oat milk over dairy",
        "Alex avoids processed sugar",
    ],
    "person_alex_pet": [
        "Alex's dog is named Luna",
        "Luna is a 3-year-old border collie",
        "Luna was adopted from a rescue shelter",
        "Luna weighs 42 pounds",
        "Luna's favorite toy is a blue tennis ball",
    ],
    "person_alex_work": [
        "Alex works at ArcMotion, a robotics startup",
        "Alex is a senior engineer",
        "Alex leads the perception team",
        "Alex has been at ArcMotion for 3 years",
        "ArcMotion makes autonomous warehouse robots",
    ],
    "person_alex_partner": [
        "Alex's partner is named Jordan",
        "Jordan works as a veterinarian",
        "Alex and Jordan met in 2014",
        "Jordan grew up in San Diego",
        "Jordan specializes in exotic animals",
    ],
    "person_alex_hobbies": [
        "Alex enjoys trail running",
        "Alex reads philosophy on weekends",
        "Alex plays recreational soccer",
        "Alex volunteers at a food bank monthly",
        "Alex is learning to fly-fish",
    ],
    "project_deadlines": [
        "The perception module deadline is June 15",
        "The warehouse pilot launches September 1",
        "The annual all-hands meeting is May 20",
        "The robotics demo for investors is July 8",
        "The safety audit is scheduled for August 3",
    ],
    "project_tech": [
        "ArcMotion uses ROS2 for robot control",
        "ArcMotion's perception stack is written in Rust",
        "ArcMotion uses GPU servers for training",
        "ArcMotion's fleet runs on NVIDIA Jetson",
        "ArcMotion's CI runs on GitLab",
    ],
    "finance": [
        "Alex's rent is $2400 per month",
        "Alex's car payment is $650 per month",
        "Alex has a 401k at Fidelity",
        "Alex's credit score is 782",
        "Alex donates $200 monthly to charity",
    ],
    "travel": [
        "Alex visited Japan in October 2024",
        "Alex is planning a trip to Iceland next spring",
        "Alex has been to 23 countries",
        "Alex's passport expires in 2029",
        "Alex prefers window seats on flights",
    ],
}


def _topic_queries() -> dict[str, list[LabeledQuery]]:
    """Hand-curated queries per topic with ground-truth relevant facts."""
    return {
        "person_alex_location": [
            LabeledQuery("Where does Alex live?", ["Alex lives in Portland, Oregon"]),
            LabeledQuery(
                "When did Alex move?", ["Alex moved to Portland in 2018"],
            ),
            LabeledQuery(
                "What's Alex's zip code?", ["Alex's zip code is 97214"],
            ),
        ],
        "person_alex_diet": [
            LabeledQuery(
                "What are Alex's dietary restrictions?",
                ["Alex is vegetarian", "Alex is allergic to shellfish"],
            ),
            LabeledQuery("Does Alex drink?", ["Alex doesn't drink alcohol"]),
            LabeledQuery(
                "What kind of milk does Alex prefer?",
                ["Alex prefers oat milk over dairy"],
            ),
        ],
        "person_alex_pet": [
            LabeledQuery(
                "What is Alex's dog's name?", ["Alex's dog is named Luna"],
            ),
            LabeledQuery(
                "What breed is Luna?", ["Luna is a 3-year-old border collie"],
            ),
            LabeledQuery(
                "Where did Luna come from?",
                ["Luna was adopted from a rescue shelter"],
            ),
        ],
        "person_alex_work": [
            LabeledQuery(
                "Where does Alex work?",
                ["Alex works at ArcMotion, a robotics startup"],
            ),
            LabeledQuery("What's Alex's role?", ["Alex is a senior engineer"]),
            LabeledQuery(
                "What does ArcMotion make?",
                ["ArcMotion makes autonomous warehouse robots"],
            ),
        ],
        "person_alex_partner": [
            LabeledQuery(
                "Who is Alex's partner?", ["Alex's partner is named Jordan"],
            ),
            LabeledQuery(
                "What does Jordan do for work?",
                ["Jordan works as a veterinarian"],
            ),
            LabeledQuery("When did Alex and Jordan meet?", ["Alex and Jordan met in 2014"]),
        ],
        "person_alex_hobbies": [
            LabeledQuery(
                "What does Alex do for fun?",
                ["Alex enjoys trail running", "Alex plays recreational soccer"],
            ),
            LabeledQuery("Does Alex volunteer?", ["Alex volunteers at a food bank monthly"]),
        ],
        "project_deadlines": [
            LabeledQuery(
                "When is the perception module due?",
                ["The perception module deadline is June 15"],
            ),
            LabeledQuery(
                "When does the warehouse pilot launch?",
                ["The warehouse pilot launches September 1"],
            ),
            LabeledQuery(
                "When is the investor demo?",
                ["The robotics demo for investors is July 8"],
            ),
        ],
        "project_tech": [
            LabeledQuery(
                "What tech does ArcMotion use for robot control?",
                ["ArcMotion uses ROS2 for robot control"],
            ),
            LabeledQuery(
                "What language is the perception stack in?",
                ["ArcMotion's perception stack is written in Rust"],
            ),
        ],
        "finance": [
            LabeledQuery("What's Alex's rent?", ["Alex's rent is $2400 per month"]),
            LabeledQuery(
                "What's Alex's credit score?", ["Alex's credit score is 782"],
            ),
        ],
        "travel": [
            LabeledQuery(
                "When did Alex go to Japan?", ["Alex visited Japan in October 2024"],
            ),
            LabeledQuery(
                "How many countries has Alex visited?", ["Alex has been to 23 countries"],
            ),
        ],
    }


def generate_dataset(
    *,
    n_facts: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
) -> tuple[list[str], list[TopicFact], list[LabeledQuery]]:
    """Generate (facts, topic_facts, queries).

    If ``n_facts`` is None, returns all facts. Otherwise samples a
    reproducible subset spread across topics. Set ``shuffle=True`` to
    randomize fact order even when returning the full set — useful for
    isolating consolidation-order artifacts from true semantic signal.
    """
    rng = random.Random(seed)
    all_topic_facts = [
        TopicFact(topic=topic, fact=fact)
        for topic, facts in TOPICS.items()
        for fact in facts
    ]
    if n_facts is not None and n_facts < len(all_topic_facts):
        rng.shuffle(all_topic_facts)
        selected = all_topic_facts[:n_facts]
    else:
        selected = list(all_topic_facts)
        if shuffle:
            rng.shuffle(selected)

    # Only keep queries whose gold-truth facts survived the sampling.
    fact_set = {tf.fact for tf in selected}
    all_queries = [q for qs in _topic_queries().values() for q in qs]
    queries: list[LabeledQuery] = []
    for q in all_queries:
        kept = [t for t in q.relevant_texts if t in fact_set]
        if kept:
            queries.append(LabeledQuery(query=q.query, relevant_texts=kept))

    return [tf.fact for tf in selected], selected, queries


def topics_by_fact(topic_facts: list[TopicFact]) -> dict[str, str]:
    """Map fact text -> topic name (for novel benchmarks)."""
    return {tf.fact: tf.topic for tf in topic_facts}
