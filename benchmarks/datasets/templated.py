"""Template-based synthetic fact generator for scale experiments.

The hand-curated ``synthetic.py`` set tops out at ~50 facts — enough for
labeled-query quality benchmarks but too small to stress SOMA's graph
growth behavior. This generator produces hundreds to thousands of
plausible facts by permuting template slots, so downstream code can
study how graph size, disk footprint, and retrieval latency scale with
store size.

The facts are deliberately similar in shape (a handful of recurring
templates) so the graph has opportunities to specialize on shared
sub-structure (names, cities, roles) — exactly the regime where
plasticity should show benefit. The generator is reproducible: pass a
``seed`` for deterministic output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PEOPLE = [
    "Alex", "Jordan", "Sam", "Morgan", "Taylor", "Casey", "Riley",
    "Quinn", "Avery", "Jamie", "Dana", "Pat", "Lee", "Robin", "Skyler",
    "Cameron", "Drew", "Parker", "Reese", "Sage",
]
CITIES = [
    "Portland", "Seattle", "Denver", "Austin", "Boston", "Chicago",
    "Miami", "Phoenix", "Nashville", "Minneapolis", "Pittsburgh",
    "Raleigh", "Richmond", "Tampa", "Orlando", "Atlanta",
]
COMPANIES = [
    "ArcMotion", "NovaGrid", "PulseLabs", "OrbitalML", "Northwind",
    "TriForce", "HelioSoft", "RiverRun", "Kestrel", "BlueSpruce",
    "Fathom", "PineCone", "Solace", "MagnaTech", "Zenith",
]
ROLES = [
    "senior engineer", "designer", "architect", "product manager",
    "data scientist", "researcher", "team lead", "staff engineer",
    "CTO", "founder",
]
PETS = ["dog", "cat", "parrot", "rabbit", "ferret", "hamster"]
BREEDS = {
    "dog": ["border collie", "labrador", "husky", "poodle", "beagle"],
    "cat": ["siamese", "persian", "ragdoll", "maine coon", "tabby"],
    "parrot": ["macaw", "cockatoo", "african grey", "conure"],
    "rabbit": ["holland lop", "flemish giant", "rex"],
    "ferret": ["albino", "sable"],
    "hamster": ["syrian", "dwarf"],
}
PET_NAMES = [
    "Luna", "Bandit", "Cocoa", "Juno", "Nimbus", "Pepper", "Willow",
    "Shadow", "Rusty", "Whiskers", "Biscuit", "Clover", "Delta", "Echo",
]
HOBBIES = [
    "rock climbing", "trail running", "pottery", "chess", "birding",
    "cycling", "woodworking", "photography", "baking", "gardening",
]
CITIES_VISITED = [
    "Japan", "Iceland", "Peru", "Morocco", "Vietnam", "Norway", "Kenya",
    "Thailand", "Portugal", "Greece",
]
MONTHS = [
    "January", "March", "April", "June", "July", "August", "October",
    "November", "December",
]


TOPIC_TEMPLATES: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "location": [
        ("{person} lives in {city}", ("person", "city")),
        ("{person}'s neighborhood is in {city}", ("person", "city")),
    ],
    "work": [
        (
            "{person} works at {company} as a {role}",
            ("person", "company", "role"),
        ),
        ("{person} leads the {role} team at {company}", ("person", "role", "company")),
    ],
    "pet": [
        (
            "{person}'s {pet} is named {pet_name}",
            ("person", "pet", "pet_name"),
        ),
        (
            "{person}'s {pet} is a {breed} named {pet_name}",
            ("person", "pet", "breed", "pet_name"),
        ),
    ],
    "hobby": [
        ("{person} enjoys {hobby}", ("person", "hobby")),
        ("{person} spends weekends on {hobby}", ("person", "hobby")),
    ],
    "travel": [
        (
            "{person} visited {country} in {month}",
            ("person", "country", "month"),
        ),
    ],
}


@dataclass(frozen=True)
class TemplatedFact:
    topic: str
    text: str
    slots: dict[str, str]


def _fill_slots(rng: random.Random, slots: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    pet_choice = rng.choice(PETS)
    for slot in slots:
        if slot == "person":
            values[slot] = rng.choice(PEOPLE)
        elif slot == "city":
            values[slot] = rng.choice(CITIES)
        elif slot == "company":
            values[slot] = rng.choice(COMPANIES)
        elif slot == "role":
            values[slot] = rng.choice(ROLES)
        elif slot == "pet":
            values[slot] = pet_choice
        elif slot == "pet_name":
            values[slot] = rng.choice(PET_NAMES)
        elif slot == "breed":
            values[slot] = rng.choice(BREEDS[pet_choice])
        elif slot == "hobby":
            values[slot] = rng.choice(HOBBIES)
        elif slot == "country":
            values[slot] = rng.choice(CITIES_VISITED)
        elif slot == "month":
            values[slot] = rng.choice(MONTHS)
        else:
            raise ValueError(f"Unknown slot: {slot}")
    return values


def generate_templated_facts(
    n: int,
    *,
    seed: int = 42,
    topics: list[str] | None = None,
) -> list[TemplatedFact]:
    """Generate ``n`` unique facts permuted from shared templates.

    Uniqueness is enforced by retry; the template space is large enough
    that duplicates are rare at n <= 2000.
    """
    rng = random.Random(seed)
    topic_list = topics or list(TOPIC_TEMPLATES.keys())
    seen: set[str] = set()
    results: list[TemplatedFact] = []
    max_retries = n * 20
    attempts = 0
    while len(results) < n and attempts < max_retries:
        attempts += 1
        topic = rng.choice(topic_list)
        template, slot_names = rng.choice(TOPIC_TEMPLATES[topic])
        slots = _fill_slots(rng, slot_names)
        text = template.format(**slots)
        if text in seen:
            continue
        seen.add(text)
        results.append(TemplatedFact(topic=topic, text=text, slots=slots))
    if len(results) < n:
        raise RuntimeError(
            f"Could only produce {len(results)} unique facts (wanted {n}); "
            "expand templates or lower n."
        )
    return results
