"""Synthetic corpus generator for scale benchmarks (100K → 1M+ entries).

The hand-curated and templated generators top out at low thousands and
~30K respectively. For enterprise-scale benchmarks (100K, 1M) we
generate N deterministically-unique short documents from a fixed
topic + filler vocab, encoded so retrieval is meaningful (queries on
topic_id should retrieve facts with that topic).

Each entry looks like::

    "doc {id:07d} about topic_{topic_id}: {filler_words}"

where filler_words is 5 random words from a fixed 200-word vocabulary
seeded by id, keeping each entry's text unique while letting the
sbert embedding cluster on the topic_id token.

This is **not** a quality benchmark — sbert can't distinguish 100K
near-identical templated strings beyond the topic_id signal. It's a
**scale-and-throughput** benchmark: how do store/retrieve/disk scale
when the index grows past where the small-N benchmarks stop?
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# noqa: E501 — vocab list is intentionally one-line literal
VOCAB = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "red", "green", "blue", "yellow", "purple", "orange", "black", "white",
    "fast", "slow", "heavy", "light", "bright", "dim", "warm", "cool",
    "soft", "hard",
    "north", "south", "east", "west", "up", "down", "left", "right",
    "above", "below",
    "winter", "spring", "summer", "fall", "morning", "noon", "evening",
    "night",
    "river", "mountain", "forest", "ocean", "desert", "valley", "plateau",
    "cliff",
    "iron", "stone", "wood", "paper", "glass", "copper", "silver", "gold",
    "tin", "lead",
    "happy", "quiet", "bold", "gentle", "fierce", "calm", "steady", "wild",
    "tame", "keen",
    "round", "square", "narrow", "wide", "deep", "shallow", "flat", "steep",
    "curved", "sharp",
    "metal", "plastic", "fabric", "leather", "rubber", "wax", "clay", "foam",
    "bone", "shell",
    "circle", "triangle", "hexagon", "octagon", "spiral", "diamond", "cube",
    "prism", "cone",
    "twilight", "midnight", "afternoon", "dawn", "dusk", "midday",
    "city", "village", "hamlet", "town", "port", "harbor", "outpost",
    "station", "hub", "depot",
    "bicycle", "locomotive", "ship", "aircraft", "submarine", "glider",
    "rover", "transport",
    "rain", "snow", "fog", "mist", "hail", "thunder", "lightning", "breeze",
    "gust", "storm",
    "history", "physics", "botany", "chemistry", "biology", "geology",
    "astronomy", "ecology",
    "music", "dance", "theater", "poetry", "sculpture", "painting",
    "weaving", "pottery",
]


@dataclass(frozen=True)
class ScaleFact:
    id: int
    topic_id: int
    text: str


def generate_scale_corpus(
    n: int,
    *,
    n_topics: int = 100,
    seed: int = 42,
    filler_words_per_doc: int = 5,
) -> list[ScaleFact]:
    """Generate ``n`` unique scale-test documents.

    Topics are uniformly distributed (each topic gets ~n/n_topics docs).
    """
    rng = random.Random(seed)
    facts: list[ScaleFact] = []
    for i in range(n):
        topic_id = i % n_topics
        # Per-doc seeded RNG so the filler stays reproducible regardless
        # of generation order.
        local = random.Random(seed * 1000003 + i)
        filler = " ".join(local.choices(VOCAB, k=filler_words_per_doc))
        text = f"doc {i:07d} about topic_{topic_id}: {filler}"
        facts.append(ScaleFact(id=i, topic_id=topic_id, text=text))
    _ = rng  # appease lint (kept seeded RNG for future expansion)
    return facts


def topic_query(topic_id: int) -> str:
    """Canonical query for a topic — used to score retrieval at scale."""
    return f"topic_{topic_id}"
