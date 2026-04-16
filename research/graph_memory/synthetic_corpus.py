"""Controlled co-occurrence synthetic corpus for the C1 retrieval study.

Layout (per the C1 plan):

* 20 topic-triples ``(A, B, C)`` of distinct made-up entity tokens.
* 10 of the triples are *transitive*: snippets mention ``(A, B)`` and
  ``(A, C)`` but never ``(B, C)``. A retriever that ignores graph
  structure can still find ``B`` from the query ``A`` (direct
  co-occurrence in text), but pulling ``B`` from the query ``C`` is
  one-hop transitive — only possible if the substrate captures the
  shared ``A`` neighbour.
* 10 of the triples are *triangles*: every pair ``(A, B)``, ``(A, C)``,
  ``(B, C)`` co-occurs. Acts as a control — both pure-vector and the
  graph-blended path should do well on these queries.
* All snippets fixed total = ``1000``. Distribution per triple ≈ 50
  snippets, evenly split across the allowed pairs.
* Plus ``unrelated`` query tokens that were never minted into a
  snippet at all — a sanity check that retrieval doesn't hallucinate
  high-confidence hits for things it never saw.

The generator is deterministic given a seed; default seed is fixed
so reruns reproduce. Outputs both the snippet list and a
machine-readable ground-truth structure the harness uses for eval.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

# Number of triples per structural class. Sum × snippets-per-triple
# must equal ``TOTAL_SNIPPETS``.
NUM_TRANSITIVE_TRIPLES: int = 10
NUM_TRIANGLE_TRIPLES: int = 10
TOTAL_SNIPPETS: int = 1000
DEFAULT_SEED: int = 20260416

TripleKind = Literal["transitive", "triangle"]


@dataclass(frozen=True)
class Triple:
    """One topic-triple ``(A, B, C)`` and its co-occurrence rule."""

    kind: TripleKind
    a: str
    b: str
    c: str

    @property
    def members(self) -> tuple[str, str, str]:
        return (self.a, self.b, self.c)

    def allowed_pairs(self) -> list[tuple[str, str]]:
        """Pairs of members that ARE allowed to co-occur in snippets."""
        if self.kind == "transitive":
            return [(self.a, self.b), (self.a, self.c)]
        # triangle: all three pairs are allowed.
        return [(self.a, self.b), (self.a, self.c), (self.b, self.c)]


@dataclass(frozen=True)
class Snippet:
    """One generated text snippet plus the pair of entities it mentions."""

    text: str
    triple_idx: int
    pair: tuple[str, str]


@dataclass(frozen=True)
class GroundTruth:
    """All evaluation hooks the harness needs.

    * ``snippets`` — full corpus, in generation order.
    * ``triples`` — the 20 entity triples with their kind.
    * ``snippets_by_entity`` — for each entity, the snippet indices
      that mention it. Used to score direct queries.
    * ``transitive_targets`` — for each ``(triple_idx, source_entity,
      target_entity)`` where ``source`` and ``target`` belong to the
      same transitive triple but never co-occur in a snippet, the set
      of snippet indices a graph-aware retriever should surface
      because they share the bridging entity. The pure-text retriever
      cannot reach those snippets via cosine alone.
    * ``unrelated_queries`` — entity tokens minted but never used. The
      eval expects retrieval to surface low-confidence / random hits
      for these.
    """

    snippets: list[Snippet]
    triples: list[Triple]
    snippets_by_entity: dict[str, list[int]] = field(default_factory=dict)
    transitive_targets: dict[tuple[int, str, str], list[int]] = field(default_factory=dict)
    unrelated_queries: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sentence templates
# ---------------------------------------------------------------------------
# Templates take exactly two entity tokens and produce a short, semantically
# coherent English sentence. Diversity here matters: if every snippet
# follows the same pattern, sbert embeddings collapse and we measure the
# template instead of the entities. Twelve templates is enough for sbert
# to actually reflect the entity composition.
_TEMPLATES: list[str] = [
    "the {0} works closely with the {1} every morning",
    "researchers found that the {0} responds to the {1}",
    "in the report, the {0} is paired with the {1}",
    "during the experiment the {0} interacted with the {1}",
    "the {0} was observed near the {1} on tuesday",
    "engineers connected the {0} module to the {1} subsystem",
    "field notes describe the {0} alongside the {1}",
    "the {0} and the {1} share a common interface",
    "the {0} relies on the {1} for stable operation",
    "the {0} was logged together with the {1}",
    "a follow-up study tracked the {0} and the {1}",
    "the {0} forms a unit with the {1}",
]


def _entity_token(triple_idx: int, role: str) -> str:
    """Stable made-up token for entity ``role`` of triple ``triple_idx``.

    Picking unique-looking pseudo-words (rather than real English nouns)
    keeps sbert from leaking real-world co-occurrence priors into the
    eval. ``zenoflux17`` is in nobody's training data; ``cat`` is.
    """
    return f"{role}{triple_idx:02d}{_token_seed_word(triple_idx, role)}"


def _token_seed_word(triple_idx: int, role: str) -> str:
    """Add a short pseudo-syllable so the token isn't pure index noise."""
    syls = ["zen", "vor", "qix", "lum", "thar", "kep", "rio", "fal", "wun", "myn"]
    base = syls[(triple_idx + hash(role)) % len(syls)]
    return base


def generate(
    *,
    seed: int = DEFAULT_SEED,
    total_snippets: int = TOTAL_SNIPPETS,
    num_transitive: int = NUM_TRANSITIVE_TRIPLES,
    num_triangle: int = NUM_TRIANGLE_TRIPLES,
    num_unrelated: int = 10,
) -> GroundTruth:
    """Build the deterministic synthetic corpus + ground-truth structure.

    Snippets are evenly distributed across all ``(triple, allowed_pair)``
    slots. ``allowed_pairs`` has 2 entries for transitive triples and 3
    for triangles, so triangle triples generate 1.5× the snippets of
    transitive ones — that's intentional and matches the C1 plan
    (transitive class has fewer co-occurrence channels).

    Returns a :class:`GroundTruth` with the corpus and all eval hooks.
    """
    if num_transitive < 1 or num_triangle < 1:
        raise ValueError("need at least one transitive and one triangle triple")
    rng = random.Random(seed)

    triples: list[Triple] = []
    for i in range(num_transitive):
        triples.append(
            Triple(
                kind="transitive",
                a=_entity_token(i, "alpha"),
                b=_entity_token(i, "beta"),
                c=_entity_token(i, "gamma"),
            )
        )
    for i in range(num_transitive, num_transitive + num_triangle):
        triples.append(
            Triple(
                kind="triangle",
                a=_entity_token(i, "alpha"),
                b=_entity_token(i, "beta"),
                c=_entity_token(i, "gamma"),
            )
        )

    # Round-robin every (triple, allowed_pair) slot so the corpus is
    # balanced across structural classes. Any leftover snippets from
    # the modulo land on the early triples — at 1000 snippets / 50
    # slots that's an even 20-per-slot split.
    slots: list[tuple[int, tuple[str, str]]] = []
    for triple_idx, triple in enumerate(triples):
        for pair in triple.allowed_pairs():
            slots.append((triple_idx, pair))

    snippets: list[Snippet] = []
    template_rng = random.Random(seed + 1)
    pair_order_rng = random.Random(seed + 2)
    for n in range(total_snippets):
        triple_idx, pair = slots[n % len(slots)]
        # Half the time, swap the pair order so the surface form
        # doesn't always lead with the same entity (otherwise sbert
        # learns position more than co-occurrence).
        entities = (pair[0], pair[1]) if pair_order_rng.random() < 0.5 else (pair[1], pair[0])
        template = _TEMPLATES[template_rng.randrange(len(_TEMPLATES))]
        text = template.format(entities[0], entities[1])
        snippets.append(Snippet(text=text, triple_idx=triple_idx, pair=pair))

    rng.shuffle(snippets)

    # ----- Ground-truth indexes ----------------------------------------
    snippets_by_entity: dict[str, list[int]] = {}
    for idx, snip in enumerate(snippets):
        for ent in snip.pair:
            snippets_by_entity.setdefault(ent, []).append(idx)

    transitive_targets: dict[tuple[int, str, str], list[int]] = {}
    for triple_idx, triple in enumerate(triples):
        if triple.kind != "transitive":
            continue
        # The "B and C never co-occur" rule means: querying B should
        # surface snippets mentioning C and vice versa, but only by
        # passing through A in the substrate. Ground truth = snippets
        # mentioning the *target* (which by construction also mention
        # A, since every snippet for a transitive triple contains A).
        for source, target in [(triple.b, triple.c), (triple.c, triple.b)]:
            target_indices = snippets_by_entity.get(target, [])
            transitive_targets[(triple_idx, source, target)] = list(target_indices)

    unrelated_queries = [
        f"unrelated{i:02d}{_token_seed_word(i, 'noise')}" for i in range(num_unrelated)
    ]

    return GroundTruth(
        snippets=snippets,
        triples=triples,
        snippets_by_entity=snippets_by_entity,
        transitive_targets=transitive_targets,
        unrelated_queries=unrelated_queries,
    )


def cooccurrence_counts(gt: GroundTruth) -> dict[tuple[str, str], int]:
    """Empirical pair co-occurrence count over the snippet corpus.

    Returns ``{(entity_x, entity_y): count}`` with both orderings
    populated so callers can do symmetric lookups. Used by the harness
    to compute the Spearman correlation between learned edge weights
    and ground-truth co-occurrence.
    """
    counts: dict[tuple[str, str], int] = {}
    for snip in gt.snippets:
        a, b = snip.pair
        counts[(a, b)] = counts.get((a, b), 0) + 1
        counts[(b, a)] = counts.get((b, a), 0) + 1
    return counts
