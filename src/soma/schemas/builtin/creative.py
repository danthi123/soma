"""Creative domain schemas — characters, world details, continuity, plot threads."""

from __future__ import annotations

from soma.schemas import field, schema

__all__ = ["Character", "WorldDetail", "Continuity", "PlotThread"]


@schema("creative.character")
class Character:
    """A fictional character with traits, relationships, and arc tracking."""

    name: str = field(filterable=True, searchable=True)
    project: str = field(filterable=True, default="")
    traits: str = field(searchable=True, default="")  # comma-separated
    relationships: str = field(searchable=True, default="")
    arc: str = field(searchable=True, default="")
    last_appearance: str = field(default="")


@schema("creative.world_detail")
class WorldDetail:
    """A canonical or proposed detail about a fictional world."""

    project: str = field(filterable=True)
    category: str = field(
        filterable=True,
        choices=[
            "geography",
            "politics",
            "magic_system",
            "technology",
            "culture",
            "history",
        ],
    )
    detail: str = field(searchable=True)
    canonical: bool = field(filterable=True, default=True)
    introduced_in: str = field(default="")


@schema("creative.continuity")
class Continuity:
    """A continuity assertion that may be contradicted by later material."""

    project: str = field(filterable=True)
    assertion: str = field(searchable=True)
    source: str = field(filterable=True, default="")
    contradicted_by: str | None = field(default=None, filterable=True)


@schema("creative.plot_thread")
class PlotThread:
    """A narrative thread with lifecycle tracking."""

    project: str = field(filterable=True)
    thread: str = field(searchable=True)
    status: str = field(
        filterable=True,
        choices=["planted", "developing", "resolved", "abandoned"],
        default="planted",
    )
    introduced_in: str = field(default="")
    resolved_in: str | None = field(default=None)
