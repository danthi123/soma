"""Panel registration. Importing this package registers every built-in panel."""

# Importing each panel module triggers its @register_panel decorator, which
# adds it to soma.ui.registry.default_registry. Order matters only for
# display consistency — the registry is alphabetized by panel name.
from soma.ui.panels import (  # noqa: F401
    chat_panel,
    config_panel,
    controls_panel,
    graph_panel,
    metrics_panel,
)

__all__ = [
    "chat_panel",
    "config_panel",
    "controls_panel",
    "graph_panel",
    "metrics_panel",
]
