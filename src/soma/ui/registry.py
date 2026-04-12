"""Plugin-style registry for UI panels.

Every panel in the UI implements the :class:`Panel` protocol (a `build`
method that creates its DPG widgets and returns a tag). Panels register
themselves with :class:`PanelRegistry` so the main app can lay them out
without importing each one explicitly.

The pattern makes it easy to add new panels — create a module, define a
class that implements `Panel`, register it with the decorator
``@register_panel("my-panel")``, and import the module from ``panels/__init__.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from soma.ui.state import UIState


class Panel(Protocol):
    """Interface every panel must implement.

    The ``build`` method is called once during window construction. It
    creates DPG widgets inside whatever parent the layout manager
    provides and returns the root widget tag so the layout can refer to
    it later (e.g., to show/hide it).

    The ``update`` method is called by the main app on every DPG frame
    (~60 Hz) while the UI is visible. Panels that respond to bus events
    in their subscriber callbacks generally keep ``update`` empty.
    """

    name: str
    display_label: str

    def build(self, parent: int | str, state: UIState) -> int | str: ...

    def update(self, state: UIState) -> None: ...


PanelFactory = Callable[[], Panel]


class PanelRegistry:
    """Maps a unique panel name to a factory that constructs a new instance."""

    def __init__(self) -> None:
        self._factories: dict[str, PanelFactory] = {}

    def register(self, name: str, factory: PanelFactory) -> None:
        if name in self._factories:
            raise ValueError(f"Panel {name!r} already registered")
        self._factories[name] = factory

    def build(self, name: str) -> Panel:
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(f"No panel named {name!r}")
        return factory()

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories


# Module-level default registry that panels can decorate into.
default_registry = PanelRegistry()


def register_panel(name: str) -> Callable[[type], type]:
    """Class decorator that registers a ``Panel`` subclass with the default registry."""

    def inner(cls: type) -> type:
        default_registry.register(name, cls)  # cls() acts as factory
        return cls

    return inner
