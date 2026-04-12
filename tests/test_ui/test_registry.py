"""Tests for ``soma.ui.registry`` — panel registration."""

from __future__ import annotations

import pytest

from soma.ui.registry import PanelRegistry


class _FakePanel:
    name = "fake"
    display_label = "Fake"

    def build(self, parent: int | str, state) -> int | str:  # type: ignore[no-untyped-def]
        return 0

    def update(self, state) -> None:  # type: ignore[no-untyped-def]
        return None


class TestPanelRegistry:
    def test_register_and_build(self) -> None:
        reg = PanelRegistry()
        reg.register("fake", _FakePanel)
        assert "fake" in reg
        panel = reg.build("fake")
        assert isinstance(panel, _FakePanel)

    def test_duplicate_name_rejected(self) -> None:
        reg = PanelRegistry()
        reg.register("fake", _FakePanel)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("fake", _FakePanel)

    def test_unknown_name_raises(self) -> None:
        reg = PanelRegistry()
        with pytest.raises(KeyError):
            reg.build("missing")

    def test_names_alphabetized(self) -> None:
        reg = PanelRegistry()
        reg.register("zeta", _FakePanel)
        reg.register("alpha", _FakePanel)
        assert reg.names() == ["alpha", "zeta"]
