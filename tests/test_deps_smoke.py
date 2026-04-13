"""Smoke tests for new dependencies required by the autonomous loop."""


def test_portalocker_importable() -> None:
    import portalocker

    assert hasattr(portalocker, "Lock")
