"""Tests for development metric tracking."""
from __future__ import annotations

from soma.developmental.tracker import DevelopmentTracker


class TestDevelopmentTracker:
    def test_init(self) -> None:
        tracker = DevelopmentTracker()
        assert tracker.step_count == 0

    def test_record_step(self) -> None:
        tracker = DevelopmentTracker()
        tracker.record(
            step=0,
            prediction_error=0.5,
            novelty=0.8,
            num_nodes=20,
            num_edges=40,
            wm_occupancy=0,
            episodic_count=0,
        )
        assert tracker.step_count == 1

    def test_history_accessible(self) -> None:
        tracker = DevelopmentTracker()
        for i in range(10):
            tracker.record(
                step=i,
                prediction_error=0.5 - i * 0.04,
                novelty=0.8 - i * 0.05,
                num_nodes=20 + i,
                num_edges=40 + i * 2,
                wm_occupancy=i,
                episodic_count=i,
            )
        assert tracker.step_count == 10
        assert len(tracker.history["prediction_error"]) == 10
        assert tracker.history["prediction_error"][-1] < tracker.history["prediction_error"][0]

    def test_summary(self) -> None:
        tracker = DevelopmentTracker()
        for i in range(5):
            tracker.record(
                step=i,
                prediction_error=0.5,
                novelty=0.3,
                num_nodes=20,
                num_edges=40,
                wm_occupancy=5,
                episodic_count=i,
            )
        summary = tracker.summary()
        assert "prediction_error" in summary
        assert "num_nodes" in summary
        assert isinstance(summary["prediction_error"]["mean"], float)

    def test_summary_last_n(self) -> None:
        tracker = DevelopmentTracker()
        for i in range(10):
            tracker.record(
                step=i,
                prediction_error=float(i),
                novelty=0.5,
                num_nodes=20,
                num_edges=40,
                wm_occupancy=0,
                episodic_count=0,
            )
        summary = tracker.summary(last_n=3)
        assert summary["prediction_error"]["count"] == 3.0
        assert summary["prediction_error"]["min"] == 7.0

    def test_extra_kwargs(self) -> None:
        tracker = DevelopmentTracker()
        tracker.record(
            step=0,
            prediction_error=0.5,
            novelty=0.8,
            num_nodes=20,
            num_edges=40,
            wm_occupancy=0,
            episodic_count=0,
            custom_metric=42.0,
        )
        assert "custom_metric" in tracker.history
        assert tracker.history["custom_metric"] == [42.0]

    def test_save_load_json(self, tmp_path: object) -> None:
        tracker = DevelopmentTracker()
        for i in range(3):
            tracker.record(
                step=i,
                prediction_error=0.1 * i,
                novelty=0.5,
                num_nodes=20,
                num_edges=40,
                wm_occupancy=0,
                episodic_count=0,
            )
        path = tmp_path / "tracker.json"  # type: ignore[operator]
        tracker.save(path)
        loaded = DevelopmentTracker.load(path)
        assert loaded.step_count == 3
        assert loaded.history["prediction_error"] == tracker.history["prediction_error"]
