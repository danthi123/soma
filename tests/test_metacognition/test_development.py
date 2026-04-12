"""Tests for ``soma.metacognition.development``."""

from __future__ import annotations

import pytest

from soma.core.node import NodeType
from soma.metacognition.development import CriticalPeriod, DevelopmentSchedule


class TestCriticalPeriod:
    def test_progress_triangular(self) -> None:
        period = CriticalPeriod(
            name="test",
            start_step=0,
            peak_step=100,
            end_step=200,
            affected_node_types=(NodeType.ASSOCIATOR,),
            plasticity_multiplier=2.0,
            synaptogenesis_multiplier=3.0,
        )
        assert period.progress(-10) == pytest.approx(0.0)
        assert period.progress(0) == pytest.approx(0.0)
        assert period.progress(50) == pytest.approx(0.5)
        assert period.progress(100) == pytest.approx(1.0)
        assert period.progress(150) == pytest.approx(0.5)
        assert period.progress(200) == pytest.approx(0.0)
        assert period.progress(300) == pytest.approx(0.0)

    def test_applies_to(self) -> None:
        period = CriticalPeriod(
            name="test",
            start_step=0,
            peak_step=50,
            end_step=100,
            affected_node_types=(NodeType.ASSOCIATOR, NodeType.INTEGRATOR),
            plasticity_multiplier=2.0,
            synaptogenesis_multiplier=1.5,
        )
        assert period.applies_to(NodeType.ASSOCIATOR, 25)
        assert period.applies_to(NodeType.INTEGRATOR, 50)
        assert not period.applies_to(NodeType.SENSOR, 25)
        assert not period.applies_to(NodeType.ASSOCIATOR, 200)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="peak"):
            CriticalPeriod(
                name="bad",
                start_step=100,
                peak_step=50,
                end_step=200,
                affected_node_types=(NodeType.ASSOCIATOR,),
                plasticity_multiplier=2.0,
                synaptogenesis_multiplier=1.5,
            )
        with pytest.raises(ValueError, match="plasticity_multiplier"):
            CriticalPeriod(
                name="bad",
                start_step=0,
                peak_step=50,
                end_step=100,
                affected_node_types=(NodeType.ASSOCIATOR,),
                plasticity_multiplier=0.5,
                synaptogenesis_multiplier=1.5,
            )
        with pytest.raises(ValueError, match="synaptogenesis_multiplier"):
            CriticalPeriod(
                name="bad",
                start_step=0,
                peak_step=50,
                end_step=100,
                affected_node_types=(NodeType.ASSOCIATOR,),
                plasticity_multiplier=1.5,
                synaptogenesis_multiplier=0.5,
            )


class TestDevelopmentSchedule:
    def test_defaults_nonempty(self) -> None:
        sched = DevelopmentSchedule()
        assert len(sched.periods) > 0

    def test_plasticity_peak(self) -> None:
        sched = DevelopmentSchedule()
        # Whitepaper default: sensory_discrimination peaks at step 5000 with
        # plasticity_multiplier=3.0 for SENSOR.
        mult_at_peak = sched.get_plasticity_multiplier(step=5_000, node_type=NodeType.SENSOR)
        assert mult_at_peak == pytest.approx(3.0)

    def test_plasticity_ramps(self) -> None:
        sched = DevelopmentSchedule()
        at_start = sched.get_plasticity_multiplier(step=0, node_type=NodeType.SENSOR)
        at_peak = sched.get_plasticity_multiplier(step=5_000, node_type=NodeType.SENSOR)
        past_end = sched.get_plasticity_multiplier(step=25_000, node_type=NodeType.SENSOR)
        assert at_start < at_peak
        assert past_end == pytest.approx(1.0)  # default outside window

    def test_plasticity_unaffected_type_returns_one(self) -> None:
        sched = DevelopmentSchedule()
        # OUTPUT has its own period, but at step=0 it's before the OUTPUT
        # period start (30_000) so multiplier is 1.0.
        mult = sched.get_plasticity_multiplier(step=0, node_type=NodeType.OUTPUT)
        assert mult == pytest.approx(1.0)

    def test_synaptogenesis_multiplier(self) -> None:
        sched = DevelopmentSchedule()
        at_peak = sched.get_synaptogenesis_multiplier(step=5_000, node_type=NodeType.SENSOR)
        assert at_peak == pytest.approx(5.0)  # whitepaper default

    def test_multiple_overlapping_take_max(self) -> None:
        shared = (
            CriticalPeriod(
                name="a",
                start_step=0,
                peak_step=100,
                end_step=200,
                affected_node_types=(NodeType.ASSOCIATOR,),
                plasticity_multiplier=2.0,
                synaptogenesis_multiplier=1.5,
            ),
            CriticalPeriod(
                name="b",
                start_step=50,
                peak_step=150,
                end_step=250,
                affected_node_types=(NodeType.ASSOCIATOR,),
                plasticity_multiplier=3.0,
                synaptogenesis_multiplier=1.5,
            ),
        )
        sched = DevelopmentSchedule(periods=shared)
        # At step 100: period a is at peak (1.0 progress, multiplier 2.0);
        # period b is at (100-50)/(150-50) = 0.5 progress, multiplier
        # 1.0 + (3.0-1.0)*0.5 = 2.0. Both equal, max is 2.0.
        assert sched.get_plasticity_multiplier(step=100, node_type=NodeType.ASSOCIATOR) == (
            pytest.approx(2.0)
        )
        # At step 150: period b at peak (3.0); period a at falling edge
        # (150-100)/(200-100) = 0.5 progress -> 1.5 multiplier. Max = 3.0.
        assert sched.get_plasticity_multiplier(step=150, node_type=NodeType.ASSOCIATOR) == (
            pytest.approx(3.0)
        )

    def test_active_periods(self) -> None:
        sched = DevelopmentSchedule()
        active_at_10k = sched.active_periods(step=10_000)
        names = {p.name for p in active_at_10k}
        assert "sensory_discrimination" in names
        assert "association_formation" in names
        # Output period doesn't start until 30_000.
        assert "output_refinement" not in names

    def test_rejects_negative_step(self) -> None:
        sched = DevelopmentSchedule()
        with pytest.raises(ValueError, match="step"):
            sched.get_plasticity_multiplier(step=-1, node_type=NodeType.SENSOR)
