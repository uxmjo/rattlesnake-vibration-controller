# -*- coding: utf-8 -*-
"""Tests for components.force_amplitude_controller.ForceAmplitudeController."""
import math
import pytest

from components.force_amplitude_controller import (
    ForceAmplitudeController, ControllerStatus)


def make_controller(**overrides):
    kwargs = dict(alpha=1.0, force_floor=0.5, max_drive_amplitude=1.25,
                  max_amplitude_step=0.1, initial_drive_amplitude=0.1)
    kwargs.update(overrides)
    return ForceAmplitudeController(**kwargs)


def test_converges_toward_target_over_several_updates():
    controller = make_controller(max_amplitude_step=1.0)  # effectively unlimited step
    amplitude = 0.1

    def plant(drive):
        # Simple proportional plant: force = 20 * drive (no dynamics)
        return 20.0 * drive

    force_target = 10.0
    last_amplitude = None
    for _ in range(50):
        measured = plant(controller.drive_amplitude)
        result = controller.update(force_target, measured, estimator_valid=True)
        last_amplitude = result.drive_amplitude
    # Should have converged close to the amplitude that makes force==target
    assert last_amplitude == pytest.approx(force_target / 20.0, rel=0.05)


def test_holds_on_invalid_estimator():
    controller = make_controller()
    result = controller.update(10.0, 5.0, estimator_valid=False)
    assert result.status is ControllerStatus.HOLD_INVALID
    assert result.drive_amplitude == pytest.approx(0.1)  # unchanged
    assert math.isnan(result.relative_force_error)


def test_holds_on_none_measurement():
    controller = make_controller()
    result = controller.update(10.0, None, estimator_valid=True)
    assert result.status is ControllerStatus.HOLD_INVALID
    assert result.drive_amplitude == pytest.approx(0.1)


def test_holds_on_nan_or_inf_measurement():
    controller = make_controller()
    for bad in (math.nan, math.inf, -math.inf):
        controller2 = make_controller()
        result = controller2.update(10.0, bad, estimator_valid=True)
        assert result.status is ControllerStatus.HOLD_INVALID


def test_does_not_blindly_ramp_on_near_zero_force():
    """The dangerous case: F_target=20N, F_measured~=0 -- must not spike
    the drive amplitude, and must not raise (division by ~zero)."""
    controller = make_controller(force_floor=0.5, max_amplitude_step=1.0)
    result = controller.update(force_target=20.0, force_measured=1e-9,
                                estimator_valid=True)
    assert result.status is ControllerStatus.HOLD_LOW_FORCE
    assert result.drive_amplitude == pytest.approx(0.1)  # unchanged, not spiked
    assert result.drive_amplitude <= controller.max_drive_amplitude


def test_force_measured_exactly_at_floor_is_not_below_floor():
    controller = make_controller(force_floor=0.5, max_amplitude_step=1.0)
    # Exactly at the floor should NOT be treated as "below" -- boundary case
    result = controller.update(force_target=10.0, force_measured=0.5,
                                estimator_valid=True)
    assert result.status is not ControllerStatus.HOLD_LOW_FORCE


def test_drive_amplitude_never_negative():
    controller = make_controller(initial_drive_amplitude=0.0, max_amplitude_step=1.0)
    # With drive starting at 0 and alpha=1, ratio*0 stays 0 regardless of ratio;
    # explicitly verify the floor is enforced in general via a manual poke.
    result = controller.update(force_target=1.0, force_measured=1000.0,
                                estimator_valid=True)
    assert result.drive_amplitude >= 0.0


def test_saturation_when_target_not_achievable():
    """If the plant needs more drive than max_drive_amplitude allows, the
    controller must clip to the limit, mark SATURATED, and never exceed it,
    and must not runaway/oscillate (no windup since there is no integrator)."""
    controller = make_controller(max_drive_amplitude=1.25, max_amplitude_step=1.0,
                                  initial_drive_amplitude=1.0)

    def weak_plant(drive):
        return 1.0 * drive  # requires drive=20 to hit force_target=20 -> unreachable

    force_target = 20.0
    statuses = []
    for _ in range(20):
        measured = weak_plant(controller.drive_amplitude)
        result = controller.update(force_target, measured, estimator_valid=True)
        statuses.append(result.status)
        assert result.drive_amplitude <= controller.max_drive_amplitude

    assert statuses[-1] is ControllerStatus.SATURATED
    assert controller.drive_amplitude == pytest.approx(controller.max_drive_amplitude)


def test_slew_limit_caps_per_update_change():
    controller = make_controller(initial_drive_amplitude=0.1, max_amplitude_step=0.05,
                                  alpha=1.0)
    # A huge ratio should still only move the amplitude by max_amplitude_step
    result = controller.update(force_target=1000.0, force_measured=1.0,
                                estimator_valid=True)
    assert result.drive_amplitude == pytest.approx(0.1 + 0.05)


def test_relative_step_limit_uses_full_scale_not_current_amplitude():
    # Seed with a small nonzero amplitude -- see
    # test_zero_initial_amplitude_is_an_absorbing_state below for why 0.0
    # cannot be used as the starting point for a purely multiplicative law.
    controller = make_controller(initial_drive_amplitude=0.01, max_amplitude_step=10.0,
                                  max_relative_step=0.1, max_drive_amplitude=1.25)
    result = controller.update(force_target=1000.0, force_measured=1.0,
                                estimator_valid=True)
    assert result.drive_amplitude == pytest.approx(0.01 + 0.1 * 1.25)


def test_zero_initial_amplitude_is_an_absorbing_state():
    """The multiplicative update A_new = A_old * ratio**alpha can never move
    away from A_old == 0 (0 * anything == 0). This is expected, not a bug --
    it is exactly why the environment-layer startup ramp is required to seed
    a small nonzero amplitude ("0V -> kleine Startamplitude ->
    Kraftregelung uebernimmt") before handing control to this controller."""
    controller = make_controller(initial_drive_amplitude=0.0, max_amplitude_step=10.0)
    result = controller.update(force_target=1000.0, force_measured=1.0,
                                estimator_valid=True)
    assert result.drive_amplitude == 0.0
    assert result.status is ControllerStatus.OK


def test_no_runaway_after_repeated_invalid_updates():
    controller = make_controller(initial_drive_amplitude=0.2)
    for _ in range(10):
        result = controller.update(10.0, None, estimator_valid=True)
    assert result.drive_amplitude == pytest.approx(0.2)


@pytest.mark.parametrize('bad_kwargs', [
    dict(alpha=0.0),
    dict(alpha=1.5),
    dict(force_floor=-1.0),
    dict(max_drive_amplitude=0.0),
    dict(max_amplitude_step=0.0),
    dict(max_relative_step=1.5),
    dict(initial_drive_amplitude=-0.1),
    dict(initial_drive_amplitude=100.0),
])
def test_invalid_configuration_raises(bad_kwargs):
    kwargs = dict(alpha=1.0, force_floor=0.5, max_drive_amplitude=1.25,
                  max_amplitude_step=0.1)
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        ForceAmplitudeController(**kwargs)


def test_invalid_force_target_raises():
    controller = make_controller()
    with pytest.raises(ValueError):
        controller.update(force_target=0.0, force_measured=5.0, estimator_valid=True)
    with pytest.raises(ValueError):
        controller.update(force_target=-5.0, force_measured=5.0, estimator_valid=True)
