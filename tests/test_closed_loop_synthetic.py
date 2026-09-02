# -*- coding: utf-8 -*-
"""Synthetic closed-loop tests wiring SineSweepGenerator +
ForceTrackingEstimator + ForceAmplitudeController together against a simple
simulated F = H(f) * U plant (no real hardware/DAQ involved), per the
"Simulation vor echter Hardware" requirement.

H(f) is a lightly-damped SDOF-like resonance magnitude, applied
quasi-statically per instantaneous frequency (valid because the sweep is
deliberately kept slow relative to the tracking filter / plant response, as
documented as an assumption of this architecture).
"""
import numpy as np
import pytest

from components.sine_sweep_generator import SineSweepGenerator
from components.force_tracking_estimator import ForceTrackingEstimator
from components.force_amplitude_controller import (
    ForceAmplitudeController, ControllerStatus)

FS = 5000.0
BLOCK_SIZE = 256


def resonant_plant_gain(freq, fr=80.0, zeta=0.05, h0=2.0):
    """F/U magnitude for a lightly-damped SDOF-like resonance."""
    ratio = freq / fr
    denom = np.sqrt((1 - ratio ** 2) ** 2 + (2 * zeta * ratio) ** 2)
    return h0 / denom


def run_closed_loop(gen, estimator, controller, target_force, n_total,
                     gain_fn, dropout_range=None, tracking_cycles=None):
    """Runs the closed loop for n_total samples in fixed-size blocks.

    dropout_range : optional (start_sample, stop_sample) during which the
        force measurement is corrupted (NaN) to simulate sensor loss.
    tracking_cycles : optional. If given, before each block the estimator's
        tracking bandwidth is updated to
        ``ForceTrackingEstimator.bandwidth_for_tracking_cycles(freq, tracking_cycles)``
        using the block's starting frequency -- i.e. an adaptive/proportional
        bandwidth that scales with the instantaneous sweep frequency, mirroring
        ``SineForceControlEnvironment``'s adaptive_tracking_bandwidth option
        -- instead of the estimator's fixed constructor-time bandwidth.

    Returns a dict of per-block diagnostic lists.
    """
    drive_amplitude = controller.drive_amplitude
    diagnostics = {'relative_error': [], 'status': [], 'drive_amplitude': [],
                    'valid': [],
                    # Per-block, always appended (unlike 'relative_error'
                    # above which skips invalid blocks) so callers can align
                    # against a known sample/time window -- 'block_error' is
                    # NaN for invalid blocks.
                    'block_frequency': [], 'block_error': []}
    sample_index = 0
    for start in range(0, n_total, BLOCK_SIZE):
        n = min(BLOCK_SIZE, n_total - start)
        samples, freq, phase = gen.generate_block(n, drive_amplitude=drive_amplitude)
        gain = gain_fn(freq)
        force = gain * drive_amplitude * np.sin(phase)
        if dropout_range is not None and dropout_range[0] <= start < dropout_range[1]:
            force = np.full_like(force, np.nan)
        if tracking_cycles is not None:
            estimator.set_tracking_bandwidth(
                ForceTrackingEstimator.bandwidth_for_tracking_cycles(
                    float(freq[0]), tracking_cycles))
        result = estimator.process_block(force, phase)
        measured = result.amplitude if result.valid else None
        ctrl_result = controller.update(target_force, measured, result.valid)
        drive_amplitude = ctrl_result.drive_amplitude

        assert np.isfinite(drive_amplitude)
        assert 0.0 <= drive_amplitude <= controller.max_drive_amplitude

        diagnostics['drive_amplitude'].append(drive_amplitude)
        diagnostics['status'].append(ctrl_result.status)
        diagnostics['valid'].append(result.valid)
        diagnostics['block_frequency'].append(float(freq[0]))
        if result.valid:
            relative_error = abs(result.amplitude - target_force) / target_force
            diagnostics['relative_error'].append(relative_error)
            diagnostics['block_error'].append(relative_error)
        else:
            diagnostics['block_error'].append(np.nan)
        sample_index += n
    return diagnostics


def test_closed_loop_holds_force_through_resonance():
    """Test case #6/#7: sweep through a resonance, controller should keep
    the force roughly at the target and never go unstable (no NaN, bounded
    drive amplitude, slew limit respected implicitly via the controller)."""
    gen = SineSweepGenerator(sample_rate=FS, sweep_type='logarithmic',
                              f_start=60.0, f_stop=100.0, sweep_rate=6.0)
    estimator = ForceTrackingEstimator(sample_rate=FS, tracking_bandwidth_hz=10.0)
    controller = ForceAmplitudeController(alpha=0.5, force_floor=0.05,
                                           max_drive_amplitude=1.25,
                                           max_amplitude_step=0.02,
                                           initial_drive_amplitude=0.05)
    # Chosen so the required drive amplitude stays within max_drive_amplitude
    # everywhere in the swept band [60, 100] Hz, including at the band edges
    # where the resonance gain has dropped off (verified numerically: worst
    # case is at f=100 Hz where gain ~3.47 N/V, requiring ~0.72 V << 1.25 V).
    target_force = 2.5
    n_total = int(gen.sweep_duration * FS)

    diagnostics = run_closed_loop(gen, estimator, controller, target_force,
                                   n_total, resonant_plant_gain)

    relative_errors = np.array(diagnostics['relative_error'])
    assert len(relative_errors) > 20
    # After the initial settle/convergence transient, error should be small.
    late_errors = relative_errors[len(relative_errors) // 2:]
    assert late_errors.mean() < 0.25
    # No saturation should have been needed for this target/gain combination.
    assert controller.drive_amplitude < controller.max_drive_amplitude
    # Drive amplitude trace must stay smooth (slew-limited): no single-step
    # jump larger than the configured max_amplitude_step.
    drive = np.array(diagnostics['drive_amplitude'])
    assert np.all(np.abs(np.diff(drive)) <= controller.max_amplitude_step + 1e-12)


def test_closed_loop_saturates_without_instability():
    """Test case #8: an unreachable target force against a weak (no
    resonance) plant must saturate cleanly at max_drive_amplitude, without
    oscillation, windup, or exceeding the limit."""
    gen = SineSweepGenerator(sample_rate=FS, sweep_type='linear',
                              f_start=50.0, f_stop=150.0, sweep_rate=50.0)
    estimator = ForceTrackingEstimator(sample_rate=FS, tracking_bandwidth_hz=10.0)
    controller = ForceAmplitudeController(alpha=0.7, force_floor=0.05,
                                           max_drive_amplitude=1.25,
                                           max_amplitude_step=0.05,
                                           initial_drive_amplitude=0.1)
    target_force = 100.0  # far beyond what a gain-of-1 plant can reach at max drive
    n_total = int(gen.sweep_duration * FS)

    def weak_plant(freq):
        return np.ones_like(freq)

    diagnostics = run_closed_loop(gen, estimator, controller, target_force,
                                   n_total, weak_plant)

    assert diagnostics['status'][-1] is ControllerStatus.SATURATED
    assert controller.drive_amplitude == pytest.approx(controller.max_drive_amplitude)
    # Approach to saturation must be monotonic -- no windup-style overshoot
    # followed by a correction back down (there is no integrator state to
    # wind up in the first place, but this verifies the observable behavior).
    drive = np.array(diagnostics['drive_amplitude'])
    assert np.all(np.diff(drive) >= -1e-12)
    # Once at the limit, it must stay pinned exactly at the limit.
    saturated_tail = drive[drive >= controller.max_drive_amplitude - 1e-9]
    assert len(saturated_tail) > 0
    np.testing.assert_allclose(saturated_tail, controller.max_drive_amplitude)


def test_closed_loop_sensor_dropout_does_not_spike_drive():
    """Test case #9: force measurement drops to NaN mid-run (sensor loss).
    The controller must hold the drive amplitude during the dropout (never
    ramp it up blindly) and resume tracking once the sensor returns."""
    gen = SineSweepGenerator(sample_rate=FS, sweep_type='linear',
                              f_start=80.0, f_stop=80.0001, sweep_rate=1.0)
    # Effectively a fixed dwell frequency for this test (start~=stop).
    estimator = ForceTrackingEstimator(sample_rate=FS, tracking_bandwidth_hz=10.0)
    controller = ForceAmplitudeController(alpha=0.5, force_floor=0.05,
                                           max_drive_amplitude=1.25,
                                           max_amplitude_step=0.02,
                                           initial_drive_amplitude=0.1)
    target_force = 5.0
    n_total = int(2.0 * FS)
    dropout_range = (int(0.8 * FS), int(1.4 * FS))

    def unity_plant(freq):
        return 5.0 * np.ones_like(freq)

    diagnostics = run_closed_loop(gen, estimator, controller, target_force,
                                   n_total, unity_plant, dropout_range=dropout_range)

    drive = np.array(diagnostics['drive_amplitude'])
    valid = np.array(diagnostics['valid'])
    # During the dropout, no valid measurements should have been produced...
    dropout_block_start = dropout_range[0] // BLOCK_SIZE
    dropout_block_end = dropout_range[1] // BLOCK_SIZE
    assert not np.any(valid[dropout_block_start + 1:dropout_block_end])
    # ...and the drive amplitude must not have changed at all while invalid.
    drive_during_dropout = drive[dropout_block_start + 1:dropout_block_end]
    np.testing.assert_allclose(drive_during_dropout, drive_during_dropout[0])
    # After recovery, the controller should resume moving toward the target.
    assert diagnostics['valid'][-1] is True


def test_adaptive_tracking_bandwidth_beats_fixed_at_sweep_turnaround():
    """Reproduces the real-world failure mode seen on an up/down/up sweep
    spanning a wide frequency range (5 - 500 Hz, 100:1): a *fixed* tracking
    bandwidth sized for good rejection at the high end (10 Hz, well below
    500 Hz) is not well below the *low* end (5 Hz) -- the demodulated
    double-frequency term leaks through there and corrupts the force
    estimate/control, specifically each time the sweep revisits the low
    end at a direction turnaround. An adaptive bandwidth (constant
    tracking_cycles) with the *same* bandwidth at the high end must track
    much better there instead, without needing a fresh cold-start settle
    (the plant/gain/target here are chosen so the controller has already
    long converged well before the turnaround, isolating the demodulation
    effect from any startup transient)."""
    f_start, f_stop = 5.0, 500.0
    fixed_bandwidth_hz = 10.0
    tracking_cycles = f_stop / (2 * np.pi * fixed_bandwidth_hz)  # matches fixed bw at f_stop
    target_force = 5.0

    def unity_plant(freq):
        return 5.0 * np.ones_like(freq)  # F = 5*drive_amplitude*sin(phase)

    def turnaround_error(tracking_cycles):
        gen = SineSweepGenerator(sample_rate=FS, sweep_type='linear',
                                  f_start=f_start, f_stop=f_stop, sweep_rate=50.0,
                                  repeat=True, num_sweeps=3, alternate_direction=True)
        estimator = ForceTrackingEstimator(sample_rate=FS, tracking_bandwidth_hz=fixed_bandwidth_hz)
        controller = ForceAmplitudeController(alpha=0.3, force_floor=0.05,
                                               max_drive_amplitude=1.25,
                                               max_amplitude_step=0.01,
                                               initial_drive_amplitude=0.2)
        leg_samples = int(gen.sweep_duration * FS)
        n_total = 3 * leg_samples
        diagnostics = run_closed_loop(gen, estimator, controller, target_force,
                                       n_total, unity_plant, tracking_cycles=tracking_cycles)
        block_freq = np.array(diagnostics['block_frequency'])
        block_error = np.array(diagnostics['block_error'])
        # Leg 1 (down) ends and leg 2 (up) starts at sample index
        # 2*leg_samples -- take a window straddling that turnaround, but
        # skip the run's very first samples (leg 0's cold-start settle,
        # not what this test is about).
        block_index = np.arange(len(block_freq)) * BLOCK_SIZE
        window = (block_index > 1.9 * leg_samples) & (block_index < 2.3 * leg_samples)
        errors = block_error[window]
        errors = errors[np.isfinite(errors)]
        assert len(errors) > 5
        return errors.mean()

    error_fixed = turnaround_error(tracking_cycles=None)
    error_adaptive = turnaround_error(tracking_cycles=tracking_cycles)

    assert error_adaptive < error_fixed
    assert error_adaptive < 0.15
