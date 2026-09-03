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
import copy

import numpy as np
import pytest

from components.sine_sweep_generator import SineSweepGenerator
from components.force_tracking_estimator import ForceTrackingEstimator
from components.force_amplitude_controller import (
    ForceAmplitudeController, ControllerStatus)
from components.feedforward_map import FeedforwardMap, compose_drive_amplitude

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


def run_closed_loop_with_feedforward(gen, estimator, trim_controller, feedforward_map,
                                      target_force, n_total, gain_fn, max_drive_v,
                                      max_drive_step_v):
    """Mirrors SineForceControlEnvironment._update_feedforward_and_compose:
    u_total = A_FF(f) * g, g from the (unmodified) ForceAmplitudeController
    reused as a trim, learned into feedforward_map from u_total whenever the
    trim status is OK (estimator valid, trim not held/saturated). Uses the
    same compose_drive_amplitude() (incl. anti-windup) as the real
    environment, rather than a separately hand-duplicated copy of that
    math -- see that function's docstring for why (a duplicated copy is
    exactly how this composition's anti-windup fix could silently drift out
    of sync between the real environment and this test harness).

    ``feedforward_map`` is the *write* side, accumulating learning every
    update as normal; composition instead reads from an internal *published*
    snapshot that is only refreshed at sweep-leg boundaries -- composing
    directly from the same instance being learned from creates a second,
    fast feedback loop (map learns from the composed command; the composed
    command immediately depends on what was just learned) that measurably
    destabilizes tracking, confirmed to make it *worse* than feedforward
    being disabled even on the very first, initially-empty leg. See
    SineForceControlEnvironment.initialize_environment_test_parameters for
    the identical split in the real environment."""
    total_drive_amplitude = feedforward_map.initial_estimate
    published_map = copy.deepcopy(feedforward_map)
    committed_leg = 0
    diagnostics = {'leg': [], 'direction': [], 'frequency': [],
                    'relative_error': [], 'feedback_pct': [], 'total_drive': [],
                    'status': []}
    for start in range(0, n_total, BLOCK_SIZE):
        n = min(BLOCK_SIZE, n_total - start)
        samples, freq, phase = gen.generate_block(n, drive_amplitude=total_drive_amplitude)
        gain = gain_fn(freq)
        force = gain * total_drive_amplitude * np.sin(phase)
        result = estimator.process_block(force, phase)
        measured = result.amplitude if result.valid else None
        ctrl_result = trim_controller.update(target_force, measured, result.valid)

        leg, direction = gen.leg_and_direction(gen.elapsed_time)
        if leg != committed_leg:
            published_map = copy.deepcopy(feedforward_map)
            committed_leg = leg
        f_end = float(freq[-1])
        composition = compose_drive_amplitude(
            published_map, f_end, ctrl_result.drive_amplitude, total_drive_amplitude,
            max_drive_v, max_drive_step_v, direction=direction)
        total_drive_amplitude = composition.total_drive_amplitude

        achieved_trim_gain = min(max(composition.achieved_trim_gain, 0.0),
                                  trim_controller.max_drive_amplitude)
        trim_controller.drive_amplitude = achieved_trim_gain

        trust = ctrl_result.status is ControllerStatus.OK
        feedforward_map.update(f_end, observed_value=total_drive_amplitude, trust=trust, direction=direction)

        assert np.isfinite(total_drive_amplitude)
        assert 0.0 <= total_drive_amplitude <= max_drive_v

        diagnostics['leg'].append(leg)
        diagnostics['direction'].append(direction)
        diagnostics['frequency'].append(f_end)
        diagnostics['total_drive'].append(total_drive_amplitude)
        diagnostics['feedback_pct'].append((achieved_trim_gain - 1.0) * 100.0)
        diagnostics['status'].append(ctrl_result.status)
        if result.valid:
            diagnostics['relative_error'].append(abs(result.amplitude - target_force) / target_force)
        else:
            diagnostics['relative_error'].append(np.nan)
    return diagnostics


def test_feedforward_learning_shrinks_feedback_correction_over_sweeps():
    """End-to-end test of the feedforward architecture described in
    components/feedforward_map.py: across several continuous UP/DOWN sweep
    legs through a resonance, the trim gain's excursion away from 1.0 (i.e.
    |feedback_correction_pct|) should shrink leg-to-leg as the feedforward
    map learns the required drive amplitude at each frequency, while the
    measured force amplitude stays close to target throughout (no reset or
    jump in behavior at the direction turnarounds)."""
    f_start, f_stop = 20.0, 300.0
    target_force = 4.0
    # Chosen (and verified numerically) so the required drive amplitude
    # stays within [value_min, max_drive_v] across the whole swept band --
    # otherwise the fast loop saturates (status != OK) and never trusts the
    # feedforward layer to learn from, which is a different, already-covered
    # scenario (see test_closed_loop_saturates_without_instability above).
    max_drive_v = 30.0
    n_legs = 5

    gen = SineSweepGenerator(sample_rate=FS, sweep_type='logarithmic',
                              f_start=f_start, f_stop=f_stop, sweep_rate=40.0,
                              repeat=True, num_sweeps=n_legs, alternate_direction=True)
    estimator = ForceTrackingEstimator(sample_rate=FS, tracking_bandwidth_hz=8.0)
    trim_controller = ForceAmplitudeController(alpha=0.5, force_floor=0.05,
                                                max_drive_amplitude=3.0,
                                                max_amplitude_step=0.5,
                                                initial_drive_amplitude=1.0)
    feedforward_map = FeedforwardMap(f_min=f_start, f_max=f_stop, initial_estimate=1.0,
                                      value_min=0.02, value_max=max_drive_v,
                                      bins_per_decade=15.0, learning_rate=0.15)

    leg_duration = gen.sweep_duration
    n_total = int(n_legs * leg_duration * FS)
    diagnostics = run_closed_loop_with_feedforward(
        gen, estimator, trim_controller, feedforward_map, target_force, n_total,
        resonant_plant_gain, max_drive_v=max_drive_v, max_drive_step_v=3.0)

    leg = np.array(diagnostics['leg'])
    feedback_pct = np.abs(np.array(diagnostics['feedback_pct']))
    relative_error = np.array(diagnostics['relative_error'])

    # Mean |feedback correction| in the first leg vs. the last leg -- must
    # shrink substantially as the feedforward map takes over.
    first_leg_feedback = feedback_pct[leg == 0].mean()
    last_leg_feedback = feedback_pct[leg == n_legs - 1].mean()
    assert last_leg_feedback < 0.5 * first_leg_feedback

    # Force tracking must stay reasonable throughout, including at the
    # direction turnarounds -- no blow-up or instability from the added
    # feedforward layer.
    finite_errors = relative_error[np.isfinite(relative_error)]
    assert len(finite_errors) > 100
    assert finite_errors[len(finite_errors) // 2:].mean() < 0.3

    # The feedforward map must have actually learned a nontrivial curve,
    # not stayed at its initial flat estimate everywhere.
    freqs, values, n_obs = feedforward_map.curve()
    assert len(freqs) > 5
    assert np.ptp(values) > 0.05


def test_trim_controller_does_not_windup_when_composition_is_slew_limited():
    """Regression test for an actuator-saturation windup bug: whenever the
    *composed* command (u_total = A_FF(f) * g) is limited by max_drive_v/
    max_drive_step_v -- physical limits the trim controller's own ratio law
    has no visibility into -- the trim's internal state `g` must track what
    was actually applied, not keep marching ahead of it. Without the
    anti-windup back-calculation in compose_drive_amplitude(), `g` races
    up toward its own ceiling every tick the true error stays large (which
    it must, while the composed signal is still catching up) and then stays
    pinned there long after the plant makes the target reachable again --
    which is exactly the "garbage learned curve values slammed to their
    hard min/max clamps, and force is completely mis-controlled" failure
    mode this reproduces at a fixed dwell frequency for clarity."""
    # Dwell at a fixed frequency (matches the existing dropout test's
    # trick: f_start ~= f_stop).
    gen = SineSweepGenerator(sample_rate=FS, sweep_type='linear',
                              f_start=80.0, f_stop=80.0001, sweep_rate=1.0)
    estimator = ForceTrackingEstimator(sample_rate=FS, tracking_bandwidth_hz=10.0)
    target_force = 5.0
    max_drive_v = 5.0
    # Deliberately tight, as in the reported real-world case, so the
    # composed command cannot follow a large step in required amplitude
    # within a single dwell -- forcing many ticks of sustained clamping.
    max_drive_step_v = 0.01
    trim_controller = ForceAmplitudeController(alpha=0.15, force_floor=0.05,
                                                max_drive_amplitude=3.0,
                                                max_amplitude_step=0.5,
                                                initial_drive_amplitude=1.0)
    feedforward_map = FeedforwardMap(f_min=79.0, f_max=81.0, initial_estimate=0.5,
                                      value_min=0.05, value_max=max_drive_v,
                                      bins_per_decade=10.0, learning_rate=0.2)

    n_step1 = int(2.0 * FS)   # easy plant: required drive ~0.5 V, quickly reachable
    n_step2 = int(3.0 * FS)   # hard plant: required drive ~0.833V (trim gain ~1.667,
                              # i.e. +67% -- reachable via the trim alone, but not
                              # within this window given the tight slew limit)
    n_step3 = int(3.0 * FS)   # back to the easy plant

    current_gain = [10.0]  # required V = target/gain = 5/10 = 0.5 V

    def stepped_plant(freq):
        return np.full_like(freq, current_gain[0])

    total_drive_amplitude = feedforward_map.initial_estimate
    # Matches the real environment: composition reads from a published
    # snapshot, refreshed only at sweep-leg boundaries (see
    # run_closed_loop_with_feedforward docstring above). This single
    # continuous dwell never changes leg, so published_map correctly stays
    # frozen at its initial state for this whole test -- this test is about
    # anti-windup under external slew-limiting, a property of
    # compose_drive_amplitude independent of whether the feedforward value
    # is live-updating, so that is exactly the right behavior here.
    published_map = copy.deepcopy(feedforward_map)
    feedback_pct_by_phase = {'easy': [], 'hard': [], 'easy_again': []}
    elapsed_samples = 0

    for phase_name, n_phase, gain in (('easy', n_step1, 10.0),
                                       ('hard', n_step2, 6.0),
                                       ('easy_again', n_step3, 10.0)):
        current_gain[0] = gain
        phase_start = elapsed_samples
        while elapsed_samples - phase_start < n_phase:
            n = min(BLOCK_SIZE, n_phase - (elapsed_samples - phase_start))
            samples, freq, phase = gen.generate_block(n, drive_amplitude=total_drive_amplitude)
            force = stepped_plant(freq) * total_drive_amplitude * np.sin(phase)
            result = estimator.process_block(force, phase)
            measured = result.amplitude if result.valid else None
            ctrl_result = trim_controller.update(target_force, measured, result.valid)

            composition = compose_drive_amplitude(
                published_map, float(freq[-1]), ctrl_result.drive_amplitude,
                total_drive_amplitude, max_drive_v, max_drive_step_v, direction='up')
            total_drive_amplitude = composition.total_drive_amplitude
            achieved_trim_gain = min(max(composition.achieved_trim_gain, 0.0),
                                      trim_controller.max_drive_amplitude)
            trim_controller.drive_amplitude = achieved_trim_gain

            trust = ctrl_result.status is ControllerStatus.OK
            feedforward_map.update(float(freq[-1]), observed_value=total_drive_amplitude,
                                    trust=trust, direction='up')

            feedback_pct_by_phase[phase_name].append((achieved_trim_gain - 1.0) * 100.0)
            elapsed_samples += n

    hard = np.array(feedback_pct_by_phase['hard'])
    easy_again = np.array(feedback_pct_by_phase['easy_again'])

    assert np.isfinite(total_drive_amplitude)
    assert 0.0 <= total_drive_amplitude <= max_drive_v

    # The true required trim gain for the 'hard' phase is ~1.667 (+67%,
    # frozen ff=0.5 * g=1.667 = 0.833V = 5/6) -- reachable via the trim
    # alone, just not within this window given the tight slew limit.
    # Without anti-windup, the trim doesn't know its last request was never
    # actually applied and keeps requesting more regardless, racing all the
    # way to its own feedforward_trim_gain_max ceiling (3.0, i.e. +200%) and
    # sitting pinned there for the rest of the 'hard' phase -- overshooting
    # the true requirement by 3x. With anti-windup, the trim's *next*
    # request is always anchored to what was actually achieved, so it stays
    # close to (and never much exceeds) the true ~67% requirement instead --
    # verified numerically to differ dramatically (~68% vs. 200% peak)
    # between the fixed and the pre-fix composition.
    assert np.max(np.abs(hard)) < 100.0

    # Symmetric check coming out of the hard phase: without anti-windup the
    # correction overshoots wildly in *both* directions once the target
    # becomes easy again (observed ~[-90%, +200%], a >250-point swing, as
    # the pinned-high trim first overshoots the now-easy target and the
    # ratio law has to claw all the way back) -- with anti-windup the swing
    # stays small (observed ~65 points) since there was never a large
    # pinned-up state to unwind.
    assert (easy_again.max() - easy_again.min()) < 100.0

    # The feedforward map must not have had a wildly wrong value slammed
    # into its bin as a direct consequence of windup during the hard phase.
    freqs, values, n_obs = feedforward_map.curve()
    assert len(freqs) > 0
    assert np.all(values < feedforward_map.value_max * 0.95)


def test_feedforward_first_leg_is_not_worse_than_disabled():
    """Regression test for a confirmed bug reported from real hardware: with
    an earlier version of this code, composing directly from the same
    FeedforwardMap instance that is simultaneously being learned from (i.e.
    publishing every single control update, not just at sweep-leg
    boundaries) created a second, fast feedback loop -- the map learns from
    the composed command, and the composed command immediately depends on
    what was just learned -- layered on top of the fast loop's own
    feedback. Two coupled feedback loops on the same timescale can oscillate
    even when each is stable alone: verified to make tracking on the very
    first, initially-empty sweep leg *dramatically worse* than feedforward
    being disabled entirely (observed: 10-20 Hz band mean error ~10.6 N vs.
    ~0.9 N disabled, a >10x regression) -- exactly the "es schwingt jetzt
    deutlich stärker" behavior reported. The fix publishes learning into a
    read-side snapshot only at sweep-leg boundaries (see
    run_closed_loop_with_feedforward and
    SineForceControlEnvironment.initialize_environment_test_parameters);
    this test locks in that a freshly-enabled feedforward map's first leg
    must track statistically indistinguishably from feedforward disabled,
    not worse."""
    def realistic_plant_gain(freq):
        """Tuned so the required drive stays in a realistic ~0.15-0.55V
        range against target_force=80N (matching real hardware observed
        drive ranges), with two mild resonances -- deliberately not flat,
        so this isn't a degenerate/trivial plant."""
        def resonance(f, fr, zeta, h0):
            ratio = f / fr
            denom = np.sqrt((1 - ratio ** 2) ** 2 + (2 * zeta * ratio) ** 2)
            return h0 / denom
        baseline = 200.0 * (freq / 50.0) ** -0.15
        r1 = resonance(freq, fr=120.0, zeta=0.08, h0=60.0)
        r2 = resonance(freq, fr=900.0, zeta=0.06, h0=50.0)
        return baseline + r1 + r2

    f_start, f_stop = 5.0, 1500.0
    target_force = 80.0
    max_drive_v = 1.25
    max_drive_step_v = 0.02
    initial_drive_v = 0.3
    alpha = 0.25
    tracking_cycles = 5.0

    def run_one_leg(feedforward_enabled, seed):
        rng = np.random.default_rng(seed)
        gen = SineSweepGenerator(sample_rate=FS, sweep_type='logarithmic',
                                  f_start=f_start, f_stop=f_stop, sweep_rate=8.0,
                                  repeat=True, num_sweeps=1, alternate_direction=True)
        estimator = ForceTrackingEstimator(
            sample_rate=FS,
            tracking_bandwidth_hz=ForceTrackingEstimator.bandwidth_for_tracking_cycles(f_start, tracking_cycles))
        if feedforward_enabled:
            feedforward_map = FeedforwardMap(f_min=f_start, f_max=f_stop, initial_estimate=initial_drive_v,
                                              value_min=0.05, value_max=max_drive_v,
                                              bins_per_decade=10.0, learning_rate=0.2)
            published_map = copy.deepcopy(feedforward_map)
            trim_controller = ForceAmplitudeController(alpha=alpha, force_floor=0.1, max_drive_amplitude=3.0,
                                                        max_amplitude_step=0.5, initial_drive_amplitude=1.0)
        else:
            trim_controller = ForceAmplitudeController(alpha=alpha, force_floor=0.1, max_drive_amplitude=max_drive_v,
                                                        max_amplitude_step=max_drive_step_v,
                                                        initial_drive_amplitude=initial_drive_v)
        total_drive_amplitude = initial_drive_v
        control_update_samples = max(1, round(0.1 * FS))
        samples_since_update = 0
        n_total = int(gen.sweep_duration * FS)
        errors_n = []

        for start in range(0, n_total, BLOCK_SIZE):
            n = min(BLOCK_SIZE, n_total - start)
            samples, freq, phase = gen.generate_block(n, drive_amplitude=total_drive_amplitude)
            target_bw = ForceTrackingEstimator.bandwidth_for_tracking_cycles(float(freq[0]), tracking_cycles)
            estimator.set_tracking_bandwidth(max(target_bw, 0.01))
            gain = realistic_plant_gain(freq)
            noise = rng.normal(0.0, 0.015 * abs(total_drive_amplitude) * np.mean(gain), size=freq.shape)
            force = gain * total_drive_amplitude * np.sin(phase) + noise
            result = estimator.process_block(force, phase)
            samples_since_update += n
            if samples_since_update >= control_update_samples:
                samples_since_update -= control_update_samples
                measured = result.amplitude if result.valid else None
                ctrl_result = trim_controller.update(target_force, measured, result.valid)
                if feedforward_enabled:
                    leg, direction = gen.leg_and_direction(gen.elapsed_time)
                    composition = compose_drive_amplitude(
                        published_map, float(freq[-1]), ctrl_result.drive_amplitude, total_drive_amplitude,
                        max_drive_v, max_drive_step_v, direction=direction)
                    total_drive_amplitude = composition.total_drive_amplitude
                    achieved = min(max(composition.achieved_trim_gain, 0.0), trim_controller.max_drive_amplitude)
                    trim_controller.drive_amplitude = achieved
                    trust = ctrl_result.status is ControllerStatus.OK
                    feedforward_map.update(float(freq[-1]), observed_value=total_drive_amplitude,
                                            trust=trust, direction=direction)
                else:
                    clipped = min(max(ctrl_result.drive_amplitude, 0.0), max_drive_v)
                    delta = min(max(clipped - total_drive_amplitude, -max_drive_step_v), max_drive_step_v)
                    total_drive_amplitude += delta
            if result.valid:
                errors_n.append(result.amplitude - target_force)
        return np.array(errors_n)

    seed = 20260903
    err_disabled = run_one_leg(feedforward_enabled=False, seed=seed)
    err_enabled = run_one_leg(feedforward_enabled=True, seed=seed)

    mean_abs_disabled = np.abs(err_disabled).mean()
    mean_abs_enabled = np.abs(err_enabled).mean()
    # Same seed/plant/all other parameters -- an enabled-but-fresh map must
    # not track meaningfully worse than disabled on its very first leg.
    assert mean_abs_enabled < mean_abs_disabled * 1.2
