# -*- coding: utf-8 -*-
"""
Validates a tracking-bandwidth configuration change against a synthetic
plant before trying it on real hardware.

Context: a real diagnostics file (test_force_control_new_environment_..._0006)
showed the closed loop oscillating around the target force, with the error
almost entirely concentrated at the low end of a 5-500 Hz sweep:

    5-10 Hz:   mean |error| ~48%
    10-20 Hz:  mean |error| ~28%
    20-50 Hz:  mean |error| ~11%
    200-500 Hz: mean |error| ~6%   (already fine)

That test used a *fixed* Tracking Bandwidth = 10 Hz with
Adaptive Tracking Bandwidth = off. At 5-10 Hz a 10 Hz bandwidth is not "well
below" the drive frequency any more (see force_tracking_estimator.py), so
the demodulated double-frequency component leaks into the amplitude
estimate and corrupts both what the fast loop reacts to and what gets
learned into the feedforward map -- independent of any feedforward/
controller tuning.

This script reproduces that real configuration as closely as possible
(sweep type/range/rate, controller_alpha, control_update_period_s,
feedforward settings) against a synthetic plant, and compares:

    baseline:    adaptive_tracking_bandwidth=False, tracking_bandwidth_hz=10.0  (what was run)
    recommended: adaptive_tracking_bandwidth=True,  tracking_cycles=1.5         (proposed fix)

using the same per-frequency-band error breakdown used to diagnose the real
file, so the comparison is directly apples-to-apples with that diagnosis.

Usage
-----
    python simulate_tracking_bandwidth_comparison.py
    python simulate_tracking_bandwidth_comparison.py --no-show --tracking-cycles 1.0
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from components.sine_sweep_generator import SineSweepGenerator
from components.force_tracking_estimator import ForceTrackingEstimator
from components.force_amplitude_controller import ForceAmplitudeController, ControllerStatus
from components.feedforward_map import FeedforwardMap, compose_drive_amplitude

FS = 5000.0
BLOCK_SIZE = 256
RNG = np.random.default_rng(20260902)

# Mirrors MIN_ADAPTIVE_TRACKING_BANDWIDTH_HZ in
# components/sine_force_control_environment.py (kept as a local constant
# here rather than importing that Qt-dependent module -- see its own
# docstring: purely a numerical safety floor, not a physical limit).
MIN_ADAPTIVE_TRACKING_BANDWIDTH_HZ = 0.01

FREQ_BANDS = [5, 10, 20, 50, 100, 200, 500]


def flat_plant_gain(freq):
    """Frequency-independent plant gain (F/u ~200 N/V) -- deliberately flat
    so any error pattern seen here is isolated to the tracking-bandwidth/
    demodulation effect being validated, not plant shape. Required drive
    for target_force=80N is then a constant ~0.4 V, matching the ~0.2-0.5V
    range actually observed in the real diagnostics file."""
    return 200.0 * np.ones_like(freq)


def measure_force(freq, u, phase, noise_std_fraction=0.02):
    true_force = flat_plant_gain(freq) * u * np.sin(phase)
    noise = RNG.normal(0.0, noise_std_fraction * np.abs(u) * flat_plant_gain(freq).mean(), size=freq.shape)
    return true_force + noise


def run_realistic_scenario(adaptive_tracking_bandwidth, tracking_bandwidth_hz=10.0,
                            tracking_cycles=1.5, n_legs=3):
    """Mirrors the real test's configuration: linear sweep 5-500 Hz at
    10 Hz/s, alternating direction, target_force=80N, controller_alpha=0.15,
    control_update_period_s=0.1s, feedforward enabled with the same limits."""
    f_start, f_stop = 5.0, 500.0
    target_force = 80.0
    max_drive_v = 1.25
    max_drive_step_v = 0.01
    initial_drive_v = 0.2
    control_update_period_s = 0.1

    gen = SineSweepGenerator(sample_rate=FS, sweep_type='linear',
                              f_start=f_start, f_stop=f_stop, sweep_rate=10.0,
                              repeat=True, num_sweeps=n_legs, alternate_direction=True)
    initial_bw = (ForceTrackingEstimator.bandwidth_for_tracking_cycles(f_start, tracking_cycles)
                  if adaptive_tracking_bandwidth else tracking_bandwidth_hz)
    estimator = ForceTrackingEstimator(sample_rate=FS, tracking_bandwidth_hz=initial_bw)

    feedforward_map = FeedforwardMap(f_min=f_start, f_max=f_stop, initial_estimate=initial_drive_v,
                                      value_min=0.05, value_max=max_drive_v,
                                      bins_per_decade=10.0, learning_rate=0.2)
    trim_controller = ForceAmplitudeController(alpha=0.15, force_floor=0.1,
                                                max_drive_amplitude=3.0,
                                                max_amplitude_step=0.5,
                                                initial_drive_amplitude=1.0)

    control_update_samples = max(1, round(control_update_period_s * FS))
    samples_since_control_update = 0
    total_drive_amplitude = initial_drive_v

    leg_duration = gen.sweep_duration
    n_total = int(n_legs * leg_duration * FS)

    log = {k: [] for k in ('time', 'leg', 'direction', 'frequency', 'measured',
                            'valid', 'relative_error', 'feedback_pct')}

    for start in range(0, n_total, BLOCK_SIZE):
        n = min(BLOCK_SIZE, n_total - start)
        samples, freq, phase = gen.generate_block(n, drive_amplitude=total_drive_amplitude)

        if adaptive_tracking_bandwidth:
            target_bw = ForceTrackingEstimator.bandwidth_for_tracking_cycles(float(freq[0]), tracking_cycles)
            estimator.set_tracking_bandwidth(max(target_bw, MIN_ADAPTIVE_TRACKING_BANDWIDTH_HZ))

        force = measure_force(freq, total_drive_amplitude, phase)
        result = estimator.process_block(force, phase)
        leg, direction = gen.leg_and_direction(gen.elapsed_time)
        f_end = float(freq[-1])

        samples_since_control_update += n
        feedback_pct = float('nan')
        if samples_since_control_update >= control_update_samples:
            samples_since_control_update -= control_update_samples
            measured = result.amplitude if result.valid else None
            ctrl_result = trim_controller.update(target_force, measured, result.valid)

            composition = compose_drive_amplitude(
                feedforward_map, f_end, ctrl_result.drive_amplitude, total_drive_amplitude,
                max_drive_v, max_drive_step_v, direction=direction)
            total_drive_amplitude = composition.total_drive_amplitude
            achieved_trim_gain = min(max(composition.achieved_trim_gain, 0.0),
                                      trim_controller.max_drive_amplitude)
            trim_controller.drive_amplitude = achieved_trim_gain

            trust = ctrl_result.status is ControllerStatus.OK
            feedforward_map.update(f_end, observed_value=total_drive_amplitude,
                                    trust=trust, direction=direction)
            feedback_pct = (achieved_trim_gain - 1.0) * 100.0

        log['time'].append(gen.elapsed_time)
        log['leg'].append(leg)
        log['direction'].append(direction)
        log['frequency'].append(f_end)
        log['measured'].append(result.amplitude if result.valid else np.nan)
        log['valid'].append(result.valid)
        log['relative_error'].append(
            (result.amplitude - target_force) / target_force if result.valid else np.nan)
        log['feedback_pct'].append(feedback_pct)

    for key in log:
        log[key] = np.array(log[key])
    return log, feedforward_map, target_force


def band_breakdown(log, label):
    freq = log['frequency']
    err = np.abs(log['relative_error']) * 100.0
    valid = log['valid']
    print('{:}:'.format(label))
    for lo, hi in zip(FREQ_BANDS[:-1], FREQ_BANDS[1:]):
        m = (freq >= lo) & (freq < hi) & valid
        if np.any(m):
            print('  {:>4}-{:<4}Hz  n={:4d}  mean|err%|={:6.2f}  median={:6.2f}'.format(
                lo, hi, int(m.sum()), np.nanmean(err[m]), np.nanmedian(err[m])))
    overall = err[valid]
    print('  overall mean|err%|={:.2f}'.format(np.nanmean(overall)))


def make_comparison_plot(log_baseline, log_recommended, out_dir, show):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    for log, label, color in ((log_baseline, 'Baseline (fixed 10 Hz)', 'tab:red'),
                               (log_recommended, 'Recommended (adaptive)', 'tab:blue')):
        valid = log['valid']
        err = np.abs(log['relative_error']) * 100.0
        order = np.argsort(log['frequency'][valid])
        ax.plot(log['frequency'][valid][order], err[valid][order], '.', color=color,
                markersize=2, alpha=0.5, label=label)
    ax.set_xscale('log')
    ax.set_ylabel('|Relative force error| (%)')
    ax.set_title('Force error vs. frequency: fixed vs. adaptive tracking bandwidth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    ax = axes[1]
    for log, label, color in ((log_baseline, 'Baseline (fixed 10 Hz)', 'tab:red'),
                               (log_recommended, 'Recommended (adaptive)', 'tab:blue')):
        order = np.argsort(log['frequency'])
        ax.plot(log['frequency'][order], np.abs(log['feedback_pct'][order]), '.', color=color,
                markersize=2, alpha=0.5, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|Feedback correction| (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tracking_bandwidth_comparison.png'), dpi=130)
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--out-dir', default='results/feedforward_simulation')
    parser.add_argument('--tracking-cycles', type=float, default=1.5,
                        help='Tracking Cycles to validate for the recommended (adaptive) config')
    parser.add_argument('--n-legs', type=int, default=3)
    args = parser.parse_args()

    print('=== Baseline: as actually run (Adaptive Tracking Bandwidth = off, 10 Hz fixed) ===')
    log_baseline, ff_baseline, target = run_realistic_scenario(
        adaptive_tracking_bandwidth=False, tracking_bandwidth_hz=10.0, n_legs=args.n_legs)
    band_breakdown(log_baseline, 'Baseline')

    print()
    print('=== Recommended: Adaptive Tracking Bandwidth = on, Tracking Cycles = {:.2f} ==='.format(
        args.tracking_cycles))
    log_recommended, ff_recommended, target = run_realistic_scenario(
        adaptive_tracking_bandwidth=True, tracking_cycles=args.tracking_cycles, n_legs=args.n_legs)
    band_breakdown(log_recommended, 'Recommended')

    print()
    low_band_baseline = np.nanmean(np.abs(log_baseline['relative_error'][
        (log_baseline['frequency'] < 20) & log_baseline['valid']])) * 100.0
    low_band_recommended = np.nanmean(np.abs(log_recommended['relative_error'][
        (log_recommended['frequency'] < 20) & log_recommended['valid']])) * 100.0
    print('Low-band (<20 Hz) mean |error|: {:.1f}% -> {:.1f}%'.format(
        low_band_baseline, low_band_recommended))

    make_comparison_plot(log_baseline, log_recommended, args.out_dir, show=not args.no_show)
    print('Plot saved to {:}'.format(os.path.abspath(args.out_dir)))
