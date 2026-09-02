# -*- coding: utf-8 -*-
"""
Standalone simulation of the feedforward learning layer described in
``components/feedforward_map.py``, demonstrating its intended behavior over
several continuous UP/DOWN sweep legs before relying on it against real
hardware -- no DAQ/Qt/multiprocessing involved, only the same building
blocks ``SineForceControlEnvironment`` itself wires together:

    SineSweepGenerator -> (simulated plant F = H(f) * u) -> ForceTrackingEstimator
                                                                    |
    FeedforwardMap.get(f) --*--> u_total --> ForceAmplitudeController (trim)
                              (composition mirrors
                               SineForceControlEnvironment
                               ._update_feedforward_and_compose)

The simulated plant ``H(f)`` has two resonances and measurement noise plus
occasional outlier spikes, so this exercises: (1) whether the measured force
amplitude stays close to target throughout a continuous UP/DOWN/UP/DOWN/UP
run, (2) whether the feedforward map actually learns a sensible curve,
(3) whether the fast loop's feedback correction shrinks over successive
sweeps as a result, (4) whether the learned curve is stable (not destroyed
by noise/outliers), and (5) whether the resonances remain resolved (not
smoothed away) rather than causing instability.

Usage
-----
    python simulate_feedforward_learning.py
    python simulate_feedforward_learning.py --no-show --out-dir results/feedforward_simulation

Produces PNG plots (frequency-response-vs-sweep, feedback-correction-vs-sweep,
force-vs-time) and prints a convergence summary.
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


def plant_gain(freq):
    """F/u magnitude (N per V) -- two lightly-damped resonances on top of a
    gently rolling-off baseline, spanning the 5-2000 Hz band. Deliberately
    not symmetric/smooth so the learned curve has real structure to recover."""
    def resonance(f, fr, zeta, h0):
        ratio = f / fr
        denom = np.sqrt((1 - ratio ** 2) ** 2 + (2 * zeta * ratio) ** 2)
        return h0 / denom

    baseline = 8.0 * (freq / 50.0) ** -0.3
    r1 = resonance(freq, fr=120.0, zeta=0.04, h0=25.0)
    r2 = resonance(freq, fr=900.0, zeta=0.03, h0=40.0)
    return baseline + r1 + r2


def measure_force(freq, u, phase, noise_std_fraction=0.01, outlier_prob=0.0015):
    """Simulates the raw force signal: true plant response + broadband
    measurement noise + occasional large transient spikes (outliers)."""
    true_force = plant_gain(freq) * u * np.sin(phase)
    noise = RNG.normal(0.0, noise_std_fraction * np.abs(u) * np.mean(plant_gain(freq)), size=freq.shape)
    signal = true_force + noise
    if RNG.random() < outlier_prob:
        spike_index = RNG.integers(0, len(signal))
        signal[spike_index] += RNG.uniform(20.0, 60.0) * np.sign(RNG.normal())
    return signal


def run_simulation(f_start=5.0, f_stop=2000.0, target_force=6.0, n_legs=5,
                    sweep_rate_oct_per_min=25.0, max_drive_v=6.0,
                    feedforward_enabled=True, learning_rate=0.15,
                    bins_per_decade=12.0):
    gen = SineSweepGenerator(sample_rate=FS, sweep_type='logarithmic',
                              f_start=f_start, f_stop=f_stop,
                              sweep_rate=sweep_rate_oct_per_min,
                              repeat=True, num_sweeps=n_legs, alternate_direction=True)
    estimator = ForceTrackingEstimator(sample_rate=FS, tracking_bandwidth_hz=6.0)

    if feedforward_enabled:
        feedforward_map = FeedforwardMap(
            f_min=f_start, f_max=f_stop, initial_estimate=0.3,
            value_min=0.02, value_max=max_drive_v,
            bins_per_decade=bins_per_decade, learning_rate=learning_rate)
        trim_controller = ForceAmplitudeController(
            alpha=0.4, force_floor=0.05, max_drive_amplitude=4.0,
            max_amplitude_step=0.5, initial_drive_amplitude=1.0)
    else:
        feedforward_map = None
        trim_controller = ForceAmplitudeController(
            alpha=0.4, force_floor=0.05, max_drive_amplitude=max_drive_v,
            max_amplitude_step=0.5, initial_drive_amplitude=0.3)

    max_drive_step_v = 3.0
    total_drive_amplitude = (feedforward_map.initial_estimate if feedforward_map is not None
                              else trim_controller.drive_amplitude)

    leg_duration = gen.sweep_duration
    n_total = int(n_legs * leg_duration * FS)

    log = {k: [] for k in ('time', 'leg', 'direction', 'frequency', 'target',
                            'measured', 'valid', 'feedforward_value',
                            'feedback_pct', 'total_drive', 'learning_applied')}

    for start in range(0, n_total, BLOCK_SIZE):
        n = min(BLOCK_SIZE, n_total - start)
        samples, freq, phase = gen.generate_block(n, drive_amplitude=total_drive_amplitude)
        force = measure_force(freq, total_drive_amplitude, phase)
        result = estimator.process_block(force, phase)
        measured = result.amplitude if result.valid else None
        ctrl_result = trim_controller.update(target_force, measured, result.valid)
        leg, direction = gen.leg_and_direction(gen.elapsed_time)
        f_end = float(freq[-1])

        if feedforward_map is not None:
            composition = compose_drive_amplitude(
                feedforward_map, f_end, ctrl_result.drive_amplitude, total_drive_amplitude,
                max_drive_v, max_drive_step_v, direction=direction)
            total_drive_amplitude = composition.total_drive_amplitude
            achieved_trim_gain = min(max(composition.achieved_trim_gain, 0.0),
                                      trim_controller.max_drive_amplitude)
            trim_controller.drive_amplitude = achieved_trim_gain  # anti-windup resync

            trust = ctrl_result.status is ControllerStatus.OK
            learn_result = feedforward_map.update(f_end, observed_value=total_drive_amplitude,
                                                    trust=trust, direction=direction)
            ff_value = composition.feedforward_value
            feedback_pct = (achieved_trim_gain - 1.0) * 100.0
            learning_applied = learn_result.updated
        else:
            ff_value = float('nan')
            clipped = min(max(ctrl_result.drive_amplitude, 0.0), max_drive_v)
            delta = min(max(clipped - total_drive_amplitude, -max_drive_step_v), max_drive_step_v)
            total_drive_amplitude += delta
            feedback_pct = float('nan')
            learning_applied = False

        log['time'].append(gen.elapsed_time)
        log['leg'].append(leg)
        log['direction'].append(direction)
        log['frequency'].append(f_end)
        log['target'].append(target_force)
        log['measured'].append(result.amplitude if result.valid else np.nan)
        log['valid'].append(result.valid)
        log['feedforward_value'].append(ff_value)
        log['feedback_pct'].append(feedback_pct)
        log['total_drive'].append(total_drive_amplitude)
        log['learning_applied'].append(learning_applied)

    for key in log:
        log[key] = np.array(log[key])
    return log, feedforward_map, n_legs


def summarize(log, n_legs):
    print('Sweep legs: {:}'.format(n_legs))
    print('{:>4}  {:>8}  {:>14}  {:>16}'.format('leg', 'dir', 'mean|err| (%)', 'mean|feedback| (%)'))
    valid = log['valid']
    for leg in range(n_legs):
        m = (log['leg'] == leg) & valid
        if not np.any(m):
            continue
        err_pct = np.abs(log['measured'][m] - log['target'][m]) / log['target'][m] * 100.0
        fb_pct = np.abs(log['feedback_pct'][(log['leg'] == leg)])
        fb_pct = fb_pct[np.isfinite(fb_pct)]
        direction = log['direction'][m][0]
        print('{:>4}  {:>8}  {:>14.2f}  {:>16.2f}'.format(
            leg, direction, err_pct.mean(), fb_pct.mean() if len(fb_pct) else float('nan')))


def make_plots(log, feedforward_map, n_legs, out_dir, show):
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_legs))

    # 1. Required command vs frequency, one line per sweep leg (should
    #    converge toward a single stable curve as legs progress).
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    for leg in range(n_legs):
        m = log['leg'] == leg
        order = np.argsort(log['frequency'][m])
        ax1.plot(log['frequency'][m][order], log['total_drive'][m][order],
                  '.', color=colors[leg], markersize=2, alpha=0.6,
                  label='Sweep leg {:} ({:})'.format(leg + 1, log['direction'][m][0] if np.any(m) else ''))
    if feedforward_map is not None:
        freqs, values, n_obs = feedforward_map.curve()
        ax1.plot(freqs, values, 'k-', linewidth=2, label='Learned A_FF(f)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Commanded drive amplitude (V peak)')
    ax1.set_title('Required command vs. frequency, by sweep leg')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3, which='both')
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'ff_sim_command_vs_frequency.png'), dpi=130)

    # 2. Feedback correction vs frequency, by leg -- should shrink toward 0.
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for leg in range(n_legs):
        m = log['leg'] == leg
        order = np.argsort(log['frequency'][m])
        ax2.plot(log['frequency'][m][order], log['feedback_pct'][m][order],
                  '.', color=colors[leg], markersize=2, alpha=0.6,
                  label='Sweep leg {:}'.format(leg + 1))
    ax2.axhline(0.0, color='k', linewidth=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Feedback correction (%)')
    ax2.set_title('Fast-loop trim correction vs. frequency, by sweep leg\n'
                   '(shrinking toward the leg-1 line means the feedforward map is learning)')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3, which='both')
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'ff_sim_feedback_vs_frequency.png'), dpi=130)

    # 3. Measured force amplitude over the whole continuous run.
    fig3, ax3 = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    ax3[0].plot(log['time'], log['frequency'], color='tab:blue')
    ax3[0].set_ylabel('Frequency (Hz)')
    ax3[0].set_yscale('log')
    ax3[0].grid(True, alpha=0.3)
    valid = log['valid']
    ax3[1].plot(log['time'][valid], log['measured'][valid], '.', color='tab:blue',
                markersize=2, label='Measured force')
    ax3[1].plot(log['time'], log['target'], '--', color='tab:gray', label='Target force')
    ax3[1].set_ylabel('Force (N peak)')
    ax3[1].set_xlabel('Time (s)')
    ax3[1].legend(fontsize=8)
    ax3[1].grid(True, alpha=0.3)
    fig3.suptitle('Continuous UP/DOWN sweeps -- force amplitude tracking')
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, 'ff_sim_force_vs_time.png'), dpi=130)

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true', help='Do not open plot windows (still saves PNGs)')
    parser.add_argument('--out-dir', default='results/feedforward_simulation',
                        help='Directory to save PNG plots into')
    parser.add_argument('--n-legs', type=int, default=5, help='Number of sweep legs (up/down/up/...)')
    args = parser.parse_args()

    print('Running feedforward-disabled baseline...')
    log_off, _, n_legs = run_simulation(feedforward_enabled=False, n_legs=args.n_legs)
    summarize(log_off, n_legs)

    print()
    print('Running feedforward-enabled simulation...')
    log_on, ff_map, n_legs = run_simulation(feedforward_enabled=True, n_legs=args.n_legs)
    summarize(log_on, n_legs)

    valid = log_on['valid']
    first_leg = (log_on['leg'] == 0) & valid
    last_leg = (log_on['leg'] == n_legs - 1) & valid
    err_first = np.abs(log_on['measured'][first_leg] - log_on['target'][first_leg]).mean()
    err_last = np.abs(log_on['measured'][last_leg] - log_on['target'][last_leg]).mean()
    fb_first = np.abs(log_on['feedback_pct'][log_on['leg'] == 0])
    fb_last = np.abs(log_on['feedback_pct'][log_on['leg'] == n_legs - 1])
    print()
    print('Summary: mean |force error| leg 1 -> leg {:}: {:.3f} N -> {:.3f} N'.format(
        n_legs, err_first, err_last))
    print('Summary: mean |feedback correction| leg 1 -> leg {:}: {:.2f}% -> {:.2f}%'.format(
        n_legs, np.nanmean(fb_first), np.nanmean(fb_last)))

    make_plots(log_on, ff_map, n_legs, args.out_dir, show=not args.no_show)
    print('Plots saved to {:}'.format(os.path.abspath(args.out_dir)))
