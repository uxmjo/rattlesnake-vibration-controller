# -*- coding: utf-8 -*-
"""
End-to-end simulation of the constant-force chirp controller against a
synthetic single-shaker / single-force-sensor system with:

  * a structural resonance at 300 Hz
  * an antiresonance at 600 Hz
  * a second structural resonance at 900 Hz

and a separate monitoring-only accelerometer channel with its own resonance
at 800 Hz, to demonstrate that force is held constant while acceleration is
free to vary strongly (never controlled to).

This script exercises the real control loop
(control_laws.constant_force_chirp_control.ConstantForceChirpController)
frame by frame across a full sweep, without any Rattlesnake multiprocessing/
Qt/hardware -- it stands in for what
components/constant_force_chirp_environment.py will do at runtime once wired
into the acquisition/signal-generation subprocesses.

Run with:  python tests/simulate_constant_force_chirp.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_laws.constant_force_chirp_control import (
    ForceChirpControlConfig,
    LinearChirpProfile,
    compute_feedforward_profile,
    ConstantForceChirpController,
    lockin_amplitude,
)
from tests.synthetic_frf import synthetic_force_voltage_frf, synthetic_accel_voltage_frf


def run_simulation(seed=0, force_noise_fraction=0.05, gain_error=1.0, plot_path=None):
    """Run one full constant-force chirp simulation.

    Parameters
    ----------
    force_noise_fraction : float
        Measurement noise on the force sensor, as a fraction of the true
        instantaneous force amplitude (models sensor/electrical noise).
    gain_error : float
        Deliberate mismatch between the identified FRF (used to build the
        feedforward) and the "true" plant FRF used to simulate the response
        (models system-ID error / mild nonlinearity/drift); 1.0 = no error.
    """
    rng = np.random.default_rng(seed)

    config = ForceChirpControlConfig(
        # target_force is chosen so that, given the synthetic plant's FRF
        # magnitude and max_voltage headroom, the target is achievable across
        # essentially the whole swept band -- except deliberately at the
        # 600 Hz antiresonance, where the voltage clamp legitimately prevents
        # reaching it (this is the safety behavior being demonstrated, not a
        # tuning failure). A stiffer/less sensitive test article, or a lower
        # max_voltage amplifier, would need a correspondingly lower
        # target_force for the same reason.
        target_force=2.0,
        start_frequency=100.0,
        end_frequency=2000.0,
        chirp_duration=30.0,
        ramp_time=1.0,
        max_voltage=10.0,
        force_abort=5.0,
        acceleration_abort=900.0,  # generous: this run intentionally lets accel swing near its own
                                    # 800 Hz resonance (unrelated to the 300/900 Hz force resonances)
                                    # to demonstrate force stays controlled while accel is not
        control_update_rate=20.0,  # 20 Hz control update -> 50 ms frames
        amplitude_smoothing=0.25,
        max_gain_step_per_update=1.15,
        # This synthetic plant's FRF rolls off smoothly from ~84 N/V (at the
        # 300 Hz resonance) down to ~0.24 N/V at 2000 Hz -- a >300:1 dynamic
        # range even away from the antiresonance. A regularization floor
        # sized for a "typical" test article (a few % of the FRF's peak,
        # config's 0.03 default) would therefore also suppress legitimate,
        # honestly-identified response at the high-frequency end of this
        # particular sweep. Real test articles vary widely in dynamic range,
        # which is exactly why `regularization` is left as an operator-set
        # parameter (mirroring Rattlesnake's existing per-test `rcond` for
        # `pseudoinverse_control`) rather than a fixed constant.
        regularization=0.005,
    )

    # --- Phase 1: system identification (simulated) ---
    # A real run gets this from Rattlesnake's existing H1 estimator
    # (components/spectral_processing.py). Here we sample the "true" plant
    # FRF on a modest system-ID frequency grid and add a touch of estimation
    # noise/coherence dropout to be realistic.
    sysid_freqs = np.linspace(config.start_frequency * 0.9, config.end_frequency * 1.05, 800)
    true_H = synthetic_force_voltage_frf(sysid_freqs)
    sysid_noise = 1 + 0.03 * (rng.normal(size=sysid_freqs.shape) + 1j * rng.normal(size=sysid_freqs.shape))
    identified_H = true_H * sysid_noise * gain_error
    coherence = np.clip(1 - np.abs(sysid_noise - 1) * 3, 0, 1)

    feedforward = compute_feedforward_profile(sysid_freqs, identified_H, config, coherence=coherence)

    profile = LinearChirpProfile(config.start_frequency, config.end_frequency, config.chirp_duration)
    controller = ConstantForceChirpController(config, feedforward, profile)

    # --- Phase 2/3: run the chirp, frame by frame, closed loop ---
    sample_rate = 20000.0
    frame_samples = int(sample_rate / config.control_update_rate)
    n_frames = int(np.ceil(config.chirp_duration * config.control_update_rate)) + int(config.ramp_time * config.control_update_rate) + 2

    voltage = 0.0
    log = {k: [] for k in ('t', 'frequency', 'target_force', 'measured_force',
                            'true_force', 'voltage', 'trim_gain', 'acceleration', 'aborted')}

    t = 0.0
    for frame in range(n_frames):
        if t > config.chirp_duration:
            break
        f_now = float(profile.frequency(np.array([t]))[0])

        # Simulate the plant response to the *previously commanded* voltage
        # (one-frame actuation delay -- a simple, honest model of the real
        # output -> amplifier -> shaker -> sensor -> acquisition latency).
        true_force_amplitude = voltage * abs(synthetic_force_voltage_frf(np.array([f_now]))[0])
        true_accel_amplitude = voltage * abs(synthetic_accel_voltage_frf(np.array([f_now]))[0])

        time_axis = np.arange(frame_samples) / sample_rate
        force_signal = true_force_amplitude * np.sin(2 * np.pi * f_now * time_axis)
        force_signal += rng.normal(0, force_noise_fraction * max(true_force_amplitude, 1e-3), frame_samples)
        accel_peak = true_accel_amplitude * (1 + rng.normal(0, 0.02))

        measured_force = lockin_amplitude(force_signal, sample_rate, f_now)
        result = controller.step(t=t, measured_force=measured_force,
                                  acceleration_peaks=[accel_peak])

        log['t'].append(t)
        log['frequency'].append(f_now)
        log['target_force'].append(config.target_force)
        log['measured_force'].append(result.measured_force)
        log['true_force'].append(true_force_amplitude)
        log['voltage'].append(voltage)  # voltage actually applied during this frame
        log['trim_gain'].append(result.trim_gain)
        log['acceleration'].append(accel_peak)
        log['aborted'].append(result.aborted)

        if result.aborted:
            voltage = 0.0
            break

        voltage = result.voltage_amplitude
        t += frame_samples / sample_rate

    for k in log:
        log[k] = np.array(log[k])

    if plot_path is not None:
        _plot(log, config, plot_path)

    return log, config


def _plot(log, config, plot_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    ax = axes[0]
    ax.plot(log['t'], log['frequency'])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('Constant-Force Chirp Simulation (300/900 Hz resonance, 600 Hz antiresonance)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.axhline(config.target_force, color='k', linestyle='--', label='F_target')
    ax.plot(log['t'], log['measured_force'], label='F_measured (lock-in)', alpha=0.8)
    ax.axhline(config.force_abort, color='r', linestyle=':', label='force_abort')
    ax.set_ylabel('Force [N pk]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(log['t'], log['voltage'], color='tab:orange')
    ax.axhline(config.max_voltage, color='r', linestyle=':', label='max_voltage')
    ax.set_ylabel('Output Voltage [V pk]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(log['t'], log['acceleration'], color='tab:green')
    ax.set_ylabel('Acceleration [monitoring]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    log, config = run_simulation(plot_path=os.path.join(out_dir, 'constant_force_chirp_simulation.png'))
    steady = log['t'] > config.ramp_time + 1.0  # skip startup ramp/settling
    err = np.abs(log['measured_force'][steady] - config.target_force) / config.target_force
    print(f"Frames simulated: {len(log['t'])}")
    print(f"Aborted: {bool(np.any(log['aborted']))}")
    print(f"Max |voltage|: {np.max(log['voltage']):.3f} V (limit {config.max_voltage} V)")
    print(f"Median force tracking error (post-settling): {np.median(err)*100:.2f} %")
    print(f"95th pct force tracking error (post-settling): {np.percentile(err,95)*100:.2f} %")
    print(f"Max acceleration reached: {np.max(log['acceleration']):.2f} (monitoring only, uncontrolled)")
    print(f"Plot written to {os.path.join(out_dir, 'constant_force_chirp_simulation.png')}")
