# -*- coding: utf-8 -*-
"""
Unit tests for control_laws/constant_force_chirp_control.py.

Covers the scenarios required for a constant-force chirp controller intended
to drive a single shaker against a single force sensor:

 1. normal operating frequency range (convergence toward target force)
 2. behavior at a structural resonance
 3. behavior at an antiresonance (near-zero FRF magnitude)
 4. robustness to strong measurement noise
 5. missing / invalid FRF (must not blindly invert)
 6. force-abort triggering
 7. voltage-abort (clamping, never exceeded)
 8. acceleration-abort triggering
 9. very low coherence (FRF line must not be trusted)
10. frequencies outside the identified range (no extrapolation)
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_laws.constant_force_chirp_control import (
    ForceChirpControlConfig,
    LinearChirpProfile,
    compute_feedforward_profile,
    interpolate_voltage_amplitude,
    interpolate_validity,
    lockin_amplitude,
    fft_bin_amplitude,
    ConstantForceChirpController,
)
from tests.synthetic_frf import synthetic_force_voltage_frf


SAMPLE_RATE = 20000.0


def make_config(**overrides):
    defaults = dict(
        target_force=10.0,
        start_frequency=100.0,
        end_frequency=2000.0,
        chirp_duration=20.0,
        ramp_time=0.5,
        max_voltage=10.0,
        force_abort=15.0,
        acceleration_abort=50.0,
        control_update_rate=20.0,
        amplitude_smoothing=0.3,
        max_gain_step_per_update=1.2,
    )
    defaults.update(overrides)
    return ForceChirpControlConfig(**defaults)


def make_frame(frequency, amplitude, sample_rate=SAMPLE_RATE, n=1000, noise_rms=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    if noise_rms > 0:
        signal = signal + rng.normal(0, noise_rms, n)
    return signal


def t_for_frequency(config, frequency):
    """Inverse of LinearChirpProfile.frequency(): the elapsed sweep time at
    which the linear chirp defined by ``config`` reaches ``frequency``. Tests
    use this to keep the synthetic force sample's frequency consistent with
    what the controller's internal chirp-phase law believes the instantaneous
    frequency is at time t -- otherwise the lock-in demodulator legitimately
    sees a frequency mismatch and under-reports the amplitude."""
    sweep_rate = (config.end_frequency - config.start_frequency) / config.chirp_duration
    return (frequency - config.start_frequency) / sweep_rate


#%% 1. Normal frequency range: closed loop should converge to target force

def test_convergence_in_normal_frequency_range():
    freqs = np.linspace(50, 2100, 2048)
    H = synthetic_force_voltage_frf(freqs)
    config = make_config()
    ff = compute_feedforward_profile(freqs, H, config)
    profile = LinearChirpProfile(config.start_frequency, config.end_frequency, config.chirp_duration)
    controller = ConstantForceChirpController(config, ff, profile)

    frequency = 1000.0  # a well-behaved, non-resonant frequency; needs ~3.4 V for 10 N here
    t = t_for_frequency(config, frequency)
    true_H = synthetic_force_voltage_frf(np.array([frequency]))[0]
    voltage = interpolate_voltage_amplitude(ff, frequency)
    measured = None
    for i in range(30):
        force_samples = make_frame(frequency, voltage * abs(true_H), seed=i)
        measured_force = lockin_amplitude(force_samples, SAMPLE_RATE, frequency)
        result = controller.step(t=t, measured_force=measured_force)
        voltage = result.voltage_amplitude
        measured = result.measured_force
        assert not result.aborted
    assert measured == pytest.approx(config.target_force, rel=0.1)


#%% 2. Resonance: FRF magnitude is large -> required voltage should be small,
#      and force should still track the target

def test_behavior_at_resonance():
    freqs = np.linspace(50, 2100, 2048)
    H = synthetic_force_voltage_frf(freqs)
    config = make_config()
    ff = compute_feedforward_profile(freqs, H, config)

    resonance_freq = 300.0
    off_resonance_freq = 1000.0
    v_res = interpolate_voltage_amplitude(ff, resonance_freq)
    v_off = interpolate_voltage_amplitude(ff, off_resonance_freq)
    # At resonance |H_FV| is large, so far less voltage is needed for the same force.
    assert v_res < v_off
    assert v_res <= config.max_voltage


#%% 3. Antiresonance: FRF magnitude near zero -> inversion must be regularized,
#      not blow up toward infinity

def test_antiresonance_is_regularized_not_infinite():
    freqs = np.linspace(50, 2100, 4096)
    H = synthetic_force_voltage_frf(freqs)
    idx_anti = np.argmin(np.abs(freqs - 600.0))
    assert np.abs(H[idx_anti]) < 0.05 * np.max(np.abs(H))  # confirm the synthetic FRF really dips here

    config = make_config()
    ff = compute_feedforward_profile(freqs, H, config)
    v_anti = interpolate_voltage_amplitude(ff, 600.0)
    naive_unregularized_voltage = config.target_force / abs(H[idx_anti])  # what a blind pinv would ask for

    assert np.isfinite(v_anti)
    assert v_anti > 0
    assert v_anti <= config.max_voltage
    # The regularization floor must keep the requested voltage far below what
    # an unregularized 1/H inversion would demand at the antiresonance.
    assert v_anti < 0.1 * naive_unregularized_voltage


#%% 4. Strong measurement noise: gain should stay bounded (no chasing noise into instability)

def test_robust_to_strong_measurement_noise():
    freqs = np.linspace(50, 2100, 2048)
    H = synthetic_force_voltage_frf(freqs)
    config = make_config(amplitude_smoothing=0.15, max_gain_step_per_update=1.1)
    ff = compute_feedforward_profile(freqs, H, config)
    profile = LinearChirpProfile(config.start_frequency, config.end_frequency, config.chirp_duration)
    controller = ConstantForceChirpController(config, ff, profile)

    frequency = 1000.0
    t = t_for_frequency(config, frequency)
    true_H = synthetic_force_voltage_frf(np.array([frequency]))[0]
    voltage = interpolate_voltage_amplitude(ff, frequency)
    gains = []
    for i in range(200):
        force_samples = make_frame(frequency, voltage * abs(true_H), noise_rms=voltage * abs(true_H) * 2.0, seed=i)
        measured_force = lockin_amplitude(force_samples, SAMPLE_RATE, frequency)
        result = controller.step(t=t, measured_force=measured_force)
        voltage = result.voltage_amplitude
        gains.append(result.trim_gain)
        assert not result.aborted
        assert result.voltage_amplitude <= config.max_voltage
    # Trim gain must stay within its configured absolute bounds despite heavy noise
    assert min(gains) >= config.min_trim_gain
    assert max(gains) <= config.max_trim_gain


#%% 5. Missing / invalid FRF: must not blindly invert into huge voltages

def test_missing_frf_does_not_produce_unbounded_voltage():
    freqs = np.linspace(50, 2100, 512)
    H = np.full(freqs.shape, np.nan, dtype=complex)  # system ID totally failed
    config = make_config()
    ff = compute_feedforward_profile(freqs, H, config)
    assert np.all(~ff.valid)
    assert np.all(np.isfinite(ff.voltage_amplitude))
    assert np.all(ff.voltage_amplitude <= config.max_voltage)
    # With no valid data at all, the safe fallback is zero commanded voltage,
    # never an unbounded/blind inversion.
    assert np.all(ff.voltage_amplitude == 0.0)


def test_partially_invalid_frf_holds_nearest_valid_gain():
    freqs = np.linspace(50, 2100, 512)
    H = synthetic_force_voltage_frf(freqs)
    H_bad = H.copy()
    bad_mask = (freqs > 1200) & (freqs < 1500)
    H_bad[bad_mask] = np.nan
    config = make_config()
    ff = compute_feedforward_profile(freqs, H_bad, config)
    assert np.all(~ff.valid[bad_mask])
    # Held gain in the bad region should match the nearest valid edge, not zero/explode
    assert np.all(np.isfinite(ff.voltage_amplitude[bad_mask]))
    edge_gain = ff.inverse_gain[np.flatnonzero(bad_mask)[0] - 1]
    assert np.allclose(ff.inverse_gain[bad_mask], edge_gain)


#%% 6. Force overshoot -> controller must abort and drive to zero

def test_force_abort_triggers_and_zeros_output():
    freqs = np.linspace(50, 2100, 2048)
    H = synthetic_force_voltage_frf(freqs)
    config = make_config(force_abort=15.0)
    ff = compute_feedforward_profile(freqs, H, config)
    profile = LinearChirpProfile(config.start_frequency, config.end_frequency, config.chirp_duration)
    controller = ConstantForceChirpController(config, ff, profile)

    frequency = 1000.0
    t = t_for_frequency(config, frequency)
    # Force far above abort level regardless of controller state
    force_samples = make_frame(frequency, amplitude=20.0)
    measured_force = lockin_amplitude(force_samples, SAMPLE_RATE, frequency)
    result = controller.step(t=t, measured_force=measured_force)
    assert result.aborted
    assert result.abort_reason is not None
    assert result.voltage_amplitude == 0.0

    # Once aborted, the controller must stay aborted / zero on subsequent steps
    measured_force2 = lockin_amplitude(make_frame(frequency, 1.0), SAMPLE_RATE, frequency)
    result2 = controller.step(t=t + 0.05, measured_force=measured_force2)
    assert result2.aborted
    assert result2.voltage_amplitude == 0.0


#%% 7. Voltage never exceeds max_voltage, even under a "runaway" request

def test_voltage_is_always_clamped():
    freqs = np.linspace(50, 2100, 2048)
    H = synthetic_force_voltage_frf(freqs)
    config = make_config(max_voltage=5.0, target_force=1000.0)  # deliberately unreachable target
    ff = compute_feedforward_profile(freqs, H, config)
    assert np.all(ff.voltage_amplitude <= config.max_voltage)

    profile = LinearChirpProfile(config.start_frequency, config.end_frequency, config.chirp_duration)
    controller = ConstantForceChirpController(config, ff, profile)
    frequency = 1000.0
    t = t_for_frequency(config, frequency)
    voltage = interpolate_voltage_amplitude(ff, frequency)
    for i in range(50):
        # Force measured is always far below the (unreachable) target -> controller
        # will keep asking for more gain, but voltage must never exceed the limit.
        force_samples = make_frame(frequency, amplitude=0.01, seed=i)
        measured_force = lockin_amplitude(force_samples, SAMPLE_RATE, frequency)
        result = controller.step(t=t, measured_force=measured_force)
        voltage = result.voltage_amplitude
        assert voltage <= config.max_voltage + 1e-9
        if not result.aborted:
            assert any('clamp' in w.lower() for w in result.warnings)


#%% 8. Acceleration overshoot -> abort even though force is fine

def test_acceleration_abort_triggers_even_with_good_force():
    freqs = np.linspace(50, 2100, 2048)
    H = synthetic_force_voltage_frf(freqs)
    config = make_config(acceleration_abort=5.0)
    ff = compute_feedforward_profile(freqs, H, config)
    profile = LinearChirpProfile(config.start_frequency, config.end_frequency, config.chirp_duration)
    controller = ConstantForceChirpController(config, ff, profile)

    frequency = 900.0  # resonance in the synthetic FRF: force is fine, accel is not (monitoring only)
    t = t_for_frequency(config, frequency)
    force_samples = make_frame(frequency, amplitude=config.target_force)  # force right on target
    measured_force = lockin_amplitude(force_samples, SAMPLE_RATE, frequency)
    result = controller.step(t=t, measured_force=measured_force,
                              acceleration_peaks=[1.0, 8.0, 2.0])  # channel 1 exceeds abort level
    assert result.aborted
    assert 'acceleration' in result.abort_reason.lower()
    assert result.voltage_amplitude == 0.0


#%% 9. Very low coherence: FRF line must not be trusted for aggressive correction

def test_low_coherence_frf_is_flagged_and_untrusted():
    freqs = np.linspace(50, 2100, 512)
    H = synthetic_force_voltage_frf(freqs)
    coherence = np.ones_like(freqs)
    low_coherence_band = (freqs > 1700) & (freqs < 1900)
    coherence[low_coherence_band] = 0.1  # sensor dropout / high noise band

    config = make_config(coherence_threshold=0.7)
    ff = compute_feedforward_profile(freqs, H, config, coherence=coherence)
    assert np.all(~ff.valid[low_coherence_band])
    assert np.all(ff.valid[~low_coherence_band & (freqs >= config.minimum_frequency) &
                           (freqs <= config.maximum_frequency)])

    profile = LinearChirpProfile(config.start_frequency, config.end_frequency, config.chirp_duration)
    controller = ConstantForceChirpController(config, ff, profile)
    frequency = 1800.0  # inside the untrusted band
    t = t_for_frequency(config, frequency)
    force_samples = make_frame(frequency, amplitude=1.0)  # far off target
    measured_force = lockin_amplitude(force_samples, SAMPLE_RATE, frequency)
    result = controller.step(t=t, measured_force=measured_force)
    assert any('unreliable' in w.lower() for w in result.warnings)


#%% 10. Frequencies outside the identified range: no extrapolation

def test_frequencies_outside_identified_range_are_not_extrapolated():
    freqs = np.linspace(200, 1800, 512)  # system ID only covered 200-1800 Hz
    H = synthetic_force_voltage_frf(freqs)
    config = make_config(start_frequency=100.0, end_frequency=2000.0,
                         minimum_frequency=200.0, maximum_frequency=1800.0)
    ff = compute_feedforward_profile(freqs, H, config)

    below_range = freqs < 200.0
    above_range = freqs > 1800.0
    assert not np.any(below_range) and not np.any(above_range)  # sanity: grid matches range here

    # Explicitly probe frequencies the sweep will visit but sys-id never covered
    v_100 = interpolate_voltage_amplitude(ff, 100.0)
    v_150 = interpolate_voltage_amplitude(ff, 150.0)
    v_200 = interpolate_voltage_amplitude(ff, 200.0)  # first identified line
    # Held value below the identified range should equal the edge, not extrapolate/diverge
    assert v_100 == pytest.approx(v_150, rel=0.15)
    assert not interpolate_validity(ff, 100.0)
    assert v_100 <= config.max_voltage


#%% Supporting: lock-in amplitude estimator sanity

def test_lockin_amplitude_recovers_known_sine():
    frequency = 500.0
    amplitude = 3.3
    samples = make_frame(frequency, amplitude, n=4000)
    estimate = lockin_amplitude(samples, SAMPLE_RATE, frequency)
    assert estimate == pytest.approx(amplitude, rel=0.02)


def test_lockin_amplitude_rejects_out_of_band_noise():
    frequency = 500.0
    amplitude = 2.0
    n = 4000
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(42)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    # Strong interferer at an unrelated frequency plus broadband noise
    signal = signal + 5.0 * np.sin(2 * np.pi * 1234.0 * t) + rng.normal(0, 1.0, n)
    estimate = lockin_amplitude(signal, SAMPLE_RATE, frequency)
    assert estimate == pytest.approx(amplitude, abs=0.3)


def test_fft_bin_amplitude_matches_lockin_amplitude():
    """The FFT-based estimator (used by the real environment, reconstructing
    the frame via inverse FFT from an already-computed *rectangular-window*
    FFT -- see fft_bin_amplitude's docstring on why the control-phase
    collector must use Window.RECTANGLE) must agree exactly with the
    time-domain lock-in estimator for the same underlying signal, including
    at a frequency that does not land on an FFT line."""
    frequency = 743.0  # deliberately off an exact FFT bin
    amplitude = 4.2
    n = 2000
    samples = make_frame(frequency, amplitude, n=n)
    time_domain_estimate = lockin_amplitude(samples, SAMPLE_RATE, frequency)

    fft_values = np.fft.rfft(samples)  # no window applied (rectangular)
    fft_estimate = fft_bin_amplitude(fft_values, SAMPLE_RATE, frequency)

    assert fft_estimate == pytest.approx(time_domain_estimate, rel=1e-9)
    assert fft_estimate == pytest.approx(amplitude, rel=0.02)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
