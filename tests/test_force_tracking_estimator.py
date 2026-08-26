# -*- coding: utf-8 -*-
"""Tests for components.force_tracking_estimator.ForceTrackingEstimator."""
import numpy as np
import pytest

from components.force_tracking_estimator import ForceTrackingEstimator
from components.sine_sweep_generator import SineSweepGenerator


def _run_blocks(estimator, force, phase, block_size):
    """Feeds `force`/`phase` through the estimator in fixed-size blocks and
    returns the list of ForceTrackingResult, one per block."""
    results = []
    n = len(force)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        results.append(estimator.process_block(force[start:end], phase[start:end]))
    return results


def test_constant_frequency_amplitude_recovered():
    """F(t) = 20*sin(2*pi*100*t + phi) -> estimator should settle to ~20 N."""
    fs = 10000.0
    f0 = 100.0
    amplitude = 20.0
    phi0 = 0.7  # arbitrary phase offset between drive reference and force
    duration = 2.0
    n = int(duration * fs)
    t = np.arange(n) / fs
    reference_phase = 2 * np.pi * f0 * t  # known generator phase (phi0 not known to generator)
    force = amplitude * np.sin(2 * np.pi * f0 * t + phi0)

    estimator = ForceTrackingEstimator(sample_rate=fs, tracking_bandwidth_hz=5.0)
    results = _run_blocks(estimator, force, reference_phase, block_size=256)

    # Last several results (after settling) should be close to 20 N
    settled = [r for r in results if r.valid]
    assert len(settled) > 0
    last = settled[-1]
    assert last.amplitude == pytest.approx(amplitude, rel=0.05)


def test_phase_offset_does_not_corrupt_amplitude():
    """Force signal may have an arbitrary phase relative to the reference;
    amplitude estimate must still be correct."""
    fs = 8000.0
    f0 = 50.0
    amplitude = 12.5
    duration = 3.0
    n = int(duration * fs)
    t = np.arange(n) / fs
    reference_phase = 2 * np.pi * f0 * t

    amplitudes_found = []
    for phi0 in [0.0, np.pi / 4, np.pi / 2, np.pi, 3.5]:
        force = amplitude * np.sin(2 * np.pi * f0 * t + phi0)
        estimator = ForceTrackingEstimator(sample_rate=fs, tracking_bandwidth_hz=5.0)
        results = _run_blocks(estimator, force, reference_phase, block_size=512)
        settled = [r for r in results if r.valid]
        amplitudes_found.append(settled[-1].amplitude)

    for a in amplitudes_found:
        assert a == pytest.approx(amplitude, rel=0.05)


def test_sweep_amplitude_tracked_via_generator_phase():
    """Simulate a constant-amplitude chirp and confirm the estimator, using
    the generator's own phase/frequency, tracks the amplitude through the sweep."""
    fs = 20000.0
    amplitude = 15.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='logarithmic',
                              f_start=20.0, f_stop=200.0, sweep_rate=120.0)
    n_total = int(gen.sweep_duration * fs)
    _, freq, phase = gen.generate_block(n_total, drive_amplitude=1.0)
    force = amplitude * np.sin(phase)

    estimator = ForceTrackingEstimator(sample_rate=fs, tracking_bandwidth_hz=3.0)
    results = _run_blocks(estimator, force, phase, block_size=512)
    settled = [r for r in results if r.valid]
    assert len(settled) > 10
    # Amplitude should stay close to the true constant amplitude throughout
    # the (quasi-stationary) sweep, well after the initial settle transient.
    late = settled[len(settled) // 2:]
    for r in late:
        assert r.amplitude == pytest.approx(amplitude, rel=0.1)


def test_noise_robustness():
    fs = 10000.0
    f0 = 100.0
    amplitude = 20.0
    duration = 3.0
    n = int(duration * fs)
    t = np.arange(n) / fs
    reference_phase = 2 * np.pi * f0 * t
    rng = np.random.default_rng(42)
    noise = rng.normal(scale=5.0, size=n)  # broadband noise, comparable to signal
    force = amplitude * np.sin(reference_phase) + noise

    estimator = ForceTrackingEstimator(sample_rate=fs, tracking_bandwidth_hz=2.0)
    results = _run_blocks(estimator, force, reference_phase, block_size=512)
    settled = [r for r in results if r.valid]
    amplitudes = np.array([r.amplitude for r in settled[len(settled) // 2:]])
    assert amplitudes.mean() == pytest.approx(amplitude, rel=0.1)
    # Should remain reasonably stable (low relative std) despite the noise
    assert amplitudes.std() / amplitudes.mean() < 0.2


def test_invalid_before_settled():
    fs = 5000.0
    estimator = ForceTrackingEstimator(sample_rate=fs, tracking_bandwidth_hz=1.0)
    n = 5
    force = np.ones(n)
    phase = np.zeros(n)
    result = estimator.process_block(force, phase)
    assert result.valid is False


def test_nan_input_marked_invalid_and_does_not_corrupt_state():
    fs = 5000.0
    estimator = ForceTrackingEstimator(sample_rate=fs, tracking_bandwidth_hz=5.0)
    good_force = np.ones(100) * 10.0
    phase = np.zeros(100)
    estimator.process_block(good_force, phase)

    bad_force = np.full(100, np.nan)
    result = estimator.process_block(bad_force, phase)
    assert result.valid is False
    assert np.isnan(result.amplitude)

    # Filter state must be untouched -- feeding the same good data again
    # should continue converging as if the NaN block never happened.
    result2 = estimator.process_block(good_force, phase)
    assert np.isfinite(result2.amplitude)


def test_inf_input_marked_invalid():
    fs = 5000.0
    estimator = ForceTrackingEstimator(sample_rate=fs, tracking_bandwidth_hz=5.0)
    force = np.full(50, np.inf)
    phase = np.zeros(50)
    result = estimator.process_block(force, phase)
    assert result.valid is False


def test_from_tracking_cycles_constructor():
    estimator = ForceTrackingEstimator.from_tracking_cycles(
        sample_rate=10000.0, drive_frequency_hz=100.0, tracking_cycles=5.0)
    expected_bandwidth = 100.0 / (2 * np.pi * 5.0)
    assert estimator.tracking_bandwidth_hz == pytest.approx(expected_bandwidth)


def test_mismatched_shapes_raise():
    estimator = ForceTrackingEstimator(sample_rate=1000.0, tracking_bandwidth_hz=5.0)
    with pytest.raises(ValueError):
        estimator.process_block(np.ones(10), np.zeros(5))


@pytest.mark.parametrize('bad_kwargs', [
    dict(sample_rate=-1.0),
    dict(tracking_bandwidth_hz=0.0),
    dict(valid_settle_time_constants=-1.0),
])
def test_invalid_configuration_raises(bad_kwargs):
    kwargs = dict(sample_rate=1000.0, tracking_bandwidth_hz=5.0)
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        ForceTrackingEstimator(**kwargs)
