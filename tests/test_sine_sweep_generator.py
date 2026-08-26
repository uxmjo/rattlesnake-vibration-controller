# -*- coding: utf-8 -*-
"""Tests for components.sine_sweep_generator.SineSweepGenerator."""
import numpy as np
import pytest

from components.sine_sweep_generator import SineSweepGenerator


def test_phase_continuity_across_blocks():
    """Concatenating several blocks must produce no phase reset/jump."""
    fs = 10000.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=50.0, f_stop=500.0, sweep_rate=100.0)
    block_sizes = [256, 512, 128, 1000, 37]
    all_phase = []
    all_samples = []
    for n in block_sizes:
        samples, freq, phase = gen.generate_block(n, drive_amplitude=1.0)
        all_phase.append(phase)
        all_samples.append(samples)
    phase = np.concatenate(all_phase)
    samples = np.concatenate(all_samples)
    # Phase must be monotonically increasing (sweep only goes up here) with
    # no jump larger than the per-sample increment expected from a
    # continuous sweep -- i.e. the diff must always be close to 2*pi*f/fs
    # for the *local* instantaneous frequency, never a large discontinuity.
    dphase = np.diff(phase)
    max_expected_dphase = 2 * np.pi * 500.0 / fs * 1.01  # small margin
    assert np.all(dphase > 0)
    assert np.all(dphase < max_expected_dphase)
    # The waveform itself must be continuous: no sample-to-sample jump
    # larger than physically possible for a bounded-frequency sine.
    dsamples = np.abs(np.diff(samples))
    assert np.all(dsamples < max_expected_dphase * 1.1)


def test_phase_continuity_matches_single_block():
    """Generating in many small blocks must give (numerically) the same
    phase trajectory as generating in one single big block."""
    fs = 5000.0
    total_samples = 4000

    gen_single = SineSweepGenerator(sample_rate=fs, sweep_type='logarithmic',
                                     f_start=20.0, f_stop=200.0, sweep_rate=60.0)
    _, _, phase_single = gen_single.generate_block(total_samples, drive_amplitude=1.0)

    gen_multi = SineSweepGenerator(sample_rate=fs, sweep_type='logarithmic',
                                    f_start=20.0, f_stop=200.0, sweep_rate=60.0)
    phases = []
    remaining = total_samples
    rng = np.random.default_rng(0)
    while remaining > 0:
        n = int(rng.integers(1, min(300, remaining) + 1))
        _, _, phase_block = gen_multi.generate_block(n, drive_amplitude=1.0)
        phases.append(phase_block)
        remaining -= n
    phase_multi = np.concatenate(phases)

    np.testing.assert_allclose(phase_multi, phase_single, rtol=0, atol=1e-9)


def test_linear_sweep_frequency_trajectory():
    fs = 20000.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=10.0, f_stop=110.0, sweep_rate=50.0)
    # Duration should be (110-10)/50 = 2.0 s
    assert gen.sweep_duration == pytest.approx(2.0)
    samples, freq, phase = gen.generate_block(int(2.0 * fs), drive_amplitude=1.0)
    t = np.arange(len(freq)) / fs
    expected_freq = 10.0 + 50.0 * t
    np.testing.assert_allclose(freq, expected_freq, rtol=1e-10)


def test_logarithmic_sweep_frequency_trajectory():
    fs = 20000.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='logarithmic',
                              f_start=20.0, f_stop=80.0, sweep_rate=30.0)
    # 2 octaves (20->80) at 30 octaves/min -> duration = 2/(30/60) = 4 s
    assert gen.sweep_duration == pytest.approx(4.0)
    samples, freq, phase = gen.generate_block(int(4.0 * fs), drive_amplitude=1.0)
    t = np.arange(len(freq)) / fs
    expected_freq = 20.0 * 2.0 ** (t / 2.0)
    np.testing.assert_allclose(freq, expected_freq, rtol=1e-10)


def test_direction_down():
    fs = 10000.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=50.0, f_stop=500.0, sweep_rate=100.0,
                              direction='down')
    samples, freq, phase = gen.generate_block(10, drive_amplitude=1.0)
    assert freq[0] == pytest.approx(500.0, abs=1.0)
    assert freq[-1] < freq[0]


def test_hold_at_end_when_not_repeating():
    fs = 1000.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=10.0, f_stop=20.0, sweep_rate=100.0,
                              repeat=False)
    # duration = 10/100 = 0.1 s = 100 samples
    samples, freq, phase = gen.generate_block(500, drive_amplitude=1.0)
    assert freq[-1] == pytest.approx(20.0)
    # Once held, frequency must stay exactly constant
    tail = freq[300:]
    np.testing.assert_allclose(tail, 20.0)


def test_repeat_wraps_frequency_but_phase_stays_continuous():
    fs = 2000.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=10.0, f_stop=20.0, sweep_rate=100.0,
                              repeat=True)
    # duration = 0.1 s = 200 samples; run for 3 full sweeps
    samples, freq, phase = gen.generate_block(650, drive_amplitude=1.0)
    # Frequency wraps back down at each period boundary
    assert freq[199] > freq[200]
    # But phase never jumps backwards or has a large discontinuity
    dphase = np.diff(phase)
    assert np.all(dphase > 0)
    max_expected = 2 * np.pi * 20.0 / fs * 1.01
    assert np.all(dphase < max_expected)


def test_reset_restarts_time_and_phase():
    gen = SineSweepGenerator(sample_rate=1000.0, sweep_type='linear',
                              f_start=10.0, f_stop=20.0, sweep_rate=10.0)
    gen.generate_block(50, drive_amplitude=1.0)
    assert gen.elapsed_time > 0
    gen.reset()
    assert gen.elapsed_time == 0.0
    samples, freq, phase = gen.generate_block(1, drive_amplitude=1.0)
    assert freq[0] == pytest.approx(10.0)


@pytest.mark.parametrize('bad_kwargs', [
    dict(sweep_type='banana'),
    dict(direction='sideways'),
    dict(sample_rate=-1.0),
    dict(f_start=-5.0),
    dict(sweep_rate=0.0),
    dict(f_start=100.0, f_stop=100.0),
])
def test_invalid_configuration_raises(bad_kwargs):
    kwargs = dict(sample_rate=1000.0, sweep_type='linear', f_start=10.0,
                  f_stop=20.0, sweep_rate=10.0)
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        SineSweepGenerator(**kwargs)


def test_drive_amplitude_scales_output():
    gen = SineSweepGenerator(sample_rate=1000.0, sweep_type='linear',
                              f_start=10.0, f_stop=20.0, sweep_rate=10.0)
    samples, _, phase = gen.generate_block(100, drive_amplitude=3.0)
    np.testing.assert_allclose(samples, 3.0 * np.sin(phase))
