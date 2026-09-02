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


def test_pre_dwell_time_default_is_backward_compatible():
    """pre_dwell_time=0.0 (the default) must behave exactly as before."""
    fs = 5000.0
    gen_default = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                                      f_start=10.0, f_stop=100.0, sweep_rate=20.0)
    gen_explicit = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                                       f_start=10.0, f_stop=100.0, sweep_rate=20.0,
                                       pre_dwell_time=0.0)
    s1, f1, p1 = gen_default.generate_block(2000, drive_amplitude=1.0)
    s2, f2, p2 = gen_explicit.generate_block(2000, drive_amplitude=1.0)
    np.testing.assert_allclose(f1, f2)
    np.testing.assert_allclose(p1, p2)
    assert gen_default.total_duration == pytest.approx(gen_default.sweep_duration)


def test_pre_dwell_time_holds_f_start():
    fs = 2000.0
    dwell = 2.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=50.0, f_stop=150.0, sweep_rate=10.0,
                              pre_dwell_time=dwell)
    n = int(3.5 * fs)  # spans past the 2s dwell into the sweep
    samples, freq, phase = gen.generate_block(n, drive_amplitude=1.0)
    t = np.arange(n) / fs
    during_dwell = t < dwell
    after_dwell = t > dwell + 0.01
    assert np.all(freq[during_dwell] == pytest.approx(50.0))
    # Once dwell ends, frequency should follow the normal linear law shifted
    # by `dwell` seconds.
    expected_after = 50.0 + 10.0 * (t[after_dwell] - dwell)
    np.testing.assert_allclose(freq[after_dwell], expected_after, atol=1e-9)


def test_pre_dwell_time_phase_continuous_at_transition():
    """No phase jump/reset when crossing from dwell into the sweep."""
    fs = 4000.0
    dwell = 1.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='logarithmic',
                              f_start=20.0, f_stop=200.0, sweep_rate=60.0,
                              pre_dwell_time=dwell)
    n = int(2.0 * fs)
    samples, freq, phase = gen.generate_block(n, drive_amplitude=1.0)
    dphase = np.diff(phase)
    assert np.all(dphase > 0)
    max_expected = 2 * np.pi * 200.0 / fs * 1.01
    assert np.all(dphase < max_expected)
    # No discontinuity specifically around the dwell->sweep boundary sample.
    boundary = int(dwell * fs)
    window = slice(boundary - 5, boundary + 5)
    assert np.all(np.diff(phase[window]) > 0)
    assert np.all(np.diff(phase[window]) < max_expected)


def test_pre_dwell_time_shifts_sweep_duration_and_total_duration():
    gen = SineSweepGenerator(sample_rate=1000.0, sweep_type='linear',
                              f_start=10.0, f_stop=20.0, sweep_rate=10.0,
                              pre_dwell_time=3.0)
    assert gen.sweep_duration == pytest.approx(1.0)  # unaffected by dwell
    assert gen.total_duration == pytest.approx(4.0)  # dwell + sweep_duration


def test_pre_dwell_time_with_repeat():
    """Dwell should only happen once at the very start, not on every repeat cycle."""
    fs = 2000.0
    dwell = 1.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=10.0, f_stop=20.0, sweep_rate=10.0,
                              pre_dwell_time=dwell, repeat=True)
    # sweep_duration = 1.0s; run for dwell + 3 full sweep cycles
    n = int((dwell + 3 * 1.0) * fs)
    samples, freq, phase = gen.generate_block(n, drive_amplitude=1.0)
    t = np.arange(n) / fs
    assert np.all(freq[t < dwell] == pytest.approx(10.0))
    # After the dwell, frequency should wrap (repeat) with period 1.0s,
    # never re-entering a second dwell.
    just_after_first_wrap = int((dwell + 1.0 + 0.01) * fs)
    assert freq[just_after_first_wrap] == pytest.approx(10.0, abs=0.5)


def test_negative_pre_dwell_time_raises():
    with pytest.raises(ValueError):
        SineSweepGenerator(sample_rate=1000.0, sweep_type='linear', f_start=10.0,
                           f_stop=20.0, sweep_rate=10.0, pre_dwell_time=-1.0)


def test_negative_num_sweeps_raises():
    with pytest.raises(ValueError):
        SineSweepGenerator(sample_rate=1000.0, sweep_type='linear', f_start=10.0,
                           f_stop=20.0, sweep_rate=10.0, repeat=True, num_sweeps=-1)


def test_num_sweeps_zero_is_backward_compatible_unlimited_repeat():
    """num_sweeps=0 (the default) must behave exactly like the old
    unlimited repeat=True."""
    fs = 2000.0
    kwargs = dict(sample_rate=fs, sweep_type='linear', f_start=10.0, f_stop=20.0,
                  sweep_rate=100.0, repeat=True)
    gen_default = SineSweepGenerator(**kwargs)
    gen_explicit = SineSweepGenerator(num_sweeps=0, **kwargs)
    s1, f1, p1 = gen_default.generate_block(700, drive_amplitude=1.0)
    s2, f2, p2 = gen_explicit.generate_block(700, drive_amplitude=1.0)
    np.testing.assert_allclose(f1, f2)
    np.testing.assert_allclose(p1, p2)


def test_num_sweeps_holds_after_configured_legs():
    fs = 2000.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=10.0, f_stop=20.0, sweep_rate=100.0,
                              repeat=True, num_sweeps=3)
    # duration = 0.1 s = 200 samples/leg; run well past the 3rd leg.
    samples, freq, phase = gen.generate_block(1000, drive_amplitude=1.0)
    t = np.arange(1000) / fs
    # Same-direction sawtooth still wraps for the first two legs...
    assert freq[199] > freq[200]
    assert freq[399] > freq[400]
    # ...but after the 3rd leg (t > 0.3s) it must hold at f_stop forever,
    # not wrap a 4th time.
    tail = freq[t > 0.3]
    np.testing.assert_allclose(tail, 20.0)


def test_alternate_direction_produces_continuous_triangle_wave():
    fs = 2000.0
    gen = SineSweepGenerator(sample_rate=fs, sweep_type='linear',
                              f_start=10.0, f_stop=20.0, sweep_rate=100.0,
                              repeat=True, num_sweeps=3, alternate_direction=True)
    # duration = 0.1 s = 200 samples/leg: up, down, up, then hold.
    samples, freq, phase = gen.generate_block(1000, drive_amplitude=1.0)
    t = np.arange(1000) / fs
    # Leg 0 (up): 10 -> 20. Leg 1 (down): 20 -> 10. Leg 2 (up): 10 -> 20.
    assert freq[0] == pytest.approx(10.0, abs=0.5)
    assert freq[199] == pytest.approx(20.0, abs=0.5)
    assert freq[200] == pytest.approx(20.0, abs=0.5)
    assert freq[399] == pytest.approx(10.0, abs=0.5)
    assert freq[400] == pytest.approx(10.0, abs=0.5)
    assert freq[599] == pytest.approx(20.0, abs=0.5)
    # After the 3rd leg it holds at 20 (its end frequency) forever.
    tail = freq[t > 0.3]
    np.testing.assert_allclose(tail, 20.0)
    # No jump anywhere in frequency at the turnarounds (unlike the
    # same-direction sawtooth repeat).
    dfreq = np.abs(np.diff(freq))
    assert np.max(dfreq) < 100.0 / fs * 1.5  # bounded by the sweep rate alone
    # And, as always, phase itself is smooth/monotonic throughout.
    dphase = np.diff(phase)
    assert np.all(dphase > 0)
    max_expected = 2 * np.pi * 20.0 / fs * 1.01
    assert np.all(dphase < max_expected)
