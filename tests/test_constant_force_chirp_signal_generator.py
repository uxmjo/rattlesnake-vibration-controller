# -*- coding: utf-8 -*-
"""
Unit tests for ConstantForceChirpSignalGenerator in
components/signal_generation.py: phase continuity across frame boundaries
(no clicks/discontinuities at the seams that assembling many short frames
into one long sweep could introduce), correct instantaneous-frequency
tracking, and that amplitude updates only affect frames generated after the
update.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.signal_generation import ConstantForceChirpSignalGenerator


SAMPLE_RATE = 20000.0


def make_generator(level=1.0, num_samples_per_frame=1000, num_signals=1,
                    start_frequency=100.0, end_frequency=2000.0, chirp_duration=5.0):
    return ConstantForceChirpSignalGenerator(
        level=level, sample_rate=SAMPLE_RATE, num_samples_per_frame=num_samples_per_frame,
        num_signals=num_signals, start_frequency=start_frequency, end_frequency=end_frequency,
        chirp_duration=chirp_duration, output_oversample=1)


def test_phase_is_continuous_across_frame_boundary():
    gen = make_generator()
    frame1, last1 = gen.generate_frame()
    frame2, last2 = gen.generate_frame()
    assert not last1

    # Reconstruct the expected signal for both frames from a single
    # continuous phase law and compare -- any frame-boundary phase reset
    # would show up as a large mismatch.
    n = frame1.shape[-1]
    dt = 1.0 / SAMPLE_RATE
    sweep_rate = (2000.0 - 100.0) / 5.0
    t_full = np.arange(2 * n) * dt
    phase_full = 2 * np.pi * (100.0 * t_full + 0.5 * sweep_rate * t_full ** 2)
    expected = np.sin(phase_full)
    actual = np.concatenate([frame1[0], frame2[0]])
    assert np.allclose(actual, expected, atol=1e-9)


def test_instantaneous_frequency_matches_local_derivative():
    gen = make_generator(start_frequency=100.0, end_frequency=2000.0, chirp_duration=5.0)
    # Advance a few frames
    for _ in range(10):
        gen.generate_frame()
    t = gen.current_time
    expected_freq = 100.0 + (2000.0 - 100.0) / 5.0 * t
    assert gen.instantaneous_frequency(t) == pytest.approx(expected_freq)

    # Cross-check against the actual local period of the generated signal
    frame, _ = gen.generate_frame()
    zero_crossings = np.flatnonzero(np.diff(np.sign(frame[0])) > 0)
    if len(zero_crossings) >= 2:
        periods = np.diff(zero_crossings) / SAMPLE_RATE
        measured_freq = 1.0 / np.mean(periods)
        assert measured_freq == pytest.approx(expected_freq, rel=0.05)


def test_update_parameters_only_affects_future_frames():
    gen = make_generator(level=1.0)
    frame1, _ = gen.generate_frame()
    assert np.max(np.abs(frame1)) == pytest.approx(1.0, rel=0.05)

    gen.update_parameters(level=5.0)
    frame2, _ = gen.generate_frame()
    assert np.max(np.abs(frame2)) == pytest.approx(5.0, rel=0.05)
    # The already-returned first frame must not retroactively change
    assert np.max(np.abs(frame1)) == pytest.approx(1.0, rel=0.05)


def test_last_frame_flag_set_once_duration_elapsed():
    gen = make_generator(chirp_duration=0.1, num_samples_per_frame=1000)  # 0.1s / (1000/20000s per frame) = 2 frames
    _, last1 = gen.generate_frame()
    _, last2 = gen.generate_frame()
    assert not last1
    assert last2


def test_multi_signal_independent_levels():
    gen = make_generator(level=[1.0, 2.0], num_signals=2)
    frame, _ = gen.generate_frame()
    assert frame.shape[0] == 2
    assert np.max(np.abs(frame[0])) == pytest.approx(1.0, rel=0.05)
    assert np.max(np.abs(frame[1])) == pytest.approx(2.0, rel=0.05)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
