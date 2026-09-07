# -*- coding: utf-8 -*-
"""
Integration-style tests for components/constant_force_chirp_environment.py:
exercises ConstantForceChirpControlProcess's command handlers directly
(initialize_parameters / run_control / stop_control) against real
multiprocessing queues, without spawning actual subprocesses or requiring
hardware/Qt display. This checks the plumbing between the data collector's
FFT-frame output convention, the control algorithm in
control_laws/constant_force_chirp_control.py, and the signal-generation
update queue -- the part that cannot be exercised by the pure-algorithm
tests in test_constant_force_chirp_control.py.
"""
import sys
import os
import multiprocessing as mp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.utilities import VerboseMessageQueue
from components.constant_force_chirp_environment import (
    ConstantForceChirpControlProcess, ConstantForceChirpControlCommands, ConstantForceChirpCommands,
)
from control_laws.constant_force_chirp_control import (
    ForceChirpControlConfig, compute_feedforward_profile,
)
from tests.synthetic_frf import synthetic_force_voltage_frf

SAMPLE_RATE = 20000.0


def make_process():
    log_q = mp.Queue()
    cmd_q = VerboseMessageQueue(log_q, 'test_force_control_cmd')
    data_in_q = mp.Queue()
    data_out_q = mp.Queue()
    env_cmd_q = VerboseMessageQueue(log_q, 'test_env_cmd')
    gui_q = mp.Queue()
    proc = ConstantForceChirpControlProcess(
        'test_force_control', cmd_q, data_in_q, data_out_q, env_cmd_q, gui_q, log_q, 'Constant Force Chirp')
    return proc, data_in_q, data_out_q, env_cmd_q, gui_q


def make_config():
    return ForceChirpControlConfig(
        target_force=2.0, start_frequency=100.0, end_frequency=2000.0, chirp_duration=30.0,
        ramp_time=1.0, max_voltage=10.0, force_abort=5.0, acceleration_abort=900.0,
        control_update_rate=20.0, amplitude_smoothing=0.25, regularization=0.005)


def make_fft_frame(frequency, amplitude, n=1000, sample_rate=SAMPLE_RATE):
    t = np.arange(n) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return np.fft.rfft(signal)  # rectangular window, matching Window.RECTANGLE requirement


def test_initialize_and_run_control_pushes_voltage_update():
    proc, data_in_q, data_out_q, env_cmd_q, gui_q = make_process()
    config = make_config()
    freqs = np.linspace(50, 2100, 2048)
    H = synthetic_force_voltage_frf(freqs)
    feedforward = compute_feedforward_profile(freqs, H, config)

    proc.initialize_parameters((config, feedforward, SAMPLE_RATE, 2))
    assert proc.running
    assert proc.acceleration_rows == [1, 2]

    # A response frame: row 0 = force (near target already), rows 1,2 = accel (benign)
    frequency_now = proc.controller.chirp_profile.frequency(np.array([0.0]))[0]
    force_fft = make_fft_frame(frequency_now, amplitude=2.0)
    accel1_fft = make_fft_frame(frequency_now, amplitude=1.0)
    accel2_fft = make_fft_frame(frequency_now, amplitude=0.5)
    response_fft = np.stack([force_fft, accel1_fft, accel2_fft])
    reference_fft = make_fft_frame(frequency_now, amplitude=1.0)[np.newaxis, :]
    data_in_q.put((response_fft, reference_fft))

    proc.run_control(None)

    voltage_update = data_out_q.get(timeout=2)
    assert isinstance(voltage_update, tuple)
    assert len(voltage_update) == 1
    assert 0.0 <= voltage_update[0] <= config.max_voltage

    gui_message = gui_q.get(timeout=2)
    env_name, (msg_type, payload) = gui_message
    assert msg_type == 'force_control_update'
    assert not proc.running is False  # still running (no abort with these benign values)


def test_force_abort_stops_process_and_zeros_output():
    proc, data_in_q, data_out_q, env_cmd_q, gui_q = make_process()
    config = make_config()  # force_abort=5.0
    freqs = np.linspace(50, 2100, 2048)
    H = synthetic_force_voltage_frf(freqs)
    feedforward = compute_feedforward_profile(freqs, H, config)
    proc.initialize_parameters((config, feedforward, SAMPLE_RATE, 0))

    frequency_now = proc.controller.chirp_profile.frequency(np.array([0.0]))[0]
    force_fft = make_fft_frame(frequency_now, amplitude=20.0)  # far above force_abort
    response_fft = np.stack([force_fft])
    reference_fft = make_fft_frame(frequency_now, amplitude=1.0)[np.newaxis, :]
    data_in_q.put((response_fft, reference_fft))

    proc.run_control(None)

    assert proc.running is False
    voltage_update = data_out_q.get(timeout=2)
    assert voltage_update == (0.0,)
    # Two gui messages expected: the per-step update, then the abort notice
    messages = []
    for _ in range(2):
        messages.append(gui_q.get(timeout=2))
    msg_types = [m[1][0] for m in messages]
    assert 'force_control_aborted' in msg_types

    command, reason = env_cmd_q.queue.get(timeout=2)
    assert command == ConstantForceChirpCommands.STOP_CONTROL
    assert reason is not None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
