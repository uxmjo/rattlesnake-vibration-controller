# -*- coding: utf-8 -*-
"""
Phase-continuous sine sweep generator.

Generates ``u(t) = A(t) * sin(phi(t))`` for a linear or logarithmic frequency
sweep, in successive blocks of arbitrary length, with the guarantee that the
phase never resets or jumps at block boundaries. The instantaneous drive
amplitude ``A(t)`` is supplied per block by the caller (e.g. a
:class:`~components.force_amplitude_controller.ForceAmplitudeController`),
so this class is only responsible for the frequency/phase trajectory and the
resulting waveform -- it does not itself decide the amplitude.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Tuple
import numpy as np

VALID_SWEEP_TYPES = ('linear', 'logarithmic')
VALID_DIRECTIONS = ('up', 'down')


class SineSweepGenerator:
    """Phase-continuous linear or logarithmic sine sweep generator.

    The instantaneous frequency ``f(t)`` is a pure, closed-form function of
    the absolute time elapsed since the generator was constructed (or since
    :meth:`reset` was called) -- it does not depend on any recursive state.
    Only the phase is accumulated recursively (via a cumulative sum of the
    per-sample frequency), carried forward from the last sample of the
    previous block, which is what guarantees phase continuity across block
    boundaries regardless of how the caller chooses to size its blocks
    (independent of ``samples_per_frame``/DAQ block size).

    Sweep laws (with ``f0``/``f1`` the effective start/stop frequency after
    applying ``direction``, and ``T`` the duration of one single sweep from
    ``f0`` to ``f1``):

    * ``linear``: ``f(t) = f0 + sign*rate*t`` for ``0 <= t <= T``, where
      ``rate`` is in Hz/s and ``T = abs(f1-f0)/rate``.
    * ``logarithmic``: ``f(t) = f0 * 2**(sign*k*t)`` for ``0 <= t <= T``,
      where ``k = octaves_per_minute/60`` (octaves/s) and
      ``T = abs(log2(f1/f0))/k``.

    After ``t > T``: if ``repeat`` is ``False`` the frequency holds at
    ``f1`` indefinitely; if ``repeat`` is ``True`` the law is evaluated at
    ``t modulo T`` so the sweep restarts from ``f0``. Note that with the
    default ``alternate_direction=False`` the *frequency* trajectory has a
    discontinuity at each wrap (an instantaneous jump from ``f1`` back to
    ``f0``, inherent to a repeating same-direction sweep), but the *phase*
    itself never resets or jumps -- it keeps accumulating continuously
    through the wrap, which is what actually matters for driving a DAC
    without glitches.

    ``num_sweeps`` (only relevant when ``repeat`` is ``True``) limits how
    many single sweep legs are performed before the frequency holds at the
    final leg's end frequency indefinitely, instead of repeating forever.
    ``0`` (the default) means unlimited, matching the original ``repeat``
    behavior exactly.

    ``alternate_direction`` (only relevant when ``repeat`` is ``True``)
    makes each successive leg reverse direction instead of restarting from
    ``f0`` -- i.e. up, down, up, down, ... -- producing a continuous
    triangle-wave frequency trajectory with no jump at the turnarounds
    (each leg ends exactly where the next one starts). This is useful e.g.
    to let a closed-loop amplitude controller settle in by running one or
    more full sweep legs before the leg(s) of interest, instead of (or in
    addition to) dwelling at a fixed frequency -- a fixed low frequency can
    have very different, harder-to-converge dynamics than the swept band.

    An optional ``pre_dwell_time`` holds the frequency at ``f0`` for that
    many seconds before the sweep law above is evaluated at all -- i.e. the
    whole law is evaluated at ``t - pre_dwell_time`` (clamped to ``>= 0``)
    instead of ``t``. This is meant to give a closed-loop amplitude
    controller time to converge at a fixed frequency before the frequency
    itself starts moving, so the recorded sweep starts from an already
    -settled amplitude. Because it is still expressed as a single function
    of the same continuously-accumulating ``t``, phase continuity across the
    dwell-to-sweep transition is automatic -- there is no special-cased
    transition logic.

    Parameters
    ----------
    sample_rate : float
        Output sample rate in Hz.
    sweep_type : str
        ``'linear'`` or ``'logarithmic'``.
    f_start : float
        Sweep start frequency in Hz (before ``direction`` is applied).
    f_stop : float
        Sweep stop frequency in Hz (before ``direction`` is applied).
    sweep_rate : float
        For ``'linear'``: sweep rate in Hz/s (must be > 0). For
        ``'logarithmic'``: sweep rate in octaves/minute (must be > 0).
    direction : str
        ``'up'`` sweeps from ``f_start`` to ``f_stop``; ``'down'`` sweeps
        from ``f_stop`` to ``f_start``. Default ``'up'``.
    repeat : bool
        If ``True``, the sweep restarts from the beginning once it reaches
        the end instead of holding. Default ``False``.
    num_sweeps : int
        Only used when ``repeat`` is ``True``. Number of single sweep legs
        to perform before holding at the final frequency indefinitely.
        ``0`` means unlimited (repeat forever). Default ``0``.
    alternate_direction : bool
        Only used when ``repeat`` is ``True``. If ``True``, each successive
        leg reverses direction (up, down, up, down, ...) instead of
        restarting from ``f0`` every time. Default ``False`` (matches
        prior behavior exactly).
    pre_dwell_time : float
        Time in seconds to hold the frequency at ``f0`` before the sweep
        (or repeating sweep) begins. Default ``0.0`` (no dwell, matches
        prior behavior exactly).
    initial_phase : float
        Starting phase in radians. Default ``0.0``.
    """

    def __init__(self,
                 sample_rate: float,
                 sweep_type: str,
                 f_start: float,
                 f_stop: float,
                 sweep_rate: float,
                 direction: str = 'up',
                 repeat: bool = False,
                 num_sweeps: int = 0,
                 alternate_direction: bool = False,
                 pre_dwell_time: float = 0.0,
                 initial_phase: float = 0.0):
        if sweep_type not in VALID_SWEEP_TYPES:
            raise ValueError('sweep_type must be one of {:}, got {!r}'.format(
                VALID_SWEEP_TYPES, sweep_type))
        if direction not in VALID_DIRECTIONS:
            raise ValueError('direction must be one of {:}, got {!r}'.format(
                VALID_DIRECTIONS, direction))
        if sample_rate <= 0:
            raise ValueError('sample_rate must be positive')
        if f_start <= 0 or f_stop <= 0:
            raise ValueError('f_start and f_stop must be positive')
        if sweep_rate <= 0:
            raise ValueError('sweep_rate must be positive')
        if f_start == f_stop:
            raise ValueError('f_start and f_stop must differ')
        if num_sweeps < 0:
            raise ValueError('num_sweeps must be non-negative (0 means unlimited)')
        if pre_dwell_time < 0:
            raise ValueError('pre_dwell_time must be non-negative')

        self.sample_rate = float(sample_rate)
        self.sweep_type = sweep_type
        self.f_start = float(f_start)
        self.f_stop = float(f_stop)
        self.sweep_rate = float(sweep_rate)
        self.direction = direction
        self.repeat = repeat
        self.num_sweeps = int(num_sweeps)
        self.alternate_direction = alternate_direction
        self.pre_dwell_time = float(pre_dwell_time)

        self._f0 = self.f_start if direction == 'up' else self.f_stop
        self._f1 = self.f_stop if direction == 'up' else self.f_start
        self._sign = 1.0 if self._f1 >= self._f0 else -1.0

        if not self.repeat:
            self._num_legs = 1
        elif self.num_sweeps <= 0:
            self._num_legs = None  # unlimited
        else:
            self._num_legs = self.num_sweeps

        if sweep_type == 'linear':
            self._duration = abs(self._f1 - self._f0) / self.sweep_rate
        else:
            octaves_per_second = self.sweep_rate / 60.0
            self._duration = abs(np.log2(self._f1 / self._f0)) / octaves_per_second
            self._k = octaves_per_second

        self.reset(initial_phase=initial_phase)

    def reset(self, initial_phase: float = 0.0) -> None:
        """Resets the generator to its initial state (elapsed time and phase).

        Parameters
        ----------
        initial_phase : float
            Starting phase in radians. Default ``0.0``.
        """
        self._elapsed_samples = 0
        self._last_phase = float(initial_phase)

    @property
    def sweep_duration(self) -> float:
        """Duration in seconds of one single sweep from start to stop
        (excluding ``pre_dwell_time``)."""
        return self._duration

    @property
    def total_duration(self) -> float:
        """``pre_dwell_time + sweep_duration``: time from generator start
        until one single sweep has completed."""
        return self.pre_dwell_time + self._duration

    @property
    def total_sweeps_duration(self):
        """``pre_dwell_time + (all legs' duration)``: time from generator
        start until every configured leg has completed and the frequency
        holds indefinitely. ``None`` if the number of legs is unlimited
        (``repeat=True`` with ``num_sweeps=0``)."""
        if self._num_legs is None:
            return None
        return self.pre_dwell_time + self._num_legs * self._duration

    @property
    def elapsed_time(self) -> float:
        """Total time in seconds elapsed since construction/:meth:`reset`."""
        return self._elapsed_samples / self.sample_rate

    def _frequency_law(self, t: np.ndarray) -> np.ndarray:
        """Evaluates the instantaneous frequency at absolute time(s) ``t``.

        Pure function of ``t`` (no recursive state), so it can be evaluated
        for any block of times independent of prior calls.
        """
        t_dwelled = np.maximum(0.0, t - self.pre_dwell_time)
        T = self._duration
        raw_leg = np.floor(t_dwelled / T)
        if self._num_legs is not None:
            leg = np.minimum(raw_leg, self._num_legs - 1)
        else:
            leg = raw_leg
        t_eff = np.clip(t_dwelled - leg * T, 0.0, T)

        if self.alternate_direction:
            odd_leg = np.mod(leg, 2) >= 1
        else:
            odd_leg = np.zeros_like(leg, dtype=bool)
        # Even legs run f0->f1 (this generator's configured direction); odd
        # legs (only reachable with alternate_direction) reverse to f1->f0,
        # meeting the next leg's start exactly so frequency stays continuous
        # through every turnaround (no jump, unlike the same-direction
        # repeat below).
        leg_start = np.where(odd_leg, self._f1, self._f0)
        leg_sign = np.where(odd_leg, -self._sign, self._sign)

        if self.sweep_type == 'linear':
            return leg_start + leg_sign * self.sweep_rate * t_eff
        else:
            return leg_start * 2.0 ** (leg_sign * self._k * t_eff)

    def generate_block(self, num_samples: int, drive_amplitude
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates the next block of ``num_samples`` output samples.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        drive_amplitude : float or np.ndarray
            Drive amplitude ``A`` applied to this block. May be a scalar
            (held constant over the block) or an array of length
            ``num_samples`` (e.g. to apply a controller update mid-block via
            interpolation) -- both broadcast against the generated waveform.

        Returns
        -------
        samples : np.ndarray
            The generated output waveform, shape ``(num_samples,)``.
        frequency : np.ndarray
            The instantaneous frequency (Hz) at each sample, shape
            ``(num_samples,)``.
        phase : np.ndarray
            The instantaneous phase (radians, unwrapped/continuously
            accumulating -- not wrapped to +/-pi) at each sample, shape
            ``(num_samples,)``.
        """
        if num_samples <= 0:
            raise ValueError('num_samples must be positive')
        n = np.arange(num_samples)
        t = (self._elapsed_samples + n) / self.sample_rate
        frequency = self._frequency_law(t)
        phase_increment = 2.0 * np.pi * frequency / self.sample_rate
        phase = self._last_phase + np.cumsum(phase_increment)
        self._last_phase = phase[-1]
        self._elapsed_samples += num_samples
        samples = np.asarray(drive_amplitude) * np.sin(phase)
        return samples, frequency, phase
