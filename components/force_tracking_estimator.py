# -*- coding: utf-8 -*-
"""
Phase-synchronous streaming force amplitude estimator.

Estimates the peak amplitude of a sinusoidal force signal by synchronous
I/Q (lock-in) demodulation against a known, externally supplied instantaneous
phase reference (e.g. the phase of a
:class:`~components.sine_sweep_generator.SineSweepGenerator`), followed by a
tracking low-pass filter. Unlike an FFT-bin based estimate, this does not
require the drive frequency to line up with any particular frequency bin and
works equally well for a fixed dwell frequency or a continuously changing
sweep frequency, as long as the sweep is slow relative to the tracking
filter's response time (see ``tracking_bandwidth_hz``).

Maintains filter state across calls to :meth:`process_block`, so it can be
fed arbitrarily sized, successive blocks of DAQ data (independent of any
FFT frame size) and will produce a continuous amplitude estimate.

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

from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter


@dataclass
class ForceTrackingResult:
    """Result of processing one block of force data.

    Attributes
    ----------
    amplitude : float
        Estimated peak force amplitude (same units as the input force
        samples, e.g. N_peak) at the end of the processed block. ``nan`` if
        ``valid`` is ``False``.
    phase_force : float
        Phase of the force signal relative to the reference phase (radians,
        ``atan2(Q, I)``) at the end of the processed block. This is a
        diagnostic quantity (drive-to-force phase lag) and is not used by
        the amplitude estimate itself. ``nan`` if ``valid`` is ``False``.
    valid : bool
        ``True`` once enough samples have been processed for the tracking
        filter to have settled (see ``valid_settle_time_constants``) *and*
        the input block contained no non-finite (NaN/Inf) samples. When
        ``False``, ``amplitude``/``phase_force`` must not be used to drive a
        controller.
    I : float
        In-phase tracking filter output at the end of the block.
    Q : float
        Quadrature tracking filter output at the end of the block.
    """
    amplitude: float
    phase_force: float
    valid: bool
    I: float
    Q: float


class ForceTrackingEstimator:
    """Streaming phase-synchronous (lock-in) force amplitude tracker.

    The I/Q tracking low-pass is a single-pole IIR filter,
    ``y[n] = alpha*x[n] + (1-alpha)*y[n-1]``, with ``alpha`` derived from the
    configured ``-3 dB`` tracking bandwidth via the standard first-order
    analog-equivalent relation

        tau = 1 / (2*pi*tracking_bandwidth_hz)
        alpha = 1 - exp(-1 / (sample_rate*tau))

    This is the single configurable trade-off knob: a narrow
    ``tracking_bandwidth_hz`` gives a low-noise but slow-to-respond estimate;
    a wide one responds quickly but passes more noise through. There is no
    other hidden smoothing constant.

    Parameters
    ----------
    sample_rate : float
        Sample rate of the force channel in Hz.
    tracking_bandwidth_hz : float
        -3 dB bandwidth of the I/Q tracking low-pass filter, in Hz. Must be
        positive and should be well below the drive frequency (so the
        demodulated double-frequency component is rejected) and well below
        the sweep rate's rate of change of frequency is not directly
        relevant here, but the sweep must be slow enough that the amplitude
        does not change materially within ``1/tracking_bandwidth_hz``
        seconds -- see module-level docs.
    valid_settle_time_constants : float
        Number of filter time constants (``tau``) that must elapse before
        :attr:`ForceTrackingResult.valid` becomes ``True``. Default ``5.0``
        (~99.3% settled for a step input).
    """

    def __init__(self,
                 sample_rate: float,
                 tracking_bandwidth_hz: float,
                 valid_settle_time_constants: float = 5.0):
        if sample_rate <= 0:
            raise ValueError('sample_rate must be positive')
        if tracking_bandwidth_hz <= 0:
            raise ValueError('tracking_bandwidth_hz must be positive')
        if valid_settle_time_constants <= 0:
            raise ValueError('valid_settle_time_constants must be positive')
        self.sample_rate = float(sample_rate)
        self.tracking_bandwidth_hz = float(tracking_bandwidth_hz)
        self._tau = 1.0 / (2.0 * np.pi * self.tracking_bandwidth_hz)
        self._alpha = 1.0 - np.exp(-1.0 / (self.sample_rate * self._tau))
        self._b = np.array([self._alpha])
        self._a = np.array([1.0, -(1.0 - self._alpha)])
        self._settle_samples = int(np.ceil(
            valid_settle_time_constants * self._tau * self.sample_rate))
        self.reset()

    @classmethod
    def from_tracking_cycles(cls,
                              sample_rate: float,
                              drive_frequency_hz: float,
                              tracking_cycles: float,
                              valid_settle_time_constants: float = 5.0
                              ) -> 'ForceTrackingEstimator':
        """Alternate constructor specifying the tracking window in drive
        cycles rather than Hz.

        Converts to an equivalent bandwidth via
        ``tracking_bandwidth_hz = drive_frequency_hz / (2*pi*tracking_cycles)``
        (i.e. a time constant of ``tracking_cycles`` drive periods), then
        delegates to the normal constructor.

        Parameters
        ----------
        sample_rate : float
            Sample rate of the force channel in Hz.
        drive_frequency_hz : float
            The (instantaneous) drive frequency the cycle count is relative
            to. Must be positive.
        tracking_cycles : float
            Desired tracking filter time constant, expressed in number of
            drive cycles. Must be positive.
        valid_settle_time_constants : float
            See :class:`ForceTrackingEstimator`.
        """
        if drive_frequency_hz <= 0:
            raise ValueError('drive_frequency_hz must be positive')
        if tracking_cycles <= 0:
            raise ValueError('tracking_cycles must be positive')
        bandwidth_hz = drive_frequency_hz / (2.0 * np.pi * tracking_cycles)
        return cls(sample_rate, bandwidth_hz, valid_settle_time_constants)

    def reset(self) -> None:
        """Resets the filter state and settle-time counter."""
        self._zi_I = np.zeros(1)
        self._zi_Q = np.zeros(1)
        self._samples_processed = 0

    def process_block(self,
                       force_samples: np.ndarray,
                       phase: np.ndarray) -> ForceTrackingResult:
        """Processes one block of force samples against a known phase.

        Parameters
        ----------
        force_samples : np.ndarray
            Force channel samples for this block (already in calibrated
            engineering units, e.g. N), shape ``(num_samples,)``.
        phase : np.ndarray
            Instantaneous reference phase in radians at each sample of this
            block (unwrapped, continuously accumulating -- e.g. as returned
            by :meth:`SineSweepGenerator.generate_block`), shape
            ``(num_samples,)``. Must be the phase of the same drive that
            produced ``force_samples`` (a constant phase offset between
            drive and force, e.g. from electromechanical or acquisition
            latency, does not corrupt the resulting amplitude -- only
            ``phase_force`` -- as long as the *frequency* is correct).

        Returns
        -------
        ForceTrackingResult
        """
        force_samples = np.asarray(force_samples, dtype=float)
        phase = np.asarray(phase, dtype=float)
        if force_samples.shape != phase.shape:
            raise ValueError('force_samples and phase must have the same shape')
        if force_samples.size == 0:
            raise ValueError('force_samples must not be empty')
        if not np.all(np.isfinite(force_samples)):
            # Do not let bad data corrupt the filter state -- simply report
            # this block as invalid and leave the filter state untouched.
            return ForceTrackingResult(amplitude=np.nan, phase_force=np.nan,
                                        valid=False, I=np.nan, Q=np.nan)
        i_raw = 2.0 * force_samples * np.sin(phase)
        q_raw = 2.0 * force_samples * np.cos(phase)
        i_filt, self._zi_I = lfilter(self._b, self._a, i_raw, zi=self._zi_I)
        q_filt, self._zi_Q = lfilter(self._b, self._a, q_raw, zi=self._zi_Q)
        self._samples_processed += force_samples.size
        i_end = float(i_filt[-1])
        q_end = float(q_filt[-1])
        amplitude = float(np.hypot(i_end, q_end))
        phase_force = float(np.arctan2(q_end, i_end))
        valid = self._samples_processed >= self._settle_samples
        return ForceTrackingResult(amplitude=amplitude, phase_force=phase_force,
                                    valid=valid, I=i_end, Q=q_end)
