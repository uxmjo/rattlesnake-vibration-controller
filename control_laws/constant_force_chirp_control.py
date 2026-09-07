# -*- coding: utf-8 -*-
"""
Constant-force chirp control law for a single shaker / single force-control
channel test.

This module is intentionally framework-agnostic: it does not import any
Rattlesnake multiprocessing, Qt, or netCDF machinery, and can therefore be
unit tested and simulated in isolation.  It is used by
``components/constant_force_chirp_environment.py``, which is the piece that
wires this algorithm into Rattlesnake's process/queue architecture (system
identification, data collection, signal generation).

Physical picture
-----------------
::

    Rattlesnake Output Voltage V(t)
            |
      Shaker Amplifier
            |
          Shaker
            |
       Force Sensor  --> F(t)  (controlled quantity)
            |
       Test Structure
            |
      Accelerometers --> a(t)  (monitoring / abort quantity only)

The excitation is a swept sine (chirp)::

    V(t) = A(t) * sin(phi(t))

with a monotonically increasing instantaneous frequency f(t).  The complex
transfer function between drive voltage and measured force,

    H_FV(f) = F(f) / V(f)

is estimated once (system identification phase, reusing Rattlesnake's
existing H1/H2 FRF estimator) and then used two ways:

1. Feedforward: invert H_FV(f) to build an initial voltage amplitude
   envelope A_ff(f) = F_target / |H_FV(f)|, regularized so that
   H_FV(f) =~ 0 (an anti-resonance, or an unidentified/low-coherence line)
   can never imply V -> infinity.
2. Slow closed-loop trim: while the chirp is running, the actual force
   amplitude at the instantaneous sweep frequency is estimated with a
   synchronous (lock-in) demodulator and compared to the target. The
   resulting error updates a *rate-limited, bounded* multiplicative
   correction gain -- never a raw proportional gain applied directly to the
   output, and never allowed to increase the drive without bound.

Frequency is the controlled test parameter; force is the controlled
response.  Acceleration and voltage are hard, non-negotiable limits that can
abort the test, but they are never targets of the control loop.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, List
import numpy as np


#%% Configuration

@dataclass
class ForceChirpControlConfig:
    """User-facing parameters for a constant-force chirp test.

    Parameters map directly onto the quantities requested for the GUI
    (see components/constant_force_chirp_environment.py).
    """

    # --- Test definition ---
    target_force: float                     # N peak, the force amplitude to hold constant
    start_frequency: float                  # Hz
    end_frequency: float                    # Hz
    chirp_duration: float                   # s, duration of the frequency sweep itself
    ramp_time: float = 0.5                  # s, amplitude ramp in/out at start/end of the sweep

    # --- Hard safety limits (independent of the control law's own behavior) ---
    max_voltage: float = 1.0                # V peak, absolute ceiling on commanded drive
    force_abort: float = np.inf             # N peak, immediate abort if measured force exceeds this
    acceleration_abort: float = np.inf      # engineering units peak, immediate abort if any monitored
                                             # accelerometer exceeds this (per-channel; see step())

    # --- Valid frequency range of the identified transfer function ---
    # Frequencies outside [minimum_frequency, maximum_frequency] are considered
    # unidentified: the controller holds the nearest valid gain rather than
    # extrapolating, and disables aggressive closed-loop correction there.
    minimum_frequency: Optional[float] = None
    maximum_frequency: Optional[float] = None

    # --- FRF regularization (feedforward inversion) ---
    coherence_threshold: float = 0.7        # below this coherence, a frequency line is untrusted
    regularization: float = 0.03            # floor on |H_FV| relative to max(|H_FV|); prevents
                                             # V -> inf at antiresonances / dropouts
    max_inverse_gain: Optional[float] = None  # V per N absolute ceiling on the feedforward inverse
                                             # gain; if None, derived from max_voltage/target_force

    # --- Closed-loop trim (during the running chirp) ---
    control_update_rate: float = 20.0       # Hz, how often the trim gain is recomputed (frame rate)
    amplitude_smoothing: float = 0.2        # 0..1, low-pass factor on the trim gain (higher = faster,
                                             # less filtered); this is NOT a proportional gain on the
                                             # raw error, it smooths the *target* gain update
    max_gain_step_per_update: float = 1.1   # max multiplicative change of the trim gain per update
                                             # (e.g. 1.1 => at most +/-10% per control update)
    min_trim_gain: float = 0.2              # absolute floor on trim gain relative to feedforward
    max_trim_gain: float = 3.0              # absolute ceiling on trim gain relative to feedforward
    force_deadband: float = 0.02            # fractional deadband around target_force (avoids
                                             # chasing measurement noise once on target)

    def __post_init__(self):
        if self.start_frequency <= 0 or self.end_frequency <= 0:
            raise ValueError("start_frequency and end_frequency must be positive")
        if self.chirp_duration <= 0:
            raise ValueError("chirp_duration must be positive")
        if self.target_force <= 0:
            raise ValueError("target_force must be positive")
        if self.max_voltage <= 0:
            raise ValueError("max_voltage must be positive")
        if not (0 < self.amplitude_smoothing <= 1):
            raise ValueError("amplitude_smoothing must be in (0, 1]")
        if self.max_gain_step_per_update <= 1.0:
            raise ValueError("max_gain_step_per_update must be > 1.0")
        if self.minimum_frequency is None:
            self.minimum_frequency = min(self.start_frequency, self.end_frequency)
        if self.maximum_frequency is None:
            self.maximum_frequency = max(self.start_frequency, self.end_frequency)
        if self.max_inverse_gain is None:
            # Default ceiling: the gain that would produce max_voltage at target_force.
            self.max_inverse_gain = self.max_voltage / self.target_force


#%% Linear chirp instantaneous-frequency / phase law
#
# This duplicates the (very small) phase law used by ChirpSignalGenerator in
# components/signal_generation.py so that the controller and the signal
# generator agree on "what frequency are we at right now" without the
# controller needing to import the signal-generation module.  See
# ConstantForceChirpSignalGenerator, which is built to be phase-consistent
# with this class.

class LinearChirpProfile:
    """Instantaneous frequency / phase law for a linear (constant df/dt) sweep."""

    def __init__(self, start_frequency: float, end_frequency: float, duration: float):
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.duration = duration
        self.sweep_rate = (end_frequency - start_frequency) / duration  # Hz/s

    def frequency(self, t: np.ndarray) -> np.ndarray:
        """Instantaneous frequency at time t (clipped to the sweep's time range)."""
        t_clipped = np.clip(t, 0.0, self.duration)
        return self.start_frequency + self.sweep_rate * t_clipped

    def phase(self, t: np.ndarray) -> np.ndarray:
        """Total phase (radians) accumulated from t=0 to t, matching frequency()."""
        t_clipped = np.clip(t, 0.0, self.duration)
        return 2 * np.pi * (self.start_frequency * t_clipped + 0.5 * self.sweep_rate * t_clipped ** 2)


#%% Feedforward inversion

@dataclass
class FeedforwardResult:
    frequencies: np.ndarray
    voltage_amplitude: np.ndarray   # V peak, already clipped to max_voltage
    valid: np.ndarray               # bool mask: frequency inside identified range AND coherent
    inverse_gain: np.ndarray        # V/N used at each frequency (post regularization/clamp)


def compute_feedforward_profile(frequencies: np.ndarray,
                                 frf_force_over_voltage: np.ndarray,
                                 config: ForceChirpControlConfig,
                                 coherence: Optional[np.ndarray] = None) -> FeedforwardResult:
    """Invert a measured H_FV(f) = F(f)/V(f) into a regularized feedforward
    voltage-amplitude profile A_ff(f) ~= F_target / |H_FV(f)|.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency line abscissa (Hz) of the identified FRF, ascending.
    frf_force_over_voltage : np.ndarray
        Complex H_FV(f) = F(f)/V(f), one value per frequency line. May
        contain NaN for lines where system identification failed/was not
        computed.
    config : ForceChirpControlConfig
    coherence : np.ndarray, optional
        Ordinary coherence of the force channel w.r.t. the drive at each
        frequency line, in [0, 1]. If omitted, only the frequency-range and
        magnitude/NaN checks are applied.

    Returns
    -------
    FeedforwardResult
        The regularized, range-limited, voltage-clamped feedforward profile,
        plus a validity mask callers should use to gate how aggressively the
        closed-loop trim is allowed to act at that frequency.

    Notes
    -----
    A frequency line is "invalid" (not to be trusted / not to be
    extrapolated from) if any of the following hold:
      * it lies outside [config.minimum_frequency, config.maximum_frequency]
      * frf_force_over_voltage is NaN or exactly zero there
      * coherence is provided and below config.coherence_threshold

    For invalid lines the inverse gain is filled by holding the nearest
    valid neighbor's gain (never extrapolated/amplified), so H_FV(f) ~= 0
    or an unidentified line can never drive V(f) -> infinity.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    H = np.asarray(frf_force_over_voltage, dtype=complex)
    if frequencies.shape != H.shape:
        raise ValueError("frequencies and frf_force_over_voltage must have the same shape")

    abs_H = np.abs(H)
    finite = np.isfinite(abs_H) & (abs_H > 0)

    in_range = (frequencies >= config.minimum_frequency) & (frequencies <= config.maximum_frequency)

    valid = finite & in_range
    if coherence is not None:
        coherence = np.asarray(coherence, dtype=float)
        valid = valid & np.isfinite(coherence) & (coherence >= config.coherence_threshold)

    # Regularized magnitude: floor |H| at a fraction of the largest *valid*
    # magnitude so that antiresonances / dropouts cannot produce a near-zero
    # denominator.
    if np.any(valid):
        h_ref = np.max(abs_H[valid])
    else:
        h_ref = 0.0
    floor = config.regularization * h_ref if h_ref > 0 else 0.0

    safe_abs_H = np.where(finite, abs_H, np.inf)  # inf -> gain 0 for invalid/NaN lines before hold-fill
    denom = np.maximum(safe_abs_H, floor) if floor > 0 else safe_abs_H
    with np.errstate(divide='ignore'):
        raw_gain = np.where(np.isfinite(denom) & (denom > 0), 1.0 / denom, 0.0)

    gain = np.minimum(raw_gain, config.max_inverse_gain)

    # Hold-fill invalid lines from the nearest valid neighbor instead of
    # extrapolating/trusting an unregularized inversion.
    gain = _nearest_valid_fill(gain, valid)

    voltage_amplitude = config.target_force * gain
    voltage_amplitude = np.minimum(voltage_amplitude, config.max_voltage)

    return FeedforwardResult(frequencies=frequencies,
                              voltage_amplitude=voltage_amplitude,
                              valid=valid,
                              inverse_gain=gain)


def _nearest_valid_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill entries where ``valid`` is False with the nearest ``valid`` entry's
    value (by index distance). If no entries are valid, returns zeros."""
    values = np.array(values, dtype=float)
    n = len(values)
    if not np.any(valid):
        return np.zeros_like(values)
    valid_idx = np.flatnonzero(valid)
    out = values.copy()
    invalid_idx = np.flatnonzero(~valid)
    if len(invalid_idx) == 0:
        return out
    # For each invalid index, find nearest valid index (vectorized via searchsorted)
    pos = np.searchsorted(valid_idx, invalid_idx)
    pos_clipped_right = np.clip(pos, 0, len(valid_idx) - 1)
    pos_clipped_left = np.clip(pos - 1, 0, len(valid_idx) - 1)
    right_idx = valid_idx[pos_clipped_right]
    left_idx = valid_idx[pos_clipped_left]
    use_right = np.abs(right_idx - invalid_idx) <= np.abs(left_idx - invalid_idx)
    nearest_idx = np.where(use_right, right_idx, left_idx)
    out[invalid_idx] = values[nearest_idx]
    return out


def interpolate_voltage_amplitude(feedforward: FeedforwardResult, frequency: float) -> float:
    """Linearly interpolate the feedforward voltage-amplitude profile at an
    arbitrary frequency (used to evaluate A_ff(f(t)) for the current sweep
    instant). Clamps to the profile's endpoints outside its range."""
    return float(np.interp(frequency, feedforward.frequencies, feedforward.voltage_amplitude))


def interpolate_validity(feedforward: FeedforwardResult, frequency: float) -> bool:
    """Nearest-neighbor lookup of the validity mask at an arbitrary frequency.

    A frequency outside the span of ``feedforward.frequencies`` altogether is
    always considered untrusted, even if the nearest edge line happened to be
    valid -- ``interpolate_voltage_amplitude`` will hold that edge value for
    such a frequency (no extrapolation), but that held value must never be
    treated as a *trusted* measurement of the actual system at that frequency.
    """
    if frequency < feedforward.frequencies[0] or frequency > feedforward.frequencies[-1]:
        return False
    idx = int(np.argmin(np.abs(feedforward.frequencies - frequency)))
    return bool(feedforward.valid[idx])


#%% Synchronous (lock-in) amplitude estimation
#
# Two equivalent ways to estimate the force amplitude at the current sweep
# frequency are provided. Which one an integration uses depends on what form
# the acquired data arrives in:
#
#   * ``lockin_amplitude`` operates on raw time-domain samples (used by the
#     unit tests and the standalone simulation in tests/, where samples are
#     synthesized directly).
#   * ``fft_bin_amplitude`` operates on an already-computed one-sided FFT of
#     the acquired frame -- which is what Rattlesnake's own
#     ``components/data_collector.py`` produces for every frame it hands to
#     downstream processes (windowed, ``rfft(frame) * window_correction``).
#     Reusing that FFT avoids computing a second, redundant transform of the
#     same data in the new environment's control process.
#
# Both are mathematically the same operation (correlating the signal against
# a reference sinusoid at the known instantaneous frequency); which one is
# numerically more convenient just depends on whether time or frequency
# domain data is already on hand.

def lockin_amplitude(samples: np.ndarray, sample_rate: float, frequency: float,
                      start_phase: float = 0.0) -> float:
    """Estimate the amplitude of a (possibly noisy) sinusoid at a known
    frequency using quadrature (lock-in) demodulation of raw time samples.

    This is used instead of simple peak-picking because it rejects
    out-of-band noise and any accelerometer/harmonic content that might leak
    into the acquired frame, at the cost of assuming the frequency is
    (approximately) known -- which it is, since we command the sweep.

    Parameters
    ----------
    samples : np.ndarray
        1D array of time-domain samples (e.g. one force-sensor frame).
    sample_rate : float
        Sample rate in Hz.
    frequency : float
        Frequency (Hz) to demodulate at (the current instantaneous chirp
        frequency).
    start_phase : float
        Phase (radians) of the reference at the first sample of ``samples``,
        for continuity across frames. Not required for an amplitude-only
        estimate but accepted for API symmetry / future phase tracking.

    Returns
    -------
    amplitude : float
        Estimated peak amplitude of the sinusoidal component at ``frequency``.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[-1]
    if n == 0:
        return 0.0
    t = np.arange(n) / sample_rate
    ref_cos = np.cos(2 * np.pi * frequency * t + start_phase)
    ref_sin = np.sin(2 * np.pi * frequency * t + start_phase)
    in_phase = 2.0 / n * np.sum(samples * ref_cos)
    quadrature = 2.0 / n * np.sum(samples * ref_sin)
    return float(np.hypot(in_phase, quadrature))


def fft_bin_amplitude(fft_values: np.ndarray, sample_rate: float, frequency: float) -> float:
    """Estimate a sinusoid's peak amplitude at ``frequency`` from an
    already-computed one-sided FFT of a real time-domain frame, by taking
    the inverse FFT to exactly reconstruct the time-domain samples and then
    running the same lock-in demodulation as :func:`lockin_amplitude`.

    Naively interpolating ``|fft_values|`` between the two bins nearest
    ``frequency`` was tried first and rejected: because a chirp's
    instantaneous frequency essentially never lands exactly on an FFT line,
    that approach suffers DFT "scalloping loss" of up to ~30% at the
    worst-case midpoint between bins for realistic control-frame lengths.
    Reconstructing the time signal via ``irfft`` and correlating directly at
    the *known* target frequency (we command the sweep, so the frequency is
    never actually unknown) sidesteps that error entirely.

    Important: this exact reconstruction requires that ``fft_values`` was
    computed with **no tapering window** (rectangular window), i.e.
    ``rfft(frame)`` with no additional scaling. Rattlesnake's data collector
    (``components/data_collector.py``) applies ``window_correction =
    sqrt(1/mean(window**2))`` to its output FFTs -- this is the correct
    normalization for CPSD/power-spectrum estimation (its purpose
    elsewhere), but it is *not* a coherent-gain correction and does not make
    a windowed frame's inverse FFT equal the original samples. The
    constant-force-chirp control-phase data collector must therefore be
    configured with ``Window.RECTANGLE`` (see
    ``components/constant_force_chirp_environment.py``) for this function to
    be exact; the system-identification phase is unaffected and can keep
    using whatever window gives the best FRF/coherence estimate there.

    Parameters
    ----------
    fft_values : np.ndarray
        One-sided FFT (``np.fft.rfft``) of a rectangular-windowed real frame.
    sample_rate : float
        Sample rate (Hz) of the original time-domain frame.
    frequency : float
        Frequency (Hz) to evaluate the amplitude at (the current
        instantaneous chirp frequency).

    Returns
    -------
    amplitude : float
        Estimated peak amplitude at ``frequency``.
    """
    fft_values = np.asarray(fft_values)
    if fft_values.shape[-1] == 0:
        return 0.0
    time_samples = np.fft.irfft(fft_values)
    return lockin_amplitude(time_samples, sample_rate, frequency)


#%% Closed-loop controller

@dataclass
class ControlStepResult:
    voltage_amplitude: float        # V peak to command for the upcoming segment
    trim_gain: float                # current multiplicative trim gain (relative to feedforward)
    measured_force: float           # lock-in force amplitude estimate used this step
    aborted: bool                   # True if a hard safety limit was violated
    abort_reason: Optional[str]
    warnings: List[str] = field(default_factory=list)


class ConstantForceChirpController:
    """Frame-based closed-loop amplitude trim on top of an FRF feedforward
    profile.

    This implements "Variant A" from the design discussion: a feedforward
    voltage profile computed once from the identified H_FV(f), corrected by
    a slow, rate-limited, bounded multiplicative gain based on a
    synchronous-demodulation estimate of the actually measured force
    amplitude. It deliberately does *not* apply an unbounded proportional
    gain directly to the raw force signal, and it never removes the
    feedforward voltage ceiling (config.max_voltage).

    Usage
    -----
    Call :meth:`step` once per acquired control frame (nominally at
    ``config.control_update_rate``), passing the current sweep time, the
    force-sensor samples for that frame, and the peak values of any
    monitored acceleration channels. The returned ``voltage_amplitude`` is
    what the signal generator should use for the *next* segment of the
    chirp.
    """

    def __init__(self, config: ForceChirpControlConfig,
                 feedforward: FeedforwardResult,
                 chirp_profile: LinearChirpProfile):
        self.config = config
        self.feedforward = feedforward
        self.chirp_profile = chirp_profile
        self.trim_gain = 1.0
        self.aborted = False
        self.abort_reason = None

    def _ramp_factor(self, t: float) -> float:
        ramp = self.config.ramp_time
        duration = self.config.chirp_duration
        if ramp <= 0:
            return 1.0
        if t < ramp:
            return t / ramp
        if t > duration - ramp:
            return max(0.0, (duration - t) / ramp)
        return 1.0

    def step(self, t: float, measured_force: float,
              acceleration_peaks: Optional[Sequence[float]] = None) -> ControlStepResult:
        """Process one control frame and return the voltage amplitude to use
        for the next output segment.

        Parameters
        ----------
        t : float
            Elapsed time (s) since the start of the frequency sweep,
            corresponding to this control update.
        measured_force : float
            Estimated peak force amplitude at the current instantaneous
            sweep frequency, from either :func:`lockin_amplitude` (raw time
            samples) or :func:`fft_bin_amplitude` (an already-computed FFT
            frame, as produced by Rattlesnake's data collector) -- see the
            module docstring above those functions.
        acceleration_peaks : sequence of float, optional
            Peak absolute value of each monitored accelerometer channel
            during this frame. Any value exceeding
            ``config.acceleration_abort`` triggers an immediate abort.
        """
        warnings: List[str] = []
        frequency_now = float(self.chirp_profile.frequency(np.array([t]))[0])

        # --- Hard abort checks: these always win, regardless of controller state ---
        if measured_force > self.config.force_abort:
            self.aborted = True
            self.abort_reason = (f"Measured force {measured_force:.3g} N exceeded "
                                  f"force_abort {self.config.force_abort:.3g} N at {frequency_now:.1f} Hz")
        if acceleration_peaks is not None:
            for i, a in enumerate(acceleration_peaks):
                if a > self.config.acceleration_abort:
                    self.aborted = True
                    self.abort_reason = (f"Channel {i} acceleration {a:.3g} exceeded "
                                          f"acceleration_abort {self.config.acceleration_abort:.3g} "
                                          f"at {frequency_now:.1f} Hz")

        if self.aborted:
            return ControlStepResult(voltage_amplitude=0.0, trim_gain=self.trim_gain,
                                      measured_force=measured_force, aborted=True,
                                      abort_reason=self.abort_reason, warnings=warnings)

        # --- Feedforward baseline at the current sweep frequency ---
        ff_voltage = interpolate_voltage_amplitude(self.feedforward, frequency_now)
        trusted = interpolate_validity(self.feedforward, frequency_now)

        # --- Slow, bounded, rate-limited trim gain update ---
        target = self.config.target_force
        error_fraction = abs(measured_force - target) / target if target > 0 else 0.0
        if measured_force <= 0 or error_fraction < self.config.force_deadband:
            desired_gain_change = 1.0
        else:
            desired_gain_change = target / max(measured_force, 1e-9)
            # In untrusted (low-coherence / unidentified) regions, do not chase
            # the error aggressively -- halve the requested correction.
            if not trusted:
                desired_gain_change = 1.0 + 0.5 * (desired_gain_change - 1.0)
                warnings.append(f"Frequency {frequency_now:.1f} Hz has an unreliable FRF estimate; "
                                 "closed-loop correction reduced")

        # Rate-limit the *step* itself
        step_limit = self.config.max_gain_step_per_update
        desired_gain_change = float(np.clip(desired_gain_change, 1.0 / step_limit, step_limit))

        # Low-pass filter (exponential smoothing) toward the rate-limited target
        target_gain = self.trim_gain * desired_gain_change
        alpha = self.config.amplitude_smoothing
        new_gain = (1 - alpha) * self.trim_gain + alpha * target_gain

        # Absolute bounds, independent of the smoothing/rate limiting above
        new_gain = float(np.clip(new_gain, self.config.min_trim_gain, self.config.max_trim_gain))
        self.trim_gain = new_gain

        commanded_voltage = ff_voltage * self.trim_gain * self._ramp_factor(t)

        if commanded_voltage >= self.config.max_voltage:
            commanded_voltage = self.config.max_voltage
            warnings.append(f"Voltage clamped to max_voltage at {frequency_now:.1f} Hz; "
                             "target force may not be reachable at this frequency")

        return ControlStepResult(voltage_amplitude=commanded_voltage, trim_gain=self.trim_gain,
                                  measured_force=measured_force, aborted=False,
                                  abort_reason=None, warnings=warnings)
