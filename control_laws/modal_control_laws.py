import numpy as np


def constant_force_amplitude(target_amplitude, measured_amplitude, current_level,
                              current_frequency, extra_parameters, frame_number):
    """Adjusts the sine signal generator level so the measured force amplitude
    tracks a constant target amplitude.

    Computes a proportional (ratio-based) update:
    ``new_level = current_level * (target_amplitude / measured_amplitude)``,
    clamped to a maximum fractional change per frame so a collapsed or
    momentarily lost force signal cannot drive a runaway gain spike.

    Parameters
    ----------
    target_amplitude : float
        The desired constant force amplitude (RMS, engineering units).
    measured_amplitude : float
        The force amplitude (RMS) measured at the drive frequency during the
        most recently completed frame.
    current_level : float
        The current output level of the sine signal generator.
    current_frequency : float
        The current drive frequency (Hz). Unused by this control law but
        provided for control laws that also adjust frequency.
    extra_parameters : str
        Optional text containing the maximum fractional level change allowed
        per frame, e.g. ``'0.25'`` for 25%. Defaults to 0.5 (50%) if blank or
        not parseable as a float.
    frame_number : int
        The current frame number. Unused by this control law but provided
        for control laws that need startup/ramp logic.

    Returns
    -------
    float
        The new output level for the sine signal generator.
    """
    try:
        max_step_fraction = float(extra_parameters)
    except (TypeError, ValueError):
        max_step_fraction = 0.5
    amplitude_floor = 1e-10
    if measured_amplitude < amplitude_floor:
        # No usable signal -- ramp up cautiously rather than dividing by ~0
        ratio = 1.0 + max_step_fraction
    else:
        ratio = target_amplitude / measured_amplitude
        ratio = np.clip(ratio, 1.0 - max_step_fraction, 1.0 + max_step_fraction)
        res = np.clip(current_level * ratio, 0.0, 1.25/np.sqrt(2))
    return max(res, 0.0)
