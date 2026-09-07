# -*- coding: utf-8 -*-
"""
Synthetic force/voltage transfer function H_FV(f) used to test and
demonstrate the constant-force chirp controller without hardware.

Models a driving-point-like FRF with two structural resonances and one
antiresonance between them, e.g.:

    Resonance at 300 Hz
    Antiresonance at 600 Hz
    Resonance at 900 Hz

as requested for the constant-force-chirp validation in
control_laws/constant_force_chirp_control.py.
"""
import numpy as np


def synthetic_force_voltage_frf(frequencies: np.ndarray,
                                 pole_frequencies=(300.0, 900.0),
                                 pole_damping=(0.02, 0.015),
                                 zero_frequencies=(600.0,),
                                 zero_damping=(0.01,),
                                 gain: float = 4.0) -> np.ndarray:
    """Evaluate a synthetic complex H_FV(f) = F(f)/V(f) [N/V] with the given
    resonance (pole) and antiresonance (zero) frequencies.

    A larger `gain` yields a larger low-frequency N/V ratio (i.e. less
    voltage needed for a given force away from the antiresonance).
    """
    frequencies = np.asarray(frequencies, dtype=float)
    s = 1j * 2 * np.pi * frequencies

    num = np.ones_like(s)
    for fz, zz in zip(zero_frequencies, zero_damping):
        wz = 2 * np.pi * fz
        num = num * (s ** 2 + 2 * zz * wz * s + wz ** 2) / wz ** 2

    den = np.ones_like(s)
    for fp, zp in zip(pole_frequencies, pole_damping):
        wp = 2 * np.pi * fp
        den = den * (s ** 2 + 2 * zp * wp * s + wp ** 2) / wp ** 2

    return gain * num / den


def synthetic_accel_voltage_frf(frequencies: np.ndarray,
                                 resonance_frequency: float = 800.0,
                                 damping: float = 0.01,
                                 gain: float = 3.0) -> np.ndarray:
    """A separate monitoring-channel FRF (acceleration/voltage) with its own
    sharp resonance, used to demonstrate that acceleration is left
    uncontrolled (and can grow large) while force is held constant -- see
    the "Resonanzen" requirement: 10 N -> 0.5 g at 500 Hz but 10 N -> 8 g at
    800 Hz in the write-up's example.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    s = 1j * 2 * np.pi * frequencies
    wp = 2 * np.pi * resonance_frequency
    return gain * wp ** 2 / (s ** 2 + 2 * damping * wp * s + wp ** 2)
