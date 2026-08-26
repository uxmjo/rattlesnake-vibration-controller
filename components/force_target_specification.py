# -*- coding: utf-8 -*-
"""
Target force profile abstraction for closed-loop force control.

Defines the interface a force controller uses to look up the desired force
amplitude at the current drive frequency: ``F_target = specification.evaluate(f)``.
The first implementation supports only a constant target force, but the
interface is deliberately frequency-aware so a later, frequency-dependent
profile (e.g. a breakpoint table) can be dropped in without changing the
controller.

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

from abc import ABC, abstractmethod

VALID_FORCE_UNITS = ('peak', 'rms')


class ForceTargetSpecification(ABC):
    """Abstract base class for a (possibly frequency-dependent) target force
    profile.

    Parameters
    ----------
    force_unit : str
        Whether the values returned by :meth:`evaluate` are ``'peak'`` or
        ``'rms'`` amplitudes. Must be declared explicitly -- there is no
        implicit peak/RMS conversion anywhere in this module or its callers;
        the caller (the amplitude controller and the tracking estimator,
        which both operate in N_peak) must be configured consistently with
        this declaration.
    """

    def __init__(self, force_unit: str):
        if force_unit not in VALID_FORCE_UNITS:
            raise ValueError('force_unit must be one of {:}, got {!r}'.format(
                VALID_FORCE_UNITS, force_unit))
        self.force_unit = force_unit

    @abstractmethod
    def evaluate(self, frequency_hz: float) -> float:
        """Returns the target force amplitude at the given frequency.

        Parameters
        ----------
        frequency_hz : float
            The current drive frequency in Hz.

        Returns
        -------
        float
            The target force amplitude, in the unit declared by
            ``force_unit`` (N_peak or N_rms).
        """
        raise NotImplementedError


class ConstantForceTarget(ForceTargetSpecification):
    """A constant target force, independent of frequency.

    Parameters
    ----------
    target_force : float
        The constant target force amplitude.
    force_unit : str
        ``'peak'`` or ``'rms'``. Default ``'peak'``.
    """

    def __init__(self, target_force: float, force_unit: str = 'peak'):
        super().__init__(force_unit)
        if target_force <= 0:
            raise ValueError('target_force must be positive')
        self.target_force = float(target_force)

    def evaluate(self, frequency_hz: float) -> float:
        return self.target_force
