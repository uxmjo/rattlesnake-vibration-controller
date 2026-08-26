# -*- coding: utf-8 -*-
"""
Slow outer-loop drive amplitude controller for closed-loop force control.

Adjusts the drive (sine sweep) amplitude so the measured force amplitude
tracks a target, using a multiplicative update in log-amplitude space:

    A_new = A_old * (F_target / max(F_measured, force_floor)) ** alpha

with ``0 < alpha <= 1``. This is deliberately not a sample-by-sample PID --
it is meant to be called at a slow, explicit update rate (see
``control_update_period_s`` in the environment layer), decoupled from the
DAQ block size.

This module never talks to hardware, queues, or Qt -- it is a small, pure
function of its inputs plus its own persistent ``drive_amplitude`` state, so
it can be unit tested in isolation and reused unchanged by the eventual
Rattlesnake environment.

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
from enum import Enum
from typing import Optional
import math


class ControllerStatus(Enum):
    """Status of the most recent :meth:`ForceAmplitudeController.update` call."""
    #: Update applied normally, no limit was active.
    OK = 'OK'
    #: Estimator was not valid (not settled, or bad/NaN data) -- amplitude held.
    HOLD_INVALID = 'HOLD_INVALID'
    #: Measured force below ``force_floor`` -- amplitude held (never blindly
    #: ramped up from a near-zero force reading).
    HOLD_LOW_FORCE = 'HOLD_LOW_FORCE'
    #: The requested amplitude exceeded ``max_drive_amplitude`` and was
    #: clipped -- target force may not be achievable at this frequency.
    SATURATED = 'SATURATED'


@dataclass
class ControllerResult:
    """Result of one :meth:`ForceAmplitudeController.update` call.

    Attributes
    ----------
    drive_amplitude : float
        The new drive amplitude to apply (also stored as
        ``controller.drive_amplitude``).
    status : ControllerStatus
        See :class:`ControllerStatus`.
    relative_force_error : float
        ``(force_measured - force_target) / force_target``, or ``nan`` when
        the update was held (status is ``HOLD_INVALID``/``HOLD_LOW_FORCE``).
    """
    drive_amplitude: float
    status: ControllerStatus
    relative_force_error: float


class ForceAmplitudeController:
    """Multiplicative log-amplitude force controller with slew/saturation limits.

    Parameters
    ----------
    alpha : float
        Controller gain exponent, ``0 < alpha <= 1``. ``alpha=1`` applies
        the full ratio-based correction each update; smaller values make the
        controller more conservative/slower per update.
    force_floor : float
        Minimum force amplitude (N_peak) treated as usable. Below this, the
        controller holds the current amplitude rather than dividing by a
        near-zero measurement (which would otherwise request an unbounded
        drive amplitude).
    max_drive_amplitude : float
        Control-limit: the controller will never request more than this
        (V_peak). This is the "soft" limit the controller itself respects;
        it is independent of any hardware/abort-level safety limit enforced
        elsewhere.
    max_amplitude_step : float
        Maximum absolute change in drive amplitude (V_peak) allowed per
        :meth:`update` call.
    max_relative_step : float, optional
        If given, an additional cap on the per-update change, expressed as a
        fraction of ``max_drive_amplitude`` (i.e. of full scale, not of the
        current amplitude -- this keeps the limit well-defined even while
        ramping up from zero). The more restrictive of
        ``max_amplitude_step`` and this fraction of ``max_drive_amplitude``
        applies. Default ``None`` (only ``max_amplitude_step`` applies).
    initial_drive_amplitude : float
        Starting drive amplitude (V_peak). Default ``0.0``.

        Note: because the update is purely multiplicative
        (``A_new = A_old * ratio**alpha``), ``A_old == 0`` is an absorbing
        state -- the controller can never move away from an exact zero
        starting amplitude on its own. Callers must seed a small nonzero
        amplitude (e.g. via the environment-layer startup ramp:
        "0 V -> small start amplitude -> hand off to this controller")
        before relying on this controller to converge.
    """

    def __init__(self,
                 alpha: float,
                 force_floor: float,
                 max_drive_amplitude: float,
                 max_amplitude_step: float,
                 max_relative_step: Optional[float] = None,
                 initial_drive_amplitude: float = 0.0):
        if not (0.0 < alpha <= 1.0):
            raise ValueError('alpha must satisfy 0 < alpha <= 1')
        if force_floor <= 0:
            raise ValueError('force_floor must be positive')
        if max_drive_amplitude <= 0:
            raise ValueError('max_drive_amplitude must be positive')
        if max_amplitude_step <= 0:
            raise ValueError('max_amplitude_step must be positive')
        if max_relative_step is not None and not (0.0 < max_relative_step <= 1.0):
            raise ValueError('max_relative_step must satisfy 0 < max_relative_step <= 1')
        if not (0.0 <= initial_drive_amplitude <= max_drive_amplitude):
            raise ValueError('initial_drive_amplitude must be in [0, max_drive_amplitude]')

        self.alpha = float(alpha)
        self.force_floor = float(force_floor)
        self.max_drive_amplitude = float(max_drive_amplitude)
        self.max_amplitude_step = float(max_amplitude_step)
        self.max_relative_step = (None if max_relative_step is None
                                   else float(max_relative_step))
        self.drive_amplitude = float(initial_drive_amplitude)

    def _hold(self, status: ControllerStatus) -> ControllerResult:
        return ControllerResult(drive_amplitude=self.drive_amplitude,
                                 status=status,
                                 relative_force_error=math.nan)

    def _max_step(self) -> float:
        step = self.max_amplitude_step
        if self.max_relative_step is not None:
            step = min(step, self.max_relative_step * self.max_drive_amplitude)
        return step

    def update(self,
               force_target: float,
               force_measured: Optional[float],
               estimator_valid: bool) -> ControllerResult:
        """Computes and applies one controller update.

        Parameters
        ----------
        force_target : float
            Desired force amplitude (N_peak) at the current drive frequency
            (e.g. from
            :meth:`~components.force_target_specification.ForceTargetSpecification.evaluate`).
            Must be positive.
        force_measured : float or None
            The latest force amplitude estimate (N_peak). May be ``None``,
            ``nan``, or ``inf`` to indicate no usable measurement -- treated
            the same as ``estimator_valid=False``.
        estimator_valid : bool
            Whether the tracking estimator that produced ``force_measured``
            has settled and saw clean data (see
            :attr:`~components.force_tracking_estimator.ForceTrackingResult.valid`).
            If ``False``, the amplitude is held unchanged -- the controller
            never ramps up while the estimate cannot be trusted.

        Returns
        -------
        ControllerResult
        """
        if force_target <= 0:
            raise ValueError('force_target must be positive')

        if (not estimator_valid or force_measured is None
                or not math.isfinite(force_measured)):
            return self._hold(ControllerStatus.HOLD_INVALID)

        if force_measured < self.force_floor:
            return self._hold(ControllerStatus.HOLD_LOW_FORCE)

        ratio = force_target / max(force_measured, self.force_floor)
        requested = self.drive_amplitude * ratio ** self.alpha

        # Slew limit
        max_step = self._max_step()
        delta = requested - self.drive_amplitude
        delta = max(-max_step, min(max_step, delta))
        requested = self.drive_amplitude + delta

        # Saturation (drive amplitude must always be in [0, max_drive_amplitude])
        saturated = False
        if requested > self.max_drive_amplitude:
            requested = self.max_drive_amplitude
            saturated = True
        if requested < 0.0:
            requested = 0.0

        self.drive_amplitude = requested
        relative_force_error = (force_measured - force_target) / force_target
        status = ControllerStatus.SATURATED if saturated else ControllerStatus.OK
        return ControllerResult(drive_amplitude=self.drive_amplitude,
                                 status=status,
                                 relative_force_error=relative_force_error)
