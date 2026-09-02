# -*- coding: utf-8 -*-
"""
Slowly-learned, frequency-dependent feedforward drive-amplitude map for
closed-loop force control.

Sits *above* the existing fast force loop (see ``force_amplitude_controller.py``
and ``sine_force_control_environment.py``) as a second, much slower adaptation
layer. It never touches the inner force loop's math -- it only learns, across
repeated continuous UP/DOWN sweeps, what drive amplitude was actually needed
to hit the target force at each frequency, so that quantity is available
immediately the next time that frequency is swept through, instead of having
to be re-discovered by the fast loop's feedback every single time.

Composition with the existing controller
-----------------------------------------
``ForceAmplitudeController`` (the "inner"/fast loop, see that module's
docstring) computes its state *multiplicatively*, in log-amplitude space::

    A_new = A_old * (F_target / F_measured) ** alpha

This module's map, ``A_FF(f)``, is combined with that controller
multiplicatively rather than additively for exactly this reason -- it is the
form that is actually consistent with the existing controller, and it avoids
the absorbing-zero-state trap documented on
``ForceAmplitudeController.__init__`` (an additive trim starting at 0 V could
never move away from 0 under that controller's math; a multiplicative trim
naturally starts at a gain of 1.0)::

    u_total(f)  =  A_FF(f) * g

where ``g`` (the *trim gain*, dimensionless, centered on 1.0) is produced by
an ordinary, completely unmodified ``ForceAmplitudeController`` instance --
reused as-is for the trim role, just reinterpreted as regulating a
dimensionless ratio instead of a voltage. In log space this is exactly the
additive decomposition the requirements describe::

    log(u_total) = log(A_FF(f)) + log(g)

so ``log(g)`` (or equivalently ``g - 1`` as a fraction, the form logged by the
environment) is the "feedback correction" -- how far the fast loop currently
has to trim away from the learned feedforward value to hit the target force.
When the feedforward map is accurate, ``g -> 1`` and the fast loop's job
shrinks to a small residual correction.

Frequency representation
-------------------------
Frequency is binned logarithmically (``bins_per_decade`` bins per decade)
across a fixed ``[f_min, f_max]`` range fixed at construction time -- cheap,
bounded memory, and a natural match for a sweep that can span several decades
(5-2000 Hz) where the underlying dynamics vary relative to frequency rather
than in absolute Hz. Querying an arbitrary frequency interpolates *linearly
in log(frequency) vs. log(value)* between the two nearest populated bins
(log-log interpolation -- consistent with the log-amplitude domain the rest
of this control path already operates in), falls back to the nearest
populated bin when only one side has data, and falls back to a configured
safe initial estimate when the map has no data at all yet. This guarantees no
hard jumps when moving from a learned region into an unlearned one.

Learning rule and safety
-------------------------
Each call to :meth:`FeedforwardMap.update` is a single, independent, bounded
nudge of one bin toward one trustworthy observation -- never a raw
accumulation of error (no ``feedforward += error``) and never applied to a
momentary sample (the caller is expected to pass an amplitude estimate that
has already been through a settled tracking-filter/lock-in estimate over
several periods, not a raw instantaneous sample -- see
``force_tracking_estimator.py``). Safety layers, all independent of each
other:

* The caller declares the observation trustworthy or not (``trust``); when
  not, nothing is updated, full stop. The environment wires this to "did the
  fast loop's own status say the estimate was valid and not saturated" --
  see ``sine_force_control_environment.py``.
* Hard clamps ``value_min``/``value_max`` on every accepted value.
* Outlier rejection: once a bin has a few real observations, a new
  observation wildly different from the map's current local prediction
  (more than ``outlier_reject_ratio`` off) is rejected outright rather than
  blended in -- a single bad measurement cannot corrupt an established bin.
* A per-update relative-step cap (``max_relative_step_per_update``) bounds
  how much any single update may move a bin, independent of the learning
  rate below -- so even a legitimate but large error is absorbed gradually
  over several updates/sweeps, never in one jump.
* Per-bin adaptive learning rate: a bin with few observations moves quickly
  toward new data (so it does not take many sweeps to form an initial
  estimate); a well-established bin moves only at the slow, configured
  ``learning_rate`` floor -- so noise averages out over many sweeps instead
  of causing the curve to jitter sample-to-sample.

None of this smooths *across* frequency (each bin is learned independently)
-- a real resonance is allowed to look like a sharp spike in the learned
curve; only noise on repeated visits to the *same* bin is averaged out.

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
from typing import Optional, Dict, Tuple
import json
import math
import time

import numpy as np

VALID_DIRECTIONS = ('up', 'down')
SHARED_KEY = 'shared'


@dataclass
class FeedforwardLearnResult:
    """Result of one :meth:`FeedforwardMap.update` call.

    Attributes
    ----------
    updated : bool
        Whether a bin was actually changed.
    reason : str
        ``'ok'`` if updated. Otherwise one of ``'not_trusted'``,
        ``'invalid_value'``, ``'outlier_rejected'`` explaining why not.
    bin_index : int or None
        Index of the bin the observation mapped to (``None`` if the
        frequency could not be binned, e.g. non-finite).
    value_before : float
        The bin's (or fallback) value before this call.
    value_after : float
        The bin's (or fallback) value after this call (equal to
        ``value_before`` when ``updated`` is ``False``).
    """
    updated: bool
    reason: str
    bin_index: Optional[int]
    value_before: float
    value_after: float


class _DirectionTable:
    """Holds the per-bin learned value/observation-count arrays for one
    direction (or the single shared direction)."""

    __slots__ = ('value', 'n_obs')

    def __init__(self, n_bins: int):
        self.value = np.full(n_bins, np.nan)
        self.n_obs = np.zeros(n_bins)

    def has_any_data(self) -> bool:
        return bool(np.any(self.n_obs > 0))

    def to_dict(self) -> dict:
        return {'value': [None if not np.isfinite(v) else float(v) for v in self.value],
                'n_obs': [float(n) for n in self.n_obs]}

    @classmethod
    def from_dict(cls, d: dict, n_bins: int) -> '_DirectionTable':
        table = cls(n_bins)
        values = d.get('value', [])
        n_obs = d.get('n_obs', [])
        m = min(n_bins, len(values), len(n_obs))
        for i in range(m):
            v = values[i]
            table.value[i] = np.nan if v is None else float(v)
            table.n_obs[i] = float(n_obs[i])
        return table


class FeedforwardMap:
    """Frequency-binned, log-log-interpolated, slowly-learned feedforward map.

    Parameters
    ----------
    f_min, f_max : float
        Frequency range the map covers, Hz. Must satisfy ``0 < f_min < f_max``.
        Frequencies outside this range are clipped to it for lookup/learning
        purposes (matches the fact that a configured sweep never leaves its
        own ``[f_start, f_stop]`` band, which should be passed here).
    initial_estimate : float
        Safe fallback value returned by :meth:`get` when the map (or the
        relevant region of it) has no learned data yet. Typically the same
        conservative small-signal starting amplitude already used to seed
        the fast loop (e.g. ``initial_drive_v``).
    value_min, value_max : float
        Hard clamps: a learned value can never leave ``[value_min, value_max]``
        regardless of what the fast loop's trim requested.
    bins_per_decade : float
        Log-frequency resolution. Default ``10`` (i.e. bin edges spaced by a
        factor of ``10**(1/10) ~= 1.259``). Higher = can resolve narrower
        resonances but each bin gets fewer repeated visits per sweep to
        average noise over; lower = smoother/noisier trade the other way.
    learning_rate : float
        Steady-state (long-run) learning rate floor, ``0 < learning_rate <= 1``,
        applied once a bin has accumulated several observations. A brand new
        bin learns faster than this (see module docstring); this is the rate
        it settles to. Default ``0.2``.
    max_relative_step_per_update : float
        Hard cap, independent of ``learning_rate``, on the fractional change
        a single :meth:`update` call may apply to a bin's value (or to the
        local baseline, for a bin's very first observation). Default ``0.3``
        (30% per update).
    outlier_reject_ratio : float
        An observation more than this factor above or below the map's
        current local prediction is rejected outright rather than blended
        in, once the target bin has at least ``outlier_reject_min_observations``
        prior observations. Default ``4.0``.
    outlier_reject_min_observations : float
        Minimum prior observation count a bin must have before outlier
        rejection engages (a brand-new bin has nothing yet to compare
        against, other than the global fallback/neighbor baseline -- see
        module docstring). Default ``2``.
    max_observations_cap : float
        Upper bound on the effective per-bin observation count used to
        compute the adaptive learning rate -- keeps a small residual
        adaptivity available indefinitely (to track slow real drift, e.g.
        temperature) instead of the per-bin rate decaying to zero forever.
        Default ``50``.
    separate_direction : bool
        If ``False`` (default), a single shared curve ``A_FF(f)`` is learned
        and used regardless of sweep direction. If ``True``, independent
        ``A_FF_up(f)``/``A_FF_down(f)`` curves are maintained; a direction
        with no local data yet falls back to the other direction's curve
        before falling back to ``initial_estimate``. Not enabled by default
        (mixing UP/DOWN data is the simpler, more sample-efficient starting
        point -- see module docstring); this flag exists so that later
        enabling per-direction learning (e.g. to capture real hysteresis) is
        a one-line change, not an architecture change.
    """

    def __init__(self,
                 f_min: float,
                 f_max: float,
                 initial_estimate: float,
                 value_min: float,
                 value_max: float,
                 bins_per_decade: float = 10.0,
                 learning_rate: float = 0.2,
                 max_relative_step_per_update: float = 0.3,
                 outlier_reject_ratio: float = 4.0,
                 outlier_reject_min_observations: float = 2.0,
                 max_observations_cap: float = 50.0,
                 separate_direction: bool = False):
        if not (0.0 < f_min < f_max):
            raise ValueError('f_min/f_max must satisfy 0 < f_min < f_max')
        if initial_estimate <= 0:
            raise ValueError('initial_estimate must be positive')
        if not (0.0 < value_min <= initial_estimate <= value_max):
            raise ValueError('must have 0 < value_min <= initial_estimate <= value_max')
        if bins_per_decade <= 0:
            raise ValueError('bins_per_decade must be positive')
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError('learning_rate must satisfy 0 < learning_rate <= 1')
        if not (0.0 < max_relative_step_per_update <= 1.0):
            raise ValueError('max_relative_step_per_update must satisfy 0 < x <= 1')
        if outlier_reject_ratio <= 1.0:
            raise ValueError('outlier_reject_ratio must be > 1')
        if max_observations_cap <= 0:
            raise ValueError('max_observations_cap must be positive')

        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.initial_estimate = float(initial_estimate)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.bins_per_decade = float(bins_per_decade)
        self.learning_rate = float(learning_rate)
        self.max_relative_step_per_update = float(max_relative_step_per_update)
        self.outlier_reject_ratio = float(outlier_reject_ratio)
        self.outlier_reject_min_observations = float(outlier_reject_min_observations)
        self.max_observations_cap = float(max_observations_cap)
        self.separate_direction = bool(separate_direction)

        n_decades = math.log10(self.f_max / self.f_min)
        n_bins = max(1, int(round(n_decades * self.bins_per_decade)))
        self._bin_edges = np.geomspace(self.f_min, self.f_max, n_bins + 1)
        self._log_bin_centers = 0.5 * (np.log10(self._bin_edges[:-1]) + np.log10(self._bin_edges[1:]))
        self.n_bins = n_bins

        keys = (VALID_DIRECTIONS if self.separate_direction else (SHARED_KEY,))
        self._tables: Dict[str, _DirectionTable] = {k: _DirectionTable(n_bins) for k in keys}

    @property
    def bin_centers(self) -> np.ndarray:
        """Geometric-mean frequency (Hz) of each bin, shape ``(n_bins,)``."""
        return 10.0 ** self._log_bin_centers

    def _key(self, direction: Optional[str]) -> str:
        if not self.separate_direction:
            return SHARED_KEY
        if direction not in VALID_DIRECTIONS:
            raise ValueError('direction must be one of {:} when separate_direction=True, got {!r}'.format(
                VALID_DIRECTIONS, direction))
        return direction

    def _bin_index(self, frequency_hz: float) -> Optional[int]:
        if not math.isfinite(frequency_hz) or frequency_hz <= 0:
            return None
        f = min(max(frequency_hz, self.f_min), self.f_max)
        idx = int(np.searchsorted(self._bin_edges, f, side='right') - 1)
        return int(np.clip(idx, 0, self.n_bins - 1))

    def _interp_table(self, table: _DirectionTable, log_f: float) -> Optional[float]:
        """Log-log interpolation/nearest-neighbor lookup within one table.
        Returns None if the table has no data at all."""
        mask = table.n_obs > 0
        if not np.any(mask):
            return None
        centers = self._log_bin_centers[mask]
        values = table.value[mask]
        if log_f <= centers[0]:
            return float(values[0])
        if log_f >= centers[-1]:
            return float(values[-1])
        # centers is sorted (bins are constructed in increasing-frequency order)
        i_hi = int(np.searchsorted(centers, log_f, side='left'))
        i_lo = i_hi - 1
        if centers[i_hi] == centers[i_lo]:
            return float(values[i_lo])
        frac = (log_f - centers[i_lo]) / (centers[i_hi] - centers[i_lo])
        log_v = np.log10(values[i_lo]) + frac * (np.log10(values[i_hi]) - np.log10(values[i_lo]))
        return float(10.0 ** log_v)

    def get(self, frequency_hz: float, direction: Optional[str] = None) -> float:
        """Returns the current best feedforward estimate at ``frequency_hz``.

        Log-log interpolates between the nearest populated bins on either
        side; holds flat at the nearest populated bin when only one side has
        data; falls back to ``initial_estimate`` when nothing is known yet.
        With ``separate_direction=True``, a direction with no data anywhere
        falls back to the other direction's curve before falling back to
        ``initial_estimate``.
        """
        if not math.isfinite(frequency_hz) or frequency_hz <= 0:
            return self.initial_estimate
        f = min(max(frequency_hz, self.f_min), self.f_max)
        log_f = math.log10(f)

        key = self._key(direction)
        value = self._interp_table(self._tables[key], log_f)
        if value is not None:
            return value
        if self.separate_direction:
            other_key = 'down' if key == 'up' else 'up'
            value = self._interp_table(self._tables[other_key], log_f)
            if value is not None:
                return value
        return self.initial_estimate

    def confidence(self, frequency_hz: float, direction: Optional[str] = None) -> float:
        """Returns a ``[0, 1]`` confidence for the bin nearest ``frequency_hz``
        (0 = never observed, 1 = at/above ``max_observations_cap`` observations).
        Purely diagnostic -- not used internally to gate learning or lookup."""
        idx = self._bin_index(frequency_hz)
        if idx is None:
            return 0.0
        table = self._tables[self._key(direction)]
        return float(min(1.0, table.n_obs[idx] / self.max_observations_cap))

    def update(self,
               frequency_hz: float,
               observed_value: float,
               trust: bool,
               direction: Optional[str] = None) -> FeedforwardLearnResult:
        """Nudges the bin containing ``frequency_hz`` toward ``observed_value``.

        Parameters
        ----------
        frequency_hz : float
            Frequency the observation was made at.
        observed_value : float
            The actuator command (e.g. drive amplitude, V_peak) observed to
            be required to hit the target force at this frequency -- must
            already be a settled, multi-period amplitude estimate, never a
            single instantaneous sample (see module docstring).
        trust : bool
            Whether the caller considers this observation trustworthy (e.g.
            tracking estimator valid, fast loop not saturated/held, no
            frequency transient in progress). If ``False``, nothing is
            updated.
        direction : str, optional
            ``'up'`` or ``'down'``; required if ``separate_direction=True``,
            ignored otherwise.

        Returns
        -------
        FeedforwardLearnResult
        """
        key = self._key(direction)
        idx = self._bin_index(frequency_hz)
        baseline = self.get(frequency_hz, direction)

        if not trust:
            return FeedforwardLearnResult(False, 'not_trusted', idx, baseline, baseline)
        if idx is None or not math.isfinite(observed_value) or observed_value <= 0:
            return FeedforwardLearnResult(False, 'invalid_value', idx, baseline, baseline)

        table = self._tables[key]
        old_value = table.value[idx] if table.n_obs[idx] > 0 else None
        reference = old_value if old_value is not None else baseline

        clipped_obs = min(max(observed_value, self.value_min), self.value_max)

        # Outlier rejection: only once the *target bin itself* has enough
        # history of its own to trust over a single new sample -- a fresh
        # bin must be allowed to accept its first observations even if they
        # look surprising relative to neighboring bins/resonances.
        if table.n_obs[idx] >= self.outlier_reject_min_observations:
            ratio = clipped_obs / reference
            if ratio > self.outlier_reject_ratio or ratio < 1.0 / self.outlier_reject_ratio:
                return FeedforwardLearnResult(False, 'outlier_rejected', idx, baseline, baseline)

        if old_value is None:
            # This bin has no observation of its own yet -- accept the first
            # one directly (matches ForceAmplitudeController-style startup:
            # nothing established yet to protect, so nothing to gain by
            # crawling toward it slowly; the neighbor/fallback `reference`
            # used for the outlier check above is not this bin's own history,
            # so it must not also throttle the step size here).
            new_value = min(max(clipped_obs, self.value_min), self.value_max)
        else:
            eff_lr = max(self.learning_rate, 1.0 / (table.n_obs[idx] + 1.0))
            eff_lr = min(eff_lr, 1.0)
            proposed = reference + eff_lr * (clipped_obs - reference)
            # Hard per-update relative-step cap, independent of eff_lr above
            # -- only meaningful once `reference` is this bin's own prior
            # value, not an unrelated neighbor's.
            max_step = self.max_relative_step_per_update * reference
            proposed = min(max(proposed, reference - max_step), reference + max_step)
            new_value = min(max(proposed, self.value_min), self.value_max)

        table.value[idx] = new_value
        table.n_obs[idx] = min(table.n_obs[idx] + 1.0, self.max_observations_cap)

        return FeedforwardLearnResult(True, 'ok', idx, baseline, float(new_value))

    # -- Persistence ---------------------------------------------------

    def to_dict(self) -> dict:
        """Serializes the full learned state (and enough configuration to
        reconstruct compatible lookup behavior) to a plain dict."""
        return {
            'format': 'rattlesnake_feedforward_map_v1',
            'timestamp': time.time(),
            'f_min': self.f_min,
            'f_max': self.f_max,
            'bins_per_decade': self.bins_per_decade,
            'initial_estimate': self.initial_estimate,
            'value_min': self.value_min,
            'value_max': self.value_max,
            'separate_direction': self.separate_direction,
            'bin_edges': [float(x) for x in self._bin_edges],
            'tables': {k: t.to_dict() for k, t in self._tables.items()},
        }

    def save(self, path: str) -> None:
        """Saves the learned curve(s) (frequency, value, observation count)
        plus enough metadata to reload, as JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> None:
        """Loads a previously saved map into this instance ("continue
        learning"), resampling onto *this* instance's own bin edges via the
        same log-log interpolation used by :meth:`get` -- so the loaded data
        does not need to have been saved with identical
        ``bins_per_decade``/``f_min``/``f_max``. Carries over a reduced
        observation count per bin (capped) so learning remains responsive
        rather than frozen at whatever rate the old data implied."""
        with open(path, 'r') as f:
            d = json.load(f)
        n_bins_saved = len(d.get('bin_edges', [])) - 1
        if n_bins_saved <= 0:
            return
        saved_centers = 0.5 * (np.log10(np.asarray(d['bin_edges'][:-1]))
                                + np.log10(np.asarray(d['bin_edges'][1:])))
        saved_separate = bool(d.get('separate_direction', False))
        saved_tables_raw = d.get('tables', {})

        source_keys = VALID_DIRECTIONS if saved_separate else (SHARED_KEY,)
        source_tables = {}
        for k in source_keys:
            raw = saved_tables_raw.get(k)
            if raw is None:
                continue
            table = _DirectionTable.from_dict(raw, n_bins_saved)
            source_tables[k] = table

        def source_value_and_conf(key, log_f):
            candidates = [key] if key in source_tables else []
            if not candidates and saved_separate:
                candidates = list(source_tables.keys())
            if not candidates and not saved_separate and SHARED_KEY in source_tables:
                candidates = [SHARED_KEY]
            for cand in candidates:
                table = source_tables[cand]
                mask = table.n_obs > 0
                if not np.any(mask):
                    continue
                centers = saved_centers[mask]
                values = table.value[mask]
                n_obs = table.n_obs[mask]
                j = int(np.argmin(np.abs(centers - log_f)))
                if log_f <= centers[0]:
                    return float(values[0]), float(n_obs[0])
                if log_f >= centers[-1]:
                    return float(values[-1]), float(n_obs[-1])
                i_hi = int(np.searchsorted(centers, log_f, side='left'))
                i_lo = i_hi - 1
                if centers[i_hi] == centers[i_lo]:
                    return float(values[i_lo]), float(n_obs[i_lo])
                frac = (log_f - centers[i_lo]) / (centers[i_hi] - centers[i_lo])
                log_v = np.log10(values[i_lo]) + frac * (np.log10(values[i_hi]) - np.log10(values[i_lo]))
                return float(10.0 ** log_v), float(min(n_obs[i_lo], n_obs[i_hi]))
            return None

        carried_over_obs_cap = min(self.max_observations_cap, self.outlier_reject_min_observations + 3.0)
        for key, table in self._tables.items():
            for i, log_f in enumerate(self._log_bin_centers):
                result = source_value_and_conf(key, log_f)
                if result is None:
                    continue
                value, n_obs = result
                table.value[i] = min(max(value, self.value_min), self.value_max)
                table.n_obs[i] = min(n_obs, carried_over_obs_cap)

    # -- Introspection for logging/plotting -----------------------------

    def curve(self, direction: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns ``(frequencies, values, n_obs)`` for only the *populated*
        bins of the given direction's table, sorted by frequency -- the raw
        learned curve, for plotting/inspection."""
        table = self._tables[self._key(direction)]
        mask = table.n_obs > 0
        return self.bin_centers[mask], table.value[mask], table.n_obs[mask]


@dataclass
class DriveComposition:
    """Result of :func:`compose_drive_amplitude`.

    Attributes
    ----------
    total_drive_amplitude : float
        ``u_total`` -- the physical command to actually send to the shaker.
    achieved_trim_gain : float
        The trim gain ``g`` that *actually* produced ``total_drive_amplitude``
        given ``feedforward_value`` -- may differ from the trim controller's
        raw requested gain whenever ``max_total``/``max_step`` intervened.
        Feed this back into the trim ``ForceAmplitudeController``'s
        ``drive_amplitude`` (see module docstring "Anti-windup" note).
    feedforward_value : float
        ``A_FF(frequency_hz)`` used for this composition (the value
        :meth:`FeedforwardMap.get` returned *before* any learning update
        from this same observation).
    """
    total_drive_amplitude: float
    achieved_trim_gain: float
    feedforward_value: float


def compose_drive_amplitude(feedforward_map: FeedforwardMap,
                             frequency_hz: float,
                             trim_gain_requested: float,
                             previous_total_drive_amplitude: float,
                             max_total_drive_amplitude: float,
                             max_step: float,
                             direction: Optional[str] = None) -> DriveComposition:
    """Composes ``u_total = A_FF(f) * g`` and applies the hard physical
    limits (``max_total_drive_amplitude``, and a slew limit ``max_step`` on
    the change from ``previous_total_drive_amplitude``), with anti-windup.

    This is the single shared implementation of the composition step used
    by both ``SineForceControlEnvironment`` and the test/simulation harnesses
    that mirror it -- previously duplicated inline in three places, which is
    exactly how an anti-windup fix could silently drift out of sync between
    them. See the module docstring's "Composition with the existing
    controller" section for the ``u_total = A_FF(f) * g`` design, and the
    Anti-windup note below for why ``achieved_trim_gain`` must be written
    back into the trim controller's own state by the caller.

    Anti-windup (back-calculation)
    -------------------------------
    ``max_total_drive_amplitude``/``max_step`` are hard *physical* limits
    the trim controller's own multiplicative ratio law has no visibility
    into (they are enforced here, externally, on the *composed* signal).
    Without correcting for this, the trim's requested gain keeps marching
    further away from reality every time this function's clamping actually
    binds, so it winds up pinned near its own configured ceiling/floor
    regardless of the true remaining error -- the classic actuator-
    saturation windup failure mode. The caller must overwrite the trim
    controller's ``drive_amplitude`` with this result's
    ``achieved_trim_gain`` (clamped to the trim controller's own valid
    range) after every call, so its *next* update starts from what was
    actually applied.
    """
    ff_value = feedforward_map.get(frequency_hz, direction=direction)
    raw_total = ff_value * trim_gain_requested

    clipped = min(max(raw_total, 0.0), max_total_drive_amplitude)
    delta = min(max(clipped - previous_total_drive_amplitude, -max_step), max_step)
    total = previous_total_drive_amplitude + delta

    achieved_trim_gain = (total / ff_value) if ff_value > 0 else trim_gain_requested
    return DriveComposition(total_drive_amplitude=total,
                             achieved_trim_gain=achieved_trim_gain,
                             feedforward_value=ff_value)
