# -*- coding: utf-8 -*-
"""Tests for components.feedforward_map.FeedforwardMap."""
import os
import numpy as np
import pytest

from components.feedforward_map import FeedforwardMap


def make_map(**overrides):
    kwargs = dict(f_min=5.0, f_max=2000.0, initial_estimate=0.1,
                  value_min=0.01, value_max=5.0, bins_per_decade=10.0,
                  learning_rate=0.1, max_relative_step_per_update=0.3,
                  outlier_reject_ratio=4.0, outlier_reject_min_observations=2.0,
                  max_observations_cap=50.0)
    kwargs.update(overrides)
    return FeedforwardMap(**kwargs)


# -- Test 1: constant error must push the FF value in the right direction --

def test_constant_positive_correction_increases_feedforward():
    ff = make_map()
    f = 100.0
    before = ff.get(f)
    for _ in range(30):
        ff.update(f, observed_value=before * 1.5, trust=True)
    after = ff.get(f)
    assert after > before
    # Should be converging toward the observed value, not overshooting past it.
    assert after <= before * 1.5 + 1e-9


def test_constant_negative_correction_decreases_feedforward():
    ff = make_map()
    f = 300.0
    before = ff.get(f)
    for _ in range(30):
        ff.update(f, observed_value=before * 0.6, trust=True)
    after = ff.get(f)
    assert after < before
    assert after >= before * 0.6 - 1e-9


# -- Test 2: no error -> no drift --

def test_no_error_does_not_drift():
    ff = make_map()
    f = 50.0
    ff.update(f, observed_value=0.2, trust=True)
    value_after_first = ff.get(f)
    for _ in range(20):
        result = ff.update(f, observed_value=value_after_first, trust=True)
        assert result.value_after == pytest.approx(value_after_first, rel=1e-9)
    assert ff.get(f) == pytest.approx(value_after_first, rel=1e-9)


# -- Test 3: a frequency-dependent error pattern is learned as a curve --

def test_learns_frequency_dependent_curve():
    ff = make_map(f_min=5.0, f_max=2000.0, learning_rate=0.3)
    low_freqs = np.geomspace(6.0, 20.0, 5)
    high_freqs = np.geomspace(500.0, 1800.0, 5)
    for _ in range(15):
        for f in low_freqs:
            ff.update(f, observed_value=0.05, trust=True)
        for f in high_freqs:
            ff.update(f, observed_value=0.5, trust=True)
    low_value = ff.get(10.0)
    high_value = ff.get(1000.0)
    assert low_value == pytest.approx(0.05, rel=0.2)
    assert high_value == pytest.approx(0.5, rel=0.2)
    assert high_value > low_value


# -- Test 4: a single extreme outlier must not destroy an established bin --

def test_single_outlier_does_not_destroy_established_bin():
    ff = make_map()
    f = 100.0
    for _ in range(10):
        ff.update(f, observed_value=0.2, trust=True)
    established = ff.get(f)
    assert established == pytest.approx(0.2, rel=0.1)

    result = ff.update(f, observed_value=500.0, trust=True)
    assert result.updated is False
    assert result.reason == 'outlier_rejected'
    assert ff.get(f) == pytest.approx(established, rel=1e-9)


# -- Test 5: learning disabled (trust=False) leaves the map unchanged --

def test_trust_false_does_not_update():
    ff = make_map()
    f = 42.0
    ff.update(f, observed_value=0.3, trust=True)
    before = ff.get(f)
    result = ff.update(f, observed_value=0.9, trust=False)
    assert result.updated is False
    assert result.reason == 'not_trusted'
    assert ff.get(f) == pytest.approx(before, rel=1e-9)


# -- Test 6: sweep direction changes (up -> down -> up) must not reset learning --

def test_direction_changes_do_not_reset_shared_map():
    """A shared (non-direction-separated) map must keep learning smoothly
    across UP/DOWN/UP direction changes -- no reset back toward the initial
    estimate and no discontinuity at the direction switches."""
    ff = make_map(separate_direction=False)
    f = 250.0
    for _ in range(5):
        ff.update(f, observed_value=0.4, trust=True, direction='up')
    after_up = ff.get(f)
    assert after_up > ff.initial_estimate  # actually learned something, not reset

    for _ in range(5):
        ff.update(f, observed_value=0.4, trust=True, direction='down')
    after_down = ff.get(f)
    # Continues converging toward (or holding at) 0.4 -- not reset back down
    # toward the initial estimate just because the direction tag changed.
    assert after_down >= after_up - 1e-9

    for _ in range(5):
        ff.update(f, observed_value=0.4, trust=True, direction='up')
    after_up2 = ff.get(f)
    assert after_up2 >= after_down - 1e-9
    assert after_up2 == pytest.approx(0.4, rel=0.15)


def test_separate_direction_tables_are_independent_but_both_persist():
    ff = make_map(separate_direction=True)
    f = 250.0
    for _ in range(10):
        ff.update(f, observed_value=0.3, trust=True, direction='up')
    for _ in range(10):
        ff.update(f, observed_value=0.6, trust=True, direction='down')
    up_value = ff.get(f, direction='up')
    down_value = ff.get(f, direction='down')
    assert up_value == pytest.approx(0.3, rel=0.15)
    assert down_value == pytest.approx(0.6, rel=0.15)
    assert up_value != pytest.approx(down_value, rel=0.05)


# -- Test 7: interpolation between two learned points --

def test_interpolates_between_two_learned_bins():
    ff = make_map(learning_rate=1.0)
    f_lo, f_hi = 10.0, 1000.0
    for _ in range(5):
        ff.update(f_lo, observed_value=0.1, trust=True)
        ff.update(f_hi, observed_value=1.0, trust=True)
    f_mid = np.sqrt(f_lo * f_hi)  # geometric mean -> log-midpoint
    mid_value = ff.get(f_mid)
    assert mid_value == pytest.approx(np.sqrt(0.1 * 1.0), rel=0.05)
    assert 0.1 < mid_value < 1.0


def test_flat_extrapolation_beyond_learned_region():
    ff = make_map(learning_rate=1.0)
    ff.update(500.0, observed_value=0.7, trust=True)
    assert ff.get(6.0) == pytest.approx(0.7, rel=1e-6)
    assert ff.get(1900.0) == pytest.approx(0.7, rel=1e-6)


# -- Test 8: hard limits are never exceeded --

def test_hard_limits_never_exceeded():
    ff = make_map(value_min=0.05, value_max=1.0, learning_rate=0.5)
    f = 80.0
    for _ in range(50):
        ff.update(f, observed_value=100.0, trust=True)
    assert ff.get(f) <= 1.0 + 1e-9
    ff2 = make_map(value_min=0.05, value_max=1.0, learning_rate=0.5)
    for _ in range(50):
        ff2.update(f, observed_value=1e-6, trust=True)
    assert ff2.get(f) >= 0.05 - 1e-9


# -- Test 9: no learning without valid data --

def test_invalid_observation_value_does_not_update():
    ff = make_map()
    f = 60.0
    before = ff.get(f)
    for bad in (float('nan'), float('inf'), -1.0, 0.0):
        result = ff.update(f, observed_value=bad, trust=True)
        assert result.updated is False
        assert result.reason == 'invalid_value'
    assert ff.get(f) == pytest.approx(before, rel=1e-9)


# -- Test 10: persistence round-trips the learned curve --

def test_save_and_load_round_trip(tmp_path):
    ff = make_map(learning_rate=0.3)
    freqs = np.geomspace(6.0, 1900.0, 12)
    for _ in range(8):
        for i, f in enumerate(freqs):
            ff.update(f, observed_value=0.05 + 0.001 * i ** 2, trust=True)

    path = os.path.join(str(tmp_path), 'ff_map.json')
    ff.save(path)
    assert os.path.exists(path)

    ff2 = make_map(learning_rate=0.3)
    ff2.load(path)
    for f in freqs:
        assert ff2.get(f) == pytest.approx(ff.get(f), rel=1e-6)


def test_load_into_different_bin_resolution_still_resembles_original(tmp_path):
    ff = make_map(bins_per_decade=10.0, learning_rate=1.0)
    freqs = np.geomspace(6.0, 1900.0, 20)
    for f in freqs:
        ff.update(f, observed_value=0.02 * f, trust=True)
    path = os.path.join(str(tmp_path), 'ff_map.json')
    ff.save(path)

    ff_coarser = make_map(bins_per_decade=4.0, learning_rate=1.0)
    ff_coarser.load(path)
    for f in freqs:
        assert ff_coarser.get(f) == pytest.approx(ff.get(f), rel=0.35)


# -- Additional structural checks --

def test_empty_map_returns_initial_estimate():
    ff = make_map(initial_estimate=0.15)
    assert ff.get(10.0) == pytest.approx(0.15)
    assert ff.confidence(10.0) == 0.0


def test_confidence_increases_toward_one_with_more_observations():
    ff = make_map(max_observations_cap=10.0, learning_rate=0.2)
    f = 40.0
    c0 = ff.confidence(f)
    for _ in range(10):
        ff.update(f, observed_value=0.2, trust=True)
    c1 = ff.confidence(f)
    assert c1 > c0
    assert c1 == pytest.approx(1.0, abs=1e-9)


def test_separate_direction_requires_direction_argument():
    ff = make_map(separate_direction=True)
    with pytest.raises(ValueError):
        ff.get(100.0, direction=None)
    with pytest.raises(ValueError):
        ff.update(100.0, 0.2, trust=True, direction='sideways')


# -- Regression: a bin must never get permanently stuck rejecting live data --

def test_persistent_disagreement_eventually_overrides_stale_reference():
    """A single surprising sample is a one-off outlier and must be rejected
    (see test_single_outlier_does_not_destroy_established_bin), but the same
    kind of "outlier" repeating past outlier_reject_persistence times in a
    row is no longer statistically an outlier -- it means the reference bin
    value itself is stale (e.g. real drift, or a changed operating point --
    see the next test). The bin must recover, not reject forever."""
    ff = make_map(outlier_reject_ratio=4.0, outlier_reject_persistence=3.0, learning_rate=0.2)
    f = 100.0
    for _ in range(10):
        ff.update(f, observed_value=0.2, trust=True)
    established = ff.get(f)
    assert established == pytest.approx(0.2, rel=0.1)

    # The true value has genuinely moved far outside the outlier ratio --
    # simulate many repeated "surprising" but consistent observations.
    reasons = []
    for _ in range(40):
        result = ff.update(f, observed_value=0.01, trust=True)
        reasons.append(result.reason)

    assert 'outlier_rejected' in reasons  # some rejections did happen...
    assert reasons.count('ok') >= 5       # ...but it was not stuck forever
    # And it must have moved substantially toward the new true value, not
    # stayed pinned at the old one.
    assert ff.get(f) < established * 0.5


def test_load_across_large_operating_point_change_recovers_not_stuck():
    """Regression test for a confirmed bug: loading a map learned under one
    operating point (e.g. a different target force) into a run where the
    true required values are now far outside outlier_reject_ratio used to
    cause every single subsequent -- correct -- live observation to be
    rejected forever (the loaded value was never itself corroborated by
    live data, yet was defended as if it had been). Verified to reproduce
    with a >4x regime change before the live_obs/outlier_reject_persistence
    fix; must now recover within a bounded number of live observations."""
    f = 100.0
    ff_old = make_map(learning_rate=0.2)
    for _ in range(10):
        ff_old.update(f, observed_value=0.40, trust=True)

    ff_new = make_map(learning_rate=0.2)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'map.json')
        ff_old.save(path)
        ff_new.load(path)

    assert ff_new.get(f) == pytest.approx(0.40, rel=1e-6)

    # New operating point needs ~6.7x less drive at this frequency -- well
    # beyond the default outlier_reject_ratio of 4.0.
    reasons = []
    for _ in range(30):
        result = ff_new.update(f, observed_value=0.06, trust=True)
        reasons.append(result.reason)
        if ff_new.get(f) < 0.10:
            break

    assert reasons.count('ok') >= 1
    # Must have made real, substantial progress toward the true value --
    # not stayed pinned at (or near) the stale loaded value of 0.40.
    assert ff_new.get(f) < 0.25
