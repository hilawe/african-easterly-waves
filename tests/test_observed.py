"""The calendar of observed slots and the rules that keep an unobserved slot from counting as zero."""
import os

import numpy as np
import pandas as pd
import pytest

from aew import observed as O
from aew.composites import catalog_support_check
from aew.leadlag import box_count_series

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lattice(start, n):
    return pd.date_range(start, periods=n, freq="3h").values


def test_a_slot_without_a_row_or_with_a_fraction_of_its_hours_rows_is_not_observed():
    t = lattice("2000-07-01", 80)                                          # ten days, every hour of day ten times
    rows = np.full(80, 40)
    rows[16] = 9                                                           # under a quarter of 40
    rows[17] = 10                                                          # exactly a quarter, which is kept
    keep = np.ones(80, bool)
    keep[30] = False                                                       # a slot with no row is simply not listed
    seen, partial = O.observed_from_counts(t[keep], rows[keep])
    assert list(partial) == [t[16]] and t[17] in seen and t[30] not in seen and t[16] not in seen
    assert seen.size == 78 and (np.diff(seen.astype("int64")) > 0).all()
    with pytest.raises(ValueError, match="listed twice"):
        O.observed_from_counts(np.r_[t[:5], t[:1]], np.full(6, 40))
    with pytest.raises(ValueError, match="at least one row"):
        O.observed_from_counts(t[:5], np.array([40, 0, 40, 40, 40]))


def test_a_trough_is_eligible_only_when_both_of_its_windows_were_observed_in_full():
    t = lattice("2000-07-01", 8 * 12)                                      # twelve days of slots
    passage = np.datetime64("2000-07-06T00:00")
    for off, eligible in ((0, False), (21, False), (24, True), (-3, True), (-24, True), (-27, False), (-48, False), (-51, True)):
        seen = t[t != passage + np.timedelta64(off, "h")]                  # one slot unobserved, at this offset from the passage
        assert bool(O.window_observed_mask([passage], seen)[0]) is eligible, off
    seen = t[t != passage + np.timedelta64(-30, "h")]                      # the antecedent window can be left out of the test
    assert not O.window_observed_mask([passage], seen)[0] and O.window_observed_mask([passage], seen, antecedent=False)[0]
    seen = t[t != passage + np.timedelta64(30, "h")]                       # a longer forward window reaches a slot the 24 h window does not
    assert O.window_observed_mask([passage], seen)[0] and not O.window_observed_mask([passage], seen, resp_h=48.0)[0]
    assert O.window_observed_mask([passage], seen, resp_h=12.0)[0]
    assert O.window_offsets(12.0, antecedent=False) == [0.0, 3.0, 6.0, 9.0] and len(O.window_offsets()) == 16
    edge = np.datetime64("2000-07-12T12:00")                               # a window past the end of the calendar is not observed
    assert not O.window_observed_mask([edge], t)[0]
    many = np.array([passage, edge, np.datetime64("2000-07-04T06:00")])
    assert list(O.window_observed_mask(many, t)) == [True, False, True]
    assert list(O.slots_observed(many, t, [0.0, -3.0])) == [True, True, True]


def test_the_exposure_of_a_grid_time_is_the_count_the_box_counter_gives_one_system_per_observed_slot():
    slots = lattice("2000-07-01", 8 * 20)
    rng = np.random.default_rng(3)
    seen = np.sort(rng.choice(slots, size=130, replace=False))
    grid = pd.date_range("2000-06-29", "2000-07-23", freq="6h").values
    counted = box_count_series(seen, np.zeros(seen.size), np.full(seen.size, 10.0), grid, 0.0, 5.0, 5.0, 15.0)
    e = O.exposure_series(grid, seen)
    assert np.allclose(e, counted) and e.max() == 2.0 and e.min() == 0.0
    full = O.exposure_series(grid, slots)
    inside = (grid >= np.datetime64("2000-07-01T06:00")) & (grid <= np.datetime64("2000-07-20T12:00"))
    assert (full[inside] == 2.0).all() and full[grid == np.datetime64("2000-07-01T00:00")][0] == 1.5
    with pytest.raises(ValueError, match="uniform"):
        O.exposure_series(np.r_[grid[:5], grid[7:9]], seen)


def test_short_gaps_are_filled_long_gaps_and_ends_stay_missing_and_counts_scale_to_full_exposure():
    v = np.array([4.0, 3.0, 0.0, 0.0, 8.0, 5.0, 0.0, 0.0, 0.0, 6.0, 0.0])
    e = np.array([2.0, 1.5, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    out, longest = O.fill_short_gaps(v, e, 2.0, max_gap=2)
    assert longest == 3
    assert np.allclose(out[[0, 1, 4, 5, 9]], [4.0, 4.0, 8.0, 10.0, 6.0])   # 3 over 1.5 and 5 over 1 scale to full exposure
    assert np.allclose(out[2:4], [4.0 + 4.0 / 3.0, 4.0 + 8.0 / 3.0])       # two gaps between 4 and 8, filled linearly
    assert np.isnan(out[6:9]).all() and np.isnan(out[10])                  # three gaps are too many, and a gap at the end has no far side
    out, _ = O.fill_short_gaps(v, e, 2.0, max_gap=3)
    assert np.allclose(out[6:9], [9.0, 8.0, 7.0]) and np.isnan(out[10])
    out, longest = O.fill_short_gaps(v[:2], e[:2], 2.0, max_gap=2)
    assert longest == 0 and np.allclose(out, [4.0, 4.0])


def test_the_support_check_holds_every_bin_edge_inside_the_reach_with_no_allowance():
    rel = np.arange(-12.0, 12.1, 2.0)                                      # outer edges at 13 degrees either side
    assert O.support_check(np.array([-41.0, 0.0, 41.0]), rel, -54.0, 54.0) == (-54.0, 54.0)
    for placed in (np.array([-41.1, 0.0]), np.array([0.0, 41.1])):
        with pytest.raises(RuntimeError, match="support assertion failed"):
            O.support_check(placed, rel, -54.0, 54.0)
    catalog = np.array([-54.0, 0.0, 54.0])                                 # the older guard allows what this one stops
    catalog_support_check(np.array([-42.9, 42.9]), float(rel.max()), catalog)
    with pytest.raises(RuntimeError):
        O.support_check(np.array([-42.9, 42.9]), rel, catalog.min(), catalog.max())
    world = np.array([-179.9, 0.0, 180.0])                                 # and it passes any catalog that spans the globe
    catalog_support_check(np.array([-170.0, 170.0]), 30.0, world)
    with pytest.raises(RuntimeError):
        O.support_check(np.array([-170.0, 170.0]), np.arange(-30.0, 30.1, 2.0), world.min(), world.max())


def test_the_tracked_calendar_reads_back_as_distinct_slots_on_the_lattice(tmp_path):
    seen = O.load_observed_slots(os.path.join(ROOT, "scripts", "met_observed_slots.csv"))
    t = pd.DatetimeIndex(seen)
    assert seen.size == 17751 and t.is_monotonic_increasing and not t.has_duplicates
    assert set(t.month) == {7, 8, 9} and set(t.hour) == set(range(0, 24, 3)) and t.year.min() == 1983 and t.year.max() == 2007
    assert np.datetime64("1992-09-15T00:00") not in seen and np.datetime64("2004-08-08T00:00") not in seen
    bad = tmp_path / "bad.csv"
    bad.write_text("time,rows\n2000-07-01 00:00,5\n2000-07-01 00:00,6\n")
    with pytest.raises(ValueError, match="distinct observed slots"):
        O.load_observed_slots(str(bad))
