"""The sign-change measurement's declared rules on synthetic fields: brackets and their
values, branches as eight-connected rows with the tilt limit, every plausible match kept,
the offset only for unambiguous correspondence, and the additional-structure test."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pilot_eastern_vertical as VT  # noqa: E402

LAT = np.arange(20.0, 9.0, -1.0)          # 11 rows, 20 to 10 N
LON = np.arange(40.0, 56.0, 1.0)          # 16 columns, 40 to 55 E


def field(axis_lon_by_row):
    """v negative west of each row's axis longitude and positive east of it."""
    v = np.zeros((LAT.size, LON.size))
    for i, a in enumerate(axis_lon_by_row):
        v[i] = np.where(LON < a, -2.0, 3.0)
    return v


def test_a_bracket_needs_both_points_valid_and_keeps_its_values():
    v = field([45.5] * LAT.size)
    ok = np.ones(v.shape, bool)
    brs = VT.brackets(v, ok, LAT, LON)
    assert len(brs) == LAT.size and all(b["lon"] == 45.5 and b["v_west"] == -2.0 and b["v_east"] == 3.0 for b in brs)
    ok[0, 6] = False                                                          # 46 E on the first row
    assert len(VT.brackets(v, ok, LAT, LON)) == LAT.size - 1


def test_rows_of_one_axis_are_one_branch_and_a_steep_tilt_splits_it():
    one = VT.branches(VT.brackets(field([45.5 + 0.0 * i for i in range(LAT.size)]), np.ones((LAT.size, LON.size), bool), LAT, LON), (LAT.size, LON.size))
    assert len(one) == 1 and one[0]["rows"] == LAT.size
    tilted = VT.branches(VT.brackets(field([42.5 + 1.0 * i for i in range(LAT.size)]), np.ones((LAT.size, LON.size), bool), LAT, LON), (LAT.size, LON.size))
    assert len(tilted) == 1                                                    # one column per row stays connected
    steep = VT.branches(VT.brackets(field([40.5 + 2.0 * (i // 1) for i in range(7)] + [55.5] * 4), np.ones((LAT.size, LON.size), bool), LAT, LON), (LAT.size, LON.size))
    assert len(steep) > 1                                                      # two columns per row splits, the stated limit


def test_every_plausible_match_is_kept_and_only_an_unambiguous_one_gets_an_offset():
    m = {"axis_lon": 45.0, "lat_range": [12.0, 18.0]}
    v7 = field([45.5] * LAT.size)
    b7 = VT.branches(VT.brackets(v7, np.ones(v7.shape, bool), LAT, LON), v7.shape)
    two = np.zeros((LAT.size, LON.size))
    two[:, :] = np.where(LON < 47, -2.0, 3.0)
    two[:, LON >= 50] = np.where(LON[LON >= 50] < 51, -2.0, 3.0)               # two separate 600 axes, near 46.5 and 50.5 E
    b6 = VT.branches(VT.brackets(two, np.ones(two.shape, bool), LAT, LON), two.shape)
    rel = VT.relevant(b7, m, 3.0)
    matches = {k: VT.plausible(VT.near({b["id"]: b for b in b7}[k], m, 5.0), b6, {"lat": 4.0, "lon": 6.0}) for k in rel}
    assert len(b6) == 2 and len(matches[rel[0]]) == 2 and VT.correspondence(rel, matches) is False
    one = VT.branches(VT.brackets(field([47.5] * LAT.size), np.ones(v7.shape, bool), LAT, LON), v7.shape)
    matches1 = {k: VT.plausible(VT.near({b["id"]: b for b in b7}[k], m, 5.0), one, {"lat": 4.0, "lon": 6.0}) for k in rel}
    assert VT.correspondence(rel, matches1) is True
    off = VT.offset(VT.near({b["id"]: b for b in b7}[rel[0]], m, 5.0), one[0], {"lat": 4.0, "lon": 6.0})
    assert off["dlon"] == 2.0 and off["dlat"] == 0.0


def test_additional_structure_needs_a_branch_near_the_read_geometry_distinct_from_the_one_near_the_surviving_output():
    m = {"axis_lon": 43.0, "lat_range": [12.0, 18.0]}
    v = np.where(LON[None, :] < 43, -2.0, 3.0) * np.ones((LAT.size, 1))
    v[:, LON >= 49] = np.where(LON[LON >= 49] < 50, -2.0, 3.0)                  # a second axis near 49.5 E
    brs = VT.branches(VT.brackets(v, np.ones(v.shape, bool), LAT, LON), v.shape)
    separate = VT.additional_structure(brs, m, (15.0, 50.0), 3.0)
    assert separate["additional_structure"] is True and separate["near_read"] != separate["near_surviving_output"]
    same = VT.additional_structure(brs, m, (15.0, 44.0), 3.0)
    assert same["additional_structure"] is False
    assert VT.additional_structure([], m, (15.0, 44.0), 3.0)["additional_structure"] is False
