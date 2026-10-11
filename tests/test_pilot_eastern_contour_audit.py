"""The connectivity audit on synthetic fields: a smooth tilted zero contour that the
branch rule fragments is one contour line, genuinely separate contours stay separate, and
a saddle cell is reported where connectivity is the extractor's choice."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pilot_eastern_contour_audit as CA  # noqa: E402
import pilot_eastern_vertical as VT  # noqa: E402

LAT = np.arange(20.0, 9.5, -1.0)
LON = np.arange(35.0, 65.5, 1.0)
LONG, LATG = np.meshgrid(LON, LAT)


def test_a_tilted_contour_fragmented_by_the_branch_rule_is_one_line():
    v = LONG - (39.5 + 2.0 * (20.0 - LATG))
    ok = np.ones(v.shape, bool)
    brs = VT.branches(VT.brackets(v, ok, LAT, LON), v.shape)
    lines = CA.contour_lines(v, ok, LAT, LON)
    assert len(brs) == 11 and len(lines) == 1
    a = CA.audit_level(brs, lines, LON, CA.saddle_cells(v, ok), LAT, [b["id"] for b in brs], [], [])
    assert a["branches_involved"] == 11 and a["lines_they_lie_on"] == 1 and a["branches_sharing_a_line"] == 11


def test_genuinely_separate_contours_stay_separate():
    v = np.sin(2 * np.pi * (LONG - 35.0) / 12.0)                   # trough-type zero lines near 41 and 53 E
    ok = np.ones(v.shape, bool)
    brs = VT.branches(VT.brackets(v, ok, LAT, LON), v.shape)
    lines = CA.contour_lines(v, ok, LAT, LON)
    ids = [b["id"] for b in brs]
    a = CA.audit_level(brs, lines, LON, CA.saddle_cells(v, ok), LAT, [ids[0]], [ids[1]], [])
    assert len(brs) == 2 and a["lines_they_lie_on"] == 2 and a["branches_sharing_a_line"] == 0 and a["additional_on_lines"] is True


def test_a_saddle_cell_is_found_where_signs_alternate():
    v = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert CA.saddle_cells(v, np.ones(v.shape, bool)).tolist() == [[True]]
    assert not CA.saddle_cells(np.array([[1.0, -1.0], [1.0, -1.0]]), np.ones((2, 2), bool)).any()
