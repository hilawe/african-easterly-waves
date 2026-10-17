"""The mask-to-axis check's synthetic parts: the vertex classes against a mask, and the
westerly mask removing only the contour that runs through masked cells."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pilot_eastern_mask_axis as MA  # noqa: E402

LAT_C = np.arange(20.0, 9.0, -2.0)                                      # north first, as the control grid
LON_C = np.arange(30.0, 45.0, 2.0)


def test_a_vertex_is_classed_by_its_nearest_corner_then_by_the_rest_of_its_quad():
    masked = np.zeros((LAT_C.size, LON_C.size), bool)
    masked[2, 3] = True                                                 # 16 N, 36 E
    assert MA.vertex_class(16.4, 36.3, LAT_C, LON_C, masked) == "location masked"
    assert MA.vertex_class(15.0, 37.6, LAT_C, LON_C, masked) == "neighboring support"
    assert MA.vertex_class(11.0, 41.0, LAT_C, LON_C, masked) == "unaffected"


def test_omitting_the_westerly_mask_restores_only_the_contour_through_masked_cells():
    latg, long_ = np.meshgrid(LAT_C, LON_C, indexing="ij")
    adv = long_ - 36.9                                                  # a zero contour near 37 E
    adv2 = long_ - 41.1                                                 # a second near 41 E
    field = np.where(long_ < 39.0, adv, adv2)
    wind = np.where((long_ >= 40.0) & (long_ <= 42.0), 5.0, -5.0)       # westerly only around the second
    fields = {"coarse_zonal_wind_smoothed": wind, "coarse_curvature": np.ones_like(field), "coarse_advection": field}
    grids = {"latgrid_c": latg, "longrid_c": long_}
    on, westerly = MA.axes_from_prepared(fields, 0.5, grids, True)
    off, _ = MA.axes_from_prepared(fields, 0.5, grids, False)
    lons_on = [float(np.mean(lo)) for _, lo in on]
    lons_off = [float(np.mean(lo)) for _, lo in off]
    assert any(abs(x - 36.9) < 0.5 for x in lons_on) and not any(abs(x - 41.1) < 0.5 for x in lons_on)
    assert any(abs(x - 36.9) < 0.5 for x in lons_off) and any(abs(x - 41.1) < 0.5 for x in lons_off)
    weak = {**fields, "coarse_curvature": np.zeros_like(field)}
    assert MA.axes_from_prepared(weak, 0.5, grids, False)[0] == []      # the weak mask still applies with the westerly mask off
