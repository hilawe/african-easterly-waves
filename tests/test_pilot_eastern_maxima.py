"""The maxima comparison's declared rules on synthetic fields: the peak definition with its
threshold, plateau and invalid-neighbor exclusion, the finals' reproduction check, the
nearer-without-added-density rule, and separation inside a shared region."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pilot_eastern_maxima as X  # noqa: E402


def _grid(ny=9, nx=11):
    lat = np.arange(ny, dtype=float)[:, None] * np.ones((1, nx)) * 2.0
    lon = np.ones((ny, 1)) * np.arange(nx, dtype=float)[None, :] * 2.0 + 21.0
    return lat, lon


def test_two_separate_peaks_are_two_maxima_and_a_low_peak_is_below_the_threshold():
    lat, lon = _grid()
    f = np.zeros(lat.shape)
    f[3, 3], f[5, 8], f[7, 1] = 5.0, 4.0, 0.5
    out, excluded = X.local_maxima(f, np.ones(f.shape, bool), 1.0, lat, lon)
    assert sorted((p["lat"], p["lon"]) for p in out) == [(6.0, 27.0), (10.0, 37.0)] and excluded == 0


def test_a_plateau_is_one_maximum_at_its_mean_position():
    lat, lon = _grid()
    f = np.zeros(lat.shape)
    f[4, 4] = f[4, 5] = f[5, 5] = 3.0
    out, _ = X.local_maxima(f, np.ones(f.shape, bool), 1.0, lat, lon)
    assert len(out) == 1 and out[0]["cells"] == 3
    assert abs(out[0]["lat"] - np.mean([8.0, 8.0, 10.0])) < 1e-9 and abs(out[0]["lon"] - np.mean([29.0, 31.0, 31.0])) < 1e-9


def test_a_peak_next_to_an_invalid_cell_or_on_the_edge_is_excluded_and_counted():
    lat, lon = _grid()
    f = np.zeros(lat.shape)
    f[4, 4] = 3.0
    f[0, 6] = 3.0                                                     # on the edge, neighbors outside the grid
    valid = np.ones(f.shape, bool)
    valid[4, 5] = False                                               # an invalid neighbor of the interior peak
    out, excluded = X.local_maxima(f, valid, 1.0, lat, lon)
    assert out == [] and excluded == 2


def test_the_finals_must_reproduce_in_digest_and_position():
    a = [(10.0, 45.0, "d1"), (12.0, 50.0, "d2")]
    assert X.same_finals(a, [(10.0 + 1e-9, 45.0, "d1"), (12.0, 50.0, "d2")])
    assert not X.same_finals(a, [(10.0, 45.0, "d1"), (12.0, 50.0, "dx")])
    assert not X.same_finals(a, [(10.0, 45.0, "d1")]) and not X.same_finals(a, [(10.0, 45.1, "d1"), (12.0, 50.0, "d2")])


def test_nearer_without_added_density_needs_no_more_points_in_the_box():
    m = {"axis_lon": 45.0, "lat_range": [10.0, 20.0]}
    box = {"lat": [10.0, 20.0], "lon": [40.0, 50.0]}
    fin = X.correspondence([{"lat": 15.0, "lon": 49.0}], m, box)
    one = X.correspondence([{"lat": 15.0, "lon": 46.0}], m, box)
    many = X.correspondence([{"lat": 15.0, "lon": 46.0}, {"lat": 12.0, "lon": 41.0}, {"lat": 18.0, "lon": 49.5}], m, box)
    assert fin["nearest_deg"] == 4.0 and fin["within"] == {"3.0": 0, "5.0": 1}
    assert X.compare(one, fin) == {"nearer": True, "nearer_without_added_density": True}
    assert X.compare(many, fin) == {"nearer": True, "nearer_without_added_density": False}
    assert X.compare(X.correspondence([], m, box), fin) == {"nearer": False, "nearer_without_added_density": False}


def test_a_shared_region_separates_only_with_more_than_one_maximum_inside_and_one_near_the_read_geometry():
    lat_c = np.arange(0.0, 30.0, 2.0)
    lon_c = np.arange(31.0, 61.0, 2.0)
    mask = np.zeros((lat_c.size, lon_c.size), bool)
    mask[5:9, 5:12] = True                                            # 10 to 16 N, 41 to 53 E
    m = {"axis_lon": 42.0, "lat_range": [10.0, 16.0]}
    near = {"lat": 12.0, "lon": 43.0, "value": 2.0, "cells_ij": [[6, 6]]}
    far = {"lat": 14.0, "lon": 51.0, "value": 3.0, "cells_ij": [[7, 10]]}
    outside = {"lat": 26.0, "lon": 43.0, "value": 3.0, "cells_ij": [[13, 6]]}
    two = X.region_maxima(mask, [near, far, outside], [{"lat": 12.2, "lon": 42.6, "value": 1.5}], lat_c, lon_c, m, [14.0, 51.0])
    assert two["A"]["count"] == 2 and two["A"]["separates"]["3.0"] is True
    assert two["B"]["count"] == 1 and two["B"]["separates"]["3.0"] is False
    assert two["A"]["maxima_inside"][1]["distance_to_surviving_output_deg"] == 0.0
    only_far = X.region_maxima(mask, [far, {"lat": 10.0, "lon": 53.0, "value": 1.0, "cells_ij": [[5, 11]]}], [], lat_c, lon_c, m, None)
    assert only_far["A"]["count"] == 2 and only_far["A"]["separates"]["3.0"] is False
