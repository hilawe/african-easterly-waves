"""The across-month tables must read a ridge by the stated slope rule, average the
persistence maps over months per band and bin, and refuse records on different grids."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("regime_composite_under_test", os.path.join(ROOT, "scripts", "eastern_regime_composite.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(label, fraction, ridges, lat, lon):
    return {"label": label, "lons": list(lon), "bands": {"10 to 14.5 N": [10.0, 14.5], "15 to 19.5 N": [15.0, 19.5]},
            "persistence": {"lat": list(lat), "fraction": fraction.tolist()},
            "lag_correlation": {"ridges": ridges}, "generated_by": {"sha256": "x" * 64}}


def test_ridge_reading_follows_the_slope_rule():
    C = _load()
    assert C.ridge_reading({"slope_days_per_degree": None, "speed_deg_per_day": None}) == "none"
    assert C.ridge_reading({"slope_days_per_degree": -0.05, "speed_deg_per_day": -20.0}) == "standing"
    assert C.ridge_reading({"slope_days_per_degree": -0.1, "speed_deg_per_day": -10.0}) == "standing", "the boundary is inclusive"
    assert C.ridge_reading({"slope_days_per_degree": -0.25, "speed_deg_per_day": -4.0}) == "westward"
    assert C.ridge_reading({"slope_days_per_degree": 0.3, "speed_deg_per_day": 3.3}) == "eastward"


def test_composite_averages_fractions_and_counts_ridges_and_refuses_mixed_grids():
    C = _load()
    lat = np.arange(19.5, 9.9, -0.5); lon = np.arange(20.0, 75.01, 0.5)
    f1 = np.zeros((len(lat), len(lon))); f1[:, (lon >= 40) & (lon <= 44.5)] = 0.8
    f2 = np.zeros((len(lat), len(lon))); f2[:, (lon >= 40) & (lon <= 44.5)] = 0.4; f2[0, 0] = np.nan
    r1 = {"10 to 14.5 N": {"slope_days_per_degree": -0.3, "speed_deg_per_day": -3.3}, "15 to 19.5 N": {"slope_days_per_degree": 0.0, "speed_deg_per_day": None}}
    r2 = {"10 to 14.5 N": {"slope_days_per_degree": None, "speed_deg_per_day": None}, "15 to 19.5 N": {"slope_days_per_degree": 0.4, "speed_deg_per_day": 2.5}}
    c = C.composite([_record("a", f1, r1, lat, lon), _record("b", f2, r2, lat, lon)])
    t = c["mean_fraction_by_band_and_bin"]
    assert abs(t["10 to 14.5 N"]["40 to 44.5 E"] - 0.6) < 1e-12 and t["10 to 14.5 N"]["45 to 49.5 E"] == 0.0
    assert c["cells_at_least_half"] == 20 * 10 and c["cells_valid"] == len(lat) * len(lon), "the mean over months, NaN ignored, counted against valid cells"
    assert c["ridge_readings_by_band"]["10 to 14.5 N"] == {"westward": 1, "standing": 0, "eastward": 0, "none": 1}
    assert c["ridge_readings_by_band"]["15 to 19.5 N"] == {"westward": 0, "standing": 1, "eastward": 1, "none": 0}
    assert c["westward_speeds_by_band"]["10 to 14.5 N"] == [-3.3]
    try:
        C.composite([_record("a", f1, r1, lat, lon), _record("b", f2, r2, lat, lon[1:])]); assert False
    except SystemExit as e:
        assert "one grid" in str(e)
