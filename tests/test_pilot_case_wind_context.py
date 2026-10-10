"""The case wind-and-cloud context. The window and the band follow the declared rules from
the stored track, the meridional-wind Hovmoller is the band mean of the raw ERA5 field
inside the window, and the cloud Hovmoller keeps a timestep without a retained file as
coverage missing, apart from cold cloud absent."""
import datetime as dt
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

nc = pytest.importorskip("netCDF4")


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _days(y, m, d, hour=0):
    return (dt.datetime(y, m, d, hour) - dt.datetime(1900, 1, 1)).total_seconds() / 86400.0


def test_window_and_band_follow_the_declared_rules_from_the_stored_track():
    W = _load("pilot_case_wind_context")
    track = {"time": [_days(1990, 8, 10, 6), _days(1990, 8, 10, 12), _days(1990, 8, 11, 0)], "lat": [12.0, 12.5, 13.0], "lon": [50.0, 48.0, 46.0]}
    win = W.window_for(track)
    assert win["lon"] == [46.0 - 25.0, 50.0 + 15.0] and win["time"] == [_days(1990, 8, 10, 6) - 3.0, _days(1990, 8, 11, 0) + 4.0]
    assert W.band_for(track) == [7.0, 17.0] and W.band_for({"lat": [33.0]}) == [28.0, 35.0] and W.band_for({"lat": [-3.0]}) == [-5.0, 2.0]
    assert W.window_for(track, lon_bounds=(-155.0, 60.0))["lon"][1] == 60.0                             # clipped to the data
    win_p = W.window_for_point(29.0, _days(2006, 9, 2, 0))
    assert win_p["lon"] == [29.0 - 40.0, 29.0 + 15.0] and win_p["time"] == [_days(2006, 9, 2, 0) - 3.0, _days(2006, 9, 2, 0) + 7.0]


def test_the_wind_hovmoller_is_the_band_mean_inside_the_window_and_the_cloud_hovmoller_keeps_missing_coverage_apart(tmp_path):
    W = _load("pilot_case_wind_context")
    times = np.array([_days(1990, 8, 10, h) for h in (0, 6, 12, 18)] + [_days(1990, 8, 11, 0)])
    lats, lons = np.arange(30.0, -1.0, -1.0), np.arange(20.0, 61.0, 1.0)
    v = np.zeros((times.size, lats.size, lons.size), np.float32)
    v[:, (lats <= 17.0) & (lats >= 7.0), :] = 2.0                                                   # the band carries 2, outside it 0
    v[2, np.where((lats <= 17.0) & (lats >= 7.0))[0][0], lons == 45.0] = 10.0                    # one band row of eleven carries 10 at 45 E at 12Z
    path = tmp_path / "v.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("valid_time", times.size); ds.createDimension("latitude", lats.size); ds.createDimension("longitude", lons.size)
        t = ds.createVariable("valid_time", "i8", ("valid_time",)); t.units = "seconds since 1970-01-01"
        t[:] = np.round((times - 25567.0) * 86400.0).astype(np.int64)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lats
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lons
        ds.createVariable("v", "f4", ("valid_time", "latitude", "longitude"))[:] = v
    hov = W.v_hovmoller(str(path), [7.0, 17.0], {"lon": [40.0, 50.0], "time": [_days(1990, 8, 10, 3), _days(1990, 8, 10, 21)]})
    assert hov["times"].tolist() == [_days(1990, 8, 10, 6), _days(1990, 8, 10, 12), _days(1990, 8, 10, 18)] and hov["lons"].tolist() == list(range(40, 51))
    assert hov["v"].shape == (3, 11) and hov["v"][0].tolist() == [2.0] * 11 and hov["v"][1][5] == pytest.approx(2.0 + 8.0 / 11.0)   # 11 band rows, one of them 10
    imagery = tmp_path / "imagery"
    imagery.mkdir()
    when = dt.datetime(1990, 8, 10, 6)
    glat, glon = np.arange(34.965, -25.0, -0.07), np.arange(-44.97, 75.0, 0.07)
    tb = np.full((1, glat.size, glon.size), 280.0, np.float32)
    tb[0][np.ix_((glat <= 17.0) & (glat >= 7.0), (glon >= 44.5) & (glon < 45.5))] = 210.0              # cold over one degree of longitude in the band
    with nc.Dataset(imagery / f"gridsat_{when:%Y%m%d%H}.nc", "w") as ds:
        ds.createDimension("time", 1); ds.createDimension("lat", glat.size); ds.createDimension("lon", glon.size)
        tv = ds.createVariable("time", "i4", ("time",)); tv.units = "hours since 1970-01-01"; tv[:] = [int((when - dt.datetime(1970, 1, 1)).total_seconds() // 3600)]
        ds.createVariable("lat", "f4", ("lat",))[:] = glat
        ds.createVariable("lon", "f4", ("lon",))[:] = glon
        var = ds.createVariable("irwin_cdr", "i2", ("time", "lat", "lon"), fill_value=np.int16(-31999)); var.scale_factor = np.float32(0.01); var.add_offset = np.float32(200.0); var.units = "Kelvin"
        var[:] = tb
    cloud = W.cloud_hovmoller(str(imagery), [7.0, 17.0], {"lon": [40.0, 50.0], "time": [_days(1990, 8, 10, 5), _days(1990, 8, 10, 11)]}, cold_k=240.0)
    assert cloud["coverage"] == ["retained", "missing"] and len(cloud["times"]) == 2 and cloud["lons"].tolist() == list(range(40, 51))
    assert cloud["cold_fraction"][0][5] == pytest.approx(1.0, abs=0.05) and cloud["cold_fraction"][0][0] == 0.0 and np.isnan(cloud["cold_fraction"][1]).all()
    assert cloud["summary"] == {"timesteps": 2, "retained": 1, "missing": 1, "timestamp_mismatch": 0}


def test_the_wind_reader_follows_the_variables_dimension_names_with_a_singleton_level_axis(tmp_path):
    W = _load("pilot_case_wind_context")
    times = np.array([_days(2006, 9, 1, h) for h in (0, 6, 12)])
    lats, lons = np.arange(20.0, -1.0, -1.0), np.arange(20.0, 41.0, 1.0)
    v = np.zeros((times.size, 1, lats.size, lons.size), np.float32)
    v[:, 0, (lats <= 15.0) & (lats >= 5.0), :] = 3.0
    path = tmp_path / "v4d.nc"
    with nc.Dataset(path, "w") as ds:
        for name, n in (("valid_time", times.size), ("pressure_level", 1), ("latitude", lats.size), ("longitude", lons.size)):
            ds.createDimension(name, n)
        t = ds.createVariable("valid_time", "i8", ("valid_time",)); t.units = "seconds since 1970-01-01"
        t[:] = np.round((times - 25567.0) * 86400.0).astype(np.int64)
        ds.createVariable("pressure_level", "f8", ("pressure_level",))[:] = [700.0]
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lats
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lons
        ds.createVariable("v", "f4", ("valid_time", "pressure_level", "latitude", "longitude"))[:] = v
    hov = W.v_hovmoller(str(path), [5.0, 15.0], {"lon": [25.0, 35.0], "time": [_days(2006, 9, 1, 0), _days(2006, 9, 1, 12)]})
    assert hov["v"].shape == (3, 11) and hov["v"].tolist() == [[3.0] * 11] * 3 and hov["lats"].tolist() == list(range(15, 4, -1))


def test_a_retained_file_whose_own_timestamp_is_not_the_requested_one_yields_no_cloud_reading(tmp_path):
    W = _load("pilot_case_wind_context")
    imagery = tmp_path / "imagery"
    imagery.mkdir()
    when = dt.datetime(1990, 8, 10, 6)
    glat, glon = np.arange(34.965, -25.0, -0.07), np.arange(-44.97, 75.0, 0.07)
    tb = np.full((1, glat.size, glon.size), 210.0, np.float32)                                    # cold everywhere, so a reading would be 1.0
    with nc.Dataset(imagery / f"gridsat_{when:%Y%m%d%H}.nc", "w") as ds:
        ds.createDimension("time", 1); ds.createDimension("lat", glat.size); ds.createDimension("lon", glon.size)
        tv = ds.createVariable("time", "i4", ("time",)); tv.units = "hours since 1970-01-01"
        tv[:] = [int((when + dt.timedelta(hours=3) - dt.datetime(1970, 1, 1)).total_seconds() // 3600)]        # named for 06Z, stamped 09Z
        ds.createVariable("lat", "f4", ("lat",))[:] = glat
        ds.createVariable("lon", "f4", ("lon",))[:] = glon
        var = ds.createVariable("irwin_cdr", "i2", ("time", "lat", "lon"), fill_value=np.int16(-31999)); var.scale_factor = np.float32(0.01); var.add_offset = np.float32(200.0); var.units = "Kelvin"
        var[:] = tb
    cloud = W.cloud_hovmoller(str(imagery), [7.0, 17.0], {"lon": [40.0, 50.0], "time": [_days(1990, 8, 10, 5), _days(1990, 8, 10, 11)]}, cold_k=240.0)
    assert cloud["coverage"] == ["timestamp_mismatch", "missing"] and np.isnan(cloud["cold_fraction"]).all()
    assert cloud["summary"] == {"timesteps": 2, "retained": 0, "missing": 1, "timestamp_mismatch": 1} and cloud["files"] == []


def test_the_window_is_clipped_to_the_wind_files_own_time_range():
    W = _load("pilot_case_wind_context")
    track = {"time": [_days(1990, 1, 2), _days(1990, 1, 2, 6)], "lat": [10.0], "lon": [45.0, 44.0]}
    win = W.window_for(track, time_bounds=(_days(1990, 1, 1), _days(1990, 12, 31, 18)))
    assert win["time"] == [_days(1990, 1, 1), _days(1990, 1, 2, 6) + 4.0]                              # the start is clipped to the file's first time
    late = {"time": [_days(1990, 12, 30), _days(1990, 12, 31, 12)], "lat": [10.0], "lon": [45.0, 44.0]}
    assert W.window_for(late, time_bounds=(_days(1990, 1, 1), _days(1990, 12, 31, 18)))["time"][1] == _days(1990, 12, 31, 18)
    assert W.window_for_point(29.0, _days(2006, 12, 30), time_bounds=(_days(2006, 1, 1), _days(2006, 12, 31, 18)))["time"][1] == _days(2006, 12, 31, 18)


def test_the_figure_marks_timestamp_rejected_rows_apart_from_missing_rows_and_counts_both(tmp_path):
    pytest.importorskip("matplotlib")
    W = _load("pilot_case_wind_context")
    t0 = _days(1990, 8, 10, 0)
    hov_v = {"times": np.array([t0, t0 + 0.25]), "lons": np.arange(40.0, 51.0), "v": np.zeros((2, 11)), "lats": np.arange(17.0, 6.0, -1)}
    hov_c = {"times": np.array([t0, t0 + 0.125, t0 + 0.25]), "lons": np.arange(40, 51), "cold_fraction": np.full((3, 11), np.nan),
             "coverage": ["timestamp_mismatch", "missing", "retained"], "cold_k": 240.0,
             "summary": {"timesteps": 3, "retained": 1, "missing": 1, "timestamp_mismatch": 1}}
    marks = W.render("t", [7.0, 17.0], {"lon": [40.0, 50.0], "time": [t0 - 0.1, t0 + 0.35]}, hov_v, hov_c, None, None, str(tmp_path / "f.png"))
    assert (tmp_path / "f.png").exists()
    by_cov = {m["coverage"]: m for m in marks["shaded"]}
    assert set(by_cov) == {"timestamp_mismatch", "missing"} and by_cov["timestamp_mismatch"]["color"] != by_cov["missing"]["color"]
    flat = marks["caption"].replace("\n", " ")
    assert "coverage missing: 1 of 3" in flat and "another time: 1 of 3" in flat
    assert marks["caption_inside_figure"] is True                                                    # the drawn caption lies inside the figure, so the counts are not clipped off


RETAINED_SUMMARIES = [(163, 108, 55), (111, 0, 111), (101, 39, 62), (79, 0, 79), (123, 123, 0), (123, 123, 0), (99, 0, 99), (85, 0, 85), (81, 81, 0)]   # timesteps, retained, missing of the nine retained records


def test_the_caption_fits_the_figure_at_the_retained_records_counts_and_a_worst_case_on_both_paths(tmp_path):
    pytest.importorskip("matplotlib")
    W = _load("pilot_case_wind_context")
    t0 = _days(1990, 8, 10, 0)
    window = {"lon": [40.0, 50.0], "time": [t0 - 0.1, t0 + 0.35]}
    hov_v = {"times": np.array([t0, t0 + 0.25]), "lons": np.arange(40.0, 51.0), "v": np.zeros((2, 11)), "lats": np.arange(17.0, 6.0, -1)}
    hov_c = {"times": np.array([t0, t0 + 0.125, t0 + 0.25]), "lons": np.arange(40, 51), "cold_fraction": np.full((3, 11), np.nan),
             "coverage": ["timestamp_mismatch", "missing", "retained"], "cold_k": 240.0}
    track = {"time": [t0, t0 + 0.25], "lat": [10.0, 10.0], "lon": [45.0, 44.0]}
    for i, (n, retained, missing) in enumerate(RETAINED_SUMMARIES + [(163, 0, 63)]):                 # the worst case has 100 timestamp-rejected rows
        hov_c["summary"] = {"timesteps": n, "retained": retained, "missing": missing, "timestamp_mismatch": n - retained - missing}
        for j, (tr, pt) in enumerate([(track, None), (None, (10.5, 45.0, t0 + 0.1))]):
            marks = W.render("t", [7.0, 17.0], window, hov_v, hov_c, tr, pt, str(tmp_path / f"f{i}{j}.png"))
            assert marks["caption_inside_figure"] is True and (tmp_path / f"f{i}{j}.png").exists()
