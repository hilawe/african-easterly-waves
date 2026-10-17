"""The seasonal wind view: the band means are the propagation study's rule on the band's
rows, the grid selects the declared longitudes and the two 20-row bands, and the figures
carry their windows, the unavailable counts and nothing of the tracker."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("season_view_under_test", os.path.join(ROOT, "scripts", "pilot_season_wind_view.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_band_means_apply_the_declared_support_rule_on_each_band_s_rows():
    S = _load()
    P = S.P
    nt, nc = 3, 4
    v = np.ones((nt, 40, nc)); v[:, :20, :] = 2.0                    # southern rows 0 to 19 hold 2, northern rows 20 to 39 hold 1
    v[0, 0:5, 1] = 100.0                                              # five southern rows at time 0, column 1 get a large value
    sp = np.full((nt, 40, nc), 90000.0)
    sp[0, 0:5, 1] = 70000.0                                           # and they are at the level: below ground, 75 percent valid, unavailable
    sp[1, 0:4, 2] = 70000.0                                           # four rows at time 1, column 2: 80 percent valid, kept as the mean of 16 rows
    block = {"v": v, "sp": sp, "row0": 10}
    rows = {"southern": np.arange(10, 30), "northern": np.arange(30, 50)}
    means = S.band_means(block, rows)
    s = means["southern"]["series"]
    assert np.isnan(s[0, 1]) and means["southern"]["support"][0, 1] == 0.75
    assert s[1, 2] == 2.0 and means["southern"]["support"][1, 2] == 0.8
    assert np.all(means["northern"]["series"] == 1.0) and np.all(means["northern"]["support"] == 1.0)
    assert means["southern"]["unavailable"] == 1 and means["southern"]["samples"] == nt * nc
    expected, _ = P.band_series(v[:, :20, :], sp[:, :20, :], P.PRIMARY_LEVEL)
    np.testing.assert_array_equal(np.isnan(s), np.isnan(expected))
    np.testing.assert_allclose(np.nan_to_num(s), np.nan_to_num(expected))


def _write_v(path, lats, lons, ntime=2):
    import netCDF4
    d = netCDF4.Dataset(path, "w")
    d.createDimension("valid_time", ntime); d.createDimension("pressure_level", 1)
    d.createDimension("latitude", len(lats)); d.createDimension("longitude", len(lons))
    t = d.createVariable("valid_time", "f8", ("valid_time",)); t.units = "seconds since 1970-01-01"; t[:] = np.arange(ntime) * 21600.0
    d.createVariable("pressure_level", "f8", ("pressure_level",))[:] = [700.0]
    d.createVariable("latitude", "f8", ("latitude",))[:] = lats
    d.createVariable("longitude", "f8", ("longitude",))[:] = lons
    v = d.createVariable("v", "f8", ("valid_time", "pressure_level", "latitude", "longitude")); v.units = "m s**-1"; v[:] = 0.0
    d.close()


def test_the_grid_selects_the_declared_longitudes_and_two_twenty_row_bands(tmp_path, monkeypatch):
    pytest.importorskip("netCDF4")
    S = _load()
    lats = np.arange(35.0, -25.5, -0.5); lons = np.arange(-45.0, 75.5, 0.5)
    _write_v(str(tmp_path / "era5_v700_2001_6h_region.nc"), lats, lons)
    monkeypatch.setattr(S.P, "DATA_DIR", str(tmp_path))
    lat, lon, rows, cols = S.grid(2001)
    assert lon[cols][0] == 20.0 and lon[cols][-1] == 75.0 and cols.size == 111
    assert lat[rows["southern"]][0] == 14.5 and lat[rows["southern"]][-1] == 5.0 and rows["southern"].size == 20
    assert lat[rows["northern"]][0] == 24.5 and lat[rows["northern"]][-1] == 15.0 and rows["northern"].size == 20
    short = np.arange(35.0, 9.5, -0.5)                                 # a file whose rows stop at 10 N cannot hold the southern band
    _write_v(str(tmp_path / "era5_v700_2002_6h_region.nc"), short, lons)
    with pytest.raises(SystemExit, match="20 rows"):
        S.grid(2002)


def test_figures_carry_their_windows_and_unavailable_counts_and_draw_nothing_of_the_tracker(tmp_path):
    pytest.importorskip("matplotlib")
    import datetime as dt
    S = _load()
    times = [dt.datetime(2001, 6, 1) + dt.timedelta(hours=6 * k) for k in range(488)]
    lons = np.arange(20.0, 75.5, 0.5)
    rng = np.random.default_rng(1)
    means = {}
    for b in ("southern", "northern"):
        series = rng.normal(0, 3, (488, lons.size)); series[10:14, 5] = np.nan
        means[b] = {"series": series, "support": np.ones((488, lons.size)), "unavailable": 4, "samples": series.size}
    figs = S.draw(2001, lons, times, means, str(tmp_path / "view"), "test")
    assert [f["name"] for f in figs] == ["season", "June", "July", "August", "September"]
    assert figs[0]["steps"] == 488 and figs[1]["steps"] == 120 and figs[1]["first"] == "2001-06-01T00" and figs[1]["last"] == "2001-06-30T18"
    assert figs[0]["unavailable_in_window"] == {"southern": 4, "northern": 4} and figs[2]["unavailable_in_window"] == {"southern": 0, "northern": 0}
    assert all(os.path.exists(f["path"]) and len(f["sha256"]) == 64 for f in figs)
    import matplotlib.image as mpimg
    img = mpimg.imread(figs[0]["path"])
    assert img.shape[0] >= 488 * 100 / S.STEPS_PER_INCH, "the season figure is tall enough for a four-day slope to span more than 30 pixels"
    src = open(os.path.join(ROOT, "scripts", "pilot_season_wind_view.py")).read()
    body = src.split("def draw(")[1].split("def main(")[0]
    assert "track" not in body.replace("tracker output drawn", "") and "candidate" not in body and "history" not in body
