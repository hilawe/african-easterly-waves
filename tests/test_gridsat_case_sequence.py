"""The case-sized GridSat-B1 sequence: bound to the product and observation, read from
retained imagery, missing coverage preserved, the 1979 gap kept, and readings tagged
before or after each storm event."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    path = os.environ.get("GRIDSAT_SEQUENCE_SCRIPT", os.path.join(ROOT, "scripts", "gridsat_case_sequence.py"))
    spec = importlib.util.spec_from_file_location("gridsat_case_sequence_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _imagery(tmp_path, stamps, cold_at=None, internal_time=None):
    """Synthetic cropped GridSat files at 0.5 degrees over 0 to 20 N, 30 W to 0, 300 K
    everywhere except a cold patch, with a missing strip. `internal_time` names a file's
    own timestamp when it should differ from its name."""
    import xarray as xr
    d = tmp_path / "imagery"
    d.mkdir()
    lat = np.arange(0.0, 20.5, 0.5)
    lon = np.arange(-30.0, 0.5, 0.5)
    for stamp in stamps:
        tb = np.full((1, lat.size, lon.size), 300.0)
        if cold_at is not None:
            la, lo = cold_at
            tb[0, (lat >= la - 1) & (lat <= la + 1), :][:, (lon >= lo - 1) & (lon <= lo + 1)] = 210.0
            i = (lat >= la - 1) & (lat <= la + 1)
            j = (lon >= lo - 1) & (lon <= lo + 1)
            tb[0][np.ix_(i, j)] = 210.0
        tb[0, :, (lon >= -22.0) & (lon <= -21.5)] = np.nan                  # a missing strip inside a 5-degree box at 20 W
        inner = internal_time or f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:00"
        t = np.array([np.datetime64(inner)])
        xr.Dataset({"irwin_cdr": (("time", "lat", "lon"), tb)}, coords={"time": t, "lat": lat, "lon": lon}).to_netcdf(d / f"gridsat_{stamp}.nc")
    return str(d)


def _product(tmp_path, year, track):
    from scipy.io import savemat
    root = tmp_path / "product"
    (root / "tracks").mkdir(parents=True)
    rel = f"tracks/era5_{year}_tracks.mat"
    savemat(str(root / rel), {"n": 1.0, "case_id": "c" * 32, "lat0": track["lat"], "lon0": track["lon"], "time0": track["time"]})
    files = {rel: {"sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()}}
    (root / "MANIFEST.json").write_text(json.dumps({"product": "test-product", "version": "0.0.1", "files": files}))
    return str(root)


def test_nearest_gridsat_time_rounds_to_three_hours():
    G = _load()
    t = G.nearest_gridsat_time((np.datetime64("1990-08-20T06:00") - np.datetime64("1900-01-01T00:00")) / np.timedelta64(1, "D"))
    assert t.strftime("%Y%m%d%H") == "1990082006"
    t = G.nearest_gridsat_time((np.datetime64("1990-08-20T13:00") - np.datetime64("1900-01-01T00:00")) / np.timedelta64(1, "D"))
    assert t.strftime("%Y%m%d%H") == "1990082012"


def test_the_sequence_reads_retained_imagery_keeps_gaps_and_tags_storm_events(tmp_path):
    G = _load()
    day0 = float((np.datetime64("1990-08-20T00:00") - np.datetime64("1900-01-01T00:00")) / np.timedelta64(1, "D"))
    track = {"time": day0 + 0.25 * np.arange(4), "lat": np.full(4, 10.0), "lon": -20.0 - 0.5 * np.arange(4)}
    imagery = _imagery(tmp_path, ["1990082000", "1990082006"], cold_at=(10.0, -20.0))   # the second and fourth steps are not covered
    product = _product(tmp_path, 1990, track)
    storm = {"sid": "S1", "first_archived": {"time": "1990-08-20T06:00:00Z"}, "first_depression_or_stronger": {"time": "1990-08-20T12:00:00Z"},
             "first_tropical_storm": None}
    out = G.sequence(product, 1990, 0, imagery, storm=storm)
    assert out["observations"] == 4 and out["covered"] == 2 and out["fetched"] == 0 and out["storm"] == "S1"
    seq = out["sequence"]
    assert [o["coverage"] for o in seq] == ["retained imagery", "retained imagery", "not covered", "not covered"]
    assert seq[0]["gridsat_time"] == "1990-08-20T00:00Z" and seq[1]["gridsat_time"] == "1990-08-20T06:00Z"
    assert seq[0]["min_k"] == 210.0 and 0 < seq[0]["cold_fraction"] < 1 and seq[0]["missing_fraction"] > 0
    assert seq[0]["cold_cells"] == 25 and seq[0]["cold_cells"] == round(seq[0]["cold_fraction"] * (seq[0]["cells"] * (1 - seq[0]["missing_fraction"])))   # a 5 by 5 patch at 0.5 degrees
    assert seq[0]["spatial_coverage"] == "complete" and seq[0]["covered_fraction_of_requested_box"] == 1.0
    assert seq[0]["file_sha256"] == hashlib.sha256(open(os.path.join(imagery, "gridsat_1990082000.nc"), "rb").read()).hexdigest()
    assert seq[0]["storm_events"] == {"first_archived": "before", "first_depression_or_stronger": "before", "first_tropical_storm": "the storm has no such event"}
    assert seq[1]["storm_events"]["first_archived"] == "at or after" and seq[1]["storm_events"]["first_depression_or_stronger"] == "before"
    assert seq[2]["storm_events"]["first_depression_or_stronger"] == "at or after"
    assert out["link"] == {"product": "test-product", "version": "0.0.1", "track_file": "tracks/era5_1990_tracks.mat",
                           "track_file_sha256": json.load(open(os.path.join(product, "MANIFEST.json")))["files"]["tracks/era5_1990_tracks.mat"]["sha256"],
                           "track_index": 0}
    assert "never a requirement" in out["readings_are"]


def test_1979_is_a_gap_and_a_changed_product_file_is_refused(tmp_path):
    G = _load()
    day0 = float((np.datetime64("1979-08-20T00:00") - np.datetime64("1900-01-01T00:00")) / np.timedelta64(1, "D"))
    track = {"time": day0 + 0.25 * np.arange(2), "lat": np.full(2, 10.0), "lon": np.array([-20.0, -20.5])}
    imagery = _imagery(tmp_path, [])
    product = _product(tmp_path, 1979, track)
    out = G.sequence(product, 1979, 0, imagery, fetch_limit=5)
    assert [o["coverage"] for o in out["sequence"]] == ["no GridSat-B1 before 1980"] * 2 and out["fetched"] == 0
    with open(os.path.join(product, "tracks", "era5_1979_tracks.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="does not have the digest the product manifest lists"):
        G.sequence(product, 1979, 0, imagery)


def test_a_fetch_is_bounded_and_never_fills(tmp_path, monkeypatch):
    G = _load()
    day0 = float((np.datetime64("1990-06-20T00:00") - np.datetime64("1900-01-01T00:00")) / np.timedelta64(1, "D"))
    track = {"time": day0 + 0.25 * np.arange(3), "lat": np.full(3, 10.0), "lon": -20.0 - 0.5 * np.arange(3)}
    imagery = _imagery(tmp_path, [])
    product = _product(tmp_path, 1990, track)
    calls = []

    def fake_fetch(when, box, out_dir):
        calls.append(when)
        scratch = tmp_path / f"f{len(calls)}"
        scratch.mkdir()
        made = _imagery(scratch, [when.strftime("%Y%m%d%H")])
        path = os.path.join(out_dir, f"gridsat_{when:%Y%m%d%H}.nc")
        os.rename(os.path.join(made, f"gridsat_{when:%Y%m%d%H}.nc"), path)
        return path
    monkeypatch.setattr(G, "fetch_missing", fake_fetch)
    out = G.sequence(product, 1990, 0, imagery, fetch_limit=2)
    assert out["fetched"] == 2 and len(calls) == 2
    assert [o["coverage"] for o in out["sequence"]] == ["retained imagery", "retained imagery", "not covered"]


def test_a_box_beyond_the_crop_reports_partial_spatial_coverage_apart_from_missing_pixels(tmp_path):
    """The reproduced gap: a box reaching past the retained crop was read as a smaller box
    with no sign of it. The clipped width per side and the covered fraction are reported,
    and the missing share counts only the returned pixels."""
    G = _load()
    day0 = float((np.datetime64("1990-08-20T00:00") - np.datetime64("1900-01-01T00:00")) / np.timedelta64(1, "D"))
    track = {"time": day0 + 0.25 * np.arange(2), "lat": np.array([18.0, 10.0]), "lon": np.array([-3.0, -20.0])}
    imagery = _imagery(tmp_path, ["1990082000", "1990082006"])            # the crop ends at 20 N and 0 E
    out = G.sequence(_product(tmp_path, 1990, track), 1990, 0, imagery)
    edge, interior = out["sequence"]
    assert edge["spatial_coverage"] == "partial" and edge["clipped_deg"] == {"north": 3.0, "south": 0.0, "east": 2.0, "west": 0.0}
    assert edge["covered_fraction_of_requested_box"] == round(0.7 * 0.8, 4) and edge["box_lat"][1] == 20.0 and edge["box_lon"][1] == 0.0
    assert edge["missing_fraction"] == 0.0                                  # no missing pixel among those returned
    assert interior["spatial_coverage"] == "complete" and interior["covered_fraction_of_requested_box"] == 1.0
    assert interior["missing_fraction"] > 0.0                               # the strip inside the box, a different thing


def test_a_file_whose_internal_timestamp_is_not_the_requested_one_yields_no_reading(tmp_path):
    """The reproduced gap: a file named for 00Z holding 03Z was read and both times merely
    recorded. The file's timestamp is checked against the requested timestep."""
    G = _load()
    day0 = float((np.datetime64("1990-08-20T00:00") - np.datetime64("1900-01-01T00:00")) / np.timedelta64(1, "D"))
    track = {"time": day0 + 0.25 * np.arange(1), "lat": np.array([10.0]), "lon": np.array([-20.0])}
    imagery = _imagery(tmp_path, ["1990082000"], internal_time="1990-08-20T03:00")
    out = G.sequence(_product(tmp_path, 1990, track), 1990, 0, imagery)
    o = out["sequence"][0]
    assert o["coverage"] == "file timestamp does not match the requested timestep" and out["covered"] == 0
    assert o["gridsat_time"] == "1990-08-20T00:00Z" and o["file_time"] == "1990-08-20T03:00:00Z"
    assert "min_k" not in o and "cold_fraction" not in o


def test_a_box_wholly_outside_the_crop_reports_no_coverage_and_no_readings_instead_of_failing(tmp_path):
    """The counterexample from the 2006 in-season audit: a track south of the retained
    crop's 25 S edge asked for a box the file does not reach on either row."""
    G = _load()
    imagery = _imagery(tmp_path, ["1990080100"])                              # the crop is 0 to 20 N, 30 W to 0
    tb, lats, lons, when, cov = G.read_box(os.path.join(imagery, "gridsat_1990080100.nc"), -30.0, -15.0)
    assert tb.shape == (0, 21) and lats.size == 0 and lons.size == 21                      # 20 W to 10 W is inside the crop, no row is
    assert cov["spatial_coverage"] == "none" and cov["covered_fraction_of_requested_box"] == 0.0 and cov["clipped_deg"]["south"] == 35.0
    assert G.readings(tb) == {"min_k": None, "cold_fraction": None, "cold_cells": 0, "missing_fraction": None, "cells": 0}
    tb, lats, lons, when, cov = G.read_box(os.path.join(imagery, "gridsat_1990080100.nc"), 10.0, 40.0)   # east of the crop on every column
    assert tb.shape == (21, 0) and cov["spatial_coverage"] == "none"
