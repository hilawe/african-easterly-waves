"""The protocol's shared entry point, on tiny synthetic years.

The manifest is validated against the code's constants and refused when it asks for
what the protocol does not support; every climatology year is validated, not only the
target year; wind files in the wrong units are refused; the calibration gate reads the
producer's own population, transformation, grid and digest fields; the inputs stage
decimates at a stride of two and reports the achieved spacing and the wind-mask fraction;
the producer record keeps protocol settings apart from dataset-specific values; the
tracking stage refuses a missing climatology year, retains a case that carries its
settings, refuses to overwrite it, and hands the retained case, the thresholds and both
flags to the tracker, whose result is what is saved.
"""
import calendar
import datetime as dt
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
pytest.importorskip("netCDF4")
pytest.importorskip("scipy")


def _load():
    path = os.environ.get("ENTRY_SCRIPT", os.path.join(ROOT, "scripts", "export_protocol_case.py"))
    spec = importlib.util.spec_from_file_location("export_protocol_case", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# THE TINY GRID must still crop to at least three coarse rows and columns: 12 N to 0 at
# one degree gives thirteen rows, the domain 1 to 11 keeps eleven fine rows and, after
# the stride of two from the buffered edge, five coarse rows
LAT = np.arange(12.0, -0.5, -1.0)
LON = np.arange(0.0, 12.5, 1.0)


def _write_year(directory, prefix, year, var, wind, units="m s**-1", lat=LAT):
    import netCDF4 as nc
    steps = (366 if calendar.isleap(year) else 365) * 4
    path = os.path.join(directory, f"{prefix}_{var}_{year}_6h_region.nc")
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("valid_time", steps)
        ds.createDimension("pressure_level", 1)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", LON.size)
        t = ds.createVariable("valid_time", "i8", ("valid_time",))
        t.units = "seconds since 1970-01-01"
        t[:] = (dt.datetime(year, 1, 1) - dt.datetime(1970, 1, 1)).total_seconds() + 21600 * np.arange(steps)
        p = ds.createVariable("pressure_level", "f8", ("pressure_level",))
        p.units = "hPa"
        p[:] = [700.0]
        la = ds.createVariable("latitude", "f8", ("latitude",))
        la[:] = lat
        lo = ds.createVariable("longitude", "f8", ("longitude",))
        lo[:] = LON
        f = ds.createVariable(var[0], "f4", ("valid_time", "pressure_level", "latitude", "longitude"))
        f.units = units
        f[:] = wind
    return path


def _manifest(tmp_path, years=(2001, 2002), **overrides):
    real = json.load(open(os.path.join(ROOT, "docs", "aewc_v2", "protocol", "manifest_2026-09-25.json")))
    m = json.loads(json.dumps(real))
    m["years"] = list(years)
    m["grid"] = {"area_nwse": [12.0, 0.0, 0.0, 12.0], "spacing_deg": 1.0, "shape": [13, 13],
                 "lat_first": 12.0, "lat_last": 0.0, "lon_first": 0.0, "lon_last": 12.0, "row_order": "descending"}
    m["domain"] = {"lat": [1.0, 11.0], "lon": [1.0, 11.0]}
    m["calibration"] = dict(real["calibration"], climatology_years=list(years), population_years=list(years))
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    m["datasets"] = {"tiny": {"directory": str(data), "prefix": "tiny", "label": "Tiny"},
                     "tiny2": {"directory": str(data), "prefix": "tiny", "label": "Tiny again"}}
    for k, v in overrides.items():
        m[k] = v
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m, indent=1))
    return path, str(data)


def _fixture_years(data, years=(2001, 2002), units="m s**-1", lat_shift=0.0):
    """Winds with a westerly band at 8 N that survives the crop, and a varying field."""
    for year in years:
        steps = 365 * 4
        base = np.arange(steps * LAT.size * LON.size, dtype=np.float32).reshape(steps, 1, LAT.size, LON.size) % 7 - 6
        u = base.copy()
        u[:, :, 4, :] = 20.0                        # the 8 N row is westerly
        lat = LAT + lat_shift if year == years[-1] else LAT
        _write_year(data, "tiny", year, "u700", u, units=units, lat=lat)
        _write_year(data, "tiny", year, "v700", base * 0.5, units=units, lat=lat)


def _calibration(tmp_path, manifest, data, prefix="tiny", years=(2001, 2002), **overrides):
    """An artifact in the producer's own layout, fitting the manifest unless overridden."""
    import hashlib
    inputs = {}
    for y in range(years[0], years[1] + 1):
        for var in ("u700", "v700"):
            p = os.path.join(data, f"{prefix}_{var}_{y}_6h_region.nc")
            if os.path.exists(p):
                inputs[os.path.basename(p)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    g = manifest["grid"]
    art = {"schema": "thresholds-v2", "prefix": prefix, "climatology_years": list(years), "population_years": list(years),
           "percentiles": {"coarse": 55.0, "fine": 66.0}, "estimator": "exact", "percentile_method": "linear",
           "transformation": {"id": "T0", "smoothing_passes": 1, "coarse_scale": None},
           "population": {"wind_mask": "none", "smoothing_passes": 1, "hemispheres": "both"},
           "coarse_resolution": 2.5, "domain": {"lat_range": list(manifest["domain"]["lat"]), "lon_range": list(manifest["domain"]["lon"])},
           "preflight": {"n_lat": g["shape"][0], "n_lon": g["shape"][1], "lat_first": g["lat_first"], "lat_last": g["lat_last"],
                         "lon_first": g["lon_first"], "lon_last": g["lon_last"], "level_hpa": 700.0},
           "subsample_stride": 1, "coarse": {"threshold": 1e-9}, "fine": {"threshold": 2e-9},
           "input_file_sha256": inputs}
    for k, v in overrides.items():
        art[k] = v
    path = tmp_path / f"cal_{prefix}_{years[0]}_{abs(hash(json.dumps(overrides, sort_keys=True, default=str)))}.json"
    path.write_text(json.dumps(art))
    return path


def test_the_manifest_is_validated_against_the_code_and_refused_otherwise(tmp_path):
    E = _load()
    path, _ = _manifest(tmp_path)
    with pytest.raises(SystemExit):                    # the tiny domain is not the code's
        E.load_manifest(str(path))
    real = json.load(open(os.path.join(ROOT, "docs", "aewc_v2", "protocol", "manifest_2026-09-25.json")))
    (tmp_path / "ok").mkdir()
    path, _ = _manifest(tmp_path / "ok", domain=real["domain"])
    manifest, digest = E.load_manifest(str(path))
    assert len(digest) == 64 and manifest["years"] == [2001, 2002]
    with pytest.raises(SystemExit):
        E.dataset_entry(manifest, "era5")
    bad = [("hours", ["12:00"]), ("domain", {"lat": [-30.0, 30.0], "lon": [-140.0, 40.0]}),
           ("sign_convention", "no reversal"), ("initialization", "warm start on June 1, both datasets"),
           ("smoothing", {"passes": 1, "where": "before cropping"}),
           ("calibration", dict(real["calibration"], estimator="exact, nearest")),
           ("membership", dict(real["membership"], priority=["AFR"])),
           ("membership", dict(real["membership"], rule="the last observation inside the outline")),
           ("decimation", {"nominal_deg": 2.5, "stride_on_one_degree_input": 9, "achieved_spacing_deg": 2.0, "order": "x"}),
           ("smoothing", {"passes": 2, "where": "x"}), ("tracker_flags", {"exclusive": True}),
           ("wind_mask", "everywhere")]
    for n, (key, value) in enumerate(bad):
        key_dir = tmp_path / f"bad_{n}_{key}"
        key_dir.mkdir()
        p, _ = _manifest(key_dir, domain=real["domain"], **({key: value} if key != "domain" else {}))
        if key == "domain":
            p, _ = _manifest(key_dir, domain=value)
        with pytest.raises(SystemExit):
            E.load_manifest(str(p))


def _accepted(tmp_path, E, years=(2001, 2002)):
    """A manifest the validator accepts on the tiny grid: the code's domain is far outside
    the tiny grid, so the tests that need the tiny domain bypass load_manifest's domain
    check by validating everything else and reading the file directly."""
    path, data = _manifest(tmp_path, years=years)
    m = json.load(open(path))
    problems = [p for p in E.validate_manifest(m) if "domain" not in p]
    assert problems == []
    import hashlib
    return m, hashlib.sha256(path.read_bytes()).hexdigest(), data


def test_every_climatology_year_and_the_wind_units_are_validated(tmp_path):
    E = _load()
    manifest, _, data = _accepted(tmp_path, E)
    _fixture_years(data, lat_shift=10.0)                       # the second year's grid is shifted
    dataset = manifest["datasets"]["tiny"]
    preflight, digests = E.validate_inputs(manifest, dataset, [2001])
    assert set(digests) == {"tiny_u700_2001_6h_region.nc", "tiny_v700_2001_6h_region.nc"}
    with pytest.raises((SystemExit, ValueError)):
        E.validate_inputs(manifest, dataset, [2001, 2002])
    (tmp_path / "knots").mkdir()
    manifest2, _, data2 = _accepted(tmp_path / "knots", E)
    _fixture_years(data2, units="knots")
    with pytest.raises(SystemExit):
        E.validate_inputs(manifest2, manifest2["datasets"]["tiny"], [2001])


def test_calibration_must_carry_the_producers_population_grid_and_digests(tmp_path):
    E = _load()
    manifest, _, data = _accepted(tmp_path, E)
    _fixture_years(data)
    dataset = manifest["datasets"]["tiny"]
    ct, ft, digest, inputs = E.check_calibration(str(_calibration(tmp_path, manifest, data)), manifest, dataset)
    assert (ct, ft) == (1e-9, 2e-9) and len(digest) == 64 and len(inputs) == 4
    refusals = [dict(prefix="eraint"), dict(climatology_years=[1981, 2010]),
                dict(percentiles={"coarse": 50.0, "fine": 66.0}),
                dict(transformation={"id": "T1", "smoothing_passes": 0, "coarse_scale": 0.9}),
                dict(population={"wind_mask": "applied", "smoothing_passes": 1, "hemispheres": "both"}),
                dict(domain={"lat_range": [-30.0, 30.0], "lon_range": [-140.0, 40.0]}),
                dict(percentile_method="nearest"), dict(input_file_sha256={}),
                dict(input_file_sha256={"tiny_u700_2001_6h_region.nc": "0" * 64 + "\n", "tiny_v700_2001_6h_region.nc": "0" * 64,
                                        "tiny_u700_2002_6h_region.nc": "0" * 64, "tiny_v700_2002_6h_region.nc": "0" * 64}),
                dict(input_file_sha256={"tiny_u700_2001_6h_region.nc": None, "tiny_v700_2001_6h_region.nc": "0" * 64,
                                        "tiny_u700_2002_6h_region.nc": "0" * 64, "tiny_v700_2002_6h_region.nc": "0" * 64}),
                dict(coarse={"threshold": float("nan")}), dict(subsample_stride=4)]
    for override in refusals:
        with pytest.raises(SystemExit):
            E.check_calibration(str(_calibration(tmp_path, manifest, data, **override)), manifest, dataset)


def test_inputs_stage_decimates_and_reports_readiness_and_the_record_separates_settings(tmp_path):
    E = _load()
    manifest, digest, data = _accepted(tmp_path, E)
    _fixture_years(data)
    dataset = manifest["datasets"]["tiny"]
    preflight, digests = E.validate_inputs(manifest, dataset, [2001])
    readiness, arrays = E.inputs_stage(manifest, dataset, 2001, preflight)
    assert readiness["timesteps"] == 1460 and readiness["native_spacing_deg"] == 1.0
    assert readiness["achieved_coarse_spacing_deg"] == 2.0                # stride two, not 2.5
    assert readiness["fine_shape"] == [11, 11] and readiness["coarse_shape"] == [5, 5]
    assert readiness["finite_fraction"]["u"] == 1.0 and 0.0 < readiness["wind_mask_fraction_coarse"] < 1.0
    record = E.producer_record(manifest, digest, "tiny", dataset, 2001, "inputs", digests, {"readiness": readiness})
    assert record["protocol_settings"]["manifest_sha256"] == digest
    assert set(record["protocol_settings"]) == set(E.PROTOCOL_KEYS) | {"manifest_sha256"}
    assert record["dataset_specific"]["dataset"] == "tiny" and "manifest_sha256" not in record["dataset_specific"]
    assert "scripts/export_protocol_case.py" in record["source_sha256"]


def test_tracking_refuses_a_missing_year_retains_a_case_with_its_settings_and_saves_what_the_tracker_returns(tmp_path, monkeypatch):
    E = _load()
    manifest, digest, data = _accepted(tmp_path, E, years=(2001, 2003))         # 2003 is missing
    _fixture_years(data)
    dataset = manifest["datasets"]["tiny"]
    preflight, digests = E.validate_inputs(manifest, dataset, [2001])
    readiness, arrays = E.inputs_stage(manifest, dataset, 2001, preflight)
    cal = _calibration(tmp_path, manifest, data, years=(2001, 2003))
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(SystemExit):
        E.tracking_stage(manifest, digest, "tiny", dataset, 2001, arrays, str(tmp_path / "climo"), str(cal), str(run), digests, readiness)
    # a complete pair of years, with the tracker faked to record its call and return one track
    (tmp_path / "second").mkdir()
    manifest2, digest2, data2 = _accepted(tmp_path / "second", E)
    _fixture_years(data2)
    dataset2 = manifest2["datasets"]["tiny"]
    preflight2, digests2 = E.validate_inputs(manifest2, dataset2, [2001])
    readiness2, arrays2 = E.inputs_stage(manifest2, dataset2, 2001, preflight2)
    cal2 = _calibration(tmp_path / "second", manifest2, data2)
    seen = {}
    expected_track = {"time": np.array([1.0, 1.25]), "meanlat": np.array([5.0, 5.5]), "meanlon": np.array([3.0, 2.5])}

    def fake_track_case(payload, ct, ft, exclusive, absorb):
        seen.update({"case_id": payload["case_id"], "ct": ct, "ft": ft, "exclusive": exclusive, "absorb": absorb,
                     "steps": int(np.asarray(payload["time"]).size), "coarse_rows": int(np.asarray(payload["lat_c"]).size)})
        return [expected_track]
    monkeypatch.setattr(E, "track_case", fake_track_case)
    run2 = tmp_path / "second" / "run"
    run2.mkdir()
    record, n = E.tracking_stage(manifest2, digest2, "tiny", dataset2, 2001, arrays2, str(tmp_path / "second" / "climo"),
                                 str(cal2), str(run2), digests2, readiness2)
    assert n == 1
    assert seen == {"case_id": record["dataset_specific"]["case_id"], "ct": 1e-9, "ft": 2e-9,
                    "exclusive": False, "absorb": False, "steps": 1460, "coarse_rows": 5}
    from scipy.io import loadmat
    case = loadmat(str(run2 / "tracker_case.mat"))
    inner = json.loads(str(np.asarray(case["producer_json"]).ravel()[0]))
    assert inner["protocol_settings"]["manifest_sha256"] == digest2
    assert inner["dataset_specific"]["calibration_sha256"] and inner["dataset_specific"]["coarse_threshold"] == 1e-9
    assert set(inner["dataset_specific"]["climatology_inputs_sha256"]) == {f"tiny_{v}_{y}_6h_region.nc" for y in (2001, 2002) for v in ("u700", "v700")}
    port = loadmat(str(run2 / "tracker_port.mat"))
    assert int(float(np.asarray(port["n"]).ravel()[0])) == 1
    assert np.array_equal(np.asarray(port["lat0"]).ravel(), expected_track["meanlat"])
    assert record["dataset_specific"]["case_sha256"] and record["dataset_specific"]["tracks_sha256"]
    # THE PHASES ARE TIMED in the produced record, each nonnegative and named
    phases = record["dataset_specific"]["phase_seconds"]
    assert set(phases) == {"validate_climatology_inputs_seconds", "climatology_load_seconds",
                           "anomaly_advection_and_case_seconds", "tracking_seconds"}
    assert all(isinstance(v, float) and v >= 0 for v in phases.values())     # tiny fixtures round to 0.0
    # nothing at a final path is overwritten, and the record is published exclusively
    with pytest.raises(SystemExit):
        E.tracking_stage(manifest2, digest2, "tiny", dataset2, 2001, arrays2, str(tmp_path / "second" / "climo"),
                         str(cal2), str(run2), digests2, readiness2)
    E.publish_record(str(run2 / "tracking_tiny_2001.json"), record)
    with pytest.raises(SystemExit):
        E.publish_record(str(run2 / "tracking_tiny_2001.json"), record)
    # a calibration whose digests do not match the climatology's bytes is refused at tracking
    (tmp_path / "third").mkdir()
    manifest3, digest3, data3 = _accepted(tmp_path / "third", E)
    _fixture_years(data3)
    dataset3 = manifest3["datasets"]["tiny"]
    preflight3, digests3 = E.validate_inputs(manifest3, dataset3, [2001])
    readiness3, arrays3 = E.inputs_stage(manifest3, dataset3, 2001, preflight3)
    wrong = {f"tiny_{v}_{y}_6h_region.nc": "1" * 64 for y in (2001, 2002) for v in ("u700", "v700")}
    cal3 = _calibration(tmp_path / "third", manifest3, data3, input_file_sha256=wrong)
    run3 = tmp_path / "third" / "run"
    run3.mkdir()
    with pytest.raises(SystemExit):
        E.tracking_stage(manifest3, digest3, "tiny", dataset3, 2001, arrays3, str(tmp_path / "third" / "climo"),
                         str(cal3), str(run3), digests3, readiness3)


def test_the_same_winds_under_two_dataset_entries_give_the_same_arrays(tmp_path):
    E = _load()
    manifest, _, data = _accepted(tmp_path, E)
    _fixture_years(data)
    outs = {}
    for name in ("tiny", "tiny2"):
        dataset = manifest["datasets"][name]
        preflight, _ = E.validate_inputs(manifest, dataset, [2001])
        readiness, arrays = E.inputs_stage(manifest, dataset, 2001, preflight)
        outs[name] = (readiness, arrays)
    assert outs["tiny"][0] == outs["tiny2"][0]
    for name in ("u", "v", "curvature"):
        assert np.array_equal(outs["tiny"][1]["coarse"][name], outs["tiny2"][1]["coarse"][name], equal_nan=True)
