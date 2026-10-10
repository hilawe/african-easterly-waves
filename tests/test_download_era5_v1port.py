"""The corrected ERA5 retrieval: the request carries the buffered one-degree grid, the
level and the synoptic hours; the expected grid is 101 by 211; an existing file is
validated and not trusted, and one on the wrong grid or without its level is moved
aside (or only reported under a dry run); a valid file passes; and a completion record
carries the file's digest."""
import calendar
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


def _load():
    spec = importlib.util.spec_from_file_location(
        "download_era5_v1port", os.environ.get("ERA5_SCRIPT", os.path.join(ROOT, "scripts", "download_era5_v1port.py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_year(path, year, lat, lon, level=700.0, var="u", level_units="hPa", fill=False,
                time_units="seconds since 1970-01-01", extra_dim=None, wind_units="m s**-1", constant=False,
                float_time=False, mask_one_time=False, two_level_wind=False, mask_level=False):
    """A full six-hourly year of one wind component, as CDS writes it, on a tiny grid."""
    import netCDF4 as nc
    steps = (366 if calendar.isleap(year) else 365) * 4
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("valid_time", steps)
        ds.createDimension("pressure_level", 1)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        if extra_dim:
            ds.createDimension(extra_dim, 1)
        if two_level_wind:
            ds.createDimension("plev2", 2)
        t = ds.createVariable("valid_time", "f8" if float_time else "i8", ("valid_time",),
                              fill_value=(-1 if mask_one_time else None))
        t.units = time_units
        t.calendar = "proleptic_gregorian"
        import datetime as dt
        jan1 = (dt.datetime(year, 1, 1) - dt.datetime(1970, 1, 1)).total_seconds()
        t[:] = jan1 + 21600 * np.arange(steps)
        if mask_one_time:
            t[7] = np.ma.masked
        p = ds.createVariable("pressure_level", "f8", ("pressure_level",), fill_value=(-1.0 if mask_level else None))
        p.units = level_units
        p[:] = [level]
        if mask_level:
            p[0] = np.ma.masked
        la = ds.createVariable("latitude", "f8", ("latitude",))
        la.units = "degrees_north"
        la[:] = lat
        lo = ds.createVariable("longitude", "f8", ("longitude",))
        lo.units = "degrees_east"
        lo[:] = lon
        dims = ("valid_time", "plev2" if two_level_wind else (extra_dim or "pressure_level"), "latitude", "longitude")
        u = ds.createVariable(var, "f4", dims, fill_value=-32767.0)
        if wind_units:
            u.units = wind_units
        if fill:
            pass                                       # every value stays the fill value
        elif constant:
            u[:] = 1.5
        else:
            u[:] = np.arange(steps * lat.size * lon.size, dtype=np.float32).reshape(steps, 1, lat.size, lon.size) % 5 - 2


def test_the_request_is_the_protocols_grid_level_and_hours():
    D = _load()
    assert D.AREA == [50.0, -155.0, -50.0, 55.0] and D.GRID == (1.0, 1.0)
    dataset, request = D.request_for(1990, "u700")
    assert dataset == "reanalysis-era5-pressure-levels"
    assert request["area"] == [50.0, -155.0, -50.0, 55.0] and request["grid"] == [1.0, 1.0]
    assert request["pressure_level"] == "700" and request["variable"] == "u_component_of_wind"
    assert request["time"] == ["00:00", "06:00", "12:00", "18:00"] and len(request["month"]) == 12
    lat, lon = D.expected_coordinates(D.AREA_TEXT, D.GRID_TEXT)
    assert (lat.size, lon.size) == (101, 211) == D.EXPECTED_SHAPE
    assert lat[0] == 50.0 and lat[-1] == -50.0 and lon[0] == -155.0 and lon[-1] == 55.0


def test_existing_files_are_validated_not_trusted(tmp_path, monkeypatch):
    D = _load()
    # a tiny area so a full year fits in a test: 2 N to 0 and 0 to 2 E at one degree, 3 by 3
    monkeypatch.setattr(D, "AREA", [2.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(D, "AREA_TEXT", "2/0/0/2")
    lat, lon = D.expected_coordinates("2/0/0/2", D.GRID_TEXT)
    good = D.output_path(1990, "u700", str(tmp_path))
    _write_year(good, 1990, lat, lon)
    shifted = D.output_path(1991, "u700", str(tmp_path))
    _write_year(shifted, 1991, lat + 0.5, lon)
    no_level = D.output_path(1992, "u700", str(tmp_path))
    _write_year(no_level, 1992, lat, lon, level=850.0)
    todo, done = D.plan([1990, 1991, 1992, 1993], ["u700"], str(tmp_path), repair=False)
    assert [y for y, _, _ in done] == [1990]
    assert [y for y, _, _ in todo] == [1991, 1992, 1993]
    assert os.path.exists(shifted) and os.path.exists(no_level)      # a dry run moves nothing
    todo, done = D.plan([1990, 1991, 1992], ["u700"], str(tmp_path), repair=True)
    assert not os.path.exists(shifted) and os.path.exists(shifted + ".invalid")
    assert not os.path.exists(no_level) and os.path.exists(no_level + ".invalid")
    assert [y for y, _, _ in todo] == [1991, 1992]


def test_completion_record_carries_the_digest_and_the_request(tmp_path, monkeypatch):
    D = _load()
    monkeypatch.setattr(D, "AREA", [2.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(D, "AREA_TEXT", "2/0/0/2")
    lat, lon = D.expected_coordinates("2/0/0/2", D.GRID_TEXT)
    path = D.output_path(1990, "v700", str(tmp_path))
    _write_year(path, 1990, lat, lon, var="v")
    dataset, request = D.request_for(1990, "v700")
    record = D.completion_record(path, 1990, "v700", 0.0, 12.5, dataset, request)
    saved = json.load(open(path + ".completion.json"))
    import hashlib
    assert saved["sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest() == record["sha256"]
    assert saved["bytes"] == os.path.getsize(path) and saved["request"]["grid"] == [1.0, 1.0]


class _FakeClient:
    """Stands in for cdsapi: writes what it is told to at the target it is handed, and
    records that target, so a test can require the partial path."""

    def __init__(self, behavior):
        self.behavior, self.targets = behavior, []

    def retrieve(self, dataset, request, target):
        self.targets.append(target)
        year, var = int(request["year"]), "u700" if "u_" in request["variable"] else "v700"
        lat, lon = _TINY
        if self.behavior == "valid":
            _write_year(target, year, lat, lon, var=var[0])
        elif self.behavior == "interrupted":
            with open(target, "wb") as fh:
                fh.write(b"\x89HDF fragment")
            raise RuntimeError("connection lost")
        elif self.behavior == "shifted":
            _write_year(target, year, lat + 0.5, lon, var=var[0])
        elif self.behavior == "fill":
            _write_year(target, year, lat, lon, var=var[0], fill=True)


_TINY = None


def _tiny(monkeypatch, D):
    global _TINY
    monkeypatch.setattr(D, "AREA", [2.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(D, "AREA_TEXT", "2/0/0/2")
    monkeypatch.setattr(D, "EXPECTED_SHAPE", (3, 3))
    _TINY = D.expected_coordinates("2/0/0/2", D.GRID_TEXT)
    return _TINY


def test_main_downloads_to_the_partial_path_and_promotes_only_a_valid_file(tmp_path, monkeypatch):
    D = _load()
    _tiny(monkeypatch, D)
    client = _FakeClient("valid")
    rc = D.main(["--years", "1990", "--variables", "u700", "--out-dir", str(tmp_path)], client=client)
    assert rc == 0
    assert client.targets == [D.output_path(1990, "u700", str(tmp_path)) + ".partial"]
    final = D.output_path(1990, "u700", str(tmp_path))
    assert os.path.exists(final) and not os.path.exists(final + ".partial")
    record = json.load(open(final + ".completion.json"))
    assert record["sha256"] and record["reconciled_at_restart"] is False


def test_main_leaves_no_final_file_after_an_interrupted_or_invalid_transfer(tmp_path, monkeypatch):
    D = _load()
    _tiny(monkeypatch, D)
    rc = D.main(["--years", "1990", "--variables", "u700", "--out-dir", str(tmp_path)], client=_FakeClient("interrupted"))
    final = D.output_path(1990, "u700", str(tmp_path))
    assert rc == 1 and not os.path.exists(final) and not os.path.exists(final + ".partial")
    assert os.path.exists(final + ".interrupted")
    rc = D.main(["--years", "1991", "--variables", "u700", "--out-dir", str(tmp_path)], client=_FakeClient("shifted"))
    final = D.output_path(1991, "u700", str(tmp_path))
    assert rc == 1 and not os.path.exists(final) and os.path.exists(final + ".invalid")
    rc = D.main(["--years", "1992", "--variables", "u700", "--out-dir", str(tmp_path)], client=_FakeClient("fill"))
    final = D.output_path(1992, "u700", str(tmp_path))
    assert rc == 1 and not os.path.exists(final) and os.path.exists(final + ".invalid")


def test_a_valid_file_without_a_record_is_reconciled_at_restart(tmp_path, monkeypatch):
    D = _load()
    lat, lon = _tiny(monkeypatch, D)
    final = D.output_path(1990, "v700", str(tmp_path))
    _write_year(final, 1990, lat, lon, var="v")
    todo, done = D.plan([1990], ["v700"], str(tmp_path), repair=False)
    assert done and not os.path.exists(final + ".completion.json")
    rc = D.main(["--years", "1990", "--variables", "v700", "--out-dir", str(tmp_path)], client=_FakeClient("valid"))
    assert rc == 0
    record = json.load(open(final + ".completion.json"))
    assert record["reconciled_at_restart"] is True and record["elapsed_seconds"] is None
    import hashlib
    assert record["sha256"] == hashlib.sha256(open(final, "rb").read()).hexdigest()


def test_the_validator_refuses_fill_payload_nan_times_a_wrong_epoch_odd_dimensions_and_no_units(tmp_path, monkeypatch):
    D = _load()
    lat, lon = _tiny(monkeypatch, D)
    import netCDF4 as nc
    p = str(tmp_path / "fill.nc")
    _write_year(p, 1990, lat, lon, fill=True)
    assert "masked or nonfinite" in D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    p = str(tmp_path / "nan_times.nc")
    _write_year(p, 1990, lat, lon, float_time=True)
    with nc.Dataset(p, "a") as ds:
        ds["valid_time"][:] = np.full(ds["valid_time"].shape, np.nan)       # every comparison with NaN is false
    assert "finite" in D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    p = str(tmp_path / "feb_epoch.nc")
    _write_year(p, 1990, lat, lon, time_units="seconds since 1970-02-01")
    assert "epoch" in D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    p = str(tmp_path / "member.nc")
    _write_year(p, 1990, lat, lon, extra_dim="member")
    assert "dimensions" in D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    p = str(tmp_path / "no_units.nc")
    _write_year(p, 1990, lat, lon, wind_units=None)
    assert "units" in D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    p = str(tmp_path / "constant.nc")
    _write_year(p, 1990, lat, lon, constant=True)
    assert "constant" in D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    # a masked timestamp keeps its fill number under the mask, and is refused as masked
    p = str(tmp_path / "masked_time.nc")
    _write_year(p, 1990, lat, lon, mask_one_time=True)
    assert "masked" in D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    # a masked pressure coordinate cannot establish the level
    p = str(tmp_path / "masked_level.nc")
    _write_year(p, 1990, lat, lon, mask_level=True)
    assert "cannot be established" in D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    # a two-level wind beside a singleton level coordinate is refused by dimension binding
    p = str(tmp_path / "two_level.nc")
    _write_year(p, 1990, lat, lon, two_level_wind=True)
    reason = D.validate_file(p, 1990, "u700", D.AREA_TEXT, D.GRID_TEXT)
    assert reason is not None and ("length" in reason or "dimensions" in reason or "shape" in reason)


def test_a_variant_area_is_a_named_option_that_needs_its_own_directory(tmp_path, monkeypatch, capsys):
    """The eastern extension's retrieval: the area is given on the command line, the
    expected shape follows from it, and the production directory is never its target."""
    D = _load()
    _tiny(monkeypatch, D)
    monkeypatch.setattr(D, "EXPECTED_SHAPE", (101, 211))               # the protocol's shape, which the tiny area is not
    with pytest.raises(SystemExit, match="not the protocol's"):
        D.main(["--years", "1990", "--variables", "u700", "--out-dir", str(tmp_path)], client=_FakeClient("valid"))
    with pytest.raises(SystemExit, match="needs its own --out-dir"):
        D.main(["--years", "1990", "--variables", "u700", "--area", "2", "0", "0", "2"], client=_FakeClient("valid"))
    client = _FakeClient("valid")
    rc = D.main(["--years", "1990", "--variables", "u700", "--out-dir", str(tmp_path), "--area", "2", "0", "0", "2"], client=client)
    out = capsys.readouterr().out
    assert rc == 0 and "declared variant area" in out and len(client.targets) == 1, out
    record = json.load(open(tmp_path / "era5_u700_1990_6h_region.nc.completion.json"))
    assert record["request"]["area"] == [2.0, 0.0, 0.0, 2.0]


def test_the_production_directory_is_refused_under_every_spelling_and_nothing_persists(tmp_path, monkeypatch, capsys):
    """A review bypassed the guard with ./, a trailing slash, an absolute path and a
    symbolic link, and found a variant call's area persisting into a later default call."""
    D = _load()
    _tiny(monkeypatch, D)
    monkeypatch.chdir(tmp_path)
    production = tmp_path / "data" / "era5" / "v1port_buffered"
    production.mkdir(parents=True)
    monkeypatch.setattr(D, "OUT_DIR", "data/era5/v1port_buffered")
    os.symlink(production, tmp_path / "alias")
    for spelling in ("data/era5/v1port_buffered", "./data/era5/v1port_buffered", "data/era5/v1port_buffered/", str(production), "alias"):
        with pytest.raises(SystemExit, match="needs its own --out-dir"):
            D.main(["--years", "1990", "--variables", "u700", "--out-dir", spelling, "--area", "2", "0", "0", "2"], client=_FakeClient("valid"))
    assert list(production.iterdir()) == []                                 # nothing was written there
    variant = tmp_path / "variant"
    D.main(["--years", "1990", "--variables", "u700", "--out-dir", str(variant), "--area", "3", "0", "0", "3"], client=_FakeClient("valid"))
    capsys.readouterr()
    client = _FakeClient("valid")
    rc = D.main(["--years", "1990", "--variables", "u700", "--out-dir", str(tmp_path / "default"), "--dry-run"], client=client)
    out = capsys.readouterr().out
    assert rc == 0 and "version 1's buffered one-degree grid" in out and "[2.0, 0.0, 0.0, 2.0]" in out and "3.0" not in out
    assert D.AREA == [2.0, 0.0, 0.0, 2.0] and D.EXPECTED_SHAPE == (3, 3)


def test_a_reconciled_record_names_the_area_the_file_was_validated_against(tmp_path, monkeypatch, capsys):
    """A review found a valid variant file without its record given, at restart, a record
    naming the production area."""
    D = _load()
    _tiny(monkeypatch, D)
    variant = tmp_path / "variant"
    client = _FakeClient("valid")
    assert D.main(["--years", "1990", "--variables", "u700", "--out-dir", str(variant), "--area", "2", "0", "0", "2"], client=client) == 0
    record = variant / "era5_u700_1990_6h_region.nc.completion.json"
    record.unlink()                                                        # a stop between the rename and the record
    capsys.readouterr()
    assert D.main(["--years", "1990", "--variables", "u700", "--out-dir", str(variant), "--area", "2", "0", "0", "2"], client=_FakeClient("valid")) == 0
    out = capsys.readouterr().out
    assert "RECONCILED" in out
    reconciled = json.load(open(record))
    assert reconciled["reconciled_at_restart"] and reconciled["request"]["area"] == [2.0, 0.0, 0.0, 2.0]
