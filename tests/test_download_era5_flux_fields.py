"""Tests for scripts/download_era5_flux_fields.py, run on synthetic files with no network use."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import download_era5_flux_fields as D  # noqa: E402


def part_dataset(var, months, year=1990, levels=D.LEVELS, drop_hour=None, nan=False, shift_lon=0.0, shift_minutes=0, mix_year=None):
    times = pd.date_range(f"{year}-{months[0]:02d}-01", f"{year}-{months[-1]:02d}-{D.DAYS[months[-1]]:02d} 18:00", freq="6h")
    times = times + pd.Timedelta(minutes=shift_minutes)
    if mix_year is not None:
        times = pd.DatetimeIndex([t.replace(year=mix_year) if t.month == months[-1] else t for t in times])
    if drop_hour is not None:
        times = times[times.hour != drop_hour]
    lat = np.arange(D.AREA[0], D.AREA[2] - D.GRID / 2, -D.GRID)
    lon = np.arange(D.AREA[1], D.AREA[3] + D.GRID / 2, D.GRID) + shift_lon
    vals = np.random.default_rng(0).normal(size=(len(times), len(levels), len(lat), len(lon))).astype("float32")
    if nan:
        vals[0, 0, 0, 0] = np.nan
    return xr.Dataset({var: (("valid_time", "pressure_level", "latitude", "longitude"), vals)},
                      coords={"valid_time": times, "pressure_level": np.array(levels, float), "latitude": lat, "longitude": lon})


class FakeClient:
    def __init__(self, **kw):
        self.calls, self.kw = [], kw

    def retrieve(self, dataset, req, target):
        months = tuple(int(m) for m in req["month"])
        self.calls.append((dataset, req["variable"], req["year"], months))
        part_dataset(next(k for k, v in D.VARIABLES.items() if v == req["variable"]), months, int(req["year"]), **self.kw).to_netcdf(target)


def test_requests_stay_under_the_cost_limit_and_cover_the_declared_season():
    assert D.YEARS == tuple(y for y in range(1990, 2008) if y != 2005) and len(D.YEARS) == 17
    assert sorted(m for p in D.PARTS for m in p) == list(D.MONTHS) == [6, 7, 8, 9]
    dataset, req = D.request("z", 1999, (6, 7))
    assert dataset == "reanalysis-era5-pressure-levels" and req["variable"] == "geopotential"
    assert req["pressure_level"][0] == "950" and req["pressure_level"][-1] == "350" and len(req["pressure_level"]) == 17
    assert req["month"] == ["06", "07"] and req["area"] == [30.0, -50.0, -5.0, 40.0] and req["grid"] == [1.0, 1.0]
    assert req["time"] == ["00:00", "06:00", "12:00", "18:00"] and D.N_TIMES == 488
    with pytest.raises(SystemExit, match="2005"):
        D.main(["--years", "2004,2005"])


@pytest.mark.parametrize("kw,why", ((dict(levels=D.LEVELS[:-1]), "levels"), (dict(drop_hour=6), "times"),
                                    (dict(nan=True), "missing values"), (dict(shift_lon=0.5), "declared grid")))
def test_check_refuses_a_part_that_is_not_the_declared_one(tmp_path, kw, why):
    f = tmp_path / "p.nc"
    part_dataset("u", (6, 7), **kw).to_netcdf(f)
    with pytest.raises(SystemExit, match=why):
        D.check(str(f), "u", 1990, (6, 7))
    part_dataset("u", (6, 7)).to_netcdf(tmp_path / "ok.nc")
    assert D.check(str(tmp_path / "ok.nc"), "u", 1990, (6, 7))["valid_time"] == 244
    with pytest.raises(SystemExit, match="not v"):
        D.check(str(tmp_path / "ok.nc"), "v", 1990, (6, 7))
    with pytest.raises(SystemExit, match="times"):
        D.check(str(tmp_path / "ok.nc"), "u", 1990, (8, 9))


def test_fetch_joins_two_checked_parts_into_one_season_and_skips_what_is_present(tmp_path):
    c = FakeClient()
    out, how = D.fetch("t", 1995, str(tmp_path), client=c)
    assert c.calls == [("reanalysis-era5-pressure-levels", "temperature", "1995", (6, 7)),
                       ("reanalysis-era5-pressure-levels", "temperature", "1995", (8, 9))]
    assert D.check(out, "t", 1995)["valid_time"] == 488 and sorted(os.listdir(tmp_path)) == ["era5_t_1995_6h_flux.nc"]
    with xr.open_dataset(out) as ds:
        assert (np.diff(ds["valid_time"].values) > np.timedelta64(0, "ns")).all()
    assert D.fetch("t", 1995, str(tmp_path), client=c) == (out, "present") and len(c.calls) == 2


def test_a_part_that_fails_its_check_is_never_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="times"):
        D.fetch("w", 1995, str(tmp_path), client=FakeClient(drop_hour=12))
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("kw", (dict(year=2005), dict(mix_year=2005), dict(shift_minutes=30)))
def test_check_refuses_another_season_a_mixed_season_and_an_offset_clock(tmp_path, kw):
    """The full calendar of the requested year is compared at full precision, so a file of another
    season, one with a month from another season, or one 30 minutes off the six-hourly clock is
    refused although each has 244 unique times in June and July at hours that look right."""
    f = tmp_path / "p.nc"
    part_dataset("t", (6, 7), **kw).to_netcdf(f)
    with pytest.raises(SystemExit, match="exactly the 244 times of months \\(6, 7\\) in 1990"):
        D.check(str(f), "t", 1990, (6, 7))
    assert len(D.calendar(1990, (6, 7))) == 244 and str(D.calendar(1990)[-1]).startswith("1990-09-30T18:00")


def test_a_part_already_present_is_checked_against_its_own_season(tmp_path):
    part0 = tmp_path / "era5_t_1995_6h_flux_part0.nc"
    part_dataset("t", (6, 7), year=1995).to_netcdf(part0)
    c = FakeClient()
    out, how = D.fetch("t", 1995, str(tmp_path), client=c)
    assert how.startswith("present") and c.calls == [("reanalysis-era5-pressure-levels", "temperature", "1995", (8, 9))]
    assert D.check(out, "t", 1995)["valid_time"] == 488
    part_dataset("t", (6, 7), year=1994).to_netcdf(tmp_path / "era5_t_1996_6h_flux_part0.nc")
    with pytest.raises(SystemExit, match="era5_t_1996_6h_flux_part0.nc does not hold exactly .* in 1996"):
        D.fetch("t", 1996, str(tmp_path), client=FakeClient())


def test_a_join_that_fails_its_check_leaves_nothing(tmp_path):
    parts = [str(tmp_path / f"p{i}.nc") for i in (0, 1)]
    part_dataset("t", (6, 7), year=1995).to_netcdf(parts[0])
    part_dataset("t", (8, 9), year=1995).to_netcdf(parts[1])
    out = str(tmp_path / "era5_t_1996_6h_flux.nc")
    with pytest.raises(SystemExit, match="in 1996"):
        D.join(parts, out, "t", 1996)
    assert sorted(os.listdir(tmp_path)) == ["p0.nc", "p1.nc"]
