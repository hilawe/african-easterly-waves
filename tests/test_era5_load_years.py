"""The years filter on load_region_6h: a tier run must read exactly its season set.

The deposit driver and the tiered budget runs pass ``years`` so the development,
held-out, and pooled records are each regenerable regardless of what else is on disk.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aew.data.era5 import load_region_6h


def _write_year(dirpath, var_key, year, value):
    time = pd.date_range(f"{year}-07-01", periods=4, freq="6h")
    lat = np.array([5.0, 5.5])
    lon = np.array([0.0, 0.5, 1.0])
    field = np.full((time.size, lat.size, lon.size), value, dtype=np.float32)
    ds = xr.Dataset(
        {"r": (("time", "latitude", "longitude"), field)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(dirpath / f"era5_{var_key}_{year}_6h_region.nc")


def test_years_filter_selects_only_requested_seasons(tmp_path):
    for year, value in ((2000, 1.0), (2001, 2.0), (2002, 3.0)):
        _write_year(tmp_path, "r700", year, value)
    glob = str(tmp_path / "era5_r700_*_6h_region.nc")

    t, lat, lon, field = load_region_6h("r700", glob, years=[2000, 2002])
    assert set(pd.DatetimeIndex(t).year) == {2000, 2002}
    assert np.allclose(np.unique(field), [1.0, 3.0])

    t_all, _, _, field_all = load_region_6h("r700", glob)
    assert set(pd.DatetimeIndex(t_all).year) == {2000, 2001, 2002}
    assert field_all.shape[0] == 12


def test_missing_year_raises(tmp_path):
    _write_year(tmp_path, "r700", 2000, 1.0)
    glob = str(tmp_path / "era5_r700_*_6h_region.nc")
    with pytest.raises(FileNotFoundError, match="1999"):
        load_region_6h("r700", glob, years=[1999, 2000])


def test_truncated_year_raises(tmp_path):
    # a file whose name claims a year it barely covers must fail loudly, not
    # silently feed one short season into a canonical run
    _write_year(tmp_path, "r700", 2000, 1.0)
    _write_year(tmp_path, "r700", 2001, 2.0)
    time = pd.date_range("2002-07-01", periods=1, freq="6h")
    ds = xr.Dataset(
        {"r": (("time", "latitude", "longitude"),
               np.full((1, 2, 3), 3.0, dtype=np.float32))},
        coords={"time": time, "latitude": np.array([5.0, 5.5]),
                "longitude": np.array([0.0, 0.5, 1.0])},
    )
    ds.to_netcdf(tmp_path / "era5_r700_2002_6h_region.nc")
    glob = str(tmp_path / "era5_r700_*_6h_region.nc")
    with pytest.raises(ValueError, match="incomplete year"):
        load_region_6h("r700", glob, years=[2000, 2001, 2002])
