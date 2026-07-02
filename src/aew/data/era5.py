"""Download ERA5 winds from the Copernicus CDS and reduce to daily means.

The original AEW work used ERA-Interim daily-averaged v700 (0.75 deg, 6-hourly). ERA-
Interim is decommissioned, so ERA5 is the supported successor; expect close but not
bit-identical results (the published threshold 3.26136 m/s is an ERA-Interim number).

Correctness note: the band-pass filter is applied to the CONTINUOUS daily record (with
a ~30-day buffer), then JAS is subset for compositing. So we download FULL YEARS, not
JAS-only -- filtering a JAS-only series would corrupt the Lanczos window at season edges.

Requires a configured ~/.cdsapirc (url + key). One CDS request per year keeps each job
within size limits; daily means are computed from the synoptic hours and concatenated.
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr

# default analysis domain (N, W, S, E) covering the AEW maps and the 5-15N composite band
DEFAULT_AREA = [35.0, -45.0, -25.0, 75.0]
SYNOPTIC_HOURS = ["00:00", "06:00", "12:00", "18:00"]
# CDS variable names: (cds_variable, pressure_level or None). A None level means the
# variable lives on the single-levels dataset (e.g. total column water vapour) rather than
# the pressure-levels dataset, and the request must not carry a pressure_level key.
VAR_CDS = {
    "v700": ("v_component_of_wind", "700"),
    "u700": ("u_component_of_wind", "700"),
    "v600": ("v_component_of_wind", "600"),
    "u600": ("u_component_of_wind", "600"),
    "v850": ("v_component_of_wind", "850"),
    "u850": ("u_component_of_wind", "850"),
    "v925": ("v_component_of_wind", "925"),
    "u925": ("u_component_of_wind", "925"),
    "r700": ("relative_humidity", "700"),
    "r600": ("relative_humidity", "600"),
    "tcwv": ("total_column_water_vapour", None),
}
# short NetCDF variable names CDS uses for the fields above (for build_* concat helpers)
CDS_SHORT_NAMES = ("u", "v", "r", "tcwv")


def cds_request(var_key, year, months=range(1, 13), hours=SYNOPTIC_HOURS,
                area=DEFAULT_AREA, grid=(0.5, 0.5)):
    """Build the (dataset, request) pair for one year of one variable.

    Pure function (no network) so the dataset routing and request shape are testable:
    pressure-level variables go to reanalysis-era5-pressure-levels with a pressure_level
    key; single-level variables (level None in VAR_CDS) go to
    reanalysis-era5-single-levels without one.
    """
    cds_var, level = VAR_CDS[var_key]
    request = {
        "product_type": "reanalysis",
        "variable": cds_var,
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": list(hours),
        "area": list(area),
        "grid": [grid[0], grid[1]],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if level is None:
        return "reanalysis-era5-single-levels", request
    request["pressure_level"] = level
    return "reanalysis-era5-pressure-levels", request


def _client():
    import cdsapi

    return cdsapi.Client()


def download_year_hourly(year, var_key, area=DEFAULT_AREA, grid=(0.5, 0.5),
                         hours=SYNOPTIC_HOURS, out_dir="data/era5/raw"):
    """Download one year of ERA5 pressure-level wind at synoptic hours (all months).

    Returns the output NetCDF path. Skips the request if the file already exists.
    """
    os.makedirs(out_dir, exist_ok=True)
    cds_var, level = VAR_CDS[var_key]
    out = os.path.join(out_dir, f"era5_{var_key}_{year}_hourly.nc")
    if os.path.exists(out):
        return out
    request = {
        "product_type": "reanalysis",
        "variable": cds_var,
        "pressure_level": level,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": list(hours),
        "area": list(area),  # N, W, S, E
        "grid": [grid[0], grid[1]],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    _client().retrieve("reanalysis-era5-pressure-levels", request, out)
    return out


def hourly_to_daily_mean(path, var_key):
    """Open an hourly file and return a daily-mean DataArray named ``var_key``."""
    ds = xr.open_dataset(path)
    # ERA5 wind component variable in the file (cds short name)
    name = [v for v in ds.data_vars if v in ("v", "u", "v_component_of_wind",
                                             "u_component_of_wind")]
    da = ds[name[0]] if name else ds[list(ds.data_vars)[0]]
    # time coordinate may be 'time' or 'valid_time'
    tname = "valid_time" if "valid_time" in da.coords else "time"
    daily = da.resample({tname: "1D"}).mean()
    daily = daily.rename(var_key)
    daily = daily.rename({tname: "time"}) if tname != "time" else daily
    ds.close()
    return daily


def build_daily_series(var_key, years, area=DEFAULT_AREA, grid=(0.5, 0.5),
                       out_dir="data/era5", raw_dir="data/era5/raw"):
    """Download all years, compute daily means, concatenate, and save one NetCDF.

    Returns the path to the combined daily-mean file.
    """
    os.makedirs(out_dir, exist_ok=True)
    combined = os.path.join(out_dir, f"era5_{var_key}_{years[0]}-{years[-1]}_daily.nc")
    if os.path.exists(combined):
        return combined
    pieces = []
    for y in years:
        raw = download_year_hourly(y, var_key, area=area, grid=grid, out_dir=raw_dir)
        pieces.append(hourly_to_daily_mean(raw, var_key))
    da = xr.concat(pieces, dim="time").sortby("time")
    da.to_netcdf(combined)
    return combined


# global longitude band for the space-time (wk_bandpass) filter: needs the full 0-360 circle
GLOBAL_BAND = [30.0, -180.0, -5.0, 179.0]  # N, W, S, E  (tropical band, all longitudes)


def download_year_6hourly_global(year, var_key="v700", area=GLOBAL_BAND,
                                 grid=(1.5, 1.5), hours=SYNOPTIC_HOURS,
                                 out_dir="data/era5/global6h"):
    """Download one year of ERA5 wind at synoptic hours on a GLOBAL-longitude tropical
    band, kept 6-hourly (no daily averaging). For wk_bandpass, which needs a full cyclic
    longitude circle to do a true zonal-wavenumber FFT. 1.5 deg mirrors the original
    v700.anom.waves grid (240 longitudes)."""
    os.makedirs(out_dir, exist_ok=True)
    cds_var, level = VAR_CDS[var_key]
    out = os.path.join(out_dir, f"era5_{var_key}_{year}_6h_global.nc")
    if os.path.exists(out):
        return out
    request = {
        "product_type": "reanalysis",
        "variable": cds_var,
        "pressure_level": level,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": list(hours),
        "area": list(area),
        "grid": [grid[0], grid[1]],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    _client().retrieve("reanalysis-era5-pressure-levels", request, out)
    return out


def download_year_6hourly_region(year, var_key, months=(6, 7, 8, 9), area=DEFAULT_AREA,
                                 grid=(0.5, 0.5), hours=SYNOPTIC_HOURS,
                                 out_dir="data/era5/region6h"):
    """Download one year of an ERA5 field at synoptic hours on the AEW analysis domain.

    For the trough-environment fields (700 hPa relative humidity, total column water
    vapour): 6-hourly, regional, June-September by default so a pre-trough sample 24 h
    before an early-July trough still has data. No band-pass touches these fields, so a
    seasonal (non-continuous) record is fine here, unlike the wave series.
    """
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"era5_{var_key}_{year}_6h_region.nc")
    if os.path.exists(out):
        return out
    dataset, request = cds_request(var_key, year, months, hours, area, grid)
    _client().retrieve(dataset, request, out)
    return out


def build_6hourly_global(var_key, years, out_dir="data/era5", raw_dir="data/era5/global6h",
                         grid=(1.5, 1.5)):
    """Download all years (6-hourly, global band), concatenate to one NetCDF (time,lat,lon)."""
    os.makedirs(out_dir, exist_ok=True)
    combined = os.path.join(out_dir, f"era5_{var_key}_{years[0]}-{years[-1]}_6h_global.nc")
    if os.path.exists(combined):
        return combined
    pieces = []
    for y in years:
        raw = download_year_6hourly_global(y, var_key, grid=grid, out_dir=raw_dir)
        ds = xr.open_dataset(raw)
        name = [v for v in ds.data_vars if v in ("v", "u", cds_from(var_key))]
        da = ds[name[0]] if name else ds[list(ds.data_vars)[0]]
        tname = "valid_time" if "valid_time" in da.coords else "time"
        da = da.rename(var_key)
        if tname != "time":
            da = da.rename({tname: "time"})
        pieces.append(da)
    out = xr.concat(pieces, dim="time").sortby("time")
    out.to_netcdf(combined)
    return combined


def cds_from(var_key):
    return VAR_CDS[var_key][0]


def load_region_6h(var_key, path_glob=None):
    """Load and concatenate 6-hourly ERA5 files for one variable into plain arrays.

    Defaults to the regional environment files (data/era5/region6h); pass ``path_glob`` to
    read another set (e.g. the global 1.5-degree wind band). Returns
    ``(times, lat, lon, field)`` with field shaped (ntime, nlat, nlon) float32, latitude
    ascending, times sorted with duplicate steps dropped.
    """
    import glob as _glob

    import pandas as pd

    pg = path_glob or f"data/era5/region6h/era5_{var_key}_*_6h_region.nc"
    paths = sorted(_glob.glob(pg))
    if not paths:
        raise FileNotFoundError(
            f"no ERA5 files match {pg!r}; run scripts/download_era5_env.py first")
    ts, blocks, lat, lon = [], [], None, None
    for p in paths:
        ds = xr.open_dataset(p)
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        ts.append(pd.DatetimeIndex(ds[tname].values))
        lat = np.asarray(ds["latitude"].values, float)
        lon = np.asarray(ds["longitude"].values, float)
        name = [v for v in ds.data_vars if v in ("r", "tcwv", "q", "u", "v")]
        da = ds[name[0]] if name else ds[list(ds.data_vars)[0]]
        blocks.append(np.asarray(da.squeeze().values, dtype=np.float32))
        ds.close()
    t = pd.DatetimeIndex(np.concatenate([x.values for x in ts]))
    field = np.concatenate(blocks, axis=0)
    o = np.argsort(t.values)
    t, field = t[o], field[o]
    uniq = np.concatenate([[True], t.values[1:] != t.values[:-1]])
    t, field = t[uniq], field[uniq]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[:, ::-1, :]
    return t, lat, lon, field
