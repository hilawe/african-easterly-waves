"""Retrieve the ERA5 pressure-level fields the energy-flux reading needs, one variable and one
season per file, so that the geopotential flux and the eddy kinetic energy budget of Diaz and
Aiyyer (2013) can be formed at their published base.

The fields are the zonal and meridional wind, geopotential, temperature and vertical velocity at
every ERA5 level from 950 to 350 hPa (the layer the paper integrates over), at 00, 06, 12 and 18
UTC from June to September, for the 17 seasons the paper's period shares with the retained
regional files (1990 to 2007 without 2005, which stays unopened). The area runs from 50 W to
40 E and from 5 S to 30 N on a 1 degree grid, five degrees beyond the longitudes the readings use
on each side, so that centered differences reach every point read.

Copernicus refuses a request costing more than 60,000, and a season of one variable on these
levels costs 99,552, so each season comes in two parts, June and July, then August and
September (49,776 each). Each part is written under a temporary name and checked, the two are
joined into one file per variable and season, and that file is moved into place only after it
opens with the declared variable, levels, latitudes and longitudes and exactly the six-hourly
times of its own season, so a partial or wrong download is never taken for a complete one. A run skips files already in place and can
be restarted.
"""

import argparse
import os
import sys
import time

YEARS = tuple(y for y in range(1990, 2008) if y != 2005)
MONTHS = (6, 7, 8, 9)
PARTS = ((6, 7), (8, 9))
DAYS = {6: 30, 7: 31, 8: 31, 9: 30}
HOURS = ("00:00", "06:00", "12:00", "18:00")
LEVELS = (950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350)
AREA = (30.0, -50.0, -5.0, 40.0)  # N, W, S, E
GRID = 1.0
VARIABLES = {"u": "u_component_of_wind", "v": "v_component_of_wind", "z": "geopotential",
             "t": "temperature", "w": "vertical_velocity"}
N_TIMES = 4 * sum(DAYS[m] for m in MONTHS)
OUT_DIR = "data/era5/flux6h"


def request(var, year, months=MONTHS):
    """The (dataset, request) pair for one variable and the given months of one season, with no
    network use."""
    return "reanalysis-era5-pressure-levels", {
        "product_type": "reanalysis",
        "variable": VARIABLES[var],
        "pressure_level": [str(p) for p in LEVELS],
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": list(HOURS),
        "area": list(AREA),
        "grid": [GRID, GRID],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def path(var, year, out_dir=OUT_DIR):
    return os.path.join(out_dir, f"era5_{var}_{year}_6h_flux.nc")


def calendar(year, months=MONTHS):
    """Every 00, 06, 12 and 18 UTC time of `months` in `year`, in order, at full precision."""
    import numpy as np

    days = [np.datetime64(f"{year}-{m:02d}-{d:02d}") for m in months for d in range(1, DAYS[m] + 1)]
    return np.array([d + np.timedelta64(h, "h") for d in days for h in (0, 6, 12, 18)], dtype="datetime64[ns]")


def check(file, var, year, months=MONTHS):
    """Refuse a file that does not hold the declared variable on the declared levels, exactly the
    times of `months` in `year` at 00, 06, 12 and 18 UTC compared at full precision, and the
    declared grid, and return its shape when it does."""
    import numpy as np
    import xarray as xr

    with xr.open_dataset(file) as ds:
        if var not in ds:
            raise SystemExit(f"REFUSED: {file} holds {sorted(ds.data_vars)}, not {var}")
        da = ds[var]
        tname = "valid_time" if "valid_time" in da.dims else "time"
        lev = sorted(int(p) for p in ds["pressure_level"].values)
        lat, lon = ds["latitude"].values, ds["longitude"].values
        want_lat = np.arange(AREA[2], AREA[0] + GRID / 2, GRID)
        want_lon = np.arange(AREA[1], AREA[3] + GRID / 2, GRID)
        if lev != sorted(LEVELS):
            raise SystemExit(f"REFUSED: {file} holds the levels {lev}")
        want = calendar(year, months)
        times = np.sort(ds[tname].values.astype("datetime64[ns]"))
        if da.sizes[tname] != len(want) or not np.array_equal(times, want):
            raise SystemExit(f"REFUSED: {file} does not hold exactly the {len(want)} times of months {months} in {year}")
        if not (np.allclose(np.sort(lat), want_lat) and np.allclose(np.sort(lon), want_lon)):
            raise SystemExit(f"REFUSED: {file} is not on the declared grid")
        if not np.isfinite(da.values).all():
            raise SystemExit(f"REFUSED: {file} holds missing values")
        return dict(da.sizes)


def fetch_part(var, year, months, part, retries=3, client=None):
    if os.path.exists(part):
        check(part, var, year, months)
        return "present"
    dataset, req = request(var, year, months)
    for attempt in range(1, retries + 1):
        tmp = part + ".tmp"
        try:
            if client is None:
                import cdsapi
                client = cdsapi.Client(quiet=True, progress=False)
            client.retrieve(dataset, req, tmp)
            check(tmp, var, year, months)
            os.replace(tmp, part)
            return f"retrieved on attempt {attempt}"
        except SystemExit:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        except Exception as ex:  # a failed retrieval is retried, never kept
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"{var} {year} {months} attempt {attempt} failed: {ex}", flush=True)
            time.sleep(60 * attempt)
    raise SystemExit(f"REFUSED: {var} {year} {months} could not be retrieved in {retries} attempts")


def join(parts, out, var, year):
    """One file per variable and season from its checked parts, sorted in time."""
    import xarray as xr

    pieces = []
    for p in parts:
        with xr.open_dataset(p) as d:
            pieces.append(d.load())
    ds = xr.concat(pieces, dim="valid_time").sortby("valid_time")
    tmp = out + ".tmp"
    ds.to_netcdf(tmp, encoding={var: {"zlib": True, "complevel": 4}})
    try:
        check(tmp, var, year)
    except SystemExit:
        os.remove(tmp)
        raise
    os.replace(tmp, out)


def fetch(var, year, out_dir=OUT_DIR, retries=3, client=None):
    out = path(var, year, out_dir)
    if os.path.exists(out):
        return out, "present"
    os.makedirs(out_dir, exist_ok=True)
    parts = [out.replace(".nc", f"_part{i}.nc") for i in range(len(PARTS))]
    how = [fetch_part(var, year, months, p, retries, client) for months, p in zip(PARTS, parts)]
    join(parts, out, var, year)
    for p in parts:
        os.remove(p)
    return out, ", ".join(how)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--variables", default="".join(VARIABLES))
    ap.add_argument("--years", default=None, help="comma-separated, default the 17 seasons")
    a = ap.parse_args(argv)
    years = YEARS if a.years is None else tuple(int(y) for y in a.years.split(","))
    if 2005 in years:
        raise SystemExit("REFUSED: the 2005 season stays unopened")
    for year in years:
        for var in a.variables:
            t0 = time.time()
            out, how = fetch(var, year, a.out_dir)
            print(f"{var} {year} {how} in {time.time() - t0:.0f} s, {os.path.getsize(out) / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
