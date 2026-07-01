"""Download GridSat-B1 brightness temperature (irwin_cdr) for a region/period.

Fetches whole 3-hourly global files from the anonymous NOAA S3 bucket (fast static
files, ~30 MB each) and crops to a lat/lon box locally, saving small per-timestep files.
(NCEI's NCSS subset endpoint was unreliable/slow; S3 + local crop is robust, as the
GridSat research recommended.) See docs/GRIDSAT_CT_PLAN.md.

irwin_cdr: Int16, physical K = raw*0.01 + 200, fill -31999. xarray's mask_and_scale
applies the scale/offset and turns the fill into NaN on open.
"""

from __future__ import annotations

import os
import tempfile
import time as _time
import urllib.request

S3 = ("https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/{year}/"
      "GRIDSAT-B1.{year}.{mm:02d}.{dd:02d}.{hh:02d}.v02r01.nc")
HOURS = (0, 3, 6, 9, 12, 15, 18, 21)
DAYS_IN = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30,
           10: 31, 11: 30, 12: 31}


def _ndays(year, month):
    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        return 29
    return DAYS_IN[month]


def _fetch_crop(url, out, box, timeout=120):
    """Download a whole global file, crop irwin_cdr to box (S,N,W,E), save, return ok."""
    import xarray as xr

    south, north, west, east = box
    tmp = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aew/0.1"})
        fd, tmp = tempfile.mkstemp(suffix=".nc")
        os.close(fd)
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        ds = xr.open_dataset(tmp)  # mask_and_scale on by default -> K, NaN fills
        sub = ds[["irwin_cdr"]].sel(lat=slice(south, north), lon=slice(west, east))
        sub.to_netcdf(out)
        ds.close()
        return True
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def download_month(year, month, area=(35.0, -45.0, -25.0, 75.0),
                   out_dir="data/gridsat", hours=HOURS, retries=3, sleep=2.0):
    """Download + crop all 3-hourly GridSat-B1 Tb for one month over a box.

    ``area`` is (north, west, south, east). Saves small cropped per-timestep files and is
    resumable (existing files skipped). Returns the list of saved paths.
    """
    north, west, south, east = area
    box = (south, north, west, east)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for dd in range(1, _ndays(year, month) + 1):
        for hh in hours:
            out = os.path.join(out_dir, f"gridsat_{year}{month:02d}{dd:02d}{hh:02d}.nc")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                paths.append(out)
                continue
            url = S3.format(year=year, mm=month, dd=dd, hh=hh)
            for attempt in range(retries):
                try:
                    if _fetch_crop(url, out, box):
                        paths.append(out)
                        break
                except Exception as ex:
                    if os.path.exists(out):
                        os.remove(out)
                    if attempt == retries - 1:
                        print(f"SKIP {year}-{month:02d}-{dd:02d} {hh:02d}Z: {ex}")
                    else:
                        _time.sleep(sleep)
    return paths
