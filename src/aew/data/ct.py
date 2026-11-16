"""Reader for the ISCCP CT (convective tracking) FAMILY dataset.

The CT family file stores Lagrangian convective-system families on a (fam, num_hits) grid:
each family is a row, each column a successive cloud-system observation. Column 0 is the
family's first system (genesis, cspos=1), the last valid column its dissipation. Per
(family, step) fields include cslat/cslon (location), csize (radius, km), cslife (family
lifetime as number of systems), tmincl (minimum cloud-top temperature, K), and itmincntr
(number of <200 K pixels).

This reads the genesis slice (column 0, extracted with ncks to keep it small) into a
Tracks of one point per family at genesis. NOTE: the day-of-month field is unreliable in
the processed file, so genesis TIME is not built here (time is left as NaT); genesis
LOCATION, size, lifetime, and depth are reliable. A wave-relative genesis composite would
need the per-step times repaired from the raw CT files.
"""

from __future__ import annotations

import numpy as np

from ..tracks import Tracks


REPEAT_FIELDS = ("fam", "ctime", "cslat", "cslon", "csize", "cslife", "tmincl", "cspos")


def repeated_rows(path):
    """True for every row of a genesis slice that repeats an earlier row in every field.

    The family file this project holds lists some families twice or three times, each copy
    identical to the first in family number and in every step of the family's life (checked
    in the file the slice was cut from, 1,293 later rows of 81,858). A row is a repeat
    here when it equals an earlier row in family number, first time, first position, size,
    lifetime, minimum temperature and step. A slice without a family number or a first
    time cannot be tested this way, and the function then stops and does not guess.
    """
    import pandas as pd
    import xarray as xr

    ds = xr.open_dataset(path)
    cols = {}
    for k in REPEAT_FIELDS:
        if k in ds.variables:
            a = np.asarray(ds[k].values)
            cols[k] = a[:, 0] if a.ndim >= 2 else a.ravel()
    ds.close()
    if "fam" not in cols or "ctime" not in cols:
        raise ValueError("the slice holds no family number or no first time, so repeated "
                         "families cannot be told from distinct ones")
    return pd.DataFrame(cols).duplicated(keep="first").values


def from_ct_genesis(path, deep_core_K=None, min_lifetime=None, drop_repeats=True):
    """Load CT family genesis points into a Tracks (time left as NaT).

    Rows that repeat an earlier family in every field are dropped unless ``drop_repeats``
    is False (see ``repeated_rows``).

    Parameters
    ----------
    path : str
        NetCDF genesis slice (column-0 of the CT family file): cslat, cslon, csize,
        cslife, tmincl per family.
    deep_core_K : float, optional
        Keep only families whose genesis minimum cloud temperature is below this (e.g.
        200.0 for deep convection).
    min_lifetime : int, optional
        Keep only families with cslife >= this (lifetime as number of systems; >=8 is
        roughly a 24 h family at 3-hourly steps).
    """
    import xarray as xr

    ds = xr.open_dataset(path)

    def _genesis(name):
        v = ds[name]
        a = np.asarray(v.values, dtype=float)
        if a.ndim >= 2:
            # full (fam, num_hits) file: genesis is column 0; guard against misuse
            if "cspos" in ds and name == "cslat":
                pass  # checked below
            a = a[:, 0]
        return a.ravel()

    # if a full family file is passed, confirm column 0 really is genesis (cspos == 1)
    if "cspos" in ds and np.asarray(ds["cspos"].values).ndim >= 2:
        col0 = np.asarray(ds["cspos"].values, dtype=float)[:, 0]
        col0 = col0[np.isfinite(col0)]
        if col0.size and not np.allclose(col0, 1.0):
            raise ValueError("CT file column 0 is not genesis (cspos != 1); pass the "
                             "genesis slice or fix the extraction")

    lat = _genesis("cslat")
    lon = _genesis("cslon")
    csize = _genesis("csize")
    cslife = _genesis("cslife")
    tmincl = _genesis("tmincl")
    # genesis time from ctime(fam, time) column 0 if present (reliable; unlike the day
    # field). xarray decodes it to datetime64.
    if "ctime" in ds:
        ct = ds["ctime"].values
        ct = ct[:, 0] if ct.ndim >= 2 else ct.ravel()
        gtime = np.asarray(ct, dtype="datetime64[ns]")
    else:
        gtime = None
    ds.close()

    good = np.isfinite(lat) & np.isfinite(lon)
    if gtime is not None:
        good &= ~np.isnat(gtime)
    if drop_repeats:
        good &= ~repeated_rows(path)
    n = int(good.sum())
    tr = Tracks(
        time=(gtime[good] if gtime is not None
              else np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")),
        lat=lat[good], lon=lon[good],
        variables={"csize_km": csize[good], "cslife": cslife[good],
                   "tmincl_K": tmincl[good]},
    )
    if deep_core_K is not None:
        tr = tr.filter(tr.variables["tmincl_K"] < deep_core_K)
    if min_lifetime is not None:
        tr = tr.filter(tr.variables["cslife"] >= min_lifetime)
    return tr
