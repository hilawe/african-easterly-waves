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


def from_ct_genesis(path, deep_core_K=None, min_lifetime=None):
    """Load CT family genesis points into a Tracks (time left as NaT).

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
