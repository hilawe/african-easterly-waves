"""Reader for the NCEI African Easterly Wave Climatology (AEWC, C00784).

Belanger et al. curvature-vorticity wave-trough trajectories. Files are per year, per
reanalysis, per level, per region (e.g. ERA-Int_ew_700hPa_2000_AFR.nc), in a contiguous
ragged-array trajectory format: a `trajectory` dimension (one per wave) and a `sample`
dimension (all wave-trough observations), with per-sample time/lat/lon of the trough
centroid plus wavelength and vorticity statistics.

This reads the per-sample trough observations into a light container (time, lat, lon,
wavelength), with optional region/season filtering. Used to composite convection relative
to the moving trough (wave-following composite).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import glob as _glob
import numpy as np
import pandas as pd


@dataclass
class Troughs:
    """AEWC wave-trough observations as parallel 1-D arrays."""

    time: np.ndarray  # datetime64
    lat: np.ndarray   # trough centroid latitude
    lon: np.ndarray   # trough centroid longitude (-180..180)
    variables: dict = field(default_factory=dict)

    def __len__(self):
        return self.time.size

    def _subset(self, keep):
        return Troughs(self.time[keep], self.lat[keep], self.lon[keep],
                       {k: v[keep] for k, v in self.variables.items()})

    def filter_region(self, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
        keep = np.ones(self.time.size, dtype=bool)
        if min_lat is not None:
            keep &= self.lat >= min_lat
        if max_lat is not None:
            keep &= self.lat <= max_lat
        if min_lon is not None:
            keep &= self.lon >= min_lon
        if max_lon is not None:
            keep &= self.lon <= max_lon
        return self._subset(keep)

    def filter_months(self, months):
        keep = np.isin(pd.DatetimeIndex(self.time).month, list(months))
        return self._subset(keep)

    def filter(self, mask):
        """Subset by an arbitrary boolean mask (e.g. a curvature-vorticity tercile)."""
        return self._subset(np.asarray(mask, dtype=bool))


def load_aewc_trajectories(paths_or_glob, lead_hours=(24.0, 48.0)):
    """Load AEWC troughs WITH trajectory linkage and lead (preconditioning) amplitude.

    The file is a contiguous ragged array: count(trajectory) gives each wave's sample
    count, and its samples are a time-ordered block. For every observation this attaches
    the same wave's curvature-vorticity amplitude ``lead_hours`` earlier along its own
    track (linear interpolation in time within the trajectory), which supports a
    lead-lag / amplitude-preconditioning test of whether the wave precedes the convection.

    Returns a Troughs with variables crv, rv, wavelength, traj_id, and crv_lag<H> for each
    H in lead_hours (NaN where the trajectory has less than H hours of prior history).
    """
    import xarray as xr

    if isinstance(paths_or_glob, str):
        paths = sorted(_glob.glob(paths_or_glob))
    else:
        paths = list(paths_or_glob)
    if not paths:
        raise FileNotFoundError(f"no AEWC files matched {paths_or_glob!r}")

    times, lats, lons, crvs, rvs, wls, tids = [], [], [], [], [], [], []
    tpws, cafs = [], []
    lags = {h: [] for h in lead_hours}
    tid_offset = 0
    for p in paths:
        ds = xr.open_dataset(p)
        for vname in ("time", "lat", "lon"):
            if ds[vname].dims != ("sample",):
                raise ValueError(f"{p}: {vname} not on 'sample' dim ({ds[vname].dims})")
        if "count" in ds and int(np.nansum(ds["count"].values)) != ds.sizes["sample"]:
            raise ValueError(f"{p}: count.sum() != n_sample; ragged layout unexpected")
        t = pd.DatetimeIndex(ds["time"].values).values.astype("datetime64[ns]")
        lat = np.asarray(ds["lat"].values, float)
        lon = np.asarray(ds["lon"].values, float)
        lon = np.where(lon > 180, lon - 360, lon)

        def _var(name):
            return (np.asarray(ds[name].values, float) if name in ds
                    else np.full(lat.shape, np.nan))

        crv = _var("meancrv")
        rv = _var("meanrv")
        wl = _var("wavelength")
        tpw = _var("meantpw")               # trough-mean total precipitable water (TCWV, mm)
        caf = _var("cloud_area_fraction")   # trough cold-cloud area fraction
        count = np.asarray(ds["count"].values, int)
        ds.close()

        thours = t.astype("int64") / 3.6e12  # hours since epoch
        traj_id = np.empty(lat.size, dtype=np.int64)
        crv_lag = {h: np.full(lat.size, np.nan) for h in lead_hours}
        s = 0
        for k, c in enumerate(count):
            sl = slice(s, s + c)
            traj_id[sl] = tid_offset + k
            tt, cc = thours[sl], crv[sl]
            if c >= 2 and not np.all(np.diff(tt) > 0):
                o = np.argsort(tt); tt, cc = tt[o], cc[o]  # ensure time-ordered block
            if c >= 2:
                for h in lead_hours:
                    tgt = tt - h
                    lag = np.interp(tgt, tt, cc)          # interpolate amplitude at t-h
                    lag[tgt < tt[0]] = np.nan              # need a full lead of history
                    crv_lag[h][sl] = lag
            s += c
        tid_offset += count.size

        times.append(t); lats.append(lat); lons.append(lon)
        crvs.append(crv); rvs.append(rv); wls.append(wl); tids.append(traj_id)
        tpws.append(tpw); cafs.append(caf)
        for h in lead_hours:
            lags[h].append(crv_lag[h])

    variables = {"crv": np.concatenate(crvs), "rv": np.concatenate(rvs),
                 "wavelength": np.concatenate(wls), "traj_id": np.concatenate(tids),
                 "tpw": np.concatenate(tpws), "cloud_area_fraction": np.concatenate(cafs)}
    for h in lead_hours:
        variables[f"crv_lag{int(h)}"] = np.concatenate(lags[h])
    tr = Troughs(time=np.concatenate(times), lat=np.concatenate(lats),
                 lon=np.concatenate(lons), variables=variables)
    # drop only rows with no valid location (keep NaN lead amplitudes; caller filters)
    good = np.isfinite(tr.lat) & np.isfinite(tr.lon)
    return tr._subset(good)


def load_aewc_troughs(paths_or_glob):
    """Load one or more AEWC yearly files into a Troughs container."""
    import xarray as xr

    if isinstance(paths_or_glob, str):
        paths = sorted(_glob.glob(paths_or_glob))
    else:
        paths = list(paths_or_glob)
    if not paths:
        raise FileNotFoundError(f"no AEWC files matched {paths_or_glob!r}")

    times, lats, lons, wls, crvs, rvs = [], [], [], [], [], []
    for p in paths:
        ds = xr.open_dataset(p)
        # ragged-array sanity: time/lat/lon are per-sample and count sums to nsample
        for v in ("time", "lat", "lon"):
            if ds[v].dims != ("sample",):
                raise ValueError(f"{p}: {v} is not on the 'sample' dim ({ds[v].dims})")
        if "count" in ds and int(np.nansum(ds["count"].values)) != ds.sizes["sample"]:
            raise ValueError(f"{p}: count.sum() != n_sample; ragged layout unexpected")
        t = pd.DatetimeIndex(ds["time"].values)
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
        lon = np.where(lon > 180, lon - 360, lon)

        def _v(name):
            return (np.asarray(ds[name].values, dtype=float)
                    if name in ds else np.full(lat.shape, np.nan))

        wl = _v("wavelength")
        crv = _v("meancrv")   # mean trough curvature vorticity (wave amplitude proxy)
        rv = _v("meanrv")     # mean trough relative vorticity
        ds.close()
        good = np.isfinite(lat) & np.isfinite(lon)
        times.append(t[good]); lats.append(lat[good]); lons.append(lon[good])
        wls.append(wl[good]); crvs.append(crv[good]); rvs.append(rv[good])

    return Troughs(
        time=pd.DatetimeIndex(np.concatenate([x.values for x in times])).values,
        lat=np.concatenate(lats), lon=np.concatenate(lons),
        variables={"wavelength": np.concatenate(wls), "crv": np.concatenate(crvs),
                   "rv": np.concatenate(rvs)},
    )
