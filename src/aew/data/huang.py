"""Reader for the Huang et al. (2018) tropical MCS dataset (PANGAEA 877914).

An independent, openly available tracked-MCS product used as a cross-check for the
in-house GridSat tracker. Built from the CLAUS infrared record, 1985-2008, with a
brightness-temperature threshold of 233 K, a 5000 km^2 area cut, and 15% area-overlap
linking plus a Kalman filter (Huang et al. 2018, Clim. Dyn., doi:10.1007/s00382-018-4071-0).

Monthly plain-text files, one row per system-time point, columns:
  ID  Lifetime(hour)  gLat(N) gLon(E)  wLat(N) wLon(E)  Size(km^2)  Eccentricity
  BTavg(K) BTmin(K)  Time(UTC=YYYY-MM-DD-HH)  Speed(km/h)  Direction(deg)

This loads them into aew.tracks.Tracks (time, lat, lon, + size/radius/Tb/track_id), with
lon wrapped to -180..180. ID is namespaced by year-month so tracks stay unique when
months are concatenated. Apply a size/radius cut to match the ISCCP CT 90 km families.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from ..tracks import Tracks

_COLS = ["id", "lifetime_h", "glat", "glon", "wlat", "wlon", "size_km2",
         "eccentricity", "bt_avg", "bt_min", "time", "speed_kmh", "direction_deg"]


def load_huang_month(path, min_radius_km=None):
    """Load one Huang monthly file into a Tracks of system-time points."""
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, names=_COLS,
                     na_values=["NaN"], engine="python")
    df = df.dropna(subset=["glat", "glon", "time"])
    time = pd.to_datetime(df["time"], format="%Y-%m-%d-%H").values
    lon = df["glon"].to_numpy(dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    size = df["size_km2"].to_numpy(dtype=float)
    # namespace track ids by the file's year-month so concatenation stays unique
    tag = os.path.basename(path).replace("MCS_record_", "").replace(".txt", "")
    track_id = np.array([f"{tag}_{int(i)}" for i in df["id"].to_numpy()])
    tr = Tracks(
        time=time,
        lat=df["glat"].to_numpy(dtype=float),
        lon=lon,
        variables={
            "size_km2": size,
            "radius_km": np.sqrt(size / np.pi),
            "bt_min": df["bt_min"].to_numpy(dtype=float),
            "bt_avg": df["bt_avg"].to_numpy(dtype=float),
            "lifetime_h": df["lifetime_h"].to_numpy(dtype=float),
            "track_id": track_id,
        },
    )
    if min_radius_km is not None:
        tr = tr.filter(tr.variables["radius_km"] >= min_radius_km)
    return tr


def load_huang(paths_or_dir, months=None, years=None, min_radius_km=None):
    """Load multiple Huang monthly files into one Tracks.

    Parameters
    ----------
    paths_or_dir : str or sequence
        A directory of MCS_record_*.txt files, or an explicit list of file paths.
    months, years : sequence of int, optional
        Restrict to these calendar months / years (filenames are MCS_record_YYYY-MM.txt).
    min_radius_km : float, optional
        Equivalent-radius cut (e.g. 90.0 to match ISCCP CT families).
    """
    if isinstance(paths_or_dir, str) and os.path.isdir(paths_or_dir):
        paths = sorted(glob.glob(os.path.join(paths_or_dir, "MCS_record_*.txt")))
    else:
        paths = list(paths_or_dir)

    def _keep(p):
        tag = os.path.basename(p).replace("MCS_record_", "").replace(".txt", "")
        try:
            y, m = tag.split("-")
            y, m = int(y), int(m)
        except ValueError:
            return False
        if years is not None and y not in years:
            return False
        if months is not None and m not in months:
            return False
        return True

    paths = [p for p in paths if _keep(p)]
    if not paths:
        raise FileNotFoundError("no Huang monthly files matched the year/month filter")

    parts = [load_huang_month(p, min_radius_km=min_radius_km) for p in paths]
    return Tracks(
        time=np.concatenate([t.time for t in parts]),
        lat=np.concatenate([t.lat for t in parts]),
        lon=np.concatenate([t.lon for t in parts]),
        variables={k: np.concatenate([t.variables[k] for t in parts])
                   for k in parts[0].variables},
    )
