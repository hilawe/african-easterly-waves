"""Minimal ISCCP-style cold-cloud tracker for GridSat-B1 Tb (prototype).

A faithful, self-contained re-implementation of the ISCCP CS/CT method (Machado &
Rossow): per-frame connected-component detection of cold-cloud systems (Tb <= 245 K
shield, with an embedded Tb <= 220 K convective core), sized by equivalent radius
R = sqrt(area/pi) with the 90 km cut, then area-overlap linking across consecutive
3-hourly frames (link to the maximum-overlap candidate above a continuity threshold).

This is the PROTOTYPE the plan calls for (docs/GRIDSAT_CT_PLAN.md). It is intentionally
small and dependency-light (scipy.ndimage) so the whole GridSat -> tracks -> Hovmoller
chain runs end to end; PyFLEXTRKR remains the production tracker, and its output is read
by the same adapter. The tracker emits the PyFLEXTRKR trackstats schema so
aew.data.pyflextrkr.from_pyflextrkr consumes it unchanged.

Caveats: coarsening Tb by averaging warms cold cores (loses the very coldest); merge/
split handling is simplified (greedy max-overlap, largest fragment continues).
"""

from __future__ import annotations

import numpy as np

EARTH_DEG_KM = 111.32


def regrid_tb(da, factor=4):
    """Coarsen a (time, lat, lon) Tb DataArray by an integer factor (block mean).

    GridSat-B1 native ~0.07 deg; factor=4 -> ~0.28 deg (~30 km), mirroring ISCCP DX
    sampling so the size thresholds mean the same thing. Block mean is appropriate for
    the 245 K shield; it warms 220 K cores (documented caveat).
    """
    return da.coarsen(lat=factor, lon=factor, boundary="trim").mean()


def _cell_area_km2(lat, lon):
    """Per-row pixel area (km^2) for an equal-angle grid (area shrinks with cos lat)."""
    dlat = abs(float(lat[1] - lat[0]))
    dlon = abs(float(lon[1] - lon[0]))
    dlat_km = dlat * EARTH_DEG_KM
    dlon_km = dlon * EARTH_DEG_KM * np.cos(np.radians(np.asarray(lat, dtype=float)))
    return dlat_km * dlon_km  # (nlat,)


def detect_frame(tb, lat, lon, shield=245.0, core=220.0):
    """Connected-component cold-cloud systems in one Tb frame.

    Returns (labels, props) where labels is the 2-D int label array (0 = background) and
    props is a dict label -> system properties (area_km2, radius_km, lat, lon, min_tb,
    has_core, npix).
    """
    from scipy import ndimage

    tb = np.asarray(tb, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    mask = np.isfinite(tb) & (tb <= shield)
    labels, n = ndimage.label(mask)  # 4-connectivity (ISCCP uses contiguous pixels)
    row_area = _cell_area_km2(lat, lon)  # (nlat,)
    props = {}
    for lab in range(1, n + 1):
        jj, ii = np.where(labels == lab)
        cell = row_area[jj]
        area = float(cell.sum())
        vals = tb[jj, ii]
        props[lab] = dict(
            area_km2=area,
            radius_km=float(np.sqrt(area / np.pi)),
            lat=float(np.average(lat[jj], weights=cell)),
            lon=float(np.average(lon[ii], weights=cell)),
            min_tb=float(np.nanmin(vals)),
            has_core=bool((vals <= core).any()),
            npix=int(jj.size),
        )
    return labels, props


def _shift(a, drow, dcol):
    """Integer 2-D shift of array ``a`` with zero fill. drow>0 moves toward higher row
    index (higher latitude if lat ascending); dcol>0 toward higher column (east)."""
    out = np.zeros_like(a)
    nr, nc = a.shape
    r_src = slice(max(0, -drow), nr - max(0, drow))
    r_dst = slice(max(0, drow), nr - max(0, -drow))
    c_src = slice(max(0, -dcol), nc - max(0, dcol))
    c_dst = slice(max(0, dcol), nc - max(0, -dcol))
    out[r_dst, c_dst] = a[r_src, c_src]
    return out


def track_systems(da, shield=245.0, core=220.0, min_radius_km=90.0,
                  require_core=True, overlap_thresh=0.1, project=True,
                  default_u=-10.0):
    """Detect + area-overlap link cold-cloud systems across a Tb time series.

    Production tracker: before computing overlap, each previous-frame system is PROJECTED
    forward to the current frame using its own estimated motion (or, for a system with no
    history yet, a default westward speed). This follows the TAMS approach and is the fix
    for fast-moving systems: AEW-embedded storms move fast westward, and pure overlap
    (no projection) fragments their tracks. Merges and splits are detected and flagged,
    with the largest fragment continuing the parent track (ISCCP convention).

    Parameters
    ----------
    da : xarray.DataArray (time, lat, lon) of brightness temperature (K).
    shield, core : Tb thresholds (K) for the system shield and the convective core.
    min_radius_km : equivalent-radius cut (90 km mirrors ISCCP CT families).
    require_core : keep only systems with an embedded core (Tb <= core).
    overlap_thresh : minimum overlap fraction (common pixels / larger system) to continue
        a track (ISCCP uses ~0.1, not 0.5).
    project : if True, project previous systems forward before overlap (recommended).
    default_u : default zonal speed (m/s, negative = westward) for systems without a
        velocity estimate yet. ~-10 m/s is typical for West African MCS.

    Returns
    -------
    (times, tracks) where tracks is a dict track_id -> list of (time_index, props).
    Each props dict gains a "flag" key: "" (continuation/start), "split" (began as a
    split off a larger system), or "merge" (another system merged into it this frame).
    """
    times = np.asarray(da["time"].values)
    lat = np.asarray(da["lat"].values, dtype=float)
    lon = np.asarray(da["lon"].values, dtype=float)
    dlat = abs(float(lat[1] - lat[0]))
    dlon = abs(float(lon[1] - lon[0]))
    # seconds between frames (assume uniform; GridSat is 3-hourly)
    if times.size > 1:
        dt_s = float((np.datetime64(times[1], "ns") - np.datetime64(times[0], "ns"))
                     / np.timedelta64(1, "s"))
    else:
        dt_s = 10800.0

    frames = []
    for it in range(times.size):
        tb = np.asarray(da.isel(time=it).values, dtype=float)
        labels, props = detect_frame(tb, lat, lon, shield, core)
        keep = {lab: p for lab, p in props.items()
                if p["radius_km"] >= min_radius_km and (not require_core or p["has_core"])}
        frames.append((labels, keep))

    tracks = {}
    next_id = 0
    prev_label_to_track = {}

    def _shift_for(tid, p):
        """Cell shift (drow, dcol) to project a prev system forward one frame."""
        pts = tracks.get(tid, [])
        if len(pts) >= 2:  # use the track's own recent motion
            (_, a), (_, b) = pts[-2], pts[-1]
            return (int(round((b["lat"] - a["lat"]) / dlat)),
                    int(round((b["lon"] - a["lon"]) / dlon)))
        # no history: default westward speed at this latitude
        dlon_deg = default_u * dt_s / (EARTH_DEG_KM * 1000.0
                                       * np.cos(np.radians(p["lat"])))
        return (0, int(round(dlon_deg / dlon)))

    for it, (labels, keep) in enumerate(frames):
        cur_label_to_track = {}
        if it == 0:
            for lab, p in keep.items():
                p["flag"] = ""
                tracks[next_id] = [(it, p)]
                cur_label_to_track[lab] = next_id
                next_id += 1
            prev_label_to_track = cur_label_to_track
            continue

        prev_labels, prev_keep = frames[it - 1]
        # stamp each previous system's PROJECTED footprint into a label map
        proj = np.zeros_like(prev_labels)
        if project:
            for pl in prev_keep:
                tid = prev_label_to_track.get(pl)
                drow, dcol = _shift_for(tid, prev_keep[pl])
                proj += np.where((_shift((prev_labels == pl).astype(prev_labels.dtype),
                                         drow, dcol) > 0) & (proj == 0), pl, 0)
        else:
            for pl in prev_keep:
                proj = np.where((prev_labels == pl) & (proj == 0), pl, proj)

        claimed = {}  # tid -> cur label that continued it (for merge detection)
        for lab, p in sorted(keep.items(), key=lambda kv: -kv[1]["npix"]):
            overlap_vals = proj[labels == lab]
            overlap_vals = overlap_vals[overlap_vals > 0]
            best_tid, best_frac, n_overlaps = None, 0.0, 0
            if overlap_vals.size:
                uniq, counts = np.unique(overlap_vals, return_counts=True)
                for pl, c in zip(uniq, counts):
                    if pl not in prev_keep:
                        continue
                    frac = c / max(p["npix"], prev_keep[pl]["npix"])
                    if frac >= overlap_thresh:
                        n_overlaps += 1  # this prev system merged into cur
                    tid = prev_label_to_track.get(pl)
                    if frac >= overlap_thresh and frac > best_frac and tid not in claimed:
                        best_frac, best_tid = frac, tid
            if best_tid is not None:
                p["flag"] = "merge" if n_overlaps > 1 else ""
                tracks[best_tid].append((it, p))
                claimed[best_tid] = lab
                cur_label_to_track[lab] = best_tid
            else:
                # new track; if it overlapped an already-claimed parent, it is a split
                p["flag"] = "split" if overlap_vals.size else ""
                tracks[next_id] = [(it, p)]
                cur_label_to_track[lab] = next_id
                next_id += 1
        prev_label_to_track = cur_label_to_track
    return times, tracks


def to_pyflextrkr_netcdf(times, tracks, path, min_duration=1):
    """Write tracks to the PyFLEXTRKR trackstats schema (tracks x times).

    Variables: base_time (epoch seconds), meanlat, meanlon, area, track_duration.
    Read back with aew.data.pyflextrkr.from_pyflextrkr. ``min_duration`` drops tracks
    shorter than this many frames.
    """
    import xarray as xr

    items = [(tid, pts) for tid, pts in sorted(tracks.items()) if len(pts) >= min_duration]
    ntr = len(items)
    maxlen = max((len(p) for _, p in items), default=1)
    base = np.full((ntr, maxlen), np.nan)
    mlat = np.full((ntr, maxlen), np.nan)
    mlon = np.full((ntr, maxlen), np.nan)
    area = np.full((ntr, maxlen), np.nan)
    dur = np.zeros(ntr, dtype=int)
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    for r, (_, pts) in enumerate(items):
        dur[r] = len(pts)
        for k, (it, p) in enumerate(pts):
            base[r, k] = (np.datetime64(times[it], "ns") - epoch) / np.timedelta64(1, "s")
            mlat[r, k] = p["lat"]
            mlon[r, k] = p["lon"]
            area[r, k] = p["area_km2"]
    ds = xr.Dataset(
        {
            "base_time": (("tracks", "times"), base),
            "meanlat": (("tracks", "times"), mlat),
            "meanlon": (("tracks", "times"), mlon),
            "area": (("tracks", "times"), area),
            "track_duration": (("tracks",), dur),
        }
    )
    ds["base_time"].attrs["units"] = "seconds since 1970-01-01"
    ds.to_netcdf(path)
    return path
