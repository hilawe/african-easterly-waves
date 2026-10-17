#!/usr/bin/env python3
"""A continuous seasonal overview of the raw 700 hPa meridional wind in the time and
longitude plane (`EASTERN_DIRECTION_RECOMMENDATION_3_2026-10-14.md`, section 8).

One season, 1 June 00Z to 30 September 18Z six-hourly, 20 to 75 E at the files' half
degree, as terrain-aware band means in the propagation study's two bands, computed by
that study's `band_series` unchanged: a cell is valid where the wind is finite and the
surface pressure exceeds the level strictly, a sample is kept where at least 80 percent
of the band's 20 rows are valid as the equal-weight mean of the valid rows, and is
otherwise unavailable, drawn as missing and counted. No temporal filter. Nothing of the
tracker and no reference coordinate is drawn. The 40 E and 60 E meridians are marked.
One full-season figure with both bands and one figure per calendar month, fixed windows.
The record keeps the series and the support fraction by time and longitude.

    python3 scripts/pilot_season_wind_view.py --year 2006 --out <json> --png-prefix <prefix>
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_eastern_propagation as P  # noqa: E402

LON_RANGE = (20.0, 75.0)
MARK_LONS = (40.0, 60.0)
MONTHS = ((6, "June"), (7, "July"), (8, "August"), (9, "September"))
VLIM = 10.0
MISSING_COLOR = "0.75"
STEPS_PER_INCH = 24.0          # six-hourly steps per inch of figure height, so a four-day slope spans 2/3 inch


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def grid(year, level=P.PRIMARY_LEVEL):
    """The season file's latitudes and longitudes, the columns of the declared longitude
    range and the rows of each declared band (20 half-degree rows each, or refused)."""
    import netCDF4
    d = netCDF4.Dataset(os.path.join(P.DATA_DIR, f"era5_v{level}_{year}_6h_region.nc"))
    lat = np.asarray(d.variables["latitude"][:], float); lon = np.asarray(d.variables["longitude"][:], float)
    d.close()
    cols = np.flatnonzero((lon >= LON_RANGE[0] - 1e-6) & (lon <= LON_RANGE[1] + 1e-6))
    rows = {b: np.flatnonzero((lat >= lo - 1e-6) & (lat <= hi + 1e-6)) for b, (lo, hi) in P.BANDS.items()}
    if cols.size == 0 or any(r.size != 20 for r in rows.values()):
        raise SystemExit("REFUSED: the file does not hold the declared longitudes or a band does not hold 20 rows")
    return lat, lon, rows, cols


def band_means(block, rows, level=P.PRIMARY_LEVEL):
    """Per band, the study's band mean on the band's rows of the season block: the series
    (time, column) and the support fraction, with the unavailable samples counted."""
    out = {}
    for b, r in rows.items():
        rs = np.asarray(r) - block["row0"]
        series, frac = P.band_series(block["v"][:, rs, :], block["sp"][:, rs, :], level)
        out[b] = {"series": series, "support": frac, "unavailable": int(np.isnan(series).sum()), "samples": int(series.size)}
    return out


def times_of(seconds):
    return [dt.datetime(1970, 1, 1) + dt.timedelta(seconds=float(s)) for s in seconds]


def draw(year, lons, times, means, png_prefix, label):
    """The full-season figure and one per calendar month, both bands side by side, time
    downward, longitude across, missing samples in gray, the marked meridians dashed.
    Returns one record per figure with its path, digest, window and bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(MISSING_COLOR)
    windows = [("season", times[0], times[-1])] + [(name, dt.datetime(year, m, 1), (dt.datetime(year, m + 1, 1) if m < 12 else dt.datetime(year + 1, 1, 1)) - dt.timedelta(hours=6)) for m, name in MONTHS]
    tarr = np.array(times); records = []
    for name, t0, t1 in windows:
        sel = np.flatnonzero((tarr >= t0) & (tarr <= t1))
        if sel.size == 0:
            continue
        height = max(4.0, sel.size / STEPS_PER_INCH)
        fig, axes = plt.subplots(1, len(means), figsize=(6.5 * len(means) + 1.5, height + 1.6), sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (b, m) in zip(axes, means.items()):
            arr = np.ma.masked_invalid(m["series"][sel])
            lo, hi = P.BANDS[b]
            mesh = ax.pcolormesh(lons, tarr[sel], arr, cmap=cmap, vmin=-VLIM, vmax=VLIM, shading="nearest")
            for x in MARK_LONS:
                ax.axvline(x, color="k", lw=0.8, ls="--")
            ax.set_title(f"{b} band, {lo:g} to {hi:g} N", fontsize=12)
            ax.set_xlabel("longitude (degrees E)")
            ax.yaxis.set_major_locator(mdates.DayLocator(interval=5 if name == "season" else 2))
            ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.invert_yaxis() if not ax.yaxis_inverted() else None
            ax.grid(True, axis="y", color="0.5", lw=0.3, ls=":")
        fig.colorbar(mesh, ax=axes.tolist(), shrink=0.4, label="700 hPa meridional wind, band mean (m/s)")
        fig.suptitle(f"{label}, {name}: raw ERA5 700 hPa meridional wind, band mean over terrain-valid rows (80 percent support, else gray), "
                     f"no filter, no tracker output drawn. Dashed: 40 E and the detector's 60 E edge.", fontsize=10, wrap=True)
        path = f"{png_prefix}_{name}.png"
        fig.savefig(path, dpi=100); plt.close(fig)
        records.append({"name": name, "path": path, "sha256": sha256_of(path), "first": times[sel[0]].strftime("%Y-%m-%dT%H"), "last": times[sel[-1]].strftime("%Y-%m-%dT%H"),
                        "steps": int(sel.size), "bands": list(means), "unavailable_in_window": {b: int(np.isnan(m["series"][sel]).sum()) for b, m in means.items()}})
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--out", "--png-prefix"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--level", type=int, default=P.PRIMARY_LEVEL)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    lat, lon, rows, cols = grid(args.year, args.level)
    all_rows = np.concatenate(list(rows.values()))
    block = P.read_season(args.year, args.level, all_rows, cols)
    lons = lon[cols]
    means = band_means(block, rows, args.level)
    times = times_of(block["time"])
    label = f"{args.year}, {args.level} hPa"
    figures = draw(args.year, lons, times, means, args.png_prefix, label)
    files = {n: os.path.join(P.DATA_DIR, f"era5_{n}_{args.year}_6h_region.nc") for n in (f"v{args.level}", "sp")}
    out = {"generated_by": "scripts/pilot_season_wind_view.py", "script_sha256": sha256_of(os.path.abspath(__file__)),
           "reuses": {"band_series": "scripts/pilot_eastern_propagation.py, unchanged", "read_season": "scripts/pilot_eastern_propagation.py, unchanged",
                      "propagation_script_sha256": sha256_of(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_eastern_propagation.py"))},
           "year": args.year, "level_hpa": args.level, "sources": {n: {"path": p, "sha256": sha256_of(p)} for n, p in files.items()},
           "declared": {"bands": {b: {"lat_lo": lo, "lat_hi": hi, "rows": int(rows[b].size), "latitudes": lat[rows[b]].tolist()} for b, (lo, hi) in P.BANDS.items()},
                        "weights": "equal over the valid rows of the band", "valid_cell": f"wind finite and surface pressure above {args.level} hPa strictly, at that time",
                        "min_valid_fraction": P.MIN_VALID_FRACTION, "lon_range": list(LON_RANGE), "lon_step": float(np.median(np.diff(lons))), "columns": int(cols.size),
                        "time": {"first": times[0].strftime("%Y-%m-%dT%H"), "last": times[-1].strftime("%Y-%m-%dT%H"), "steps": len(times), "hours": 6},
                        "filter": "none", "marked_lons": list(MARK_LONS), "drawn": "band-mean meridional wind only; no axis, candidate, history, track or reference coordinate"},
           "lons": lons.tolist(), "times": [t.strftime("%Y-%m-%dT%H") for t in times],
           "bands": {b: {"unavailable": m["unavailable"], "samples": m["samples"], "unavailable_fraction": m["unavailable"] / m["samples"],
                         "support_by_longitude_mean": np.nanmean(m["support"], axis=0).tolist(), "support_by_time_mean": np.nanmean(m["support"], axis=1).tolist(),
                         "support": np.round(m["support"], 3).tolist(), "series": [[None if not np.isfinite(x) else round(float(x), 3) for x in row] for row in m["series"]]}
                     for b, m in means.items()},
           "figures": figures,
           "label": "a contextual field reading aid, not a census of waves and not a test of the eastern premise; a visible slope is apparent movement in this band mean at these times and establishes no individual continuity"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    for b, m in means.items():
        print(f"{b}: {m['unavailable']} of {m['samples']} samples unavailable ({m['unavailable'] / m['samples']:.3%})")
    print(f"{len(figures)} figures. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
