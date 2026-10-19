#!/usr/bin/env python3
"""The eastern regime characterization's figures and derived numbers
(`EASTERN_REGIME_CHARACTERIZATION_2026-10-18.md`, section 3). Per calendar month, from the
retained regional files: the meridional wind at 700, 850 and 600 hPa in the time and
longitude plane as band means over four five-degree latitude bands (the propagation
study's `band_series`, unchanged), the month's mean maps with the terrain mask and the
northerly persistence per cell, and the lag and offset correlation of each band's
anomalies. A reading aid and a record of numbers, not a detector. 2005 is never opened.

    python3 scripts/eastern_regime_views.py render --year 1987 --month 9 --out-dir DIR
    python3 scripts/eastern_regime_views.py summary --out-dir DIR --draw DRAW.json --table OUT.md
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_eastern_propagation as P  # noqa: E402

BOX_LAT = (5.0, 35.0)
BOX_LON = (20.0, 75.0)
BANDS = {"10 to 14.5 N": (10.0, 14.5), "15 to 19.5 N": (15.0, 19.5), "20 to 24.5 N": (20.0, 24.5), "25 to 29.5 N": (25.0, 29.5)}
LEVELS_V = (700, 850, 600)
NORTHERLY_MS = -4.0
LAG_STEPS = 12          # 3 days at six hours
OFFSET_CELLS = 20       # 10 degrees at half a degree
X_RANGE = (40.0, 65.0)
RIDGE_MIN_CORR = 0.3
MIN_PAIRS = 20
VLIM = 10.0
MARK_LONS = (40.0, 60.0)
MISSING_COLOR = "0.75"
MONTH_NAMES = {6: "June", 7: "July", 8: "August", 9: "September"}


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------------------------ loading

def month_window(year, month):
    start = dt.datetime(year, month, 1)
    end = (dt.datetime(year, month + 1, 1) if month < 12 else dt.datetime(year + 1, 1, 1)) - dt.timedelta(hours=6)
    return start, end


def load_month(year, month, data_dir=P.DATA_DIR):
    """Every six-hourly step of the calendar month over the box, for v at 600, 700 and
    850 hPa, u and v at 700 and 925 hPa, and the surface pressure, with the time axis
    checked across files. Refuses the held-out season."""
    import netCDF4
    if year == P.HELD_OUT:
        raise SystemExit("REFUSED: the held-out season is never opened")
    start, end = month_window(year, month)
    out = {"year": year, "month": month}
    ref_t = None
    for name, var in (("v700", "v"), ("v850", "v"), ("v600", "v"), ("u700", "u"), ("u925", "u"), ("v925", "v"), ("sp", "sp")):
        path = os.path.join(data_dir, f"era5_{name}_{year}_6h_region.nc")
        d = netCDF4.Dataset(path)
        try:
            t = np.asarray(d.variables["valid_time"][:], float)
            times = [dt.datetime(1970, 1, 1) + dt.timedelta(seconds=float(s)) for s in t]
            sel = np.flatnonzero([(start <= x <= end) for x in times])
            if sel.size == 0 or (sel[-1] - sel[0] + 1) != sel.size:
                raise SystemExit(f"REFUSED: {path} does not hold the month contiguously")
            lat = np.asarray(d.variables["latitude"][:], float); lon = np.asarray(d.variables["longitude"][:], float)
            rows = np.flatnonzero((lat >= BOX_LAT[0] - 1e-6) & (lat <= BOX_LAT[1] + 1e-6)); cols = np.flatnonzero((lon >= BOX_LON[0] - 1e-6) & (lon <= BOX_LON[1] + 1e-6))
            if rows.size == 0 or cols.size == 0 or (rows[-1] - rows[0] + 1) != rows.size or (cols[-1] - cols[0] + 1) != cols.size:
                raise SystemExit(f"REFUSED: {path} does not hold the box contiguously")
            a = d.variables[var]
            if a.ndim == 4:
                level = int(name[1:])
                if abs(float(d.variables["pressure_level"][0]) - level) > 1e-6:
                    raise SystemExit(f"REFUSED: {path} is not at {level} hPa")
                arr = np.asarray(a[sel[0]:sel[-1] + 1, 0, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], float)
            else:
                arr = np.asarray(a[sel[0]:sel[-1] + 1, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], float)
            if ref_t is None:
                ref_t = t[sel]; out["times"] = [times[i] for i in sel]; out["lat"] = lat[rows]; out["lon"] = lon[cols]
            elif not np.array_equal(t[sel], ref_t) or not np.array_equal(lat[rows], out["lat"]) or not np.array_equal(lon[cols], out["lon"]):
                raise SystemExit(f"REFUSED: {path} has a different time or space axis from the first file")
            out[name] = arr
            out.setdefault("files", {})[name] = {"path": path, "sha256": sha256_of(path)}
        finally:
            d.close()
    expected = int((end - start).total_seconds() // 21600) + 1
    if len(out["times"]) != expected:
        raise SystemExit(f"REFUSED: the month holds {len(out['times'])} steps, not {expected}")
    return out


def band_rows(lat):
    """{band: row indices}, ten half-degree rows each, or refused."""
    rows = {}
    for b, (lo, hi) in BANDS.items():
        r = np.flatnonzero((lat >= lo - 1e-6) & (lat <= hi + 1e-6))
        if r.size != 10:
            raise SystemExit(f"REFUSED: band {b} holds {r.size} rows, not 10")
        rows[b] = r
    return rows


def valid_cells(v, sp, level):
    return np.isfinite(v) & (np.asarray(sp, float) > level * 100.0)


# ------------------------------------------------------------------ derived numbers

def band_means(v, sp, level, rows):
    out = {}
    for b, r in rows.items():
        series, frac = P.band_series(v[:, r, :], sp[:, r, :], level)
        out[b] = {"series": series, "unavailable": int(np.isnan(series).sum()), "samples": int(series.size)}
    return out


def persistence(v700, sp, threshold=NORTHERLY_MS):
    """Per cell, over the month: the fraction of valid steps at or below the threshold, the
    mean duration in days of the uncensored runs of consecutive such steps (an invalid
    step breaks a run), the count of uncensored runs, and the count of runs censored by the
    month's ends. Cells with no valid step are NaN."""
    valid = valid_cells(v700, sp, 700)
    north = valid & (v700 <= threshold)
    nt, ny, nx = v700.shape
    n_valid = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(n_valid > 0, north.sum(axis=0) / n_valid, np.nan)
    mean_days = np.full((ny, nx), np.nan); n_runs = np.zeros((ny, nx), int); n_censored = np.zeros((ny, nx), int)
    for j in range(ny):
        for i in range(nx):
            col = north[:, j, i]
            if not col.any():
                mean_days[j, i] = 0.0 if n_valid[j, i] > 0 else np.nan
                continue
            padded = np.concatenate([[False], col, [False]]).astype(int)
            starts = np.flatnonzero(np.diff(padded) == 1); ends = np.flatnonzero(np.diff(padded) == -1)
            lengths = []
            for s, e in zip(starts, ends):
                if s == 0 or e == nt:
                    n_censored[j, i] += 1
                else:
                    lengths.append(e - s)
            n_runs[j, i] = len(lengths)
            mean_days[j, i] = (np.mean(lengths) * 0.25) if lengths else np.nan
    return {"fraction": fraction, "mean_run_days": mean_days, "uncensored_runs": n_runs, "censored_runs": n_censored, "valid_steps": n_valid, "invalid_at_any_step": int((n_valid < nt).sum())}


def lag_offset_correlation(series, lons, lags=LAG_STEPS, offsets=OFFSET_CELLS, x_range=X_RANGE, min_pairs=MIN_PAIRS):
    """corr[offset, lag]: the Pearson correlation between the band-mean series at
    longitude x and at x plus the offset, the second taken `lag` steps later, averaged over
    the x inside `x_range` for which both columns exist, each x needing at least
    `min_pairs` finite pairs. NaN where no x qualifies. The Pearson coefficient centers
    each series on its own mean over the paired steps, so the month's mean at each
    longitude is removed by construction and no separate subtraction is made (a planted
    removal of such a subtraction survived the tests on 2026-10-18, which showed it inert)."""
    S = np.asarray(series, float)
    nt, nx = S.shape
    xs = np.flatnonzero((lons >= x_range[0] - 1e-6) & (lons <= x_range[1] + 1e-6))
    corr = np.full((2 * offsets + 1, 2 * lags + 1), np.nan)
    for di, d in enumerate(range(-offsets, offsets + 1)):
        for li, L in enumerate(range(-lags, lags + 1)):
            vals = []
            for x in xs:
                if not (0 <= x + d < nx):
                    continue
                if L >= 0:
                    a = S[:nt - L, x]; b = S[L:, x + d]
                else:
                    a = S[-L:, x]; b = S[:nt + L, x + d]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < min_pairs:
                    continue
                aa, bb = a[ok], b[ok]
                if aa.std() == 0.0 or bb.std() == 0.0:
                    continue
                vals.append(float(np.corrcoef(aa, bb)[0, 1]))
            if vals:
                corr[di, li] = float(np.mean(vals))
    return corr


def ridge(corr, offsets=OFFSET_CELLS, lags=LAG_STEPS, min_corr=RIDGE_MIN_CORR):
    """The lag of maximum correlation at each offset whose maximum exceeds `min_corr`, in
    days against degrees, fitted by least squares (lag = slope times offset + intercept).
    The apparent speed is one over the slope in degrees per day, negative westward. None
    when fewer than three offsets qualify or the slope is zero."""
    pts = []
    for di in range(corr.shape[0]):
        row = corr[di]
        if np.all(np.isnan(row)):
            continue
        li = int(np.nanargmax(row))
        if row[li] > min_corr:
            pts.append(((di - offsets) * 0.5, (li - lags) * 0.25, float(row[li])))
    if len(pts) < 3:
        return {"points": len(pts), "slope_days_per_degree": None, "speed_deg_per_day": None, "mean_ridge_corr": None}
    d = np.array([p[0] for p in pts]); L = np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(d, L, 1)
    speed = (1.0 / slope) if abs(slope) > 1e-9 else None
    return {"points": len(pts), "slope_days_per_degree": float(slope), "intercept_days": float(intercept), "speed_deg_per_day": (None if speed is None else float(speed)),
            "mean_ridge_corr": float(np.mean([p[2] for p in pts])), "offset_span_deg": [float(d.min()), float(d.max())]}


# ------------------------------------------------------------------ figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_hovmoller(path, label, level, lons, times, means):
    plt = _plt()
    import matplotlib.dates as mdates
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(MISSING_COLOR)
    tarr = np.array(times)
    fig, axes = plt.subplots(1, len(means), figsize=(4.2 * len(means) + 1.5, max(6.0, len(times) / 14.0) + 1.4), sharey=True, constrained_layout=True)
    for ax, (b, m) in zip(np.atleast_1d(axes), means.items()):
        mesh = ax.pcolormesh(lons, tarr, np.ma.masked_invalid(m["series"]), cmap=cmap, vmin=-VLIM, vmax=VLIM, shading="nearest")
        for x in MARK_LONS:
            ax.axvline(x, color="k", lw=0.8, ls="--")
        ax.set_title(f"{b}, unavailable {m['unavailable']} of {m['samples']}", fontsize=11)
        ax.set_xlabel("longitude (degrees E)")
        ax.yaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.grid(True, axis="y", color="0.5", lw=0.3, ls=":")
    fig.colorbar(mesh, ax=np.atleast_1d(axes).tolist(), shrink=0.5, label=f"{level} hPa meridional wind, band mean (m/s)")
    fig.suptitle(f"{label}: raw ERA5 {level} hPa meridional wind, band means over terrain-valid rows (a cell valid where the surface pressure is strictly above {level} hPa, "
                 f"a sample kept at 80 percent support, else gray). No filter, nothing of any tracker drawn. Dashed: 40 E and 60 E.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


def draw_mean_maps(path, label, M, pers):
    plt = _plt()
    lat, lon = M["lat"], M["lon"]
    ok700 = valid_cells(M["v700"], M["sp"], 700); ok925 = valid_cells(M["v925"], M["sp"], 925)
    all700 = ok700.all(axis=0); all925 = ok925.all(axis=0)
    with np.errstate(invalid="ignore"):
        mv700 = np.where(all700, M["v700"].mean(axis=0), np.nan); mu700 = np.where(all700, M["u700"].mean(axis=0), np.nan)
        mu925 = np.where(all925, M["u925"].mean(axis=0), np.nan); mv925 = np.where(all925, M["v925"].mean(axis=0), np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(MISSING_COLOR)
    step = 4
    ax = axes[0, 0]
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(mv700), cmap=cmap, vmin=-VLIM, vmax=VLIM, shading="nearest")
    ax.quiver(lon[::step], lat[::step], mu700[::step, ::step], mv700[::step, ::step], scale=250, width=0.0025, color="k")
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="mean 700 hPa meridional wind (m/s)")
    ax.set_title("700 hPa mean wind vectors over the mean meridional wind (gray, invalid at some step)", fontsize=10)
    ax = axes[0, 1]
    spd = np.hypot(mu925, mv925)
    vir = plt.get_cmap("viridis").copy(); vir.set_bad(MISSING_COLOR)
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(spd), cmap=vir, vmin=0, vmax=20, shading="nearest")
    ax.quiver(lon[::step], lat[::step], mu925[::step, ::step], mv925[::step, ::step], scale=300, width=0.0025, color="k")
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="mean 925 hPa wind speed (m/s)")
    ax.set_title("925 hPa mean wind vectors over the mean speed (gray, invalid at some step, its own mask)", fontsize=10)
    ax = axes[1, 0]
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(pers["fraction"]), cmap="Blues", vmin=0, vmax=1, shading="nearest")
    fig.colorbar(mesh, ax=ax, shrink=0.8, label=f"fraction of valid steps with v700 at or below {NORTHERLY_MS:g} m/s")
    ax.set_title("700 hPa northerly persistence, fraction of the month's valid steps per cell", fontsize=10)
    ax = axes[1, 1]
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(pers["mean_run_days"]), cmap="Oranges", vmin=0, vmax=10, shading="nearest")
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="mean duration of uncensored northerly runs (days)")
    ax.set_title(f"700 hPa northerly runs (v at or below {NORTHERLY_MS:g} m/s), mean duration of runs inside the month", fontsize=10)
    for ax in axes.ravel():
        for x in MARK_LONS:
            ax.axvline(x, color="k", lw=0.8, ls="--")
        ax.set_xlabel("longitude (degrees E)"); ax.set_ylabel("latitude (degrees N)"); ax.set_xlim(BOX_LON); ax.set_ylim(BOX_LAT)
    fig.suptitle(f"{label}: monthly means and persistence from raw ERA5, every six-hourly step; terrain mask strict (surface pressure above the level); below-ground influence through regridding not removed. Dashed: 40 E and 60 E.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


def draw_lag_maps(path, label, lagmaps, ridges):
    plt = _plt()
    fig, axes = plt.subplots(1, len(lagmaps), figsize=(4.5 * len(lagmaps) + 1.5, 5.2), constrained_layout=True)
    offs = np.arange(-OFFSET_CELLS, OFFSET_CELLS + 1) * 0.5; lagd = np.arange(-LAG_STEPS, LAG_STEPS + 1) * 0.25
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(MISSING_COLOR)
    for ax, (b, corr) in zip(np.atleast_1d(axes), lagmaps.items()):
        mesh = ax.pcolormesh(offs, lagd, np.ma.masked_invalid(corr.T), cmap=cmap, vmin=-1, vmax=1, shading="nearest")
        r = ridges[b]
        if r["slope_days_per_degree"] is not None:
            xx = np.array(r["offset_span_deg"]); ax.plot(xx, r["slope_days_per_degree"] * xx + r["intercept_days"], color="k", lw=1.2)
        if r["speed_deg_per_day"] is not None:
            sp = f"ridge speed {r['speed_deg_per_day']:+.1f} deg/day"
        elif r["slope_days_per_degree"] is not None:
            sp = f"ridge slope {r['slope_days_per_degree']:+.2f} day/deg, standing"
        else:
            sp = "no ridge"
        ax.set_title(f"{b}: {sp} ({r['points']} offsets)", fontsize=10)
        ax.set_xlabel("zonal offset (degrees)"); ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    np.atleast_1d(axes)[0].set_ylabel("time lag (days, positive later at the offset)")
    fig.colorbar(mesh, ax=np.atleast_1d(axes).tolist(), shrink=0.7, label="correlation of 700 hPa band-mean anomalies")
    fig.suptitle(f"{label}: correlation of the 700 hPa band-mean anomaly at x with the anomaly at x plus the offset, lagged, averaged over x in 40 to 65 E. A ridge along zero lag is standing structure at the band scale, a tilted ridge is apparent movement; the line fits the ridge where its maximum exceeds {RIDGE_MIN_CORR:g}.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


# ------------------------------------------------------------------ the month

def render_month(year, month, out_dir, figures=True, gzip_out=False):
    """The month record and, by default, its five figures. With figures=False only the record
    is written (the standing climatology's archive run, `EASTERN_STANDING_CLIMATOLOGY_2026-10-18.md`,
    section 2), and with gzip_out=True the record is written compressed as `.json.gz`."""
    M = load_month(year, month)
    label = f"{MONTH_NAMES.get(month, month)} {year}"
    tag = f"eval60_regime_{year}_{month:02d}"
    paths = {k: os.path.join(out_dir, f"{tag}_{k}.png") for k in ("v700", "v850", "v600", "means", "lag")} if figures else {}
    paths["json"] = os.path.join(out_dir, f"{tag}.json" + (".gz" if gzip_out else ""))
    for p in paths.values():
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    rows = band_rows(M["lat"])
    means = {lvl: band_means(M[f"v{lvl}"], M["sp"], lvl, rows) for lvl in LEVELS_V}
    pers = persistence(M["v700"], M["sp"])
    lagmaps = {b: lag_offset_correlation(means[700][b]["series"], M["lon"]) for b in BANDS}
    ridges = {b: ridge(lagmaps[b]) for b in BANDS}
    if figures:
        for lvl in LEVELS_V:
            draw_hovmoller(paths[f"v{lvl}"], label, lvl, M["lon"], M["times"], means[lvl])
        draw_mean_maps(paths["means"], label, M, pers)
        draw_lag_maps(paths["lag"], label, lagmaps, ridges)
    rec = {"generated_by": {"script": "scripts/eastern_regime_views.py", "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_REGIME_CHARACTERIZATION_2026-10-18.md"},
           "year": year, "month": month, "label": label, "steps": len(M["times"]), "first": M["times"][0].strftime("%Y-%m-%dT%H"), "last": M["times"][-1].strftime("%Y-%m-%dT%H"),
           "box": {"lat": list(BOX_LAT), "lon": list(BOX_LON)}, "bands": {b: list(v) for b, v in BANDS.items()}, "files": M["files"],
           "figures": {k: {"path": p, "sha256": sha256_of(p)} for k, p in paths.items() if k != "json"} if figures else None,
           "lons": M["lon"].tolist(), "times": [t.strftime("%Y-%m-%dT%H") for t in M["times"]],
           "band_means_700": {b: np.round(means[700][b]["series"], 2).tolist() for b in BANDS},
           "unavailable": {str(lvl): {b: means[lvl][b]["unavailable"] for b in BANDS} for lvl in LEVELS_V}, "samples_per_band": int(means[700][next(iter(BANDS))]["samples"]),
           "persistence": {"threshold_ms": NORTHERLY_MS, "fraction": np.round(pers["fraction"], 3).tolist(), "mean_run_days": np.round(pers["mean_run_days"], 2).tolist(),
                           "uncensored_runs": pers["uncensored_runs"].tolist(), "censored_runs": pers["censored_runs"].tolist(), "invalid_at_any_step": pers["invalid_at_any_step"], "lat": M["lat"].tolist()},
           "lag_correlation": {"x_range": list(X_RANGE), "offsets_deg": (np.arange(-OFFSET_CELLS, OFFSET_CELLS + 1) * 0.5).tolist(), "lags_days": (np.arange(-LAG_STEPS, LAG_STEPS + 1) * 0.25).tolist(),
                               "maps": {b: np.round(lagmaps[b], 3).tolist() for b in BANDS}, "ridges": ridges}}
    if gzip_out:
        with gzip.open(paths["json"], "wt", encoding="utf-8") as fh:
            json.dump(rec, fh, default=float)
    else:
        with open(paths["json"], "w", encoding="utf-8") as fh:
            json.dump(rec, fh, default=float)
    return rec


def summary_rows(records):
    """Per month, derived from the records: unavailable samples per level, per band the
    ridge speed and its mean correlation, and per band the count of cells whose northerly
    persistence fraction is at least 0.5 with the mean uncensored run duration over them."""
    rows = []
    for rec in records:
        lat = np.array(rec["persistence"]["lat"]); frac = np.array(rec["persistence"]["fraction"], float); runs = np.array(rec["persistence"]["mean_run_days"], float)
        row = {"label": rec["label"], "steps": rec["steps"], "unavailable_700": sum(rec["unavailable"]["700"].values()), "unavailable_850": sum(rec["unavailable"]["850"].values()), "unavailable_600": sum(rec["unavailable"]["600"].values()),
               "invalid_at_any_step": rec["persistence"]["invalid_at_any_step"], "bands": {}}
        for b, (lo, hi) in rec["bands"].items():
            r = (lat >= lo - 1e-6) & (lat <= hi + 1e-6)
            f = frac[r]; d = runs[r]
            persistent = np.isfinite(f) & (f >= 0.5)
            rg = rec["lag_correlation"]["ridges"][b]
            row["bands"][b] = {"persistent_cells": int(persistent.sum()), "cells": int(np.isfinite(f).sum()), "mean_run_days_over_persistent": (float(np.nanmean(d[persistent])) if persistent.any() else None),
                               "ridge_speed": rg["speed_deg_per_day"], "ridge_points": rg["points"], "ridge_corr": rg["mean_ridge_corr"]}
        rows.append(row)
    return rows


def summary_table(rows):
    f = lambda x, n=1: "" if x is None else f"{x:.{n}f}"
    L = ["# the eastern regime months, derived from the month records (section 3)",
         "unavailable is the count of band-mean samples without 80 percent support, summed over the four bands; persistent cells are those whose fraction of valid steps at or below -4 m/s is at least 0.5, with the mean uncensored run duration over them; the ridge speed is the fitted apparent speed of the 700 hPa band-mean anomalies (negative westward), none where fewer than three offsets reach a correlation of 0.3", ""]
    bands = list(rows[0]["bands"])
    L.append("| month | steps | unavailable 700, 850, 600 | cells invalid at some step | " + " | ".join(f"{b}: persistent cells of valid, run days, ridge speed (points, corr)" for b in bands) + " |")
    L.append("| --- |" + " --- |" * (3 + len(bands)))
    for r in rows:
        cells = []
        for b in bands:
            x = r["bands"][b]
            cells.append(f"{x['persistent_cells']} of {x['cells']}, {f(x['mean_run_days_over_persistent'])}, {f(x['ridge_speed'])} ({x['ridge_points']}, {f(x['ridge_corr'], 2)})")
        L.append(f"| {r['label']} | {r['steps']} | {r['unavailable_700']}, {r['unavailable_850']}, {r['unavailable_600']} | {r['invalid_at_any_step']} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("render", "summary"))
    ap.add_argument("--year", type=int); ap.add_argument("--month", type=int)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--draw", default=None); ap.add_argument("--table", default=None)
    ap.add_argument("--no-figures", action="store_true", help="write the month record only"); ap.add_argument("--gzip", action="store_true", help="write the record compressed")
    a = ap.parse_args(argv)
    if a.mode == "render":
        if a.year is None or a.month is None:
            raise SystemExit("REFUSED: --year and --month are required")
        rec = render_month(a.year, a.month, a.out_dir, figures=not a.no_figures, gzip_out=a.gzip)
        print(json.dumps({"label": rec["label"], "steps": rec["steps"], "unavailable": rec["unavailable"], "ridges": {b: r["speed_deg_per_day"] for b, r in rec["lag_correlation"]["ridges"].items()}, "invalid_at_any_step": rec["persistence"]["invalid_at_any_step"]}))
        return 0
    if not a.draw or not a.table:
        raise SystemExit("REFUSED: --draw and --table are required for the summary")
    if os.path.exists(a.table):
        raise SystemExit(f"REFUSED: {a.table} exists and a record is never overwritten")
    draw = json.load(open(a.draw))
    months = [tuple(x) for x in draw["development_months"]] + [tuple(x) for x in draw["draw"]]
    records = [json.load(open(os.path.join(a.out_dir, f"eval60_regime_{y}_{m:02d}.json"))) for y, m in months]
    table = summary_table(summary_rows(records))
    with open(a.table, "w", encoding="utf-8") as fh:
        fh.write(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
