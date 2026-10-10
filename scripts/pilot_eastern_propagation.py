#!/usr/bin/env python3
"""A bounded statistical propagation study of retained ERA5 meridional wind east of 40 E.

The question, as declared in `EASTERN_PROPAGATION_STUDY_2026-10-09.md`: under the declared
sampling and filtering, do the retained winds show a repeatable westward-propagating
statistical signal east of 40 E? It does not decide whether eastern disturbances exist,
regression continuity is not individual feature identity, and an uncertain speed is not
stationarity.

Every choice below is fixed before any real season is examined. Per season, separately:
band means of v over valid cells (surface pressure above the level, strictly, at that
time), a sample kept only where at least MIN_VALID_FRACTION of the band's cells are valid;
a Lanczos band-pass applied only inside contiguous runs of kept samples, its output kept
only where the whole filter window lies inside the run, so terrain-invalid samples never
enter the filter and season and gap edges are discarded; lagged sums between each base
series and every analysis longitude, a pair kept only where both members are supported in
the same season. Seasons are pooled by summing their sums, and resampled whole for the
uncertainty, which carries the dependence within a season. The speed is the slope of the
tracked regression peak's longitude against lag. Synthetic traveling and standing signals
and the 10 E western comparison validate the diagnostic.

    python3 scripts/pilot_eastern_propagation.py --benchmark 1990
    python3 scripts/pilot_eastern_propagation.py --out <record.json> --png-prefix <path>
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "data/era5/region6h"
SEASONS = tuple(y for y in range(1983, 2008) if y != 2005)   # 24 retained seasons, 2005 held out and never opened
HELD_OUT = 2005
LEVELS = (700, 600)                        # 700 hPa primary, 600 hPa the single sensitivity
PRIMARY_LEVEL = 700
BANDS = {"southern": (5.0, 14.5), "northern": (15.0, 24.5)}  # inclusive, 0.5-degree rows, 20 each
LON_RANGE = (-20.0, 70.0)
EASTERN_BASES = (45.0, 50.0, 55.0)
CONTROL_BASE = 10.0
DT_DAYS = 0.25
PERIODS_DAYS = (2.5, 6.0)                  # the primary temporal band
LANCZOS_HALF = 40                          # steps each side, 10 days
MIN_VALID_FRACTION = 0.8
LAG_STEPS = tuple(range(-12, 13))          # -3 to +3 days; positive lag: the field later than the base
SPEED_MAX_STEP = 4                         # the speed uses lags -1 to +1 day
PEAK_WINDOW_DEG = 5.0
AMPLITUDE_FRACTION = 0.5
MIN_TRACKED_LAGS = 5
N_BOOT = 2000
SEED = 20261009
PRIMARY_TESTS = len(BANDS) * len(EASTERN_BASES)
ALPHA = 0.05
AEW_REFERENCE_DEG_PER_DAY = 4.0
MAX_FAILED_FRACTION = 0.10
NUMERICAL_ZERO = 1e-9                      # a least-squares slope this small is a flat track, exactly zero


def lanczos_bandpass(n=LANCZOS_HALF, periods=PERIODS_DAYS, dt_days=DT_DAYS):
    """Lanczos band-pass weights (Duchon 1979) for periods between the two given, in days."""
    f1, f2 = dt_days / periods[1], dt_days / periods[0]          # cycles per step
    k = np.arange(-n, n + 1, dtype=float)
    w = np.empty_like(k)
    nz = k != 0
    w[nz] = (np.sin(2 * np.pi * f2 * k[nz]) - np.sin(2 * np.pi * f1 * k[nz])) / (np.pi * k[nz])
    w[~nz] = 2.0 * (f2 - f1)
    sigma = np.ones_like(k)
    sigma[nz] = np.sin(np.pi * k[nz] / n) / (np.pi * k[nz] / n)
    return w * sigma


def filter_supported(series, weights):
    """The band-pass of each column inside its contiguous finite runs only, kept where the
    whole window lies inside the run, NaN elsewhere."""
    x = np.asarray(series, float)
    two_d = x.ndim == 2
    x = x if two_d else x[:, None]
    n = (len(weights) - 1) // 2
    out = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j])
        edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= 2 * n + 1:
                out[a + n:b - n, j] = np.convolve(x[a:b, j], weights[::-1], mode="valid")
    return out if two_d else out[:, 0]


def band_series(v, sp_pa, level_hpa, min_fraction=MIN_VALID_FRACTION):
    """Band means over (time, row, column) blocks: a cell is valid where v is finite and the
    surface pressure exceeds the level strictly; a sample is kept where at least
    `min_fraction` of the rows are valid, as the mean of the valid ones. Returns the series
    (time, column) and the per-cell validity fraction."""
    v = np.asarray(v, float)
    valid = np.isfinite(v) & (np.asarray(sp_pa, float) > level_hpa * 100.0)
    frac = valid.mean(axis=1)
    total = np.where(valid, v, 0.0).sum(axis=1)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return np.where(frac >= min_fraction, mean, np.nan), frac


def lag_sums(x, y, lags=LAG_STEPS):
    """Per lag and column, the sums over pairs (x[t], y[t + lag]) where both are finite:
    n, sum x, sum y, sum xy, sum xx, sum yy."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    T, L = y.shape
    out = np.zeros((6, len(lags), L))
    for i, lag in enumerate(lags):
        if lag >= 0:
            xs, ys = x[:T - lag], y[lag:]
        else:
            xs, ys = x[-lag:], y[:T + lag]
        m = np.isfinite(xs)[:, None] & np.isfinite(ys)
        xm = np.where(m, xs[:, None], 0.0); ym = np.where(m, ys, 0.0)
        out[:, i] = [m.sum(0), xm.sum(0), ym.sum(0), (xm * ym).sum(0), (xm * xm).sum(0), (ym * ym).sum(0)]
    return out


def regression_map(sums):
    """The lagged regression of the field on the base, in the field's units per one standard
    deviation of the base, and the lagged correlation."""
    n, sx, sy, sxy, sxx, syy = sums
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        return cov / np.sqrt(vx), cov / np.sqrt(vx * vy)


def track_speed(bmap, lons, base_lon, lags=LAG_STEPS):
    """The positive regression lobe at the base, followed outward lag by lag within
    PEAK_WINDOW_DEG of the previous peak while it keeps at least AMPLITUDE_FRACTION of its
    lag-zero amplitude, over lags within SPEED_MAX_STEP. Returns the least-squares slope of
    peak longitude against lag in degrees per day (negative westward) and the track, or
    (None, track) when fewer than MIN_TRACKED_LAGS lags are tracked."""
    lons = np.asarray(lons, float)
    i0 = list(lags).index(0)
    win = np.abs(lons - base_lon) <= PEAK_WINDOW_DEG
    row = np.where(win, bmap[i0], -np.inf)
    p0 = int(np.argmax(row))
    a0 = bmap[i0, p0]
    track = {0: float(lons[p0])}
    days = lambda tr: {str(k * DT_DAYS): tr[k] for k in sorted(tr)}
    if not np.isfinite(a0) or a0 <= 0:
        return None, days(track)
    for direction in (1, -1):
        prev = p0
        for step in range(1, SPEED_MAX_STEP + 1):
            i = i0 + direction * step
            row = np.where(np.abs(lons - lons[prev]) <= PEAK_WINDOW_DEG, bmap[i], -np.inf)
            p = int(np.argmax(row))
            if not np.isfinite(bmap[i, p]) or bmap[i, p] < AMPLITUDE_FRACTION * a0:
                break
            track[direction * step] = float(lons[p])
            prev = p
    if len(track) < MIN_TRACKED_LAGS:
        return None, days(track)
    keys = sorted(track)
    slope = float(np.polyfit(np.array(keys) * DT_DAYS, np.array([track[k] for k in keys]), 1)[0])
    return (0.0 if abs(slope) < NUMERICAL_ZERO else slope), days(track)


def classify(lo, hi, failed_fraction):
    if lo is None or failed_fraction > MAX_FAILED_FRACTION:
        return "inconclusive"
    if hi < 0:
        return "westward propagation"
    if lo > 0:
        return "eastward propagation"
    if lo > -AEW_REFERENCE_DEG_PER_DAY and hi < AEW_REFERENCE_DEG_PER_DAY:
        return "no resolved propagation"
    return "inconclusive"


def estimate(season_sums, lons, base_lon, n_boot=N_BOOT, seed=SEED, tests=PRIMARY_TESTS):
    """The pooled estimate and the season bootstrap for one base series. `season_sums` has
    shape (seasons, 6, lags, columns)."""
    season_sums = np.asarray(season_sums, float)
    bmap, rmap = regression_map(season_sums.sum(axis=0))
    speed, track = track_speed(bmap, lons, base_lon)
    rng = np.random.default_rng(seed)
    S = season_sums.shape[0]
    boot, failed = [], 0
    for _ in range(n_boot):
        pick = rng.integers(0, S, S)
        s, _t = track_speed(regression_map(season_sums[pick].sum(axis=0))[0], lons, base_lon)
        if s is None:
            failed += 1
        else:
            boot.append(s)
    boot = np.array(boot)
    q = ALPHA / tests / 2.0
    ci95 = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot.size else [None, None]
    ciadj = [float(np.percentile(boot, 100 * q)), float(np.percentile(boot, 100 * (1 - q)))] if boot.size else [None, None]
    ff = failed / n_boot
    return {"speed_deg_per_day": speed, "track": track, "ci95": ci95, "ci_adjusted": ciadj, "adjusted_level": 1 - ALPHA / tests,
            "failed_fraction": ff, "category": classify(ciadj[0] if speed is not None else None, ciadj[1], ff),
            "lag0_peak_m_s": float(bmap[list(LAG_STEPS).index(0)][int(np.argmin(np.abs(np.asarray(lons) - base_lon)))]),
            "bmap": bmap, "rmap": rmap}


def read_season(year, level, rows, cols):
    """One season's v at the level and surface pressure, as one contiguous block spanning
    the given rows and columns, with the time axis checked (1 Jun 00Z to 30 Sep 18Z,
    six-hourly, identical in both). Returns the block and its first row and column."""
    import netCDF4
    if year == HELD_OUT:
        raise SystemExit("REFUSED: the held-out season is never opened")
    out = {}
    for name, var in ((f"v{level}", "v"), ("sp", "sp")):
        d = netCDF4.Dataset(os.path.join(DATA_DIR, f"era5_{name}_{year}_6h_region.nc"))
        try:
            t = np.asarray(d.variables["valid_time"][:], float)
            a = d.variables[var]
            rs, cs = slice(int(min(rows)), int(max(rows)) + 1), slice(int(min(cols)), int(max(cols)) + 1)
            arr = np.asarray(a[:, 0, rs, cs] if a.ndim == 4 else a[:, rs, cs], float)
            if a.ndim == 4 and abs(float(d.variables["pressure_level"][0]) - level) > 1e-6:
                raise SystemExit(f"REFUSED: {name} {year} is not at {level} hPa")
            out[var] = arr
            out.setdefault("time", t)
            if not np.array_equal(out["time"], t):
                raise SystemExit(f"REFUSED: {year} v and sp times differ")
        finally:
            d.close()
    first = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=float(out["time"][0]))
    if first != dt.datetime(year, 6, 1) or len(out["time"]) != 488 or set(np.diff(out["time"])) != {21600.0}:
        raise SystemExit(f"REFUSED: {year} is not 1 Jun 00Z to 30 Sep 18Z six-hourly")
    out["row0"], out["col0"] = int(min(rows)), int(min(cols))
    return out


def grid_indices():
    import netCDF4
    d = netCDF4.Dataset(os.path.join(DATA_DIR, f"era5_v{PRIMARY_LEVEL}_{SEASONS[0]}_6h_region.nc"))
    lat = np.asarray(d.variables["latitude"][:], float); lon = np.asarray(d.variables["longitude"][:], float)
    d.close()
    cols = np.flatnonzero((lon >= LON_RANGE[0] - 1e-6) & (lon <= LON_RANGE[1] + 1e-6))
    rows = {b: np.flatnonzero((lat >= lo - 1e-6) & (lat <= hi + 1e-6)) for b, (lo, hi) in BANDS.items()}
    if any(r.size != 20 for r in rows.values()):
        raise SystemExit("REFUSED: a band does not hold 20 rows")
    return lat, lon[cols], rows, cols


def process_season(year, level, rows, cols, lons, weights):
    """Per band: the filtered series, the supported-sample counts, the analyzed dates at each
    base, and the lag sums for each base."""
    allrows = np.concatenate(list(rows.values()))
    data = read_season(year, level, allrows, cols)
    ci = np.asarray(cols) - data["col0"]
    out = {}
    for band, r in rows.items():
        ri = np.asarray(r) - data["row0"]
        series, _ = band_series(data["v"][:, ri][:, :, ci], data["sp"][:, ri][:, :, ci], level)
        filt = filter_supported(series, weights)
        rec = {"supported": np.isfinite(filt).sum(axis=0), "raw_valid": np.isfinite(series).sum(axis=0),
               "sumsq": np.nansum(filt ** 2, axis=0), "sums": {}, "dates": {}}
        for base in EASTERN_BASES + (CONTROL_BASE,):
            j = int(np.argmin(np.abs(lons - base)))
            x = filt[:, j]
            rec["sums"][base] = lag_sums(x, filt)
            ok = np.flatnonzero(np.isfinite(x))
            to_date = lambda k: f"{dt.datetime(1970, 1, 1) + dt.timedelta(seconds=float(data['time'][k])):%Y-%m-%dT%H}"
            rec["dates"][base] = None if ok.size == 0 else {"first": to_date(ok[0]), "last": to_date(ok[-1]), "samples": int(ok.size)}
        out[band] = rec
    return out


def draw(results, lons, level, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lags = np.array(LAG_STEPS) * DT_DAYS
    bases = (CONTROL_BASE,) + EASTERN_BASES
    fig, axes = plt.subplots(len(BANDS), len(bases), figsize=(20, 8), sharex=True, sharey=True)
    for i, band in enumerate(BANDS):
        for j, base in enumerate(bases):
            ax = axes[i, j]; e = results[band][base]
            vmax = max(abs(np.nanmax(e["bmap"])), abs(np.nanmin(e["bmap"])), 1e-6)
            m = ax.pcolormesh(lons, lags, e["bmap"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            tr = {float(k): v for k, v in e["track"].items()}
            ax.plot([tr[k] for k in sorted(tr)], sorted(tr), "k.-", ms=4)
            ax.axvline(base, color="0.2", lw=0.8); ax.axvline(40.0, color="0.2", lw=0.8, ls="--"); ax.axhline(0, color="0.5", lw=0.5)
            lo, hi = e["ci_adjusted"]
            sp = "none" if e["speed_deg_per_day"] is None else f"{e['speed_deg_per_day']:.1f}"
            ax.set_title(f"{band}, base {base:g} E\nspeed {sp} deg/day, adj. CI [{'' if lo is None else f'{lo:.1f}'}, {'' if hi is None else f'{hi:.1f}'}]\n{e['category']}", fontsize=8)
            fig.colorbar(m, ax=ax, shrink=0.8, label="m/s per base s.d." if j == len(bases) - 1 else None)
    for ax in axes[-1]:
        ax.set_xlabel("longitude (degrees east)")
    for ax in axes[:, 0]:
        ax.set_ylabel("lag (days), field after base")
    fig.suptitle(f"ERA5 {level} hPa v, {PERIODS_DAYS[0]:g} to {PERIODS_DAYS[1]:g} day band-pass, lagged regression on each base, 24 seasons (2005 excluded), terrain-valid samples only", fontsize=10)
    path = f"{png}_v{level}.png"
    fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)
    return path


def draw_coverage(amp, cov, lons, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for (level, band), a in amp.items():
        axes[0].plot(lons, a, label=f"{band} {level} hPa")
        axes[1].plot(lons, cov[(level, band)], label=f"{band} {level} hPa")
    axes[0].set_ylabel("band-passed v standard deviation (m/s)"); axes[1].set_ylabel("supported fraction of the 24 x 488 samples")
    for ax in axes:
        ax.axvline(40.0, color="0.3", ls="--", lw=0.8); ax.set_xlabel("longitude (degrees east)"); ax.legend(fontsize=7)
    path = f"{png}_amplitude_coverage.png"
    fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", type=int, help="time one season at both levels and print only the elapsed time")
    ap.add_argument("--out"); ap.add_argument("--png-prefix")
    args = ap.parse_args(argv)
    weights = lanczos_bandpass()
    lat, lons, rows, cols = grid_indices()
    if args.benchmark is not None:
        t0 = time.time()
        for level in LEVELS:
            process_season(args.benchmark, level, rows, cols, lons, weights)
        print(f"benchmark: one season, both levels, {time.time() - t0:.1f} s")
        return 0
    if not args.out or not args.png_prefix:
        raise SystemExit("REFUSED: the full pass needs --out and --png-prefix")
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    t0 = time.time()
    record = {"generated_by": "scripts/pilot_eastern_propagation.py", "seasons": list(SEASONS), "held_out": HELD_OUT,
              "declared": {"levels": list(LEVELS), "primary_level": PRIMARY_LEVEL, "bands": BANDS, "lon_range": list(LON_RANGE),
                           "eastern_bases": list(EASTERN_BASES), "control_base": CONTROL_BASE, "periods_days": list(PERIODS_DAYS),
                           "lanczos_half_steps": LANCZOS_HALF, "min_valid_fraction": MIN_VALID_FRACTION, "lags_days": [l * DT_DAYS for l in LAG_STEPS],
                           "speed_lags_days": [-SPEED_MAX_STEP * DT_DAYS, SPEED_MAX_STEP * DT_DAYS], "peak_window_deg": PEAK_WINDOW_DEG,
                           "amplitude_fraction": AMPLITUDE_FRACTION, "min_tracked_lags": MIN_TRACKED_LAGS, "n_boot": N_BOOT, "seed": SEED,
                           "primary_tests": PRIMARY_TESTS, "alpha": ALPHA, "aew_reference_deg_per_day": AEW_REFERENCE_DEG_PER_DAY,
                           "max_failed_fraction": MAX_FAILED_FRACTION}, "results": {}, "coverage": {}, "amplitude": {}, "analyzed_dates": {}}
    amp, cov, figs = {}, {}, []
    for level in LEVELS:
        per = {band: {base: [] for base in EASTERN_BASES + (CONTROL_BASE,)} for band in BANDS}
        sup = {band: np.zeros(lons.size) for band in BANDS}; raw = {band: np.zeros(lons.size) for band in BANDS}
        ssq = {band: np.zeros(lons.size) for band in BANDS}
        for year in SEASONS:
            s = process_season(year, level, rows, cols, lons, weights)
            for band in BANDS:
                sup[band] += s[band]["supported"]; raw[band] += s[band]["raw_valid"]; ssq[band] += s[band]["sumsq"]
                for base in per[band]:
                    per[band][base].append(s[band]["sums"][base])
                    record["analyzed_dates"].setdefault(str(level), {}).setdefault(band, {}).setdefault(f"{base:g}", {})[str(year)] = s[band]["dates"][base]
        res = {}
        for band in BANDS:
            res[band] = {}
            for base in per[band]:
                e = estimate(np.stack(per[band][base]), lons, base)
                res[band][base] = e
            total = len(SEASONS) * 488
            cov[(level, band)] = sup[band] / total
            amp[(level, band)] = np.sqrt(ssq[band] / np.maximum(sup[band], 1))
            record["coverage"][f"{level}_{band}"] = {"supported_fraction": [round(float(c), 4) for c in cov[(level, band)]],
                                                     "terrain_valid_fraction_before_filter": [round(float(c), 4) for c in raw[band] / total]}
            record["amplitude"][f"{level}_{band}"] = [round(float(a), 3) for a in amp[(level, band)]]
        figs.append(draw(res, lons, level, args.png_prefix))
        record["results"][str(level)] = {band: {f"{base:g}": {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items()}
                                                for base, e in res[band].items()} for band in BANDS}
    figs.append(draw_coverage(amp, cov, lons, args.png_prefix))
    record["lons"] = [float(x) for x in lons]
    record["figures"] = figs
    record["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "x") as fh:
        json.dump(record, fh)
    print(f"done in {record['elapsed_seconds']} s, record {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
