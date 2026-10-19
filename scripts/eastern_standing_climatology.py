#!/usr/bin/env python3
"""The standing climatology of the eastern summer regime at 700 hPa
(`EASTERN_STANDING_CLIMATOLOGY_2026-10-18.md`, section 2): every retained season except the
closed one, rendered as compressed month records by the characterization's renderer
without figures, then aggregated by calendar month into the persistence climatology (mean,
standard deviation and count over seasons of each cell's monthly fraction of valid steps
at or below -4 m/s, and the mean run duration), the band-bin table, the ridge climatology,
and the candidate recurrence by longitude at a declared prominence floor. Fractions,
counts and positions, nothing linked and nothing explained.

    python3 scripts/eastern_standing_climatology.py render --out-dir DIR [--years 1983 2007]
    python3 scripts/eastern_standing_climatology.py aggregate --records DIR --out-dir OUT --reference REFDIR
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_regime_views as V  # noqa: E402
import eastern_regime_composite as C  # noqa: E402
import eastern_candidate_positions as CP  # noqa: E402
import eastern_t1_ledger as T  # noqa: E402
import pilot_eastern_propagation as P  # noqa: E402

YEARS = [y for y in range(1983, 2008) if y != P.HELD_OUT]
MONTHS = (6, 7, 8, 9)
PROMINENCE_FLOOR = 2.0
PROMINENCE_SENSITIVITY = 4.0
RECURRENCE_BIN_DEG = 2.0
LON_BINS = C.LON_BINS
PEAK_FLOOR = 0.5            # a peak of the mean persistence map is at or above this fraction
PEAK_WINDOW_CELLS = 5       # and the maximum of its window, 2.5 degrees each way on the half-degree grid
PEAK_LIMIT = 10
MONTH_NAMES = {6: "June", 7: "July", 8: "August", 9: "September"}


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_record(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ render

def render_all(out_dir, years=YEARS, months=MONTHS):
    """Every month record not yet present, compressed, without figures. The closed season
    is refused by the renderer itself."""
    done, made = [], []
    for y in years:
        for m in months:
            path = os.path.join(out_dir, f"eval60_regime_{y}_{m:02d}.json.gz")
            if os.path.exists(path):
                done.append((y, m)); continue
            V.render_month(y, m, out_dir, figures=False, gzip_out=True)
            made.append((y, m))
    return {"already_present": done, "rendered": made}


# ------------------------------------------------------------------ consistency with the committed ten months

def check_reference(records_dir, reference_dir, pairs):
    """For each (year, month), the archive record's 700 hPa band means and persistence
    fraction must equal the committed characterization record's (both stored rounded)."""
    out = {}
    for y, m in pairs:
        a = load_record(os.path.join(records_dir, f"eval60_regime_{y}_{m:02d}.json.gz"))
        b = load_record(os.path.join(reference_dir, f"eval60_regime_{y}_{m:02d}.json"))
        same_bands = all(np.allclose(np.array(a["band_means_700"][k], float), np.array(b["band_means_700"][k], float), equal_nan=True) for k in a["band_means_700"])
        same_pers = np.allclose(np.array(a["persistence"]["fraction"], float), np.array(b["persistence"]["fraction"], float), equal_nan=True)
        out[f"{y}-{m:02d}"] = {"band_means_equal": bool(same_bands), "persistence_equal": bool(same_pers)}
    return out


def check_archive_identity(files, records):
    """The archive is exactly the declared seasons times the declared months, each once,
    every file named for the record it holds, all on one grid with one band set. A count of
    96 accepted a duplicated season in place of a missing one (review of 2026-10-18, F2)."""
    expected = sorted((y, m) for y in YEARS for m in MONTHS)
    actual = sorted((int(r["year"]), int(r["month"])) for r in records)
    if actual != expected:
        missing = sorted(set(expected) - set(actual)); unexpected = sorted(set(actual) - set(expected))
        duplicated = sorted({p for p in actual if actual.count(p) > 1})
        raise SystemExit(f"REFUSED: the archive is not the declared months once each: missing {missing}, unexpected {unexpected}, duplicated {duplicated}")
    for path, r in zip(files, records):
        want = f"eval60_regime_{r['year']}_{r['month']:02d}.json.gz"
        if os.path.basename(path) != want:
            raise SystemExit(f"REFUSED: {path} holds the record of {r['year']}-{r['month']:02d}")
    first = records[0]
    for r in records[1:]:
        if r["lons"] != first["lons"] or r["persistence"]["lat"] != first["persistence"]["lat"] or r["bands"] != first["bands"]:
            raise SystemExit(f"REFUSED: {r['year']}-{r['month']:02d} is not on the first record's grid and band set")


def reference_months(reference_dir):
    """The ten characterization months, read from the month draw record in the reference
    directory (the six drawn and the four development months), each of which must have its
    committed record there. Whatever files happen to exist is not the reference set
    (review of 2026-10-18, F2)."""
    draw_path = os.path.join(reference_dir, "eval60_regime_month_draw.json")
    if not os.path.exists(draw_path):
        raise SystemExit(f"REFUSED: {draw_path} is missing, so the reference months are not declared")
    with open(draw_path, encoding="utf-8") as fh:
        draw = json.load(fh)
    pairs = [(int(y), int(m)) for y, m in list(draw["draw"]) + list(draw["development_months"])]
    missing = [p for p in pairs if not os.path.exists(os.path.join(reference_dir, f"eval60_regime_{p[0]}_{p[1]:02d}.json"))]
    if len(pairs) != 10 or len(set(pairs)) != 10 or missing:
        raise SystemExit(f"REFUSED: the reference set is not the ten declared months with their records: {len(pairs)} declared, missing {missing}")
    return pairs


# ------------------------------------------------------------------ aggregate

def at_least_half(frac):
    """Per cell, whether the mean over seasons of the stored persistence fraction is at
    least one half, decided on the stored thousandths (the month records round the fraction
    to three decimals): the sum of the finite cells' thousandths at or above 500 times
    their count. A floating mean of three-decimal values can fall below 0.5 by one unit in
    the last place when the exact mean is 0.500 (two cells in the archive, review of
    2026-10-18, F1), and the four-decimal mean map in the record shows 0.49996 as 0.5000,
    so neither is the decision."""
    th = np.rint(np.asarray(frac, float) * 1000.0)
    n = np.isfinite(th).sum(axis=0)
    total = np.nansum(th, axis=0)
    return (n > 0) & (total >= 500.0 * n)


def persistence_climatology(records):
    """Per calendar month and over all: the mean, standard deviation and count over
    seasons of the monthly persistence fraction per cell, the mean of the mean uncensored
    run duration, and the at-least-one-half membership decided by `at_least_half`. NaN
    cells contribute nothing."""
    by_m = {}
    for rec in records:
        by_m.setdefault(rec["month"], []).append(rec)
    out = {}
    groups = {str(m): recs for m, recs in by_m.items()}
    groups["all"] = records
    for key, recs in groups.items():
        frac = np.array([r["persistence"]["fraction"] for r in recs], float)
        runs = np.array([r["persistence"]["mean_run_days"] for r in recs], float)
        with np.errstate(invalid="ignore"):
            n = np.isfinite(frac).sum(axis=0)
            mean = np.where(n > 0, np.nanmean(frac, axis=0), np.nan)
            std = np.where(n > 1, np.nanstd(frac, axis=0, ddof=1), np.nan)
            run_mean = np.where(np.isfinite(runs).sum(axis=0) > 0, np.nanmean(runs, axis=0), np.nan)
        out[key] = {"seasons": len(recs), "mean": mean, "std": std, "count": n, "run_mean": run_mean, "half_mask": at_least_half(frac)}
    return out


def band_bin_table(records, lat, lon):
    """Per calendar month and band, the mean and standard deviation over seasons of the
    persistence fraction averaged over the cells of each five-degree longitude bin."""
    out = {}
    bands = records[0]["bands"]
    for rec in records:
        frac = np.array(rec["persistence"]["fraction"], float)
        for b, (lo, hi) in bands.items():
            rows = (lat >= lo - 1e-6) & (lat <= hi + 1e-6)
            for (x0, x1) in LON_BINS:
                cols = (lon >= x0 - 1e-6) & (lon <= x1 + 1e-6)
                block = frac[np.ix_(rows, cols)]
                val = None if np.all(np.isnan(block)) else float(np.nanmean(block))
                out.setdefault(str(rec["month"]), {}).setdefault(b, {}).setdefault(f"{x0:g} to {x1:g} E", []).append(val)
    table = {}
    for m, bands_ in out.items():
        for b, bins in bands_.items():
            for name, vals in bins.items():
                v = [x for x in vals if x is not None]
                table.setdefault(m, {}).setdefault(b, {})[name] = {"mean": float(np.mean(v)) if v else None, "std": float(np.std(v, ddof=1)) if len(v) > 1 else None, "seasons": len(v)}
    return table


def peaks(mean, std, count, run_mean, lat, lon, floor=PEAK_FLOOR, window=PEAK_WINDOW_CELLS, limit=PEAK_LIMIT):
    """Cells of a mean persistence map at or above the floor that are the maximum of their
    window (plus or minus `window` cells each way, NaN ignored). A tied cell is kept only
    if it is the first equal-valued cell of its window in row order, which keeps a plateau
    narrower than the window once and does not guarantee one entry per connected plateau
    (a connected plateau wider than the window can be listed at more than one cell; review
    of 2026-10-18, F9). Ordered by the mean, at most `limit`. Positions of the strongest
    standing persistence, nothing grouped or named."""
    out = []
    ny, nx = mean.shape
    for j in range(ny):
        for i in range(nx):
            v = mean[j, i]
            if not np.isfinite(v) or v < floor:
                continue
            j0, i0 = max(0, j - window), max(0, i - window)
            block = mean[j0: j + window + 1, i0: i + window + 1]
            if v < np.nanmax(block):
                continue
            first = np.argwhere(block == v)[0]
            if (int(first[0]) + j0, int(first[1]) + i0) != (j, i):
                continue
            out.append({"lon": float(lon[i]), "lat": float(lat[j]), "mean": round(float(v), 3), "std": (None if not np.isfinite(std[j, i]) else round(float(std[j, i]), 3)),
                        "seasons": int(count[j, i]), "run_mean_days": (None if not np.isfinite(run_mean[j, i]) else round(float(run_mean[j, i]), 2))})
    out.sort(key=lambda q: (-q["mean"], q["lat"], q["lon"]))
    return out[:limit]


def composite_comparison(all_mean, composite, lat, lon):
    """The ten-month composite's bin means (its record) against the all-months bin means
    computed from the climatology's all-months mean map over the same band rows and bin
    columns; the difference is composite minus climatology."""
    rows = []
    for b, (lo, hi) in V.BANDS.items():
        rsel = (lat >= lo - 1e-6) & (lat <= hi + 1e-6)
        for (x0, x1) in LON_BINS:
            name = f"{x0:g} to {x1:g} E"
            csel = (lon >= x0 - 1e-6) & (lon <= x1 + 1e-6)
            block = all_mean[np.ix_(rsel, csel)]
            clim = None if np.all(np.isnan(block)) else float(np.nanmean(block))
            comp = composite["mean_fraction_by_band_and_bin"][b][name]
            diff = None if clim is None or comp is None else float(comp) - clim
            rows.append({"band": b, "bin": name, "composite": comp, "climatology": clim, "difference": diff})
    d = np.array([r["difference"] for r in rows if r["difference"] is not None])
    summary = {"bins": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)), "min": float(d.min()), "max": float(d.max()),
               "composite_higher_by_0.10": int((d >= 0.10).sum()), "composite_lower_by_0.10": int((d <= -0.10).sum())} if d.size else None
    return {"rows": rows, "summary": summary}


def ridge_climatology(records):
    out = {}
    for rec in records:
        for b, r in rec["lag_correlation"]["ridges"].items():
            k = C.ridge_reading(r)
            d = out.setdefault(str(rec["month"]), {}).setdefault(b, {"westward": 0, "standing": 0, "eastward": 0, "none": 0, "westward_speeds": []})
            d[k] += 1
            if k == "westward":
                d["westward_speeds"].append(r["speed_deg_per_day"])
    for m in out.values():
        for d in m.values():
            d["median_westward_speed"] = statistics.median(d["westward_speeds"]) if d["westward_speeds"] else None
            del d["westward_speeds"]
    return out


def candidate_recurrence(records, floors=(PROMINENCE_FLOOR, PROMINENCE_SENSITIVITY)):
    """Per calendar month, band and floor: the count of candidates at or above the floor
    per two-degree longitude bin divided by the band-days, and the mean crowdedness at the
    floor."""
    lon = np.array(records[0]["lons"], float)
    edges = np.arange(20.0, 76.0 + 1e-9, RECURRENCE_BIN_DEG)
    acc = {}
    for rec in records:
        for b in rec["bands"]:
            dm = T.daily_means(rec, b)
            d = acc.setdefault(str(rec["month"]), {}).setdefault(b, {"band_days": 0, **{f"{f:g}": {"hist": np.zeros(len(edges) - 1), "count": 0} for f in floors}})
            for day, v in dm.items():
                d["band_days"] += 1
                cs = CP.candidates_on(v, lon, context=f" ({rec['year']}-{rec['month']:02d}, {b}, {day})")
                for f in floors:
                    keep = [c["lon"] for c in cs if c["prominence"] >= f]
                    d[f"{f:g}"]["count"] += len(keep)
                    if keep:
                        d[f"{f:g}"]["hist"] += np.histogram(keep, bins=edges)[0]
    out = {"bin_edges_deg": edges.tolist(), "by_month_band": {}}
    for m, bands in acc.items():
        for b, d in bands.items():
            entry = {"band_days": d["band_days"]}
            for f in floors:
                h = d[f"{f:g}"]["hist"] / max(d["band_days"], 1)
                entry[f"{f:g}"] = {"recurrence_per_band_day": h.tolist(), "mean_crowdedness": d[f"{f:g}"]["count"] / max(d["band_days"], 1),
                                   "top_bins": [{"lon_lo": float(edges[i]), "lon_hi": float(edges[i + 1]), "recurrence": float(h[i])} for i in np.argsort(h)[::-1][:3]]}
            out["by_month_band"].setdefault(m, {})[b] = entry
    return out


def tables_text(pc, bt, rc, cr, ref, months_rendered, pk=None, cc=None):
    f = lambda x, n=2: "" if x is None else f"{x:.{n}f}"
    L = ["# the standing climatology, derived (section 2)",
         f"{months_rendered} months over {len(YEARS)} seasons; a persistence fraction is a cell's monthly fraction of valid steps with the 700 hPa meridional wind at or below -4 m/s; bins are the characterization's five-degree longitude bins; a ridge reads by the characterization's convention; a candidate is an interior local minimum of the daily-mean band mean at or below -4 m/s with prominence at or above the floor", ""]
    L.append("## the ten characterization months rendered again, against the committed records")
    L.append("")
    L.append("| month | band means equal | persistence equal |"); L.append("| --- | --- | --- |")
    for k, v in ref.items():
        L.append(f"| {k} | {'yes' if v['band_means_equal'] else 'NO'} | {'yes' if v['persistence_equal'] else 'NO'} |")
    bands = list(next(iter(bt.values())).keys())
    for m in sorted(bt, key=int):
        L += ["", f"## {MONTH_NAMES[int(m)]}, persistence fraction by band and bin, mean over seasons (standard deviation)", "",
              "| band | " + " | ".join(next(iter(bt[m].values())).keys()) + " |", "| --- |" + " --- |" * len(LON_BINS)]
        for b in bands:
            L.append(f"| {b} | " + " | ".join(f"{f(v['mean'])} ({f(v['std'])})" for v in bt[m][b].values()) + " |")
    L += ["", "## ridge readings by calendar month and band, seasons counted", "", "| month | band | westward | standing | eastward | none | median westward speed (degrees per day) |", "| --- |" + " --- |" * 6]
    for m in sorted(rc, key=int):
        for b in bands:
            d = rc[m][b]; L.append(f"| {MONTH_NAMES[int(m)]} | {b} | {d['westward']} | {d['standing']} | {d['eastward']} | {d['none']} | {f(d['median_westward_speed'], 1)} |")
    for fl in (f"{PROMINENCE_FLOOR:g}", f"{PROMINENCE_SENSITIVITY:g}"):
        L += ["", f"## candidate recurrence at prominence {fl} m/s or more, the three two-degree bins with the highest recurrence per band-day, and the mean crowdedness", "",
              "| month | band | band-days | mean crowdedness | first bin (recurrence) | second | third |", "| --- |" + " --- |" * 6]
        for m in sorted(cr["by_month_band"], key=int):
            for b in bands:
                e = cr["by_month_band"][m][b]; tb = e[fl]["top_bins"]
                cells = [f"{t['lon_lo']:g} to {t['lon_hi']:g} E ({t['recurrence']:.2f})" for t in tb]
                L.append(f"| {MONTH_NAMES[int(m)]} | {b} | {e['band_days']} | {e[fl]['mean_crowdedness']:.2f} | " + " | ".join(cells) + " |")
    L += ["", "## persistence climatology, cells with a mean fraction of at least one half by calendar month", ""]
    for key in sorted(pc, key=lambda k: (k == "all", k)):
        d = pc[key]; mask = d["half_mask"]; n = int(mask.sum()); tot = int(np.sum(np.isfinite(d["mean"])))
        L.append(f"- {MONTH_NAMES[int(key)] if key != 'all' else 'all four months'} ({d['seasons']} records): {n} of {tot} cells (decided on the stored thousandths), median standard deviation over seasons among those cells {f(float(np.nanmedian(d['std'][mask])) if n else None)}")
    if pk is not None:
        L += ["", f"## peaks of the mean persistence map by calendar month, cells at or above {PEAK_FLOOR:g} that are the maximum within {PEAK_WINDOW_CELLS / 2:g} degrees, at most {PEAK_LIMIT} by mean", "",
              "| month | peak | longitude E | latitude N | mean | standard deviation over seasons | seasons | mean run duration (days) |", "| --- |" + " --- |" * 7]
        for key in sorted(pk, key=lambda k: (k == "all", k)):
            for n_, q in enumerate(pk[key], 1):
                L.append(f"| {MONTH_NAMES[int(key)] if key != 'all' else 'all four months'} | {n_} | {q['lon']:g} | {q['lat']:g} | {q['mean']:.2f} | {f(q['std'])} | {q['seasons']} | {f(q['run_mean_days'])} |")
    if cc is not None:
        L += ["", "## the ten-month composite against the climatology, persistence fraction by band and bin (composite minus the all-months climatology, the climatology's bin mean computed from its all-months mean map)", "",
              "| band | " + " | ".join(f"{x0:g} to {x1:g} E" for (x0, x1) in LON_BINS) + " |", "| --- |" + " --- |" * len(LON_BINS)]
        for b in V.BANDS:
            cells = [r for r in cc["rows"] if r["band"] == b]
            L.append(f"| {b} | " + " | ".join("" if r["difference"] is None else f"{r['composite']:.2f} vs {r['climatology']:.2f} ({r['difference']:+.2f})" for r in cells) + " |")
        sm = cc["summary"]
        if sm:
            L.append("")
            L.append(f"bins compared {sm['bins']}; composite minus climatology mean {sm['mean']:+.3f}, median {sm['median']:+.3f}, range {sm['min']:+.2f} to {sm['max']:+.2f}; bins where the composite is higher by at least 0.10: {sm['composite_higher_by_0.10']}, lower by at least 0.10: {sm['composite_lower_by_0.10']}")
    return "\n".join(L) + "\n"


def draw_maps(path, pc, lat, lon, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pc[key]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
    panels = [("mean persistence fraction", d["mean"], "Blues", 0, 1), ("standard deviation over seasons", d["std"], "Purples", 0, 0.4), ("mean run duration (days)", d["run_mean"], "Oranges", 0, 8)]
    for ax, (title, arr, cmap, lo, hi) in zip(axes, panels):
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.75")
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(arr), cmap=cm, vmin=lo, vmax=hi, shading="nearest")
        fig.colorbar(mesh, ax=ax, shrink=0.85)
        for x in (40.0, 60.0):
            ax.axvline(x, color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10); ax.set_xlabel("longitude (degrees E)"); ax.set_ylabel("latitude (degrees N)")
    name = MONTH_NAMES[int(key)] if key != "all" else "all four months"
    fig.suptitle(f"{name}, {d['seasons']} month records over {len(YEARS)} seasons (2005 closed): the 700 hPa northerly persistence climatology (fraction of valid six-hourly steps at or below -4 m/s), its interannual standard deviation, and the mean uncensored run duration. Terrain mask strict. Positions and fractions, nothing explained.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


def draw_recurrence(path, cr, floor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.array(cr["bin_edges_deg"]); mid = (edges[:-1] + edges[1:]) / 2
    bands = list(next(iter(cr["by_month_band"].values())).keys())
    fig, axes = plt.subplots(1, len(bands), figsize=(4.5 * len(bands) + 1, 4.6), sharey=True, constrained_layout=True)
    for ax, b in zip(np.atleast_1d(axes), bands):
        for m in sorted(cr["by_month_band"], key=int):
            h = cr["by_month_band"][m][b][f"{floor:g}"]["recurrence_per_band_day"]
            ax.plot(mid, h, lw=1.4, label=MONTH_NAMES[int(m)])
        for x in (40.0, 60.0):
            ax.axvline(x, color="k", lw=0.8, ls="--")
        ax.set_title(b, fontsize=11); ax.set_xlabel("longitude (degrees E)"); ax.grid(True, color="0.8", lw=0.4)
    np.atleast_1d(axes)[0].set_ylabel("candidates per band-day"); np.atleast_1d(axes)[0].legend(fontsize=9)
    fig.suptitle(f"Recurrence by longitude of prominent northerly minima of the daily-mean 700 hPa band mean (prominence at least {floor:g} m/s), by calendar month over {len(YEARS)} seasons (2005 closed). Candidates per band-day in two-degree bins, nothing linked across days.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("render", "aggregate"))
    ap.add_argument("--out-dir", required=True); ap.add_argument("--records", default=None); ap.add_argument("--reference", default=None)
    ap.add_argument("--years", nargs=2, type=int, default=None); ap.add_argument("--composite", default=None, help="the ten-month composite record to compare against")
    a = ap.parse_args(argv)
    if a.mode == "render":
        os.makedirs(a.out_dir, exist_ok=True)
        years = YEARS if a.years is None else [y for y in YEARS if a.years[0] <= y <= a.years[1]]
        res = render_all(a.out_dir, years)
        print(json.dumps({"rendered": len(res["rendered"]), "already_present": len(res["already_present"])}))
        return 0
    if not a.records or not a.reference:
        raise SystemExit("REFUSED: --records and --reference are required to aggregate")
    paths = {"persistence": os.path.join(a.out_dir, "eval60_clim_persistence.json"), "candidates": os.path.join(a.out_dir, "eval60_clim_candidates.json"), "tables": os.path.join(a.out_dir, "eval60_clim_tables.md"),
             "recurrence": os.path.join(a.out_dir, "eval60_clim_recurrence.png")}
    for m in MONTHS:    # one map figure per calendar month and one recurrence figure, as declared; the all-months climatology and the sensitivity floor stay in the records and tables
        paths[f"map_{m}"] = os.path.join(a.out_dir, f"eval60_clim_map_{m:02d}.png")
    for p in paths.values():
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
    records = [load_record(p) for p in files]
    if len(records) != len(YEARS) * len(MONTHS):
        raise SystemExit(f"REFUSED: {len(records)} month records, not {len(YEARS) * len(MONTHS)}")
    if any(r["year"] == P.HELD_OUT for r in records):
        raise SystemExit("REFUSED: the closed season is present")
    check_archive_identity(files, records)
    lat = np.array(records[0]["persistence"]["lat"], float); lon = np.array(records[0]["lons"], float)
    ref_pairs = reference_months(a.reference)
    ref = check_reference(a.records, a.reference, ref_pairs)
    pc = persistence_climatology(records); bt = band_bin_table(records, lat, lon); rc = ridge_climatology(records); cr = candidate_recurrence(records)
    pk = {k: peaks(d["mean"], d["std"], d["count"], d["run_mean"], lat, lon) for k, d in pc.items()}
    cc = None
    if a.composite:
        with open(a.composite, encoding="utf-8") as fh:
            composite = json.load(fh)
        cc = composite_comparison(pc["all"]["mean"], composite, lat, lon)
        gen_extra = {"composite": {"path": a.composite, "sha256": sha256_of(a.composite)}}
    else:
        gen_extra = {}
    for m in MONTHS:
        draw_maps(paths[f"map_{m}"], pc, lat, lon, str(m))
    draw_recurrence(paths["recurrence"], cr, PROMINENCE_FLOOR)
    gen = {"script": "scripts/eastern_standing_climatology.py", "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_STANDING_CLIMATOLOGY_2026-10-18.md",
           "month_records": {os.path.basename(p): sha256_of(p) for p in files}, "renderer_sha256": sha256_of(os.path.join(HERE, "eastern_regime_views.py")), **gen_extra}
    with open(paths["persistence"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "lat": lat.tolist(), "lon": lon.tolist(), "reference_check": ref,
                   "climatology": {k: {"seasons": d["seasons"], "mean": np.round(d["mean"], 4).tolist(), "std": np.round(d["std"], 4).tolist(), "count": d["count"].tolist(), "run_mean": np.round(d["run_mean"], 3).tolist(),
                                       "at_least_half": {"rule": "sum of the finite seasons' stored thousandths at or above 500 times their count; the four-decimal mean map is not the decision", "count": int(d["half_mask"].sum()),
                                                         "valid": int(np.isfinite(d["mean"]).sum()), "cells": [[float(lat[j]), float(lon[i])] for j, i in zip(*np.where(d["half_mask"]))]}} for k, d in pc.items()},
                   "band_bins": bt, "ridges": rc, "peaks": pk, "composite_comparison": cc, "figures": {k: {"path": v, "sha256": sha256_of(v)} for k, v in paths.items() if v.endswith(".png")}}, fh, default=float)
    with open(paths["candidates"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "floors": [PROMINENCE_FLOOR, PROMINENCE_SENSITIVITY], "bin_deg": RECURRENCE_BIN_DEG, "recurrence": cr}, fh, default=float)
    text = tables_text(pc, bt, rc, cr, ref, len(records), pk, cc)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
