#!/usr/bin/env python3
"""The candidate-positions layer (`EASTERN_CANDIDATE_POSITIONS_2026-10-18.md`, section 2):
every local minimum at or below -4 m/s of the daily-mean 700 hPa band mean in the ten
characterization months, with its depth, its connected region at or below -4 m/s and its
prominence, the crowdedness per band-day, and the crosswalk from every anchored day of
the ledger to the candidates (coincident, competing inside the corridor, deeper outside
it). Positions on days. Nothing is linked across days and nothing is named a structure.

    python3 scripts/eastern_candidate_positions.py run --ledger LEDGER.json --out-dir DIR
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_t1_ledger as T  # noqa: E402

FLOOR_MS = -4.0
PROMINENCE_VIEWS = (2.0, 4.0)   # sensitivity views of the crowdedness and the competing count, not a change of the candidate definition
VLIM = 10.0
MISSING_COLOR = "0.75"


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def candidates_on(values, lons, floor=FLOOR_MS, context=""):
    """The candidates of one daily-mean series: interior local minima (strictly below both
    neighbors, or the first cell of a flat run that is below both its ends) at or below the
    floor, each with its depth, the connected run of cells at or below the floor that holds
    it (bounds and the count of candidates in it) and its prominence (the depth below the
    shallower of the ridges separating it from the neighboring candidates or the run's
    ends). Box-edge cells are never candidates."""
    v = np.asarray(values, float); lons = np.asarray(lons, float); n = len(v)
    if not np.isfinite(v).all():
        # A daily series with a missing cell has no defined ridges across the gap, so the
        # layer refuses it rather than spanning or inventing a prominence (review of
        # 2026-10-18, F6). The ten committed month records are complete.
        raise SystemExit(f"REFUSED: a daily series with {int((~np.isfinite(v)).sum())} missing cells has no defined candidates{context}")
    mins = []
    j = 1
    while j < n - 1:
        k = j
        while k + 1 < n - 1 and v[k + 1] == v[j]:
            k += 1
        left, right = v[j - 1], v[k + 1]
        if v[j] < left and v[j] < right and v[j] <= floor:
            mins.append(j)
        j = k + 1
    below = v <= floor
    out = []
    for j in mins:
        a = j
        while a - 1 >= 0 and below[a - 1]:
            a -= 1
        b = j
        while b + 1 < n and below[b + 1]:
            b += 1
        shared = [m for m in mins if a <= m <= b]
        prev = max([m for m in mins if m < j], default=None); nxt = min([m for m in mins if m > j], default=None)
        # The ridge on a side is the highest value between the candidate and its neighboring
        # candidate on that side, or, with no neighbor there, the first cell beyond the run on
        # that side (the run's own end cell when the run reaches the box edge).
        ridge_w = v[prev:j + 1].max() if prev is not None else (v[a - 1] if a - 1 >= 0 else v[a])
        ridge_e = v[j:nxt + 1].max() if nxt is not None else (v[b + 1] if b + 1 < n else v[b])
        prominence = float(min(ridge_w, ridge_e) - v[j])
        out.append({"lon": float(lons[j]), "depth": float(v[j]), "region": {"lo": float(lons[a]), "hi": float(lons[b]), "width_deg": float(lons[b] - lons[a]), "candidates_in_region": len(shared)}, "prominence": prominence})
    return out


def layer(records):
    """{(year, month): {band: {date: [candidates]}}} over the ten month records."""
    out = {}
    for key, rec in records.items():
        lons = np.array(rec["lons"], float)
        out[key] = {}
        for b in rec["bands"]:
            dm = T.daily_means(rec, b)
            out[key][b] = {d: candidates_on(v, lons, context=f" ({key[0]}-{key[1]:02d}, {b}, {d})") for d, v in dm.items()}
    return out


def crosswalk(ledger, lay):
    """Per anchored day of every entry: coincident, competing, deeper outside, and the
    candidate count inside the corridor; per entry the shares."""
    rows = []
    for e in ledger:
        cands_by_day = lay[(e["year"], e["month"])][e["band"]]
        days = []
        for r in e["days"]:
            if not r["anchored"]:
                continue
            cs = cands_by_day[r["date"]]
            inside = [c for c in cs if r["lo"] - 1e-6 <= c["lon"] <= r["hi"] + 1e-6]
            coincident = any(abs(c["lon"] - r["anchor_lon"]) < 1e-6 for c in cs)
            others = [c for c in inside if abs(c["lon"] - r["anchor_lon"]) >= 1e-6]
            competing = len(others) >= 1
            deeper_outside = any(c["depth"] < r["v"] for c in cs if not (r["lo"] - 1e-6 <= c["lon"] <= r["hi"] + 1e-6))
            days.append({"date": r["date"], "anchor_lon": r["anchor_lon"], "anchor_v": r["v"], "edge": r["edge"], "coincident": coincident, "competing": competing, "deeper_outside": deeper_outside, "candidates_inside": len(inside), "candidates_on_day": len(cs),
                         "competing_at_prominence": {f"{pv:g}": any(c["prominence"] >= pv for c in others) for pv in PROMINENCE_VIEWS}})
        n = len(days)
        rows.append({"id": e["id"], "year": e["year"], "month": e["month"], "band": e["band"], "anchored_days": n, "days": days,
                     "coincident": sum(d["coincident"] for d in days), "competing": sum(d["competing"] for d in days), "deeper_outside": sum(d["deeper_outside"] for d in days),
                     "competing_at_prominence": {f"{pv:g}": sum(d["competing_at_prominence"][f"{pv:g}"] for d in days) for pv in PROMINENCE_VIEWS},
                     "all_coincident_never_competing": n > 0 and all(d["coincident"] and not d["competing"] for d in days), "any_deeper_outside": any(d["deeper_outside"] for d in days)})
    return rows


def summarize(lay, xw):
    months = {}
    for (y, m), bands in lay.items():
        for b, by_day in bands.items():
            counts = [len(cs) for cs in by_day.values()]
            widths = [c["region"]["width_deg"] for cs in by_day.values() for c in cs]
            proms = [c["prominence"] for cs in by_day.values() for c in cs]
            months[f"{y}-{m:02d}|{b}"] = {"band_days": len(counts), "candidates": int(sum(counts)), "mean_crowdedness": float(np.mean(counts)), "max_crowdedness": int(max(counts)),
                                         "share_two_or_more": float(np.mean([c >= 2 for c in counts])), "share_none": float(np.mean([c == 0 for c in counts])),
                                         "crowdedness_at_prominence": {f"{pv:g}": {"mean": float(np.mean([sum(1 for c in cs if c["prominence"] >= pv) for cs in by_day.values()])),
                                                                                   "share_two_or_more": float(np.mean([sum(1 for c in cs if c["prominence"] >= pv) >= 2 for cs in by_day.values()]))} for pv in PROMINENCE_VIEWS},
                                         "width_median": statistics.median(widths) if widths else None, "width_max": max(widths) if widths else None,
                                         "prominence_median": statistics.median(proms) if proms else None}
    total_days = sum(r["anchored_days"] for r in xw)
    return {"by_month_band": months,
            "crosswalk": {"entries": len(xw), "anchored_days": total_days, "coincident_days": sum(r["coincident"] for r in xw), "competing_days": sum(r["competing"] for r in xw), "deeper_outside_days": sum(r["deeper_outside"] for r in xw),
                          "entries_all_coincident_never_competing": sum(1 for r in xw if r["all_coincident_never_competing"]), "entries_any_deeper_outside": sum(1 for r in xw if r["any_deeper_outside"]),
                          "entries_any_competing": sum(1 for r in xw if r["competing"] > 0), "edge_days_deeper_outside": sum(1 for r in xw for d in r["days"] if d["edge"] and d["deeper_outside"]), "edge_days": sum(1 for r in xw for d in r["days"] if d["edge"]),
                          "competing_days_at_prominence": {f"{pv:g}": sum(r["competing_at_prominence"][f"{pv:g}"] for r in xw) for pv in PROMINENCE_VIEWS},
                          "entries_any_competing_at_prominence": {f"{pv:g}": sum(1 for r in xw if r["competing_at_prominence"][f"{pv:g}"] > 0) for pv in PROMINENCE_VIEWS},
                          "nonedge_days_deeper_outside": sum(1 for r in xw for d in r["days"] if not d["edge"] and d["deeper_outside"]), "nonedge_days": sum(1 for r in xw for d in r["days"] if not d["edge"])}}


def summary_table(s):
    f = lambda x, n=2: "" if x is None else f"{x:.{n}f}"
    L = ["# the candidate-positions layer, derived (section 2)",
         "a candidate is an interior local minimum of the daily-mean 700 hPa band mean at or below -4 m/s; crowdedness is the candidate count per band-day; a region is the connected run at or below -4 m/s holding the candidate; prominence is the depth below the shallower separating ridge", "",
         "| month, band | band-days | candidates | mean crowdedness | max | share with two or more | share with none | region width median, max (degrees) | prominence median (m/s) | at prominence 2 or more: mean, share with two or more | at prominence 4 or more: mean, share with two or more |",
         "| --- |" + " --- |" * 10]
    for k, v in s["by_month_band"].items():
        p2 = v["crowdedness_at_prominence"]["2"]; p4 = v["crowdedness_at_prominence"]["4"]
        L.append(f"| {k.replace('|', ', ')} | {v['band_days']} | {v['candidates']} | {f(v['mean_crowdedness'])} | {v['max_crowdedness']} | {f(v['share_two_or_more'])} | {f(v['share_none'])} | {f(v['width_median'], 1)}, {f(v['width_max'], 1)} | {f(v['prominence_median'], 1)} | {f(p2['mean'])}, {f(p2['share_two_or_more'])} | {f(p4['mean'])}, {f(p4['share_two_or_more'])} |")
    c = s["crosswalk"]
    L += ["", "## the crosswalk to the ledger's anchored days", "",
          f"- entries {c['entries']}, anchored days {c['anchored_days']}, coincident with a candidate {c['coincident_days']}, with a competing candidate inside the corridor {c['competing_days']}, with a deeper candidate outside the corridor {c['deeper_outside_days']}",
          f"- entries whose anchored days are all coincident and never competing {c['entries_all_coincident_never_competing']}, entries with any competing day {c['entries_any_competing']}, entries with any deeper-outside day {c['entries_any_deeper_outside']}",
          f"- deeper-outside days among edge days {c['edge_days_deeper_outside']} of {c['edge_days']}, among non-edge days {c['nonedge_days_deeper_outside']} of {c['nonedge_days']}",
          f"- sensitivity, competing counted only for candidates of prominence 2 or more: {c['competing_days_at_prominence']['2']} days in {c['entries_any_competing_at_prominence']['2']} entries; of prominence 4 or more: {c['competing_days_at_prominence']['4']} days in {c['entries_any_competing_at_prominence']['4']} entries"]
    return "\n".join(L) + "\n"


def crosswalk_table(xw):
    L = ["# the crosswalk, one row per ledger entry (section 2)",
         "coincident, the anchor equals a candidate; competing, another candidate inside that day's corridor; deeper outside, a candidate outside the corridor deeper than the anchor; counts over the entry's anchored days", "",
         "| entry | month | band | anchored days | coincident | competing | competing at prominence 2, 4 | deeper outside | all coincident, never competing | any deeper outside |", "| --- |" + " --- |" * 9]
    for r in xw:
        L.append(f"| {r['id']} | {r['year']}-{r['month']:02d} | {r['band']} | {r['anchored_days']} | {r['coincident']} | {r['competing']} | {r['competing_at_prominence']['2']}, {r['competing_at_prominence']['4']} | {r['deeper_outside']} | {'yes' if r['all_coincident_never_competing'] else 'no'} | {'yes' if r['any_deeper_outside'] else 'no'} |")
    return "\n".join(L) + "\n"


def draw_overlay(path, record, by_band, entries_of_month):
    import datetime as dt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(MISSING_COLOR)
    lons = np.array(record["lons"], float); times = [dt.datetime.strptime(t, "%Y-%m-%dT%H") for t in record["times"]]
    bands = list(record["bands"])
    fig, axes = plt.subplots(1, len(bands), figsize=(4.2 * len(bands) + 1.5, max(6.0, len(times) / 14.0) + 1.4), sharey=True, constrained_layout=True)
    for ax, b in zip(np.atleast_1d(axes), bands):
        S = np.ma.masked_invalid(np.array(record["band_means_700"][b], float))
        mesh = ax.pcolormesh(lons, np.array(times), S, cmap=cmap, vmin=-VLIM, vmax=VLIM, shading="nearest")
        for x in (40.0, 60.0):
            ax.axvline(x, color="k", lw=0.8, ls="--")
        for d, cs in by_band[b].items():
            t = dt.datetime.strptime(d, "%Y-%m-%d") + dt.timedelta(hours=12)
            for c in cs:
                ax.plot(c["lon"], t, marker="v", ms=3.5, color="lime", mec="k", mew=0.4)
        for e in [e for e in entries_of_month if e["band"] == b]:
            for r in e["days"]:
                if r["anchored"]:
                    t = dt.datetime.strptime(r["date"], "%Y-%m-%d") + dt.timedelta(hours=12)
                    ax.plot(r["anchor_lon"], t, marker="o", ms=4, mfc="none", mec="yellow", mew=1.0)
        ax.set_title(b, fontsize=11); ax.set_xlabel("longitude (degrees E)")
        ax.yaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.grid(True, axis="y", color="0.5", lw=0.3, ls=":")
    fig.colorbar(mesh, ax=np.atleast_1d(axes).tolist(), shrink=0.5, label="700 hPa meridional wind, band mean (m/s)")
    fig.suptitle(f"{record['label']}: every candidate (a local minimum of the daily mean at or below -4 m/s) as a green triangle, the ledger's anchors as open yellow circles. Positions on days, nothing linked.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("run",))
    ap.add_argument("--ledger", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    led = json.load(open(a.ledger))
    months = sorted({(e["year"], e["month"]) for e in led["ledger"]})
    records = {(y, m): json.load(open(os.path.join(a.out_dir, f"eval60_regime_{y}_{m:02d}.json"))) for y, m in months}
    paths = {"json": os.path.join(a.out_dir, "eval60_cand_positions.json"), "summary": os.path.join(a.out_dir, "eval60_cand_summary.md"), "crosswalk": os.path.join(a.out_dir, "eval60_cand_crosswalk.md")}
    for y, m in months:
        paths[f"overlay_{y}_{m:02d}"] = os.path.join(a.out_dir, f"eval60_cand_overlay_{y}_{m:02d}.png")
    for p in paths.values():
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    lay = layer(records)
    xw = crosswalk(led["ledger"], lay)
    s = summarize(lay, xw)
    for y, m in months:
        draw_overlay(paths[f"overlay_{y}_{m:02d}"], records[(y, m)], lay[(y, m)], [e for e in led["ledger"] if (e["year"], e["month"]) == (y, m)])
    out = {"generated_by": {"script": "scripts/eastern_candidate_positions.py", "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_CANDIDATE_POSITIONS_2026-10-18.md"},
           "ledger": {"path": a.ledger, "sha256": sha256_of(a.ledger)},
           "month_records": {f"{y}-{m:02d}": {"path": os.path.join(a.out_dir, f"eval60_regime_{y}_{m:02d}.json"), "sha256": sha256_of(os.path.join(a.out_dir, f"eval60_regime_{y}_{m:02d}.json"))} for y, m in months},
           "rules": {"floor_ms": FLOOR_MS, "field": "daily mean of the four six-hourly 700 hPa band-mean samples", "edge_cells_excluded": True},
           "overlays": {k: {"path": v, "sha256": sha256_of(v)} for k, v in paths.items() if k.startswith("overlay")},
           "layer": {f"{y}-{m:02d}": {b: by_day for b, by_day in lay[(y, m)].items()} for y, m in months}, "crosswalk": xw, "summary": s}
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(out, fh, default=float)
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        fh.write(summary_table(s))
    with open(paths["crosswalk"], "w", encoding="utf-8") as fh:
        fh.write(crosswalk_table(xw))
    print(summary_table(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
