#!/usr/bin/env python3
"""The anchored ledger of the transient northerly regions (`EASTERN_T1_LEDGER_2026-10-18.md`,
section 2). Reads the entries file (one reader's identifications with a first and a last
date and center) and the ten month records of the characterization, anchors each entry day
by day to the strongest northerly cell of the daily-mean 700 hPa band mean inside the
interpolated interval, computes segment, fitted and net speeds, the companion check and the
censoring from the numbers, draws one overlay per month, and writes the ledger and its
summary tables. Coordinates and slopes, not identities.

    python3 scripts/eastern_t1_ledger.py run --entries ENTRIES.json --out-dir DIR
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import statistics

import numpy as np

HALF_WIDTH = 5.0           # degrees, the stated interval is the center plus and minus this
FLOOR_MS = -4.0            # a day is anchored only at or below this
COMPANION_MS = 4.0         # the companion southerly is at or above this
COMPANION_EAST = (2.0, 12.0)
STANDING_DEG_PER_DAY = 1.0
BOUNDARY_TOL = 1e-9
EAST_EDGE = 70.0
WEST_EDGE = 25.0
MIN_FIT_DAYS = 3
VLIM = 10.0
MISSING_COLOR = "0.75"


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


EXPECTED_HOURS = ("00", "06", "12", "18")


def daily_means(record, band):
    """{date: daily mean of the four six-hourly band-mean samples}. A date without exactly
    the four hours 00, 06, 12 and 18 is refused, and a cell with any missing sample on a
    date is unavailable (NaN) rather than a mean of fewer samples (review of 2026-10-18,
    F8)."""
    S = np.array(record["band_means_700"][band], float)
    days = [t[:10] for t in record["times"]]; hours = [t[11:13] for t in record["times"]]
    out = {}
    for d in sorted(set(days)):
        idx = [i for i, t in enumerate(days) if t == d]
        if sorted(hours[i] for i in idx) != list(EXPECTED_HOURS):
            raise SystemExit(f"REFUSED: {d} does not hold exactly the four six-hourly samples ({[hours[i] for i in idx]})")
        block = S[idx]
        out[d] = np.where(np.isfinite(block).all(axis=0), block.mean(axis=0), np.nan)
    return out


def _date(s):
    return dt.date.fromisoformat(s)


def interval_for(entry, date, half=HALF_WIDTH):
    """The straight-line interpolation of the two stated centers in time, plus and minus
    the half width."""
    d0, d1 = _date(entry["first"]["date"]), _date(entry["last"]["date"])
    c0, c1 = float(entry["first"]["center"]), float(entry["last"]["center"])
    if d1 < d0:
        raise SystemExit(f"REFUSED: entry {entry['id']} ends before it starts")
    if date < d0 or date > d1:
        raise SystemExit(f"REFUSED: {date} is outside entry {entry['id']}")
    span = (d1 - d0).days
    frac = 0.0 if span == 0 else (date - d0).days / span
    c = c0 + frac * (c1 - c0)
    return c - half, c + half


def anchor_entry(entry, dm, lons):
    """Per day of the entry: the interval, the anchor (longitude and value) when the daily
    minimum inside the interval is at or below the floor, the edge flag, and the companion
    check east of the anchor."""
    d0, d1 = _date(entry["first"]["date"]), _date(entry["last"]["date"])
    days = []
    d = d0
    while d <= d1:
        key = d.isoformat()
        if key not in dm:
            raise SystemExit(f"REFUSED: entry {entry['id']} needs {key}, which the month record does not hold")
        lo, hi = interval_for(entry, d)
        sel = np.flatnonzero((lons >= lo - 1e-6) & (lons <= hi + 1e-6))
        v = dm[key]
        row = {"date": key, "lo": lo, "hi": hi, "anchored": False, "anchor_lon": None, "v": None, "edge": False, "companion_max": None, "companion": False}
        if sel.size and np.isfinite(v[sel]).any():
            j = sel[int(np.nanargmin(v[sel]))]
            row["v"] = float(v[j])
            if v[j] <= FLOOR_MS:
                row["anchored"] = True; row["anchor_lon"] = float(lons[j]); row["edge"] = bool(j == sel[0] or j == sel[-1])
                east = np.flatnonzero((lons >= lons[j] + COMPANION_EAST[0] - 1e-6) & (lons <= lons[j] + COMPANION_EAST[1] + 1e-6))
                if east.size and np.isfinite(v[east]).any():
                    row["companion_max"] = float(np.nanmax(v[east])); row["companion"] = bool(row["companion_max"] >= COMPANION_MS)
        days.append(row)
        d += dt.timedelta(days=1)
    return days


def speeds(days):
    """Segment speeds between consecutive anchored days, the fitted speed over at least
    MIN_FIT_DAYS anchored days with its residual root mean square, and the net speed."""
    anc = [(i, r["anchor_lon"]) for i, r in enumerate(days) if r["anchored"]]
    seg = [{"from": days[a][ "date"], "to": days[b]["date"], "deg_per_day": (lb - la) / (b - a)} for (a, la), (b, lb) in zip(anc, anc[1:])]
    out = {"anchored_days": len(anc), "segments": seg, "fitted": None, "fit_rms": None, "net": None, "direction": None}
    if len(anc) >= 2:
        (a, la), (b, lb) = anc[0], anc[-1]
        out["net"] = (lb - la) / (b - a)
    if len(anc) >= MIN_FIT_DAYS:
        x = np.array([a for a, _ in anc], float); y = np.array([l for _, l in anc], float)
        slope, intercept = np.polyfit(x, y, 1)
        out["fitted"] = float(slope); out["fit_rms"] = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))
        # A slope whose exact value is 1.0 can come back as 0.9999999999999986 from the
        # fit, so the strict "under 1" rule is applied with a tolerance of 1e-9 degrees per
        # day (review of 2026-10-18, F6).
        out["direction"] = "standing" if abs(slope) < STANDING_DEG_PER_DAY - BOUNDARY_TOL else ("westward" if slope < 0 else "eastward")
    return out


def censoring(days, record):
    anc = [r for r in days if r["anchored"]]
    if not anc:
        return {"first_day_is_month_start": False, "last_day_is_month_end": False, "first_anchor_at_east_edge": False, "last_anchor_at_west_edge": False}
    first_date, last_date = record["times"][0][:10], record["times"][-1][:10]
    return {"first_day_is_month_start": anc[0]["date"] == first_date, "last_day_is_month_end": anc[-1]["date"] == last_date,
            "first_anchor_at_east_edge": anc[0]["anchor_lon"] >= EAST_EDGE, "last_anchor_at_west_edge": anc[-1]["anchor_lon"] <= WEST_EDGE}


def build_ledger(entries_doc, records):
    """records: {(year, month): record}. Returns the ledger entries with their anchors."""
    out = []
    for e in entries_doc["entries"]:
        rec = records[(e["year"], e["month"])]
        dm = daily_means(rec, e["band"]); lons = np.array(rec["lons"], float)
        days = anchor_entry(e, dm, lons)
        sp = speeds(days)
        out.append({**e, "days": days, "speeds": sp, "censoring_from_numbers": censoring(days, rec),
                    "companion_days": sum(1 for r in days if r["companion"]), "edge_days": sum(1 for r in days if r["edge"]),
                    "first_anchor": next((r for r in days if r["anchored"]), None), "last_anchor": next((r for r in reversed(days) if r["anchored"]), None)})
    return out


def summarize(ledger):
    fitted = [e for e in ledger if e["speeds"]["fitted"] is not None]
    by_dir = {}
    for e in fitted:
        by_dir.setdefault(e["speeds"]["direction"], []).append(e["speeds"]["fitted"])
    def stats(v):
        return {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)} if v else {"n": 0}
    months = {}
    for e in ledger:
        key = f"{e['year']}-{e['month']:02d}"
        m = months.setdefault(key, {"entries": 0, "with_fit": 0, "westward": 0, "standing": 0, "eastward": 0, "anchored_days": 0, "days": 0, "edge_entries": 0, "companion_entries": 0})
        m["entries"] += 1; m["anchored_days"] += e["speeds"]["anchored_days"]; m["days"] += len(e["days"])
        if e["speeds"]["fitted"] is not None:
            m["with_fit"] += 1; m[e["speeds"]["direction"]] += 1
        if e["edge_days"]:
            m["edge_entries"] += 1
        if e["companion_days"]:
            m["companion_entries"] += 1
    bands = {}
    for e in ledger:
        b = bands.setdefault(e["band"], {"entries": 0, "with_fit": 0, "westward": 0, "standing": 0, "eastward": 0, "westward_speeds": []})
        b["entries"] += 1
        if e["speeds"]["fitted"] is not None:
            b["with_fit"] += 1; b[e["speeds"]["direction"]] += 1
            if e["speeds"]["direction"] == "westward":
                b["westward_speeds"].append(e["speeds"]["fitted"])
    for b in bands.values():
        b["median_fitted_westward"] = statistics.median(b["westward_speeds"]) if b["westward_speeds"] else None
        del b["westward_speeds"]
    return {"entries": len(ledger), "with_fit": len(fitted), "fewer_than_min_fit_days": len(ledger) - len(fitted), "by_band": bands,
            "by_direction": {k: stats(v) for k, v in by_dir.items()},
            "first_anchor_east_of_70": sum(1 for e in ledger if e["censoring_from_numbers"]["first_anchor_at_east_edge"]),
            "last_anchor_west_of_42": sum(1 for e in ledger if e["last_anchor"] and e["last_anchor"]["anchor_lon"] <= 42.0),
            "entries_with_edge_days": sum(1 for e in ledger if e["edge_days"]), "edge_days_total": sum(e["edge_days"] for e in ledger), "days_total": sum(len(e["days"]) for e in ledger),
            "anchored_days_total": sum(e["speeds"]["anchored_days"] for e in ledger),
            "entries_with_companion_days": sum(1 for e in ledger if e["companion_days"]), "entries_flagged_companion": sum(1 for e in ledger if e["flags"]["companion"]),
            "flagged_companion_with_companion_days": sum(1 for e in ledger if e["flags"]["companion"] and e["companion_days"]),
            "unflagged_with_companion_days": sum(1 for e in ledger if not e["flags"]["companion"] and e["companion_days"]),
            "by_month": months}


def ledger_table(ledger):
    f = lambda x, n=1: "" if x is None else f"{x:.{n}f}"
    L = ["# the anchored ledger of the transient northerly regions (section 2), one row per entry",
         "anchored days are days whose daily minimum inside the interpolated interval is at or below -4 m/s; the fitted speed is the least-squares slope over at least three anchored days with its residual rms; net is last anchor minus first over the days between; edge days are days whose anchor sits on the interval's first or last cell; companion days are anchored days with a daily mean of at least +4 m/s between 2 and 12 degrees east of the anchor", "",
         "| entry | month | band | first date, center | last date, center | anchored of days | first anchor | last anchor | fitted (rms) | net | direction | segments (degrees per day) | edge days | companion days (flag) | reading flags | censoring from numbers | note |",
         "| --- |" + " --- |" * 16]
    for e in ledger:
        fa = "" if not e["first_anchor"] else f"{e['first_anchor']['date'][5:]} {e['first_anchor']['anchor_lon']:g} E"
        la = "" if not e["last_anchor"] else f"{e['last_anchor']['date'][5:]} {e['last_anchor']['anchor_lon']:g} E"
        sp = e["speeds"]
        fl = e["flags"]; rf = ", ".join(k for k, v in fl.items() if v and k != "companion") or ""
        segs = ", ".join(f"{x['deg_per_day']:+.1f}" for x in sp["segments"])
        cn = e["censoring_from_numbers"]; cf = ", ".join(k.replace("_", " ") for k, v in cn.items() if v) or ""
        L.append(f"| {e['id']} | {e['year']}-{e['month']:02d} | {e['band']} | {e['first']['date'][5:]}, {e['first']['center']:g} | {e['last']['date'][5:]}, {e['last']['center']:g} | {sp['anchored_days']} of {len(e['days'])} | {fa} | {la} | {f(sp['fitted'])} ({f(sp['fit_rms'])}) | {f(sp['net'])} | {sp['direction'] or ''} | {segs} | {e['edge_days']} | {e['companion_days']} ({'yes' if fl['companion'] else 'no'}) | {rf} | {cf} | {e['note']} |")
    return "\n".join(L) + "\n"


def summary_table(s):
    f = lambda x, n=1: "" if x is None else f"{x:.{n}f}"
    L = ["# the ledger's summary (section 2), derived from the ledger", "",
         f"- entries {s['entries']}, with a fitted speed {s['with_fit']}, with fewer than {MIN_FIT_DAYS} anchored days {s['fewer_than_min_fit_days']}",
         f"- days asked {s['days_total']}, anchored {s['anchored_days_total']}, edge days {s['edge_days_total']} in {s['entries_with_edge_days']} entries",
         f"- first anchor at or east of {EAST_EDGE:g} E: {s['first_anchor_east_of_70']} entries; last anchor at or west of 42 E: {s['last_anchor_west_of_42']} entries",
         f"- companion days in {s['entries_with_companion_days']} entries; of the {s['entries_flagged_companion']} entries flagged companion by the reading, {s['flagged_companion_with_companion_days']} have companion days, and {s['unflagged_with_companion_days']} unflagged entries have companion days", ""]
    L.append("| direction (fitted, 1 degree per day rule) | entries | median (degrees per day) | min | max |")
    L.append("| --- |" + " --- |" * 4)
    for k in ("westward", "standing", "eastward"):
        v = s["by_direction"].get(k, {"n": 0})
        L.append(f"| {k} | {v['n']} | {f(v.get('median'))} | {f(v.get('min'))} | {f(v.get('max'))} |")
    L.append("")
    L.append("| band | entries | with fit | westward | standing | eastward | median fitted westward speed |")
    L.append("| --- |" + " --- |" * 6)
    for b, v in s["by_band"].items():
        L.append(f"| {b} | {v['entries']} | {v['with_fit']} | {v['westward']} | {v['standing']} | {v['eastward']} | {f(v['median_fitted_westward'])} |")
    L.append("")
    L.append("| month | entries | with fit | westward | standing | eastward | anchored of days | entries with edge days | entries with companion days |")
    L.append("| --- |" + " --- |" * 8)
    for k, m in s["by_month"].items():
        L.append(f"| {k} | {m['entries']} | {m['with_fit']} | {m['westward']} | {m['standing']} | {m['eastward']} | {m['anchored_days']} of {m['days']} | {m['edge_entries']} | {m['companion_entries']} |")
    return "\n".join(L) + "\n"


def draw_overlay(path, record, entries_of_month):
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
        for e in [e for e in entries_of_month if e["band"] == b]:
            for r in e["days"]:
                t = dt.datetime.strptime(r["date"], "%Y-%m-%d") + dt.timedelta(hours=12)
                ax.plot([r["lo"], r["hi"]], [t, t], color="k", lw=0.6, alpha=0.6)
                if r["anchored"]:
                    ax.plot(r["anchor_lon"], t, marker="o", ms=4, color="yellow", mec="k", mew=0.6)
                else:
                    ax.plot((r["lo"] + r["hi"]) / 2, t, marker="o", ms=4, mfc="none", mec="k", mew=0.6)
            first = e["days"][0]; t0 = dt.datetime.strptime(first["date"], "%Y-%m-%d") + dt.timedelta(hours=12)
            ax.annotate(e["id"], (first["hi"] + 0.5, t0), fontsize=7, color="k")
        ax.set_title(b, fontsize=11); ax.set_xlabel("longitude (degrees E)")
        ax.yaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.grid(True, axis="y", color="0.5", lw=0.3, ls=":")
    fig.colorbar(mesh, ax=np.atleast_1d(axes).tolist(), shrink=0.5, label="700 hPa meridional wind, band mean (m/s)")
    fig.suptitle(f"{record['label']}: the saved 700 hPa band means with the ledger's entries, each day's interval as a bar, anchored days as filled markers at the anchor, unanchored days as open markers at the interval's center. Coordinates of one reader's entries, no identity.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("run",))
    ap.add_argument("--entries", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    doc = json.load(open(a.entries))
    months = sorted({(e["year"], e["month"]) for e in doc["entries"]})
    records = {(y, m): json.load(open(os.path.join(a.out_dir, f"eval60_regime_{y}_{m:02d}.json"))) for y, m in months}
    paths = {"json": os.path.join(a.out_dir, "eval60_t1_ledger.json"), "table": os.path.join(a.out_dir, "eval60_t1_ledger.md"), "summary": os.path.join(a.out_dir, "eval60_t1_summary.md")}
    for y, m in months:
        paths[f"overlay_{y}_{m:02d}"] = os.path.join(a.out_dir, f"eval60_t1_overlay_{y}_{m:02d}.png")
    for p in paths.values():
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    ledger = build_ledger(doc, records)
    s = summarize(ledger)
    for y, m in months:
        draw_overlay(paths[f"overlay_{y}_{m:02d}"], records[(y, m)], [e for e in ledger if (e["year"], e["month"]) == (y, m)])
    out = {"generated_by": {"script": "scripts/eastern_t1_ledger.py", "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_T1_LEDGER_2026-10-18.md"},
           "entries_file": {"path": a.entries, "sha256": sha256_of(a.entries)},
           "month_records": {f"{y}-{m:02d}": {"path": os.path.join(a.out_dir, f"eval60_regime_{y}_{m:02d}.json"), "sha256": sha256_of(os.path.join(a.out_dir, f"eval60_regime_{y}_{m:02d}.json")), "producer_sha256_prefix": records[(y, m)]["generated_by"]["sha256"][:16]} for y, m in months},
           "rules": {"half_width_deg": HALF_WIDTH, "floor_ms": FLOOR_MS, "companion_ms": COMPANION_MS, "companion_east_deg": list(COMPANION_EAST), "standing_deg_per_day": STANDING_DEG_PER_DAY, "min_fit_days": MIN_FIT_DAYS, "east_edge": EAST_EDGE, "west_edge": WEST_EDGE},
           "overlays": {k: {"path": v, "sha256": sha256_of(v)} for k, v in paths.items() if k.startswith("overlay")}, "ledger": ledger, "summary": s}
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(paths["table"], "w", encoding="utf-8") as fh:
        fh.write(ledger_table(ledger))
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        fh.write(summary_table(s))
    print(summary_table(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
