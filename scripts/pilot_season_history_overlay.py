#!/usr/bin/env python3
"""The frozen candidate's retained in-season eastern histories placed on the two-season
wind overview, descriptively (`EASTERN_DIRECTION_RECOMMENDATION_4_2026-10-14.md`,
section 8).

Reads an overview record (its times, longitudes and band-mean series, reused unchanged)
and a frozen replay. Selects the finished histories whose first STORED time lies in June
to September of the record's year and whose first STORED longitude is strictly east of
40 E. Draws the stored finished positions (`lat`, `lon`, not the raw ones) on the panel of
the band each observation's latitude falls in, 5.0 to 14.5 N southern and 15.0 to 24.5 N
northern, inside 20 to 75 E and the window, as contiguous visible portions: a polyline
breaks at an omitted observation (a latitude between 14.5 and 15 N or outside both bands,
a longitude outside the range, a time outside the window), at a step gap, and at an exit
from a band. The start marker is drawn only at the actual first stored observation and
only when it is displayed. Two attributes have separate encodings, line color for a first
latitude within 5 to 20 N inclusive and an end marker for a last stored longitude west of
40 E. The record keeps the whole cohort with displayed and omitted observations per
history, the visible portions, the band changes, the attributes, the rule, the replay's
digest and thresholds and the overview record's digest. No placement is classified here;
the reader's annotations live in the document.

    python3 scripts/pilot_season_history_overlay.py --overview <season view json> \\
        --replay <frozen replay json.gz> --year 2006 --out <json> --png-prefix <prefix>
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_season_wind_view as V  # noqa: E402

BANDS = V.P.BANDS                                   # southern 5.0 to 14.5 N, northern 15.0 to 24.5 N
LON_RANGE = V.LON_RANGE
TROPICAL = (5.0, 20.0)                              # first latitude within this range, inclusive
WEST_LINE = 40.0
EPOCH_1900 = dt.datetime(1900, 1, 1)
COLOR_TROPICAL = "#008000"
COLOR_OTHER = "k"
RULE = ("finished histories whose first stored time lies in June to September of the season and whose first stored longitude "
        "is strictly east of 40 E; stored finished positions (lat, lon) used for selection, attributes and display, raw positions not drawn")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def to_datetime(days_since_1900):
    return EPOCH_1900 + dt.timedelta(days=float(days_since_1900))


def band_of(lat):
    for b, (lo, hi) in BANDS.items():
        if lo - 1e-9 <= lat <= hi + 1e-9:
            return b
    return None


def select_cohort(finished, year):
    """Indices and histories under the declared rule."""
    out = []
    for i, h in enumerate(finished):
        t0 = to_datetime(h["time"][0])
        if t0.year == year and 6 <= t0.month <= 9 and float(h["lon"][0]) > WEST_LINE:
            out.append((i, h))
    return out


def omission_reason(lat, lon, t, t_first, t_last):
    if not (t_first <= t <= t_last):
        return "outside_window"
    if not (LON_RANGE[0] <= lon <= LON_RANGE[1]):
        return "outside_longitudes"
    if band_of(lat) is None:
        return "between_bands" if BANDS["southern"][1] < lat < BANDS["northern"][0] else "outside_bands"
    return None


def portions_of(h, t_first, t_last):
    """Per band, the contiguous visible portions of one history, and the omissions."""
    lats = [float(x) for x in h["lat"]]; lons = [float(x) for x in h["lon"]]; steps = [int(s) for s in h["steps"]]
    times = [to_datetime(t) for t in h["time"]]
    per_band = {b: [] for b in BANDS}
    omitted = {"outside_window": 0, "outside_longitudes": 0, "between_bands": 0, "outside_bands": 0}
    bands_seq = []
    current, current_band, prev_step = None, None, None
    for k, (la, lo, st, tm) in enumerate(zip(lats, lons, steps, times)):
        reason = omission_reason(la, lo, tm, t_first, t_last)
        if reason is not None:
            omitted[reason] += 1
            if current:
                per_band[current_band].append(current)
            current, current_band = None, None
            prev_step = st
            continue
        b = band_of(la)
        if current is None or b != current_band or st != prev_step + 1:
            if current:
                per_band[current_band].append(current)
            current = {"obs": [], "steps": [], "dates": [], "lons": [], "lats": []}
            current_band = b
        current["obs"].append(k); current["steps"].append(st); current["dates"].append(tm.strftime("%Y-%m-%dT%H")); current["lons"].append(lo); current["lats"].append(la)
        if not bands_seq or bands_seq[-1] != b:
            bands_seq.append(b)
        prev_step = st
    if current:
        per_band[current_band].append(current)
    displayed = {b: sum(len(p["obs"]) for p in ps) for b, ps in per_band.items()}
    return {"portions": per_band, "displayed": displayed, "omitted": omitted, "band_sequence": bands_seq, "band_changes": max(0, len(bands_seq) - 1),
            "start_displayed": bool(per_band and any(ps and ps[0]["obs"][0] == 0 for ps in per_band.values()))}


def history_record(index, h, t_first, t_last):
    lat0, lon0, lon_last = float(h["lat"][0]), float(h["lon"][0]), float(h["lon"][-1])
    rec = {"index": index, "birth": h.get("birth"), "n_points": len(h["lon"]),
           "first": {"date": to_datetime(h["time"][0]).strftime("%Y-%m-%dT%H"), "lat": lat0, "lon": lon0},
           "last": {"date": to_datetime(h["time"][-1]).strftime("%Y-%m-%dT%H"), "lat": float(h["lat"][-1]), "lon": lon_last},
           "starts_within_5_to_20_N": TROPICAL[0] <= lat0 <= TROPICAL[1], "ends_west_of_40_E": lon_last < WEST_LINE}
    rec.update(portions_of(h, t_first, t_last))
    return rec


def draw(year, overview, records, png_prefix, label):
    """The overview's layout, the saved band means under the visible portions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    lons = np.asarray(overview["lons"], float)
    times = [dt.datetime.strptime(t, "%Y-%m-%dT%H") for t in overview["times"]]
    tarr = np.array(times)
    series = {b: np.array([[np.nan if x is None else x for x in row] for row in overview["bands"][b]["series"]], float) for b in BANDS}
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(V.MISSING_COLOR)
    windows = [("season", times[0], times[-1])] + [(name, dt.datetime(year, m, 1), (dt.datetime(year, m + 1, 1) if m < 12 else dt.datetime(year + 1, 1, 1)) - dt.timedelta(hours=6)) for m, name in V.MONTHS]
    figs = []
    for name, t0, t1 in windows:
        sel = np.flatnonzero((tarr >= t0) & (tarr <= t1))
        if sel.size == 0:
            continue
        height = max(4.0, sel.size / V.STEPS_PER_INCH)
        fig, axes = plt.subplots(1, len(BANDS), figsize=(6.5 * len(BANDS) + 1.5, height + 1.8), sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        drawn = {b: 0 for b in BANDS}
        for ax, b in zip(axes, BANDS):
            lo, hi = BANDS[b]
            mesh = ax.pcolormesh(lons, tarr[sel], np.ma.masked_invalid(series[b][sel]), cmap=cmap, vmin=-V.VLIM, vmax=V.VLIM, shading="nearest", alpha=0.85)
            for x in V.MARK_LONS:
                ax.axvline(x, color="k", lw=0.8, ls="--")
            for r in records:
                color = COLOR_TROPICAL if r["starts_within_5_to_20_N"] else COLOR_OTHER
                for p in r["portions"][b]:
                    pd = [dt.datetime.strptime(d, "%Y-%m-%dT%H") for d in p["dates"]]
                    inside = [k for k, d in enumerate(pd) if t0 <= d <= t1]
                    if not inside:
                        continue
                    drawn[b] += 1
                    ax.plot([p["lons"][k] for k in inside], [pd[k] for k in inside], "-", color=color, lw=1.0, alpha=0.9)
                    ax.plot([p["lons"][k] for k in inside], [pd[k] for k in inside], ".", color=color, ms=2.5)
                    if r["start_displayed"] and p["obs"][0] == 0 and 0 in inside:
                        ax.plot(p["lons"][0], pd[0], "^", color=color, ms=7, mec="w", mew=0.5)
                    if r["ends_west_of_40_E"] and p["obs"][-1] == r["n_points"] - 1 and (len(pd) - 1) in inside:
                        ax.plot(p["lons"][-1], pd[-1], "x", color=color, ms=7, mew=1.5)
            ax.set_title(f"{b} band, {lo:g} to {hi:g} N, {drawn[b]} visible portions", fontsize=12)
            ax.set_xlabel("longitude (degrees E)")
            ax.yaxis.set_major_locator(mdates.DayLocator(interval=5 if name == "season" else 2))
            ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            ax.grid(True, axis="y", color="0.5", lw=0.3, ls=":")
        fig.colorbar(mesh, ax=axes.tolist(), shrink=0.4, label="700 hPa meridional wind, band mean (m/s)")
        handles = [Line2D([], [], color=COLOR_TROPICAL, lw=1.2, label="history whose first latitude lies within 5 to 20 N"),
                   Line2D([], [], color=COLOR_OTHER, lw=1.2, label="other eastern-start history"),
                   Line2D([], [], color="k", marker="^", ls="", label="first stored observation, when displayed"),
                   Line2D([], [], color="k", marker="x", ls="", label="last observation of a history ending west of 40 E, when displayed")]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
        fig.suptitle(f"{label}, {name}: the frozen candidate's in-season eastern-start histories (stored finished positions) on the band-mean "
                     f"700 hPa meridional wind. Lines break at omitted observations, step gaps and band exits. Dashed: 40 E and the detector's 60 E edge.", fontsize=10, wrap=True)
        path = f"{png_prefix}_{name}.png"
        fig.savefig(path, dpi=100); plt.close(fig)
        figs.append({"name": name, "path": path, "sha256": sha256_of(path), "first": times[sel[0]].strftime("%Y-%m-%dT%H"), "last": times[sel[-1]].strftime("%Y-%m-%dT%H"), "visible_portions": drawn})
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--overview", "--replay", "--out", "--png-prefix"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    overview = json.load(open(args.overview))
    if int(overview["year"]) != args.year:
        raise SystemExit("REFUSED: the overview record is not the declared year")
    with gzip.open(args.replay, "rb") as fh:
        replay = json.loads(fh.read())
    t_first = dt.datetime.strptime(overview["times"][0], "%Y-%m-%dT%H"); t_last = dt.datetime.strptime(overview["times"][-1], "%Y-%m-%dT%H")
    cohort = select_cohort(replay["finished"], args.year)
    records = [history_record(i, h, t_first, t_last) for i, h in cohort]
    figs = draw(args.year, overview, records, args.png_prefix, f"{args.year}")
    displayed_any = [r for r in records if any(r["displayed"][b] > 0 for b in BANDS)]
    counts = {"cohort": len(records), "finished_in_replay": len(replay["finished"]),
              "starts_within_5_to_20_N": sum(r["starts_within_5_to_20_N"] for r in records), "ends_west_of_40_E": sum(r["ends_west_of_40_E"] for r in records),
              "with_any_displayed_observation": len(displayed_any), "with_none_displayed": len(records) - len(displayed_any),
              "histories_displayed_per_band": {b: sum(1 for r in records if r["displayed"][b] > 0) for b in BANDS},
              "histories_on_both_panels": sum(1 for r in records if all(r["displayed"][b] > 0 for b in BANDS)),
              "observations": {"total": sum(r["n_points"] for r in records), "displayed_per_band": {b: sum(r["displayed"][b] for r in records) for b in BANDS},
                               "omitted_by_reason": {k: sum(r["omitted"][k] for r in records) for k in ("outside_window", "outside_longitudes", "between_bands", "outside_bands")}},
              "start_marker_displayed": sum(1 for r in records if r["start_displayed"]), "band_changes_total": sum(r["band_changes"] for r in records)}
    out = {"generated_by": "scripts/pilot_season_history_overlay.py", "script_sha256": sha256_of(os.path.abspath(__file__)), "year": args.year,
           "sources": {"overview": {"path": args.overview, "sha256": sha256_of(args.overview)}, "replay": {"path": args.replay, "sha256": sha256_of(args.replay),
                       "thresholds": replay.get("thresholds"), "configuration": replay.get("configuration")}},
           "rule": RULE, "positions_drawn": "stored finished positions (lat, lon); raw_lat and raw_lon are not drawn",
           "bands": {b: list(v) for b, v in BANDS.items()}, "between_bands": "latitudes above 14.5 and below 15.0 N are omitted", "lon_range": list(LON_RANGE),
           "window": {"first": overview["times"][0], "last": overview["times"][-1]}, "attributes": {"starts_within_5_to_20_N": "first stored latitude within 5 to 20 N inclusive, drawn in color",
                                                                                                  "ends_west_of_40_E": "last stored longitude strictly west of 40 E, an end marker when that observation is displayed"},
           "counts": counts, "histories": records, "figures": figs,
           "label": "a descriptive placement of stored histories on a latitude band mean; coincidence in this plane is not continuity, identity, validation or coverage, and an interval with no displayed selected history says nothing about detector output"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(counts, indent=1))
    print(f"{len(figs)} figures. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
