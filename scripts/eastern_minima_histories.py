#!/usr/bin/env python3
"""Histories of eastern minima (`EASTERN_MINIMA_HISTORIES_2026-10-19.md`, section 2): daily
prominent minima of three representations (RB1 the anomaly from the season's own month
mean, RB9 the anomaly from the centered 9-day running mean, RRAW the band mean itself)
linked day to day by a nearest rule (LN) or a predicted rule (LP), each history classed
westward, standing or other, scored on synthetic constructions under a declared pass rule,
and applied to the 96 month records for every combination. A history is a sequence of
minima joined by the stated rule, never an identified feature.

    python3 scripts/eastern_minima_histories.py constructions --out-dir DIR
    python3 scripts/eastern_minima_histories.py archive --constructions PATH --records DIR \\
        --ledger PATH --out-dir DIR
    python3 scripts/eastern_minima_histories.py noise --records DIR --out-dir DIR
    (each mode takes --tag r1 to write its records under versioned names)

Minima are found on the exact stored-precision path of the anomaly module (review of
2026-10-19, F1), so that every floor, plateau and prominence decision is made on integers.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_anomaly_minima as A  # noqa: E402
import eastern_standing_climatology as SC  # noqa: E402

YEARS = SC.YEARS
REPS = ("RB1", "RB9", "RRAW")
LINKERS = ("LN", "LP")
ORDER = (("RB1", "LN"), ("RB1", "LP"), ("RB9", "LN"), ("RB9", "LP"), ("RRAW", "LN"), ("RRAW", "LP"))
REP_SPEC = {"RB1": ("B1", A.REPAIRED_FLOOR), "RB9": ("B2_9", A.REPAIRED_FLOOR), "RRAW": ("zero", -4.0)}
S_MAX = 6.0
S_PRED = 3.0
W_MIN_LEN = 4
W_NET = -8.0
W_FRAC = 0.75
S_MIN_LEN = 5
S_RANGE = 6.0
NET_SENSITIVITY = (-6.0, -10.0)
TOL_DEG = 1.0
PASS_RECOVERY = 0.80
EDGE_DEG = 3.0
BOX = (20.0, 75.0)
C6_WOBBLE = (3.0, 14.0)
NOISE_SEEDS = tuple(A.NOISE_SEED + k for k in range(24))     # added after the declaration (section 5 of the record)


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def combo(rep, linker):
    return f"{rep}|{linker}"


# ------------------------------------------------------------------ minima, linking, classes

def rep_minima(rep, H, month_of, lons):
    """The day-by-day prominent minima of a representation from the daily sums in hundredths,
    None on days without a background, every comparison exact."""
    kind, floor = REP_SPEC[rep]
    rows, scales = A.exact_anomaly(kind, H, month_of)
    return A.exact_anomaly_minima(rows, scales, lons, floor)


def link(minima_by_day, linker, s_max=S_MAX, s_pred=S_PRED):
    """Histories of the daily minima. Each history ending on one day may take one minimum on
    the next day within `s_max` degrees. Under LN pairs are taken in order of distance.
    Under LP a history of at least two members is linked only within `s_pred` of its
    predicted position (last position plus last step), and pairs are taken in order of the
    distance to the prediction, or to the last position for a history of one member.
    One-to-one, ties broken by the history's last longitude and then the minimum's. A day
    that is unavailable (None) or holds no minimum ends every history."""
    histories, open_ = [], []
    for d, mins in enumerate(minima_by_day):
        if not mins:
            open_ = []
            if mins is None:
                continue
        pairs = []
        for hi in open_:
            h = histories[hi]; x = h["lons"][-1]
            pred = (x + (x - h["lons"][-2])) if (linker == "LP" and len(h["lons"]) >= 2) else None
            for mj, m in enumerate(mins):
                if abs(m["lon"] - x) > s_max:
                    continue
                if pred is not None:
                    key = abs(m["lon"] - pred)
                    if key > s_pred:
                        continue
                else:
                    key = abs(m["lon"] - x)
                pairs.append((key, x, m["lon"], hi, mj))
        pairs.sort()
        taken_h, taken_m, new_open = set(), set(), []
        for _, _, _, hi, mj in pairs:
            if hi in taken_h or mj in taken_m:
                continue
            taken_h.add(hi); taken_m.add(mj)
            h = histories[hi]; h["days"].append(d); h["lons"].append(mins[mj]["lon"]); h["depths"].append(mins[mj]["depth"])
            new_open.append(hi)
        for mj, m in enumerate(mins or []):
            if mj not in taken_m:
                histories.append({"days": [d], "lons": [m["lon"]], "depths": [m["depth"]]})
                new_open.append(len(histories) - 1)
        open_ = new_open
    return histories


def classify(h, net=W_NET):
    """westward: at least 4 days, net displacement at most `net`, at least 0.75 of the steps
    westward. standing: at least 5 days and a range of at most 6 degrees. other: the rest."""
    x = h["lons"]; n = len(x)
    if n >= W_MIN_LEN and (x[-1] - x[0]) <= net and float(np.mean(np.diff(x) < 0)) >= W_FRAC:
        return "westward"
    if n >= S_MIN_LEN and (max(x) - min(x)) <= S_RANGE:
        return "standing"
    return "other"


# ------------------------------------------------------------------ construction scores

def follows(h, tr_c, tol=TOL_DEG):
    """True when at least half of the history's members lie within `tol` of the transient's
    daily-mean center on their days."""
    hits = sum(1 for d, x in zip(h["days"], h["lons"]) if np.isfinite(tr_c[d]) and abs(x - tr_c[d]) <= tol)
    return 2 * hits >= len(h["days"])


def passage_score(histories, classes, tr_c, minima_by_day, tol=TOL_DEG):
    """The share of the transient's present days (center between 24 and 71 E, minima
    available) on which a westward history has a member within `tol` of the center, and the
    count of westward histories that do not follow the transient."""
    west = [h for h, c in zip(histories, classes) if c == "westward"]
    member = {}
    for h in west:
        for d, x in zip(h["days"], h["lons"]):
            member.setdefault(d, []).append(x)
    n = k = 0
    for d, c in enumerate(tr_c):
        if minima_by_day[d] is None or not np.isfinite(c) or not (A.PRESENT[0] <= c <= A.PRESENT[1]):
            continue
        n += 1; k += any(abs(x - c) <= tol for x in member.get(d, []))
    return {"present_days": n, "share": (k / n if n else None), "unmatched_westward": sum(1 for h in west if not follows(h, tr_c, tol)), "westward": len(west)}


def construction_cases():
    """The background-removal cases unchanged, then C6: the standing feature wobbling with
    an amplitude of 3 degrees and a period of 14 days while the transient passes through it
    at 1, 2 or 3 degrees per day with a depth of 5 or 10 m/s."""
    cases = list(A.construction_cases())
    amp, per = C6_WOBBLE
    for depth in (5.0, 10.0):
        for c in (1.0, 2.0, 3.0):
            cases.append((f"C6 wobble {amp:g} deg {per:g} days, transient depth {depth:g}, {c:g} deg/day", "passage",
                          {"standing": (lambda a, p: (lambda t: A.STANDING_AT + a * np.sin(2 * np.pi * t / p)))(amp, per),
                           "transient": (A.TRANSIENT_START_DAY, A.TRANSIENT_FROM, c, depth)},
                          {"over_standing": True, "depth": depth, "speed": c, "wobble": True}))
    return cases


def run_constructions():
    month_of = A.month_index(); out = {}
    for name, kind, kw, meta in construction_cases():
        H, s_c, tr_c = A.synth_hundredths(**kw)
        res = {"kind": kind, "meta": dict(meta)}
        for rep in REPS:
            mins = rep_minima(rep, H, month_of, A.LONS)
            for linker in LINKERS:
                hs = link(mins, linker)
                cls = [classify(h) for h in hs]
                sens = {f"{net:g}": sum(1 for h in hs if classify(h, net) == "westward") for net in NET_SENSITIVITY}
                if kind == "passage":
                    r = passage_score(hs, cls, tr_c, mins)
                else:
                    r = {"westward": cls.count("westward")}
                r.update({"standing": cls.count("standing"), "histories": len(hs), "westward_at_net": sens})
                res[combo(rep, linker)] = r
        out[name] = res
    return out


def apply_rule(results):
    verdict = {}
    for rep, linker in ORDER:
        key = combo(rep, linker); fails = []
        for name, r in results.items():
            if r["meta"].get("variant") or r["kind"] == "noise":
                continue
            d = r[key]
            if r["kind"] == "passage":
                if d["share"] is None or d["share"] < PASS_RECOVERY or d["unmatched_westward"] > 0:
                    fails.append(name)
            elif d["westward"] > 0:
                fails.append(name)
        verdict[key] = {"passes": not fails, "failing_cases": fails}
    selected = next((combo(r, l) for r, l in ORDER if verdict[combo(r, l)]["passes"]), None)
    return verdict, selected


def constructions_text(results, verdict, selected):
    keys = [combo(r, l) for r, l in ORDER]
    f = lambda x: "" if x is None else f"{x:.2f}"
    L = ["# minima histories, the constructions (section 2)",
         "westward is at least 4 days, a net displacement of at most -8 degrees and at least 0.75 of the steps westward; a passage is recovered on a present day when a westward history has a member within 1 degree of the transient's center", "",
         "## the rule", "", "| combination | passes | failing cases |", "| --- | --- | --- |"]
    for k in keys:
        L.append(f"| {k} | {'yes' if verdict[k]['passes'] else 'no'} | {', '.join(verdict[k]['failing_cases']) or 'none'} |")
    L += ["", f"selected: {selected or 'none, so no combination is primary'}", "",
          "## passage, the share of present days recovered (westward histories not following the transient in brackets)", "", "| case | " + " | ".join(keys) + " |", "| --- |" + " --- |" * len(keys)]
    for name, r in results.items():
        if r["kind"] == "passage":
            L.append(f"| {name} | " + " | ".join(f"{f(r[k]['share'])} [{r[k]['unmatched_westward']}]" for k in keys) + " |")
    L += ["", "## displacement, drift, wobble and noise, westward histories (at net -6 and -10 degrees in brackets)", "", "| case | " + " | ".join(keys) + " |", "| --- |" + " --- |" * len(keys)]
    for name, r in results.items():
        if r["kind"] != "passage":
            L.append(f"| {name} | " + " | ".join(f"{r[k]['westward']} [{r[k]['westward_at_net']['-6']}, {r[k]['westward_at_net']['-10']}]" for k in keys) + " |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ the noise ensemble, added after the declaration

def b1_anomaly_std(D, month_of):
    return float(np.nanstd(D - A.background("B1", D, month_of)))


def matched_level(real_spreads, reference_spread, base=None):
    """2 m/s (the declared noise level) times the median of the real own-month anomaly spreads
    over the reference noise's own-month anomaly spread at 2 m/s."""
    base = A.NOISE_STD if base is None else base
    return base * float(np.median(real_spreads)) / reference_spread


def matched_scales(records):
    """Per band, the noise standard deviation at which the construction noise's own-month
    anomaly has the median spread of the band's real own-month anomaly over the seasons:
    2 m/s times the ratio of the real spread to that of the construction noise at 2 m/s.
    The real spread includes any signal. This defines a comparison amplitude, not a bound on
    noise amplitude or linked-history counts (review of 2026-10-19, F2)."""
    by = {}
    for r in records:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    month_of = A.month_index()
    ref = A.noise_field(A.NOISE_STD).reshape(A.SEASON_DAYS, 4, -1).mean(axis=1)
    ref_std = b1_anomaly_std(ref, month_of)
    out = {}
    for b in records[0]["bands"]:
        spreads = [b1_anomaly_std(A.season_daily(by[y], b)[1], month_of) for y in YEARS]
        out[b] = {"real_b1_anomaly_std": float(np.median(spreads)), "noise_b1_anomaly_std_at_2": ref_std, "matched_noise_std": matched_level(spreads, ref_std)}
    return out


def noise_ensemble(stds, seeds=NOISE_SEEDS):
    """For each labeled noise standard deviation, the westward histories per noise-only
    season under every combination, over the seeded seasons (mean, maximum, the count of
    seasons with any, and the per-season counts in seed order)."""
    month_of = A.month_index(); out = {}
    for label, std in stds.items():
        counts = {combo(r, l): [] for r, l in ORDER}
        for seed in seeds:
            H, _, _ = A.synth_hundredths(noise=A.noise_field(std, seed=seed))
            for rp in REPS:
                mins = rep_minima(rp, H, month_of, A.LONS)
                for linker in LINKERS:
                    counts[combo(rp, linker)].append(sum(1 for h in link(mins, linker) if classify(h) == "westward"))
        out[label] = {"std": std, "seasons": len(seeds), "per_combination": {k: {"mean": float(np.mean(v)), "max": int(max(v)), "seasons_with_any": int(sum(1 for x in v if x > 0)), "per_season": [int(x) for x in v]} for k, v in counts.items()}}
    return out


def noise_text(scales, ens):
    keys = [combo(r, l) for r, l in ORDER]
    L = ["# minima histories, the noise ensemble (added after the declaration, section 5 of the record; a comparison with a specified noise-only model, not a false-rate bound)",
         f"{len(NOISE_SEEDS)} noise-only seasons per level (seeds {NOISE_SEEDS[0]} to {NOISE_SEEDS[-1]}), the declared recipe; westward histories per season, mean (maximum, seasons with any)", "",
         "## the matched noise levels, per band", "", "| band | real own-month anomaly spread (median over seasons) | construction noise's at 2 m/s | matched noise level |", "| --- | --- | --- | --- |"]
    for b, d in scales.items():
        L.append(f"| {b} | {d['real_b1_anomaly_std']:.2f} | {d['noise_b1_anomaly_std_at_2']:.2f} | {d['matched_noise_std']:.2f} |")
    L += ["", "## westward histories per noise-only season", "", "| noise level | " + " | ".join(keys) + " |", "| --- |" + " --- |" * len(keys)]
    for label, d in ens.items():
        L.append(f"| {label} ({d['std']:.2f} m/s) | " + " | ".join(f"{d['per_combination'][k]['mean']:.2f} ({d['per_combination'][k]['max']}, {d['per_combination'][k]['seasons_with_any']})" for k in keys) + " |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ the archive

def season_histories(records):
    """{(year, band): (dates, {combo: [(history, class)]})} for every season and band."""
    by = {}
    for r in records:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    bands = list(records[0]["bands"]); lons = np.array(records[0]["lons"], float); month_of = A.month_index()
    out = {}
    for y in YEARS:
        for b in bands:
            dates, H = A.season_hundredths(by[y], b)
            per = {}
            for rep in REPS:
                mins = rep_minima(rep, H, month_of, lons)
                for linker in LINKERS:
                    hs = link(mins, linker)
                    per[combo(rep, linker)] = [(h, classify(h)) for h in hs]
            out[(y, b)] = (dates, per)
    return out


def summarize_westward(h, dates, year, band):
    x = h["lons"]; n = len(x)
    return {"year": year, "band": band, "first_date": dates[h["days"][0]], "month": int(dates[h["days"][0]][5:7]), "days": n,
            "first_lon": x[0], "last_lon": x[-1], "net_deg": x[-1] - x[0], "speed_deg_per_day": (x[-1] - x[0]) / (n - 1),
            "cut_east": x[0] >= BOX[1] - EDGE_DEG, "cut_west": x[-1] <= BOX[0] + EDGE_DEG,
            "arc_to_channel": x[0] >= 50.0 and x[-1] <= 42.0, "ends_at_or_west_of_40": x[-1] <= 40.0}


def archive_tables(hist):
    keys = [combo(r, l) for r, l in ORDER]
    out = {}
    for k in keys:
        west, standing, sens = {}, {}, {}
        for (y, b), (dates, per) in hist.items():
            for h, c in per[k]:
                if c == "westward":
                    west.setdefault(b, []).append(summarize_westward(h, dates, y, b))
                elif c == "standing":
                    standing[b] = standing.get(b, 0) + 1
                counts = sens.setdefault(b, {f"{n:g}": 0 for n in NET_SENSITIVITY})
                for net in NET_SENSITIVITY:
                    counts[f"{net:g}"] += classify(h, net) == "westward"
        bands = sorted({b for (_, b) in hist})
        res = {}
        for b in bands:
            w = west.get(b, [])
            arr = lambda key: np.array([s[key] for s in w], float)
            q = lambda v: ([float(np.percentile(v, p)) for p in (25, 50, 75)] if v.size else None)
            res[b] = {"westward": len(w), "westward_per_season": len(w) / len(YEARS),
                      "by_month_of_first_day": {str(m): sum(1 for s in w if s["month"] == m) for m in A.MONTHS},
                      "days_q": q(arr("days")), "speed_q": q(arr("speed_deg_per_day")), "first_lon_q": q(arr("first_lon")), "last_lon_q": q(arr("last_lon")),
                      "cut_east": sum(s["cut_east"] for s in w), "cut_west": sum(s["cut_west"] for s in w),
                      "arc_to_channel": sum(s["arc_to_channel"] for s in w), "ends_at_or_west_of_40": sum(s["ends_at_or_west_of_40"] for s in w),
                      "standing_per_season": standing.get(b, 0) / len(YEARS), "westward_at_net": sens.get(b, {"-6": 0, "-10": 0})}
        out[k] = {"bands": res, "westward_histories": [s for b in bands for s in west.get(b, [])]}
    return out


def ledger_match(entries, hist, tol=TOL_DEG):
    """Per combination, the ledger's entries by fitted direction class, matched when ONE
    westward history has a member within `tol` of the entry's anchor on at least half of
    its anchored days. Hits are counted per history, never pooled across histories (review
    of 2026-10-19, F3)."""
    keys = [combo(r, l) for r, l in ORDER]
    member = {k: {} for k in keys}
    for (y, b), (dates, per) in hist.items():
        for k in keys:
            for hid, (h, c) in enumerate(per[k]):
                if c != "westward":
                    continue
                for d, x in zip(h["days"], h["lons"]):
                    member[k].setdefault((dates[d], b), []).append(((y, b, hid), x))
    out = {k: {} for k in keys}
    for e in entries:
        cls = (e.get("speeds") or {}).get("direction") or "no fit"
        anchored = [d for d in e["days"] if d["anchored"]]
        for k in keys:
            t = out[k].setdefault(cls, {"entries": 0, "matched": 0})
            t["entries"] += 1
            hits = {}
            for d in anchored:
                for hkey in {hk for hk, x in member[k].get((d["date"], e["band"]), []) if abs(x - d["anchor_lon"]) <= tol}:
                    hits[hkey] = hits.get(hkey, 0) + 1
            t["matched"] += bool(anchored) and any(2 * n >= len(anchored) for n in hits.values())
    return out


def ordered_keys(selected):
    """The combinations in the declared order, with a selected one moved to the front."""
    keys = [combo(r, l) for r, l in ORDER]
    return ([selected] + [k for k in keys if k != selected]) if selected in keys else keys


def archive_text(tabs, led, selected):
    keys = ordered_keys(selected)
    f = lambda v, n=1: "" if v is None else ", ".join(f"{x:.{n}f}" for x in v)
    L = ["# minima histories, the archive (section 2)", f"primary combination: {selected or 'none (no combination passed the constructions)'}; every combination reported; quartiles are 25th, 50th and 75th percentiles", ""]
    L += ["## westward histories by band", "", "| combination | band | westward | per season | June, July, August, September (first day) | days, quartiles | speed (deg/day), quartiles | first lon, quartiles | last lon, quartiles | cut at east edge | cut at west edge | from 50 E or east to 42 E or west | ending at 40 E or west | at net -6 and -10 | standing per season |",
          "| --- |" + " --- |" * 14]
    for k in keys:
        for b, d in tabs[k]["bands"].items():
            bm = d["by_month_of_first_day"]
            L.append(f"| {k} | {b} | {d['westward']} | {d['westward_per_season']:.2f} | {bm['6']}, {bm['7']}, {bm['8']}, {bm['9']} | {f(d['days_q'], 0)} | {f(d['speed_q'])} | {f(d['first_lon_q'])} | {f(d['last_lon_q'])} | {d['cut_east']} | {d['cut_west']} | {d['arc_to_channel']} | {d['ends_at_or_west_of_40']} | {d['westward_at_net']['-6']}, {d['westward_at_net']['-10']} | {d['standing_per_season']:.2f} |")
    L += ["", "## the ledger's entries matched by a westward history on at least half of their anchored days, by fitted direction class (descriptive, entry-weighted)", "", "| combination | " + " | ".join(f"{c} matched of entries" for c in ("westward", "standing", "eastward")) + " |", "| --- |" + " --- |" * 3]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{led[k].get(c, {}).get('matched', 0)} of {led[k].get(c, {}).get('entries', 0)}" for c in ("westward", "standing", "eastward")) + " |")
    return "\n".join(L) + "\n"


def draw_first_last(path, tabs, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = tabs[key]["westward_histories"]; bands = list(tabs[key]["bands"])
    fig, axes = plt.subplots(1, len(bands), figsize=(4.4 * len(bands) + 1.5, 4.6), sharex=True, sharey=True, constrained_layout=True)
    sc = None
    for ax, b in zip(np.atleast_1d(axes), bands):
        w = [s for s in ws if s["band"] == b]
        if w:
            sc = ax.scatter([s["first_lon"] for s in w], [s["last_lon"] for s in w], c=[s["speed_deg_per_day"] for s in w], cmap="viridis_r", vmin=-6, vmax=-1, s=10)
        ax.plot([20, 75], [20, 75], color="0.6", lw=0.8)
        ax.axhline(42.0, color="k", lw=0.7, ls="--"); ax.axvline(50.0, color="k", lw=0.7, ls=":")
        ax.set_title(f"{b} ({len(w)})", fontsize=10); ax.set_xlabel("first longitude (degrees E)"); ax.grid(True, color="0.85", lw=0.4)
    np.atleast_1d(axes)[0].set_ylabel("last longitude (degrees E)")
    if sc is not None:
        fig.colorbar(sc, ax=axes, shrink=0.85, label="net speed (degrees per day)")
    fig.suptitle(f"Westward histories of daily minima under {key}, {len(YEARS)} seasons (2005 closed): first against last longitude, colored by net speed. Dashed at 42 E, dotted at 50 E. A history is a linked sequence under a stated rule, not an identified feature.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


# ------------------------------------------------------------------ main

def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive", "noise"))
    ap.add_argument("--out-dir", required=True); ap.add_argument("--constructions", default=None); ap.add_argument("--records", default=None); ap.add_argument("--ledger", default=None)
    ap.add_argument("--tag", default="", help="a version suffix for the records written, such as r1")
    a = ap.parse_args(argv)
    sfx = f"_{a.tag}" if a.tag else ""
    gen = {"script": "scripts/eastern_minima_histories.py", "tag": a.tag, "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_MINIMA_HISTORIES_2026-10-19.md",
           "anomaly_script_sha256": sha256_of(os.path.join(HERE, "eastern_anomaly_minima.py"))}
    if a.mode == "constructions":
        pj, pm = os.path.join(a.out_dir, f"eval60_hist_constructions{sfx}.json"), os.path.join(a.out_dir, f"eval60_hist_constructions{sfx}.md")
        refuse_existing([pj, pm])
        results = run_constructions(); verdict, selected = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict, "selected": selected}, fh, default=float)
        text = constructions_text(results, verdict, selected)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if a.mode == "noise":
        if not a.records:
            raise SystemExit("REFUSED: --records is required for the noise ensemble's matched levels")
        pj, pm = os.path.join(a.out_dir, f"eval60_hist_noise{sfx}.json"), os.path.join(a.out_dir, f"eval60_hist_noise{sfx}.md")
        refuse_existing([pj, pm])
        files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
        records = [SC.load_record(p) for p in files]
        SC.check_archive_identity(files, records)
        scales = matched_scales(records)
        stds = {"declared 2 m/s": A.NOISE_STD, **{f"matched to {b}": d["matched_noise_std"] for b, d in scales.items()}}
        ens = noise_ensemble(stds)
        gen["month_records"] = {os.path.basename(p): sha256_of(p) for p in files}
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "seeds": list(NOISE_SEEDS), "scales": scales, "ensemble": ens}, fh, default=float)
        text = noise_text(scales, ens)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.constructions and a.records and a.ledger):
        raise SystemExit("REFUSED: --constructions, --records and --ledger are required for the archive")
    with open(a.constructions, encoding="utf-8") as fh:
        cons = json.load(fh)
    selected = cons.get("selected")
    paths = {k: os.path.join(a.out_dir, v) for k, v in (("json", f"eval60_hist_archive{sfx}.json"), ("tables", f"eval60_hist_tables{sfx}.md"), ("figure", f"eval60_hist_first_last{sfx}.png"))}
    refuse_existing(paths.values())
    files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
    records = [SC.load_record(p) for p in files]
    if len(records) != len(YEARS) * len(A.MONTHS):
        raise SystemExit(f"REFUSED: {len(records)} month records, not {len(YEARS) * len(A.MONTHS)}")
    SC.check_archive_identity(files, records)
    hist = season_histories(records)
    tabs = archive_tables(hist)
    with open(a.ledger, encoding="utf-8") as fh:
        entries = json.load(fh)["ledger"]
    led = ledger_match(entries, hist)
    draw_first_last(paths["figure"], tabs, selected or combo(*ORDER[0]))
    gen.update({"constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}, "month_records": {os.path.basename(p): sha256_of(p) for p in files},
                "ledger": {"path": a.ledger, "sha256": sha256_of(a.ledger)}})
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "selected": selected, "tables": tabs, "ledger": led, "figure": {"path": paths["figure"], "combination": selected or combo(*ORDER[0]), "sha256": sha256_of(paths["figure"])}}, fh, default=float)
    text = archive_text(tabs, led, selected)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
