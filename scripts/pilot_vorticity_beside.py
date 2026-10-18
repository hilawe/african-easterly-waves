#!/usr/bin/env python3
"""A second-field reading over saved listings, for the twelfth direction discussion
(`EASTERN_DIRECTION_RECOMMENDATION_12_2026-10-17.md`). The reference sheets of 1983 and
1987 read a cyclonic vorticity line on the eastern side of the northerly band moving with
it. The listings already carry every relative-vorticity maximum per row and step beside
the selected wind minimum, so this asks, from the saved files alone, whether a cyclonic
maximum sits east of the selection at each step of a westward phase and whether its
position changes with the selection's. It is a reading of saved positions with a stated
predicate, not a correspondence rule, and it changes no record.

    python3 scripts/pilot_vorticity_beside.py --phases PH.json --listing LIST.json --rows 21 19 --out OUT.md
"""
import argparse
import hashlib
import json
import os
import statistics


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def beside(steps, row, i_start, i_end, min_zeta=2.0, max_east=8.0):
    """Per step of the span, the selected wind minimum's longitude and the nearest
    cyclonic maximum (at least `min_zeta` in 1e-5 per second) east of it within `max_east`
    degrees, or None when there is none."""
    out = []
    for i in range(i_start, i_end + 1):
        m = next((m for m in steps[i]["rows"] if int(m["row_lat"]) == row), None)
        if m is None or m["status"] != "assigned":
            out.append({"time": steps[i]["time"], "sel": None, "zmax": None, "offset": None}); continue
        sel = m["i1"]["lon"]
        cands = [z for z in m.get("zeta_maxima", []) if z["zeta_1e5"] >= min_zeta and 0 < z["lon"] - sel <= max_east]
        z = min(cands, key=lambda z: z["lon"] - sel) if cands else None
        out.append({"time": steps[i]["time"], "sel": sel, "zmax": None if z is None else z["lon"], "zeta": None if z is None else z["zeta_1e5"], "offset": None if z is None else z["lon"] - sel})
    return out


def summarize(out):
    offs = [o["offset"] for o in out if o["offset"] is not None]
    with_max = [o for o in out if o["offset"] is not None]
    net_sel = (with_max[-1]["sel"] - with_max[0]["sel"]) if len(with_max) >= 2 else None
    net_z = (with_max[-1]["zmax"] - with_max[0]["zmax"]) if len(with_max) >= 2 else None
    return {"steps": len(out), "with_maximum": len(offs), "offset_median": statistics.median(offs) if offs else None,
            "offset_min": min(offs) if offs else None, "offset_max": max(offs) if offs else None,
            "net_change_selection_between_first_and_last_step_with_maximum": net_sel, "net_change_maximum_between_those_steps": net_z}


def classify_steps(res, rec, row):
    """Keep the reader's west, east or standing label inside a chain. Outside chains,
    use unassessable when the row-step is unassigned or every touching pair is
    unassessable. Use break otherwise. An individually supported step can be
    unassessable here because its neighboring steps lack support."""
    n = len(rec["steps"])
    lab = {i: m for i, m in res["motion"].get(str(row), res["motion"].get(row, []))}
    pairs = {p["k"]: p["class"] for p in res["pairs"].get(str(row), res["pairs"].get(row, []))}
    out = []
    for i in range(n):
        if i in lab:
            out.append(lab[i]); continue
        m = next((m for m in rec["steps"][i]["rows"] if int(m["row_lat"]) == row), None)
        touching = [pairs[k] for k in (i - 1, i) if k in pairs]
        if m is None or m.get("status") != "assigned" or (touching and all(c == "unassessable" for c in touching)):
            out.append("unassessable")
        else:
            out.append("break")
    return out


CLASS_ORDER = ("west", "standing", "east", "break", "unassessable")


def beside_all(res, rec, rows, min_zeta=2.0, max_east=8.0, comove=1.5):
    """Per class of row-step, over the given rows: the steps, those with a cyclonic
    maximum east of the selection within the distance, the offset's median, and over
    consecutive pairs whose first step is in the class and which are continuous with a
    maximum at both ends, how many have the maximum moving with the selection within
    `comove` degrees."""
    acc = {c: {"class": c, "steps": 0, "with_maximum": 0, "offsets": [], "continuous_pairs": 0, "pairs_with_both": 0, "pairs_comoving": 0} for c in CLASS_ORDER}
    n = len(rec["steps"])
    for row in rows:
        cls = classify_steps(res, rec, row)
        b = beside(rec["steps"], row, 0, n - 1, min_zeta, max_east)
        pairs = {p["k"]: p["class"] for p in res["pairs"].get(str(row), res["pairs"].get(row, []))}
        for i in range(n):
            a = acc[cls[i]]
            a["steps"] += 1
            if b[i]["offset"] is not None:
                a["with_maximum"] += 1; a["offsets"].append(b[i]["offset"])
            if i + 1 < n and pairs.get(i, "").startswith("continuous"):
                a["continuous_pairs"] += 1
            if i + 1 < n and pairs.get(i, "").startswith("continuous") and b[i]["offset"] is not None and b[i + 1]["offset"] is not None:
                a["pairs_with_both"] += 1
                if abs((b[i + 1]["zmax"] - b[i]["zmax"]) - (b[i + 1]["sel"] - b[i]["sel"])) <= comove:
                    a["pairs_comoving"] += 1
    out = []
    for c in CLASS_ORDER:
        a = acc[c]
        out.append({"class": c, "steps": a["steps"], "with_maximum": a["with_maximum"], "offset_median": statistics.median(a["offsets"]) if a["offsets"] else None,
                    "offset_le_3": sum(1 for o in a["offsets"] if o <= 3.0), "continuous_pairs_total": a["continuous_pairs"], "pairs_with_both": a["pairs_with_both"], f"pairs_comoving_within_{comove:g}": a["pairs_comoving"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", required=True); ap.add_argument("--listing", required=True); ap.add_argument("--rows", nargs="+", type=int, default=None)
    ap.add_argument("--set", default="primary"); ap.add_argument("--min-zeta", type=float, default=2.0); ap.add_argument("--max-east", type=float, default=8.0); ap.add_argument("--out", required=True)
    ap.add_argument("--all-rows", action="store_true", help="part one of the history-layer package: every assigned row and step classed by the reader's labels, over a grid of strength and distance thresholds")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSED: {a.out} exists and a record is never overwritten")
    ph = json.load(open(a.phases)); rec = json.load(open(a.listing))
    assert ph["record"]["sha256"] == sha256_of(a.listing), "the phases output was not computed from this listing"
    res = ph["results"][a.set]
    if a.all_rows:
        rows = sorted({int(r) for r in res["pairs"]}, reverse=True)
        L = [f"# cyclonic vorticity maxima east of the selected wind minimum at every row and step, {os.path.basename(a.listing)} (sha256 {ph['record']['sha256'][:16]}), set {a.set}, rows {rows[0]} to {rows[-1]} N",
             "Every listed row-step keeps its reader motion label inside a chain. Outside chains, an unassigned step or a step with every touching pair unassessable is unassessable. The remaining steps are breaks. Individual support does not guarantee an assessable pair. The nearest relative-vorticity maximum must meet the strength threshold (1e-5 per second) and lie east of the selection within the distance. Co-movement is counted over consecutive continuous pairs with qualifying maxima at both ends, using a displacement difference of at most 1.5 degrees, and the count of all continuous pairs in the class is printed beside it so that the conditional and the joint fractions can both be read. This is a reading of saved positions.", ""]
        for z in (1.0, 2.0, 3.0):
            for d in (4.0, 6.0, 8.0):
                L.append(f"## strength at least {z:g}, distance at most {d:g} degrees")
                L.append("| class | steps | with a maximum | offset median | offsets of 3 degrees or less | continuous pairs (all) | continuous pairs with both | co-moving within 1.5 |")
                L.append("| --- |" + " --- |" * 7)
                for t in beside_all(res, rec, rows, z, d):
                    med = "" if t["offset_median"] is None else f"{t['offset_median']:g}"
                    L.append(f"| {t['class']} | {t['steps']} | {t['with_maximum']} | {med} | {t['offset_le_3']} | {t['continuous_pairs_total']} | {t['pairs_with_both']} | {t['pairs_comoving_within_1.5']} |")
                L.append("")
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")
        print("\n".join(L[:14]))
        return 0
    if not a.rows:
        raise SystemExit("REFUSED: --rows is required without --all-rows")
    L = [f"# cyclonic vorticity maxima east of the selected wind minimum along westward phases, {os.path.basename(a.listing)} (sha256 {ph['record']['sha256'][:16]}), set {a.set}",
         f"predicate: at every step of each row's longest westward phase of 48 h or more, the nearest relative-vorticity maximum of at least {a.min_zeta:g} (1e-5 per second) east of the selection within {a.max_east:g} degrees; a reading of saved positions, not a correspondence rule", ""]
    L.append("| row | phase | steps | with a maximum | offset median (deg) | offset range | selection net change (deg) | maximum net change (deg) |")
    L.append("| --- |" + " --- |" * 7)
    for row in a.rows:
        phases = [p for p in res["phases"][str(row)] if p["direction"] == "west" and p["hours"] >= 48]
        if not phases:
            L.append(f"| {row} N | no westward phase of 48 h or more | | | | | | |"); continue
        p = max(phases, key=lambda p: p["hours"])
        out = beside(rec["steps"], row, p["i_start"], p["i_end"], a.min_zeta, a.max_east)
        sm = summarize(out)
        med = "" if sm["offset_median"] is None else f"{sm['offset_median']:g}"
        rng = "" if sm["offset_min"] is None else f"{sm['offset_min']:g} to {sm['offset_max']:g}"
        dsel = sm["net_change_selection_between_first_and_last_step_with_maximum"]
        dmax = sm["net_change_maximum_between_those_steps"]
        dsel_s = "" if dsel is None else f"{dsel:+g}"
        dmax_s = "" if dmax is None else f"{dmax:+g}"
        span = f"{p['start'][5:]} to {p['end'][5:]} ({p['hours']} h, {p['lon_start']:g} to {p['lon_end']:g} E)"
        L.append(f"| {row} N | {span} | {sm['steps']} | {sm['with_maximum']} | {med} | {rng} | {dsel_s} | {dmax_s} |")
        cells = []
        for o in out:
            if o["sel"] is None:
                continue
            mx = "-" if o["zmax"] is None else f"{o['zmax']:g} ({o['zeta']:.1f})"
            cells.append(f"{o['time'][5:]} sel {o['sel']:g} max {mx}")
        L.append("")
        L.append(f"  {row} N by step: " + ", ".join(cells))
        L.append("")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(l for l in L if l.startswith("| ") and not l.startswith("| row") and not l.startswith("| ---")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
