#!/usr/bin/env python3
"""The 600 and 700 hPa sign-change measurement of `EASTERN_VERTICAL_600_700_2026-10-11.md`.

On the integer-degree points of the retained region files (both levels on one grid,
terrain-valid where surface pressure exceeds the level strictly), it finds meridional-wind
sign-change brackets (v negative on the west, zero or positive on the east of two adjacent
columns), joins brackets on neighboring rows whose columns differ by at most one into
branches, and measures, per scored map and control: the branches relevant to the read
geometry at each level, every plausible 600 hPa match of each relevant 700 hPa branch
inside the declared window, the offset only where correspondence is unambiguous, the
additional-structure test at the shared-region maps and the branches on each recovered
700 hPa curvature region. A bracket is not a trough or a wave. No match is chosen map by
map and no sequence is read as motion.

    python3 scripts/pilot_eastern_vertical.py --declaration <vertical json> --year 2006 --reference <2006 declaration> \\
        --sp-file <sp> [--lineage-reading ... --control-run ... --control-case ... --calibration ... --wide-dir ... \\
        --finals-trace path:digest] --out <json>
"""
import argparse
import datetime as dt
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_alledge_replay as A  # noqa: E402
import pilot_eastern_measurement as EM  # noqa: E402
import pilot_field_sequence as FS  # noqa: E402

EPOCH = dt.datetime(1900, 1, 1)


def brackets(v, valid, lat1, lon1):
    """Every sign-change bracket: (row, column, latitude, mean longitude, west value, east
    value), v negative on the west and zero or positive on the east, both points valid."""
    v = np.asarray(v, float); ok = np.asarray(valid, bool) & np.isfinite(v)
    out = []
    for i in range(v.shape[0]):
        for j in range(v.shape[1] - 1):
            if ok[i, j] and ok[i, j + 1] and v[i, j] < 0.0 <= v[i, j + 1]:
                out.append({"i": i, "j": j, "lat": float(lat1[i]), "lon": float((lon1[j] + lon1[j + 1]) / 2.0),
                            "v_west": round(float(v[i, j]), 3), "v_east": round(float(v[i, j + 1]), 3)})
    return out


def branches(brs, shape):
    """Brackets on neighboring rows whose columns differ by at most one form one branch
    (eight-connected), each with its brackets, extent and row count."""
    from scipy import ndimage
    grid = np.zeros((shape[0], shape[1] - 1), bool)
    for b in brs:
        grid[b["i"], b["j"]] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3), int))
    out = {k: [] for k in range(1, n + 1)}
    for b in brs:
        out[int(labels[b["i"], b["j"]])].append(b)
    res = []
    for k in range(1, n + 1):
        bs = out[k]
        res.append({"id": k, "brackets": bs, "rows": len({b["i"] for b in bs}),
                    "lat_range": [min(b["lat"] for b in bs), max(b["lat"] for b in bs)], "lon_range": [min(b["lon"] for b in bs), max(b["lon"] for b in bs)],
                    "v_west_range": [min(b["v_west"] for b in bs), max(b["v_west"] for b in bs)], "v_east_range": [min(b["v_east"] for b in bs), max(b["v_east"] for b in bs)]})
    return res


def near(branch, m, t):
    return [b for b in branch["brackets"] if EM.ref_distance(b["lat"], b["lon"], m) <= t]


def relevant(brs, m, t):
    return [br["id"] for br in brs if near(br, m, t)]


def in_window(p, q, win):
    return abs(p["lat"] - q["lat"]) <= win["lat"] + 1e-9 and abs(p["lon"] - q["lon"]) <= win["lon"] + 1e-9


def plausible(anchors, brs600, win):
    """Every 600 hPa branch with a bracket inside the window of an anchor bracket."""
    return [br["id"] for br in brs600 if any(in_window(a, b, win) for a in anchors for b in br["brackets"])]


def offset(anchors, br600, win):
    sel = [b for b in br600["brackets"] if any(in_window(a, b, win) for a in anchors)]
    la = np.mean([b["lat"] for b in sel]) - np.mean([a["lat"] for a in anchors])
    lo = np.mean([b["lon"] for b in sel]) - np.mean([a["lon"] for a in anchors])
    return {"dlat": round(float(la), 3), "dlon": round(float(lo), 3), "brackets_600_in_window": len(sel), "anchors_700": len(anchors)}


def correspondence(rel700, matches):
    """UNAMBIGUOUS when there is exactly one relevant 700 branch with exactly one plausible
    600 match that matches no other relevant 700 branch."""
    if len(rel700) != 1:
        return False
    only = matches[rel700[0]]
    return len(only) == 1


def near_point(branch, lat, lon, t):
    return any(math.hypot(b["lat"] - lat, b["lon"] - lon) <= t for b in branch["brackets"])


def additional_structure(brs, m, surviving, t):
    """Branches within t of the read geometry (R) and of the surviving output (S), and
    whether some branch in R is not in S."""
    R = [br["id"] for br in brs if near(br, m, t)]
    S = [br["id"] for br in brs if surviving is not None and near_point(br, surviving[0], surviving[1], t)]
    return {"near_read": R, "near_surviving_output": S, "additional_structure": bool(R) and bool(set(R) - set(S))}


def summarize_branch(br):
    return {k: br[k] for k in ("id", "rows", "lat_range", "lon_range", "v_west_range", "v_east_range")}


def day_of(d):
    return (dt.datetime.strptime(d[:13], "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--declaration", "--sp-file", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--reference")
    ap.add_argument("--lineage-reading"); ap.add_argument("--control-run"); ap.add_argument("--control-case"); ap.add_argument("--calibration")
    ap.add_argument("--wide-dir"); ap.add_argument("--finals-trace")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    decl = json.load(open(args.declaration))
    g, win, tols = decl["grid"], decl["window_deg"], decl["tolerances_deg"]
    lat1 = np.arange(g["lat"][1], g["lat"][0] - 0.5, -1.0); lon1 = np.arange(g["lon"][0], g["lon"][1] + 0.5, 1.0)
    scored, ref = [], None
    if args.reference:
        if A.X.digest(args.reference) != decl["scored_maps_from"]["sha256"]:
            raise SystemExit("REFUSED: the reference is not the declared one")
        ref = json.load(open(args.reference))["reference"]
        scored = [m for m in ref["maps"] if m["scored"]]
    controls = [c for c in decl["controls"] if c["year"] == args.year]
    cdates = {c["name"]: EM.M.inspected_dates(c["window"]) for c in controls}
    dates = sorted({m["date"] for m in scored} | {d for ds in cdates.values() for d in ds})
    days = [day_of(d) for d in dates]
    V = {lev: FS.field_on(decl["v_files"][str(args.year)][str(lev)], "v", "m s**-1", lat1, lon1, days, float(lev)) for lev in (600, 700)}
    SP = FS.surface_pressure_on(args.sp_file, lat1, lon1, days)
    per_date = {}
    for n, d in enumerate(dates):
        per_date[d] = {}
        for lev in (600, 700):
            valid = np.isfinite(V[lev][n]) & (SP[n] > lev * 100.0)
            brs = branches(brackets(V[lev][n], valid, lat1, lon1), V[lev][n].shape)
            per_date[d][lev] = {"branches": brs, "below_ground_points": int((np.isfinite(SP[n]) & ~(SP[n] > lev * 100.0)).sum())}
    maps_out = []
    for m in scored:
        b600, b700 = per_date[m["date"]][600]["branches"], per_date[m["date"]][700]["branches"]
        row = {"date": m["date"], "segment": m["segment"], "west_of_40E": EM.west_of_40(m)}
        for t in tols:
            row[f"relevant_{t}"] = {"700": relevant(b700, m, t), "600": relevant(b600, m, t)}
        rel = row[f"relevant_{tols[0]}"]["700"]
        by700 = {br["id"]: br for br in b700}; by600 = {br["id"]: br for br in b600}
        matches = {k: plausible(near(by700[k], m, max(tols)), b600, win) for k in rel}
        row["plausible_600_matches"] = {str(k): v for k, v in matches.items()}
        row["unambiguous"] = correspondence(rel, matches)
        if row["unambiguous"]:
            k = rel[0]
            row["offset"] = offset(near(by700[k], m, max(tols)), by600[matches[k][0]], win)
        row["branches_700"] = [summarize_branch(by700[k]) for k in rel]
        row["branches_600_matched"] = [summarize_branch(by600[k]) for k in sorted({x for v in matches.values() for x in v})]
        maps_out.append(row)
    regions = []
    if args.lineage_reading:
        regions = shared_regions(args, per_date, ref, decl)
    ctrl_out = []
    for c in controls:
        box = c["box"]
        for d in cdates[c["name"]]:
            bx = {}
            for lev in (600, 700):
                bx[lev] = [br for br in per_date[d][lev]["branches"] if any(EM.M.in_box(b["lat"], b["lon"], box) for b in br["brackets"])]
            pairs = {}
            for br in bx[700]:
                anchors = [b for b in br["brackets"] if EM.M.in_box(b["lat"], b["lon"], box)]
                pairs[br["id"]] = plausible(anchors, bx[600], win)
            counts600 = {}
            for v in pairs.values():
                for x in v:
                    counts600[x] = counts600.get(x, 0) + 1
            one_to_one = [(k, v[0]) for k, v in pairs.items() if len(v) == 1 and counts600[v[0]] == 1]
            by600 = {br["id"]: br for br in bx[600]}; by700 = {br["id"]: br for br in bx[700]}
            offs = [offset([b for b in by700[k]["brackets"] if EM.M.in_box(b["lat"], b["lon"], box)], by600[j], win) for k, j in one_to_one]
            ctrl_out.append({"control": c["name"], "date": d, "branches_in_box": {"700": len(bx[700]), "600": len(bx[600])},
                             "plausible_pairs": {str(k): v for k, v in pairs.items()}, "one_to_one_offsets": offs,
                             "branches_700_without_match": sum(1 for v in pairs.values() if not v)})
    out = {"generated_by": "scripts/pilot_eastern_vertical.py", "script_sha256": A.X.digest(os.path.abspath(__file__)), "year": args.year,
           "declaration": {"path": args.declaration, "sha256": A.X.digest(args.declaration)}, "grid": g, "window_deg": win,
           "below_ground_points": {d: {str(lev): per_date[d][lev]["below_ground_points"] for lev in (600, 700)} for d in dates},
           "scored_maps": maps_out, "shared_regions": regions, "controls": ctrl_out,
           "limits": "a bracket is not a trough or a wave; matches are retained, not chosen; no motion is read from a sequence of matches"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"{len(maps_out)} scored maps, {sum(1 for r in maps_out if r['unambiguous'])} unambiguous; {len(regions)} shared-region records; {len(ctrl_out)} control maps. Record {args.out}.")
    return 0


def shared_regions(args, per_date, ref, decl):
    """At each shared-region map of the lineage reading (3 degrees): the additional-structure
    test at both levels, then each level's branches on the recovered 700 hPa region."""
    import pilot_eastern_maxima as MX
    import pilot_eastern_scale as SC
    run = A.P.load_run(args.control_run, year=args.year)
    case_b = A.SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, args.year)
    ds = run["record"]["dataset_specific"]
    ct, ft, _thr = A.thresholds_for_run((float(ds["coarse_threshold"]), float(ds["fine_threshold"])), args.calibration, False, True, run["domain"])
    absorb = bool(ds["tracker_flags"]["absorb"])
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    lat_c, lon_c = np.asarray(grids["latgrid_c"], float)[:, 0], np.asarray(grids["longrid_c"], float)[0, :]
    times = np.asarray(case_b["time"], float).ravel()
    path, dg = args.finals_trace.rsplit(":", 1)
    finals = MX.finals_from_trace(path, dg)
    lr = json.load(open(args.lineage_reading))
    maps_by = {(m["date"], m["segment"]): m for m in ref["maps"]}
    t = decl["tolerances_deg"][0]
    out = []
    for row in lr["readings"]["tolerance"]["uncovered_scored_maps"]:
        d = row["date"]
        k = SC.map_steps(times, [d])[d]
        fields = A.prepared_fields(wide, k, *sel)
        lin = {}
        waves = A.detect_from_prepared(fields, float(times[k]), ct, ft, grids, absorb=absorb, lineage=lin)
        mine = [(float(w["lat_mean"]), float(w["lon_mean"]), A.R.region_digest(w["region"])) for w in waves]
        if not MX.same_finals(mine, finals[d]):
            raise SystemExit(f"REFUSED: at {d} the recomputed detector does not reproduce the retained finals")
        coarse = {e["input"]: e for e in lin["coarse"]}
        seen = set()
        for ax in row["relevant_axes"]:
            nearx = [x for x in (ax.get("dropped_near") or []) if x["arc_deg"] == 0.0]
            if not nearx:
                continue
            want = next(p["region_sha256"] for p in ax["positions"] if p["stage"] == "coarse region")
            mask = np.asarray(coarse[ax["axis"]]["select"]["region"], bool)
            if A.R.region_digest(mask) != want:
                raise SystemExit(f"REFUSED: at {d} the recomputed region of axis {ax['axis']} is not the recorded one")
            if want in seen:
                continue
            seen.add(want)
            m = maps_by[(d, row["segment"])]
            surv = nearx[0]["output_position"]
            rec = {"date": d, "segment": row["segment"], "region_sha256": want, "cells": int(mask.sum()), "surviving_output": surv}
            for lev in (700, 600):
                brs = per_date[d][lev]["branches"]
                on_region = [br["id"] for br in brs if any(mask[MX.coarse_cell_of(b["lat"], b["lon"], lat_c, lon_c)] for b in br["brackets"])]
                rec[str(lev)] = {**additional_structure(brs, m, surv, t), "branches_on_region": on_region,
                                 "branches": [summarize_branch(br) for br in brs if br["id"] in set(on_region) | set(additional_structure(brs, m, surv, t)["near_read"])]}
            out.append(rec)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
