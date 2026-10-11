#!/usr/bin/env python3
"""The 600 hPa support comparison of `EASTERN_600_SUPPORT_2026-10-11.md`, one season per run.

Each retained 700 hPa relative-vorticity maximum (arm B of the closed maxima comparison)
receives one of three states from 600 hPa relative vorticity in the 3 by 3 block of
whole-degree cells centered on it. SUPPORTED when a valid cell is above B's own threshold,
UNSUPPORTED when all nine cells are valid and none is, and UNASSESSABLE otherwise, which is
unavailable evidence and never a rejection. This is local above-threshold vorticity
support, not center matching, with inherited numbers in a new experimental role.

The 600 hPa winds are read at exact whole-degree coordinates of the half-degree region
files, with the field-sequence tool's terrain rule and derivative. Before any support is
assigned, B is recomputed at 700 hPa from the same files by the same sampling, terrain
rule, derivative and peak rule, and the run refuses unless that reproduces every retained
B maximum. The unfiltered measures must also equal the retained record's.

    python3 scripts/pilot_eastern_support600.py --declaration <support json> --year 2006 --out <json>
    python3 scripts/pilot_eastern_support600.py --declaration <support json> --year 1990 --scored-record <2006 output> --out <json>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_alledge_replay as A  # noqa: E402
import pilot_eastern_maxima as MX  # noqa: E402
import pilot_eastern_measurement as EM  # noqa: E402
import pilot_field_sequence as FS  # noqa: E402

STATES = ("supported", "unsupported", "unassessable")
WIND_UNITS = "m s**-1"


def grid(decl):
    """Whole-degree latitudes (north to south, as the wide case orders them) and longitudes."""
    g = decl["grid"]
    return np.arange(g["lat"][1], g["lat"][0] - 0.5, -1.0), np.arange(g["lon"][0], g["lon"][1] + 0.5, 1.0)


def vorticity(u, v, sp, level_hpa, lat1, lon1):
    """Relative vorticity in units of 1e-5 per second, NaN where not valid, and the cell
    classes, by the field-sequence tool's terrain rule and derivative (as its panels form B)."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    wcls = FS.terrain_classes(sp, np.where(np.isfinite(u) & np.isfinite(v), u, np.nan), level_hpa)
    zeta, zcls = FS.relative_vorticity(u, v, lat1, lon1, wcls)
    return np.where(zcls == "valid", zeta * 1e5, np.nan), zcls


def support_state(z, cls, i, j, threshold):
    """The support state of the 700 hPa maximum at cell (i, j) from the 600 hPa block around
    it, with the count of valid cells, of cells above the threshold, and of invalid cells by
    reason."""
    ny, nx = z.shape
    valid, above, reasons = 0, 0, {}
    for a in (i - 1, i, i + 1):
        for b in (j - 1, j, j + 1):
            if not (0 <= a < ny and 0 <= b < nx):
                why = "outside_grid"
            elif str(cls[a, b]) != "valid":
                why = str(cls[a, b])
            elif not np.isfinite(z[a, b]):
                why = "not_finite"
            else:
                valid += 1
                above += int(z[a, b] > threshold)
                continue
            reasons[why] = reasons.get(why, 0) + 1
    state = "supported" if above else ("unsupported" if valid == 9 else "unassessable")
    return {"state": state, "valid_cells": valid, "cells_above": above, "invalid_reasons": reasons}


def cell_of(p, lat1, lon1):
    i, j = np.flatnonzero(np.abs(lat1 - p["lat"]) < 1e-9), np.flatnonzero(np.abs(lon1 - p["lon"]) < 1e-9)
    if i.size != 1 or j.size != 1:
        raise SystemExit(f"REFUSED: the maximum at {p['lat']}, {p['lon']} is not a cell of the grid")
    return int(i[0]), int(j[0])


def same_maxima(mine, retained, tol):
    key = lambda p: (p["lat"], p["lon"])  # noqa: E731
    a, b = sorted(mine, key=key), sorted(retained, key=key)
    return len(a) == len(b) and all(x["lat"] == y["lat"] and x["lon"] == y["lon"] and x["cells"] == y["cells"] and abs(x["value"] - y["value"]) <= tol
                                    for x, y in zip(a, b))


def tally(points):
    out = {s: 0 for s in STATES}
    for p in points:
        out[p["support"]["state"]] += 1
    out["total"] = len(points)
    return out


def supported_share_of_assessable(t):
    n = t["supported"] + t["unsupported"]
    return None if n == 0 else t["supported"] / n


def controls_against(scored_share, control_share):
    """The declared criterion for one control: it counts against the rule when the
    control's supported share of assessable maxima is at least the scored boxes'."""
    if scored_share is None or control_share is None:
        return None
    return bool(control_share >= scored_share)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--scored-record", help="the 2006 output, for the scored boxes' share in the 1990 controls' criterion")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    decl = json.load(open(args.declaration))
    y = str(args.year)
    md_spec, rec_spec = decl["maxima_declaration"], decl["maxima_records"][y]
    if A.X.digest(md_spec["path"]) != md_spec["sha256"]:
        raise SystemExit("REFUSED: the maxima declaration is not the declared one")
    if A.X.digest(rec_spec["path"]) != rec_spec["sha256"]:
        raise SystemExit(f"REFUSED: the {y} maxima record is not the declared one")
    md, rec = json.load(open(md_spec["path"])), json.load(open(rec_spec["path"]))
    if rec["declaration"]["sha256"] != md_spec["sha256"]:
        raise SystemExit("REFUSED: the maxima record was not made under the maxima declaration")
    if float(decl["support"]["threshold_per_second"]) != float(md["thresholds"]["B_per_second"]):
        raise SystemExit("REFUSED: the support threshold is not B's own")
    thr = float(md["thresholds"]["B_per_second"]) * 1e5                  # as B's peak rule compares it
    dom, tols = md["domain"], tuple(decl["tolerances_deg"])
    lat1, lon1 = grid(decl)
    latg, long_ = np.meshgrid(lat1, lon1, indexing="ij")
    dates = sorted(rec["maxima_by_date"])
    days = [FS.to_day(d) for d in dates]
    f = decl["files"][y]
    sp = FS.surface_pressure_on(f["sp"], lat1, lon1, days)
    u7, v7 = (FS.field_on(f[k], k[0], WIND_UNITS, lat1, lon1, days, 700.0) for k in ("u700", "v700"))
    u6, v6 = (FS.field_on(f[k], k[0], WIND_UNITS, lat1, lon1, days, 600.0) for k in ("u600", "v600"))
    reasons_total = {}
    for n, d in enumerate(dates):
        z7, c7 = vorticity(u7[n], v7[n], sp[n], 700.0, lat1, lon1)       # the reproduction gate, at 700 hPa only
        mine, _ = MX.local_maxima(z7, c7 == "valid", thr, latg, long_)
        mine = [p for p in mine if MX.in_domain(p, dom)]
        if not same_maxima(mine, rec["maxima_by_date"][d]["B"], float(decl["reproduction_gate"]["value_tolerance"])):
            raise SystemExit(f"REFUSED: at {d} B recomputed from the region files is not the retained B")
        z6, c6 = vorticity(u6[n], v6[n], sp[n], 600.0, lat1, lon1)
        for p in rec["maxima_by_date"][d]["B"]:
            p["support"] = support_state(z6, c6, *cell_of(p, lat1, lon1), thr)
            for k, c in p["support"]["invalid_reasons"].items():
                reasons_total[k] = reasons_total.get(k, 0) + c
    state_at = {d: {(p["lat"], p["lon"]): p for p in r["B"]} for d, r in rec["maxima_by_date"].items()}
    out_sum = {"maxima": tally([p for r in rec["maxima_by_date"].values() for p in r["B"]]), "invalid_cells_by_reason": reasons_total}

    scored_rows, box_points = [], []
    if rec["scored_maps"]:
        ref_spec = decl["reference"]
        if A.X.digest(ref_spec["path"]) != ref_spec["sha256"]:
            raise SystemExit("REFUSED: the reference is not the declared one")
        ref = json.load(open(ref_spec["path"]))["reference"]
        maps_by = {(m["date"], m["segment"]): m for m in ref["maps"] if m["scored"]}
        for row in rec["scored_maps"]:
            m = maps_by[(row["date"], row["segment"])]
            box = ref["segments"][row["segment"]]["case_box"]
            pts = rec["maxima_by_date"][row["date"]]["B"]
            b = MX.correspondence(pts, m, box, tols)
            if any(b[k] != row["B"][k] for k in ("nearest_deg", "within", "in_case_box")):
                raise SystemExit(f"REFUSED: at {row['date']} the unfiltered B measures are not the retained ones")
            sup = [p for p in pts if p["support"]["state"] == "supported"]
            s = MX.correspondence(sup, m, box, tols)
            dist = {id(p): EM.ref_distance(p["lat"], p["lon"], m) for p in pts}
            inbox = [p for p in pts if MX.in_box(p["lat"], p["lon"], box)]
            box_points += inbox
            near = {str(t): tally([p for p in pts if dist[id(p)] <= t]) for t in tols}
            away = {str(t): tally([p for p in inbox if dist[id(p)] > t]) for t in tols}
            lost = {str(t): bool(b["within"][str(t)] and not s["within"][str(t)]) for t in tols}
            scored_rows.append({"date": row["date"], "segment": row["segment"], "F": row["F"], "B": b, "supported": {**s, **MX.compare(s, row["F"])},
                                "near": near, "away_in_case_box": away, "lost": lost})
    regions = []
    for g in rec["shared_regions"]:
        inside = []
        for x in g["B"]["maxima_inside"]:
            p = state_at[g["date"]].get((x["lat"], x["lon"]))
            if p is None:
                raise SystemExit(f"REFUSED: a maximum inside the region at {g['date']} is not a retained B maximum")
            inside.append({**x, "support": p["support"]})
        sep_b = {str(t): bool(len(inside) > 1 and any(x["distance_to_read_deg"] <= t for x in inside)) for t in tols}
        if sep_b != g["B"]["separates"]:
            raise SystemExit(f"REFUSED: at {g['date']} the region's unfiltered separation is not the retained one")
        sup = [x for x in inside if x["support"]["state"] == "supported"]
        regions.append({"reading": g["reading"], "date": g["date"], "segment": g["segment"], "region_sha256": g["region_sha256"], "maxima_inside": tally(inside),
                        "separates_B": sep_b, "separates_supported": {str(t): bool(len(sup) > 1 and any(x["distance_to_read_deg"] <= t for x in sup)) for t in tols}})
    control_rows = []
    for c in rec["controls"]:
        pts = [p for p in rec["maxima_by_date"][c["date"]]["B"] if MX.in_box(p["lat"], p["lon"], c["box"])]
        if len(pts) != c["counts_in_box"]["B"]:
            raise SystemExit(f"REFUSED: at {c['date']} the {c['control']} box count is not the retained one")
        control_rows.append({"control": c["control"], "date": c["date"], "finals": c["counts_in_box"]["finals"], "B": tally(pts)})

    if scored_rows:
        segs = sorted({r["segment"] for r in scored_rows})
        moving = set(decl["segments"]["moving"])
        def maps_within(rows, key):
            return {str(t): sum(1 for r in rows if r[key]["within"][str(t)]) for t in tols}
        def add(ts):
            return {k: sum(t[k] for t in ts) for k in (*STATES, "total")}
        by_seg = {}
        for sg in segs + ["moving", "standing"]:
            rows = [r for r in scored_rows if (r["segment"] == sg) or (sg == "moving" and r["segment"] in moving) or (sg == "standing" and r["segment"] not in moving)]
            by_seg[sg] = {"maps": len(rows), "maps_within": {k: maps_within(rows, k) for k in ("F", "B", "supported")},
                          "near": {str(t): add([r["near"][str(t)] for r in rows]) for t in tols},
                          "away_in_case_box": {str(t): add([r["away_in_case_box"][str(t)] for r in rows]) for t in tols}}
        boxes = tally(box_points)
        out_sum["scored"] = {
            "maps": len(scored_rows), "by_segment": by_seg, "case_boxes": boxes, "case_boxes_supported_share_of_assessable": supported_share_of_assessable(boxes),
            "lost_maps": {str(t): [{"date": r["date"], "segment": r["segment"], "near_states": r["near"][str(t)]} for r in scored_rows if r["lost"][str(t)]] for t in tols},
            "nearer_without_added_density_supported": sum(1 for r in scored_rows if r["supported"]["nearer_without_added_density"])}
        bench = decl["benchmarks"]["frozen_detector_moving_within"]
        mv = by_seg["moving"]["maps_within"]["supported"]
        away3 = by_seg["moving"]["away_in_case_box"]["3.0"]["unsupported"] + by_seg["standing"]["away_in_case_box"]["3.0"]["unsupported"]
        reg3 = sum(g["maxima_inside"]["unsupported"] for g in regions if g["reading"] == "tolerance")
        out_sum["against"] = {"loss": bool(mv["3.0"] < bench["3.0"] or mv["5.0"] < bench["5.0"]),
                              "no_reduction": bool(away3 == 0 or reg3 == 0), "away_unsupported_at_3": away3, "region_unsupported_at_3": reg3}
    if regions:
        out_sum["regions"] = {name: {"regions": sum(1 for g in regions if g["reading"] == name),
                                     "maxima_inside": {k: sum(g["maxima_inside"][k] for g in regions if g["reading"] == name) for k in (*STATES, "total")},
                                     "separates_B": sum(1 for g in regions if g["reading"] == name and g["separates_B"][t]),
                                     "separates_supported": sum(1 for g in regions if g["reading"] == name and g["separates_supported"][t])}
                              for name, t in (("tolerance", "3.0"), ("alternative", "5.0"))}
    scored_share = out_sum.get("scored", {}).get("case_boxes_supported_share_of_assessable")
    scored_from = None
    if args.scored_record:
        sr = json.load(open(args.scored_record))
        scored_share = sr["summary"]["scored"]["case_boxes_supported_share_of_assessable"]
        scored_from = {"path": args.scored_record, "sha256": A.X.digest(args.scored_record)}
    if control_rows:
        out_sum["controls"] = {}
        for name in sorted({c["control"] for c in control_rows}):
            rows = [c for c in control_rows if c["control"] == name]
            t = {k: sum(c["B"][k] for c in rows) for k in (*STATES, "total")}
            share = supported_share_of_assessable(t)
            out_sum["controls"][name] = {"maps": len(rows), "finals": sum(c["finals"] for c in rows), "B": t, "supported_share_of_assessable": share,
                                         "scored_share_compared": scored_share, "counts_against": controls_against(scored_share, share)}
    for r in rec["maxima_by_date"].values():
        r.pop("A", None)
    res = {"generated_by": "scripts/pilot_eastern_support600.py", "script_sha256": A.X.digest(os.path.abspath(__file__)), "year": args.year,
           "declaration": {"path": args.declaration, "sha256": A.X.digest(args.declaration)}, "maxima_record": rec_spec, "scored_record": scored_from,
           "reproduced_B_at_700hPa_at_every_map": True, "summary": out_sum, "scored_maps": scored_rows, "shared_regions": regions, "controls": control_rows,
           "maxima_by_date": rec["maxima_by_date"], "seconds": round(time.time() - t0, 1),
           "limits": "local above-threshold vorticity support with inherited numbers in a new role, not center matching; unassessable is unavailable evidence, not rejection; "
                     "neither reduced counts nor vertical agreement establishes physical validity; references provisional; standing correspondence is not false detection"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(out_sum, indent=1))
    print(f"Record {args.out} in {res['seconds']} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
