#!/usr/bin/env python3
"""The mask-to-axis check of `EASTERN_MASK_AXIS_CHECK_2026-10-11.md`, one season per run.

With the prepared coarse fields, the curvature threshold and the contour extraction held
fixed, it extracts trough axes with the production westerly mask (mask on) and with only
that mask omitted (mask off). The mask-on axes must first reproduce the retained axes at
the scored maps, and the production function's axes and finals at the controls. It then
reports reference-near axis coverage, fragmentation, additional axis structure and the
class of every mask-off vertex against the production mask. It stops at contour
extraction, and axes are never matched between the two runs.

    python3 scripts/pilot_eastern_mask_axis.py --year 2006 --maxima <record> --reference <declaration> --trace <trace.json.gz>:<digest12> \\
        --control-run <dir> --control-case <mat> --wide-dir <dir> --calibration <json> --out <json>
"""
import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_alledge_replay as A  # noqa: E402
import pilot_eastern_maxima as MX  # noqa: E402
import pilot_eastern_measurement as EM  # noqa: E402

TOLS = (3.0, 5.0)
CLASSES = ("location masked", "neighboring support", "unaffected")


def axes_from_prepared(fields, ct, grids, westerly_mask=True):
    """The axes of `detect_from_prepared`, line for line up to the contour extraction, with
    the westerly mask optional. Returns the axes and the production westerly mask."""
    from aew.v1port.detection import MAX_ZONAL_WIND, trough_axes
    wind, curvature_c, advection_c = fields["coarse_zonal_wind_smoothed"], fields["coarse_curvature"], fields["coarse_advection"]
    westerly = wind > MAX_ZONAL_WIND
    if westerly_mask:
        advection_c = np.where(westerly, np.nan, advection_c)
        curvature_c = np.where(westerly, np.nan, curvature_c)
    weak_c = curvature_c < ct
    advection_c = np.where(weak_c, np.nan, advection_c)
    return trough_axes(grids["latgrid_c"], grids["longrid_c"], advection_c), westerly


def vertex_class(lat, lon, lat_c, lon_c, masked):
    """The class of one vertex against a mask on the grid: the quad holding it, then whether
    its nearest corner, or only another corner, is masked."""
    lat_c, lon_c = np.asarray(lat_c, float), np.asarray(lon_c, float)
    def bracket(vals, x):
        order = np.argsort(vals)
        s = vals[order]
        k = int(np.clip(np.searchsorted(s, x, side="right") - 1, 0, s.size - 2))
        return int(order[k]), int(order[k + 1])
    i0, i1 = bracket(lat_c, lat)
    j0, j1 = bracket(lon_c, lon)
    corners = [(i, j) for i in (i0, i1) for j in (j0, j1)]
    near = min(corners, key=lambda c: (lat_c[c[0]] - lat) ** 2 + (lon_c[c[1]] - lon) ** 2)
    if masked[near]:
        return "location masked"
    return "neighboring support" if any(masked[c] for c in corners) else "unaffected"


def length_deg(la, lo):
    return float(np.sum(np.hypot(np.diff(la), np.diff(lo))))


def in_box(la, lo, box):
    return (la >= box["lat"][0]) & (la <= box["lat"][1]) & (lo >= box["lon"][0]) & (lo <= box["lon"][1])


def describe_axes(axes, m, box, lat_c, lon_c, masked, classify):
    """Per tolerance, the near axes (a vertex within it) and the other axes with a vertex in
    the box, with counts, vertices, lengths and, when asked, the vertex classes."""
    out = {}
    for t in TOLS:
        near, away = [], []
        for la, lo in axes:
            la, lo = np.asarray(la, float), np.asarray(lo, float)
            d = np.array([EM.ref_distance(a, b, m) for a, b in zip(la, lo)]) if m is not None else None
            if d is not None and (d <= t).any():
                near.append((la, lo))
            elif in_box(la, lo, box).any():
                away.append((la, lo))
        e = {"near": len(near), "near_vertices": int(sum(a[0].size for a in near)), "near_length_deg": round(sum(length_deg(*a) for a in near), 3),
             "away": len(away), "away_vertices": int(sum(a[0].size for a in away))}
        if classify:
            for name, group in (("near", near), ("away", away)):
                c = {k: 0 for k in CLASSES}
                for la, lo in group:
                    for a, b in zip(la, lo):
                        c[vertex_class(a, b, lat_c, lon_c, masked)] += 1
                e[f"{name}_vertex_classes"] = c
        out[str(t)] = e
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--maxima", "--control-run", "--control-case", "--wide-dir", "--calibration", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--reference"); ap.add_argument("--trace", action="append", default=[]); ap.add_argument("--baseline")
    args = ap.parse_args(argv)
    t0 = time.time()
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    mrec = json.load(open(args.maxima))
    run = A.P.load_run(args.control_run, year=args.year)
    case_b = A.SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, args.year)
    if wide["record"]["control_run"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the wide case was not bound to this control case")
    ds = run["record"]["dataset_specific"]
    ct, ft, _ = A.thresholds_for_run((float(ds["coarse_threshold"]), float(ds["fine_threshold"])), args.calibration, False, True, run["domain"])
    if ct != float(mrec["coarse_threshold"]):
        raise SystemExit("REFUSED: the run's coarse threshold is not the bound record's")
    absorb = bool(ds["tracker_flags"]["absorb"])
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    lat_c, lon_c = np.asarray(grids["lat_c"], float).ravel(), np.asarray(grids["lon_c"], float).ravel()
    times = np.asarray(case_b["time"], float).ravel()
    retained_axes, finals = {}, {}
    for spec in args.trace:
        path, dg = spec.rsplit(":", 1)
        finals.update(MX.finals_from_trace(path, dg))
        with gzip.open(path, "rb") as fh:
            for s in json.loads(fh.read())["trace"]["steps"]:
                if s.get("lineage"):
                    retained_axes[s["date"][:13]] = s["lineage"]["axes"]
    if args.baseline:
        finals.update(MX.finals_from_baseline(args.baseline))
    ref = maps_by = None
    if args.reference:
        ref = json.load(open(args.reference))["reference"]
        maps_by = {(m["date"], m["segment"]): m for m in ref["maps"] if m["scored"]}
    items = [("scored", r["date"], r["segment"]) for r in mrec["scored_maps"]] + [("control", c["date"], c["control"]) for c in mrec["controls"]]
    steps = MX.map_steps(times, sorted({d for _, d, _ in items}))
    gates = {"scored_axes_equal_retained": 0, "control_axes_equal_production": 0, "control_finals_reproduced": 0}
    rows = []
    for kind, d, name in items:
        k = steps[d]
        fields = A.prepared_fields(wide, k, *sel)
        on, westerly = axes_from_prepared(fields, ct, grids, True)
        off, _ = axes_from_prepared(fields, ct, grids, False)
        if kind == "scored":
            want = retained_axes.get(d)
            if want is None or len(want) != len(on) or any(list(map(float, la)) != w["lat"] or list(map(float, lo)) != w["lon"] for (la, lo), w in zip(on, want)):
                raise SystemExit(f"STOPPED: at {d} the mask-on axes do not reproduce the retained axes")
            gates["scored_axes_equal_retained"] += 1
            m, box = maps_by[(d, name)], ref["segments"][name]["case_box"]
        else:
            lin = {}
            waves = A.detect_from_prepared(fields, float(times[k]), ct, ft, grids, absorb=absorb, lineage=lin)
            if len(lin["axes"]) != len(on) or any(not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])) for a, b in zip(lin["axes"], on)):
                raise SystemExit(f"STOPPED: at {d} the mask-on axes are not the production function's")
            gates["control_axes_equal_production"] += 1
            mine = [(float(w["lat_mean"]), float(w["lon_mean"]), A.R.region_digest(w["region"])) for w in waves]
            if d not in finals or not MX.same_finals(mine, finals[d]):
                raise SystemExit(f"STOPPED: at {d} the production finals do not reproduce the retained finals")
            gates["control_finals_reproduced"] += 1
            m, box = None, next(c["box"] for c in mrec["controls"] if c["date"] == d and c["control"] == name)
        rows.append({"kind": kind, "date": d, "group": name, "westerly_cells_in_box": int((westerly & in_box(np.asarray(grids["latgrid_c"]), np.asarray(grids["longrid_c"]), box)).sum()),
                     "box_cells": int(in_box(np.asarray(grids["latgrid_c"]), np.asarray(grids["longrid_c"]), box).sum()),
                     "mask_on": describe_axes(on, m, box, lat_c, lon_c, westerly, False), "mask_off": describe_axes(off, m, box, lat_c, lon_c, westerly, True)})
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    summary = {}
    for g, rs in groups.items():
        summary[g] = {"maps": len(rs), "westerly_share_of_box_cells": round(sum(r["westerly_cells_in_box"] for r in rs) / max(1, sum(r["box_cells"] for r in rs)), 3)}
        for t in map(str, TOLS):
            s = {}
            for run_name in ("mask_on", "mask_off"):
                es = [r[run_name][t] for r in rs]
                s[run_name] = {"covered": sum(1 for e in es if e["near"] > 0), "near_axes": sum(e["near"] for e in es), "near_vertices": sum(e["near_vertices"] for e in es),
                               "near_length_deg": round(sum(e["near_length_deg"] for e in es), 2), "away_axes": sum(e["away"] for e in es),
                               "maps_with_several_near_axes": sum(1 for e in es if e["near"] > 1)}
            for side in ("near", "away"):
                s[f"mask_off_{side}_vertex_classes"] = {c: sum(r["mask_off"][t][f"{side}_vertex_classes"][c] for r in rs) for c in CLASSES}
            summary[g][t] = s
    out = {"generated_by": "scripts/pilot_eastern_mask_axis.py", "script_sha256": A.X.digest(os.path.abspath(__file__)), "year": args.year,
           "coarse_threshold": ct, "gates": gates, "summary": summary, "maps": rows, "seconds": round(time.time() - t0, 1),
           "limits": "stops at contour extraction; axes are not matched between mask on and mask off; counts are net; no identity, region, merge or history claim"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"gates": gates, "seconds": out["seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
