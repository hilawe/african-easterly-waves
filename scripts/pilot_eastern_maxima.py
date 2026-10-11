#!/usr/bin/env python3
"""The maxima comparison of `EASTERN_MAXIMA_COMPARISON_2026-10-10.md`, one season per run.

At each declared map it recomputes the frozen candidate's prepared fields with the
replay's own preparation, reruns the detector and refuses unless it reproduces the
retained final candidates in position and region digest. It then finds local maxima in
(A) the prepared coarse curvature anomaly with the detector's westerly mask and (B) 700
hPa relative vorticity from the wide case's winds with the field-sequence tool's terrain
and stencil rule, and measures F (the final candidates), A and B against the read
geometry at the scored maps, inside the shared regions the baseline implicated, and in
the standing-control boxes. Every choice is the declaration's. Nothing is tuned.

    python3 scripts/pilot_eastern_maxima.py --declaration <maxima json> --year 2006 --control-run <dir> --control-case <mat> \\
        --wide-dir <dir> --calibration <audit> --sp-file <era5 sp> --finals-trace <trace.json.gz>:<digest12> \\
        [--finals-baseline <json.gz>] [--reference <2006 declaration> --measurement <baseline record> --lineage-reading <record>] --out <json>
"""
import argparse
import datetime as dt
import gzip
import hashlib
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
VORT_SUBGRID = {"lat": [-10.0, 35.0], "lon": [5.0, 64.0]}     # wider than the declared domain, inside the surface pressure's coverage


def local_maxima(field, valid, threshold, latgrid, longrid):
    """Local maxima by the declared rule: above the threshold, at least each of the eight
    neighbors, all eight valid; an eight-connected plateau is one maximum at its mean
    position. Returns the maxima and the count of peak-shaped cells excluded because a
    neighbor is invalid."""
    from scipy import ndimage
    f = np.asarray(field, float)
    ok = np.asarray(valid, bool) & np.isfinite(f)
    ny, nx = f.shape
    pad = np.full((ny + 2, nx + 2), np.nan)
    pad[1:-1, 1:-1] = np.where(ok, f, np.nan)
    neigh = [pad[1 + di:ny + 1 + di, 1 + dj:nx + 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    all_valid = np.all([np.isfinite(n) for n in neigh], axis=0)
    ge = np.all([~np.isfinite(n) | (np.where(ok, f, -np.inf) >= n) for n in neigh], axis=0)
    shape = ok & (f > threshold) & ge
    peaks = shape & all_valid
    labels, n = ndimage.label(peaks, structure=np.ones((3, 3), int))
    lat, lon = np.asarray(latgrid, float), np.asarray(longrid, float)
    out = []
    for k in range(1, n + 1):
        m = labels == k
        out.append({"lat": float(lat[m].mean()), "lon": float(lon[m].mean()), "value": float(f[m].max()), "cells": int(m.sum()),
                    "cells_ij": [[int(i), int(j)] for i, j in zip(*np.nonzero(m))]})
    return out, int((shape & ~all_valid).sum())


def in_domain(p, dom):
    return dom["lat"][0] <= p["lat"] <= dom["lat"][1] and dom["lon"][0] <= p["lon"] <= dom["lon"][1]


def in_box(lat, lon, box):
    return box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1]


def correspondence(points, m, box, tols=(3.0, 5.0)):
    """Nearest distance to the read geometry, counts within each tolerance and in the case box."""
    d = [EM.ref_distance(p["lat"], p["lon"], m) for p in points]
    return {"nearest_deg": round(min(d), 3) if d else None, "within": {str(t): sum(1 for x in d if x <= t) for t in tols},
            "in_case_box": sum(1 for p in points if in_box(p["lat"], p["lon"], box))}


def compare(rep, fin):
    """NEARER, and NEARER WITHOUT ADDED DENSITY, of a representation against the finals."""
    if rep["nearest_deg"] is None:
        return {"nearer": False, "nearer_without_added_density": False}
    nearer = fin["nearest_deg"] is None or rep["nearest_deg"] < fin["nearest_deg"]
    return {"nearer": bool(nearer), "nearer_without_added_density": bool(nearer and rep["in_case_box"] <= fin["in_case_box"])}


def coarse_cell_of(lat, lon, lat_c, lon_c):
    """The nearest coarse node to a position, as row and column indices."""
    return int(np.argmin(np.abs(lat_c - lat))), int(np.argmin(np.abs(lon_c - lon)))


def region_maxima(mask, a_max, b_max, lat_c, lon_c, m, surviving, tols=(3.0, 5.0)):
    """The maxima of each representation inside one shared region, with their distances to
    the read geometry and to the surviving coarse output, and whether it separates."""
    out = {}
    for name, pts in (("A", a_max), ("B", b_max)):
        inside = []
        for p in pts:
            if name == "A":
                hit = any(mask[i, j] for i, j in p["cells_ij"])
            else:
                i, j = coarse_cell_of(p["lat"], p["lon"], lat_c, lon_c)
                hit = bool(mask[i, j])
            if hit:
                inside.append({"lat": p["lat"], "lon": p["lon"], "value": p["value"], "distance_to_read_deg": round(EM.ref_distance(p["lat"], p["lon"], m), 3),
                               "distance_to_surviving_output_deg": None if surviving is None else round(math.hypot(p["lat"] - surviving[0], p["lon"] - surviving[1]), 3)})
        out[name] = {"maxima_inside": inside, "count": len(inside),
                     "separates": {str(t): bool(len(inside) > 1 and any(x["distance_to_read_deg"] <= t for x in inside)) for t in tols}}
    return out


def same_finals(mine, retained, tol=1e-6):
    """The recomputed final candidates equal the retained ones: same count and order,
    identical region digests, positions within `tol` degrees."""
    return len(mine) == len(retained) and all(a[2] == b[2] and abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol for a, b in zip(mine, retained))


def finals_from_trace(path, digest12):
    with gzip.open(path, "rb") as fh:
        blob = fh.read()
    if not hashlib.sha256(blob).hexdigest().startswith(digest12):
        raise SystemExit(f"REFUSED: {path} does not carry the digest {digest12}")
    t = json.loads(blob)["trace"]
    return {e["date"][:13]: [(float(c["lat_mean"]), float(c["lon_mean"]), c["region_sha256"]) for c in e["candidates"]] for e in t["steps"]}


def finals_from_baseline(path):
    with gzip.open(path, "rb") as fh:
        b = json.loads(fh.read())
    out = {}
    for s in b["steps"].values():
        d = dt.datetime.strptime(s["date"], "%Y-%m-%d %HZ").strftime("%Y-%m-%dT%H")
        out[d] = [(float(f[0]), float(f[1]), f[3]) for f in s["final"]]
    return out


def map_steps(times, dates):
    out = {}
    for d in dates:
        day = (dt.datetime.strptime(d[:13], "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0
        hit = [k for k in range(times.size) if abs(times[k] - day) < 1.0 / 1440.0]
        if len(hit) != 1:
            raise SystemExit(f"REFUSED: the map {d} is not exactly one step of the case")
        out[d] = hit[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--declaration", "--control-run", "--control-case", "--wide-dir", "--calibration", "--sp-file", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--finals-trace", action="append", default=[], help="a trace and the first twelve characters of its logged digest, as path:digest")
    ap.add_argument("--finals-baseline")
    ap.add_argument("--reference"); ap.add_argument("--measurement"); ap.add_argument("--lineage-reading")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    decl = json.load(open(args.declaration))
    dom, thr = decl["domain"], decl["thresholds"]
    finals = {}
    for spec in args.finals_trace:
        path, dg = spec.rsplit(":", 1)
        finals.update(finals_from_trace(path, dg))
    if args.finals_baseline:
        finals.update(finals_from_baseline(args.finals_baseline))
    scored, ref = [], None
    if args.reference:
        if hashlib.sha256(open(args.reference, "rb").read()).hexdigest() != decl["scored_maps_from"]["sha256"]:
            raise SystemExit("REFUSED: the reference is not the declared one")
        ref = json.load(open(args.reference))["reference"]
        scored = [m for m in ref["maps"] if m["scored"]]
    controls = [c for c in decl["controls"] if c["year"] == args.year]
    control_dates = {c["name"]: M_dates(c["window"]) for c in controls}
    dates = sorted({m["date"] for m in scored} | {d for ds in control_dates.values() for d in ds})
    missing = [d for d in dates if d not in finals]
    if missing:
        raise SystemExit(f"REFUSED: no retained finals for {missing}")
    run = A.P.load_run(args.control_run, year=args.year)
    case_b = A.SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, args.year)
    if wide["record"]["control_run"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the wide case was not bound to this control case")
    ds = run["record"]["dataset_specific"]
    ct, ft, _thr = A.thresholds_for_run((float(ds["coarse_threshold"]), float(ds["fine_threshold"])), args.calibration, False, True, run["domain"])
    if ct != float(thr["A"]):
        raise SystemExit(f"REFUSED: the run's coarse threshold {ct} is not the declared {thr['A']}")
    absorb = bool(ds["tracker_flags"]["absorb"])
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    lat_c, lon_c = np.asarray(grids["latgrid_c"], float)[:, 0], np.asarray(grids["longrid_c"], float)[0, :]
    times = np.asarray(case_b["time"], float).ravel()
    steps = map_steps(times, dates)
    wlat, wlon = np.asarray(wide["latgrid"], float), np.asarray(wide["longrid"], float)
    rows = np.flatnonzero((wlat[:, 0] >= VORT_SUBGRID["lat"][0]) & (wlat[:, 0] <= VORT_SUBGRID["lat"][1]))
    cols = np.flatnonzero((wlon[0, :] >= VORT_SUBGRID["lon"][0]) & (wlon[0, :] <= VORT_SUBGRID["lon"][1]))
    step_days = [float(times[steps[d]]) for d in dates]
    sp_w = FS.surface_pressure_on(args.sp_file, wlat[rows, 0], wlon[0, cols], step_days)
    crow = np.flatnonzero((lat_c >= -25.0) & (lat_c <= 35.0)); ccol = np.flatnonzero((lon_c >= -45.0) & (lon_c <= 75.0))
    sp_c = FS.surface_pressure_on(args.sp_file, lat_c[crow], lon_c[ccol], step_days)
    from aew.v1port.detection import MAX_ZONAL_WIND
    per_date = {}
    for n, d in enumerate(dates):
        k = steps[d]
        fields = A.prepared_fields(wide, k, *sel)
        lin = {}
        waves = A.detect_from_prepared(fields, float(times[k]), ct, ft, grids, absorb=absorb, lineage=lin)
        mine = [(float(w["lat_mean"]), float(w["lon_mean"]), A.R.region_digest(w["region"])) for w in waves]
        if not same_finals(mine, finals[d]):
            raise SystemExit(f"REFUSED: at {d} the recomputed detector does not reproduce the retained finals")
        curv = np.where(fields["coarse_zonal_wind_smoothed"] > MAX_ZONAL_WIND, np.nan, fields["coarse_curvature"])
        a_all, a_edge = local_maxima(curv, np.isfinite(curv), float(thr["A"]), grids["latgrid_c"], grids["longrid_c"])
        a_max = [p for p in a_all if in_domain(p, dom)]
        for p in a_max:
            i, j = p["cells_ij"][0]
            ii, jj = np.flatnonzero(crow == i), np.flatnonzero(ccol == j)
            p["below_700hPa_ground"] = None if not (ii.size and jj.size) else bool(not sp_c[n][ii[0], jj[0]] > 70000.0)
        pa = FS.panel_arrays(wide, k, rows, cols, "zeta", sp_w[n])
        b_all, b_edge = local_maxima(pa["shaded"], pa["classes"] == "valid", float(thr["B_per_second"]) * 1e5, wlat[rows, :][:, cols], wlon[rows, :][:, cols])
        b_max = [p for p in b_all if in_domain(p, dom)]
        per_date[d] = {"step": k, "finals": [{"lat": a, "lon": b} for a, b, _ in mine], "A": a_max, "B": b_max, "lineage": lin,
                       "excluded_next_to_invalid": {"A": a_edge, "B": b_edge}}
    tols = tuple(decl["tolerances_deg"])
    scored_rows = []
    meas = json.load(open(args.measurement)) if args.measurement else None
    for m in scored:
        box = ref["segments"][m["segment"]]["case_box"]
        r = per_date[m["date"]]
        f, a, b = (correspondence(r[key], m, box, tols) for key in ("finals", "A", "B"))
        if meas is not None:
            want = meas["summary"]["tolerance"]["segments"][m["segment"]]["nearest_candidate_distance_by_map"][m["date"]]
            if want != f["nearest_deg"]:
                raise SystemExit(f"REFUSED: at {m['date']} the finals' nearest distance {f['nearest_deg']} is not the baseline's {want}")
        scored_rows.append({"date": m["date"], "segment": m["segment"], "west_of_40E": EM.west_of_40(m), "F": f, "A": {**a, **compare(a, f)}, "B": {**b, **compare(b, f)},
                            "A_below_700hPa_ground_in_box": sum(1 for p in r["A"] if in_box(p["lat"], p["lon"], box) and p.get("below_700hPa_ground"))})
    regions = []
    if args.lineage_reading:
        lr = json.load(open(args.lineage_reading))
        maps_by = {(m["date"], m["segment"]): m for m in scored}
        for name in ("tolerance", "alternative"):
            for row in lr["readings"][name]["uncovered_scored_maps"]:
                m = maps_by[(row["date"], row["segment"])]
                lin = per_date[row["date"]]["lineage"]
                coarse = {e["input"]: e for e in lin["coarse"]}
                seen = set()
                for ax in row["relevant_axes"]:
                    near = [x for x in (ax.get("dropped_near") or []) if x["arc_deg"] == 0.0]
                    if not near:
                        continue
                    want = next(p["region_sha256"] for p in ax["positions"] if p["stage"] == "coarse region")
                    mask = np.asarray(coarse[ax["axis"]]["select"]["region"], bool)
                    if A.R.region_digest(mask) != want:
                        raise SystemExit(f"REFUSED: at {row['date']} the recomputed region of axis {ax['axis']} is not the recorded one")
                    if want in seen:
                        continue
                    seen.add(want)
                    regions.append({"reading": name, "date": row["date"], "segment": row["segment"], "region_sha256": want, "cells": int(mask.sum()),
                                    "axes_sharing": [a2["axis"] for a2 in row["relevant_axes"] if any(p.get("region_sha256") == want for p in a2["positions"])],
                                    "surviving_output": near[0]["output_position"],
                                    **region_maxima(mask, per_date[row["date"]]["A"], per_date[row["date"]]["B"], lat_c, lon_c, m, near[0]["output_position"], tols)})
    control_rows = []
    for c in controls:
        for d in control_dates[c["name"]]:
            r = per_date[d]
            control_rows.append({"control": c["name"], "date": d, "box": c["box"],
                                 "counts_in_box": {key: sum(1 for p in r[key] if in_box(p["lat"], p["lon"], c["box"])) for key in ("finals", "A", "B")}})
    summary = {}
    if scored_rows:
        unc = [x for x in scored_rows if not x["F"]["within"]["3.0"]]
        summary["scored"] = {
            "maps": len(scored_rows),
            "maps_with_a_point_within": {key: {t: sum(1 for x in scored_rows if x[key]["within"][t]) for t in ("3.0", "5.0")} for key in ("F", "A", "B")},
            "points_in_case_boxes_total": {key: sum(x[key]["in_case_box"] for x in scored_rows) for key in ("F", "A", "B")},
            "uncovered_maps_at_3": len(unc),
            "at_uncovered_maps_a_point_within_3": {key: sum(1 for x in unc if x[key]["within"]["3.0"]) for key in ("A", "B")},
            "nearer": {key: sum(1 for x in scored_rows if x[key]["nearer"]) for key in ("A", "B")},
            "nearer_without_added_density": {key: sum(1 for x in scored_rows if x[key]["nearer_without_added_density"]) for key in ("A", "B")},
            "nearer_without_added_density_at_uncovered": {key: sum(1 for x in unc if x[key]["nearer_without_added_density"]) for key in ("A", "B")}}
    if regions:
        for name in ("tolerance", "alternative"):
            rs = [g for g in regions if g["reading"] == name]
            t = "3.0" if name == "tolerance" else "5.0"
            summary[f"shared_regions_{name}"] = {"regions": len(rs), **{f"separates_{key}": sum(1 for g in rs if g[key]["separates"][t]) for key in ("A", "B")},
                                                  **{f"maxima_inside_{key}": [g[key]["count"] for g in rs] for key in ("A", "B")}}
    if control_rows:
        summary["controls"] = {c["name"]: {key: sum(x["counts_in_box"][key] for x in control_rows if x["control"] == c["name"]) for key in ("finals", "A", "B")}
                               for c in controls}
        summary["controls_maps"] = {c["name"]: len(control_dates[c["name"]]) for c in controls}
    for r in per_date.values():
        r.pop("lineage")
        for key in ("A", "B"):
            for p in r[key]:
                p.pop("cells_ij", None)
    out = {"generated_by": "scripts/pilot_eastern_maxima.py", "script_sha256": A.X.digest(os.path.abspath(__file__)), "year": args.year,
           "declaration": {"path": args.declaration, "sha256": A.X.digest(args.declaration)}, "coarse_threshold": ct,
           "inputs": {"finals_trace": args.finals_trace, "finals_baseline": args.finals_baseline, "reference": args.reference, "measurement": args.measurement,
                      "lineage_reading": args.lineage_reading, "sp_file": args.sp_file},
           "finals_reproduced_at_every_map": True, "summary": summary, "scored_maps": scored_rows, "shared_regions": regions, "controls": control_rows,
           "maxima_by_date": per_date,
           "limits": "a field-representation comparison at declared maps; a nearer point alone is not improvement; standing-control maxima are additional candidate structure, not duplicates or false detections"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"Record {args.out}.")
    return 0


def M_dates(window):
    return EM.M.inspected_dates(window)


if __name__ == "__main__":
    raise SystemExit(main())
