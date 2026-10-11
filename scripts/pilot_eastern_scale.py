#!/usr/bin/env python3
"""The spatial-scale comparison of `EASTERN_SCALE_COMPARISON_2026-10-10.md`.

B is 700 hPa relative vorticity on the wide case's 1-degree grid with the field-sequence
tool's terrain and stencil rule, and C is B after a 600 km radial mean, an experimental
transfer of QTrack's spatial scale, not a QTrack reproduction. The footprint follows the
installed QTrack package's distance rule (version 0.0.4, `core.py` `rad_mask`), and a
center is computed only when its whole footprint lies inside the subgrid, checked center
by center, in place of the package's fixed 6-row and 11-column loop boundary. Uncomputed
values are missing, never zero. C is COMPLETE where every footprint cell is valid in B,
and a PARTIAL mean over valid cells is kept apart.

    python3 scripts/pilot_eastern_scale.py --declaration <scale json> --year 2006 --wide-dir <dir> --sp-file <era5 sp> \\
        --reference <2006 declaration> --support-only --out <support.json>
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

LAT_MET = 6371 * 2 * np.pi * 1000 / 360          # meters per degree, the package's constant
EPOCH = dt.datetime(1900, 1, 1)


def footprint_offsets(lat_center, dlat, dlon, radius_km):
    """Row and column offsets of the cells closer than `radius_km` to a center at
    `lat_center`, by the package's rule: k rows away is k * |dlat| * LAT_MET, and m
    columns away in a row at latitude lat_r is m * |dlon| * cos(lat_r) * LAT_MET. `dlat`
    carries the grid's row direction (negative for rows running south)."""
    out = []
    kmax = int(math.ceil(radius_km * 1000.0 / (abs(dlat) * LAT_MET))) + 1
    for k in range(-kmax, kmax + 1):
        lat_r = lat_center + k * dlat
        dy = abs(k) * abs(dlat) * LAT_MET
        dx1 = abs(dlon) * math.cos(math.radians(lat_r)) * LAT_MET
        mmax = int(math.ceil(radius_km * 1000.0 / dx1)) + 1 if dx1 > 0 else 0
        for m in range(-mmax, mmax + 1):
            if math.sqrt(dy * dy + (abs(m) * dx1) ** 2) / 1000.0 < radius_km:
                out.append((k, m))
    return out


def radial_mean(field, valid, lat1, lon1, radius_km):
    """C from B: per center, whether the whole footprint is inside the grid (`inside`),
    the count of valid footprint cells and the footprint size, the COMPLETE mean (every
    footprint cell inside and valid, else missing) and the PARTIAL mean over valid cells
    (inside the grid, at least one valid cell, else missing)."""
    f = np.asarray(field, float)
    ok = np.asarray(valid, bool) & np.isfinite(f)
    ny, nx = f.shape
    lat1 = np.asarray(lat1, float); lon1 = np.asarray(lon1, float)
    dlat, dlon = float(lat1[1] - lat1[0]), float(lon1[1] - lon1[0])
    inside = np.zeros((ny, nx), bool)
    nvalid = np.zeros((ny, nx), int)
    size = np.zeros((ny, nx), int)
    total = np.zeros((ny, nx), float)
    fv = np.where(ok, f, 0.0)
    for i in range(ny):
        offs = footprint_offsets(float(lat1[i]), dlat, dlon, radius_km)
        ks = [k for k, _ in offs]; ms = [m for _, m in offs]
        if min(ks) + i < 0 or max(ks) + i > ny - 1:
            continue
        jlo, jhi = -min(ms), nx - 1 - max(ms)
        if jlo > jhi:
            continue
        cols = np.arange(jlo, jhi + 1)
        inside[i, cols] = True
        size[i, cols] = len(offs)
        for k, m in offs:
            nvalid[i, cols] += ok[i + k, cols + m]
            total[i, cols] += fv[i + k, cols + m]
    complete = inside & (nvalid == size) & (size > 0)
    partial_ok = inside & (nvalid > 0) & ~complete
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(nvalid > 0, total / np.maximum(nvalid, 1), np.nan)
    return {"inside": inside, "complete": complete, "valid_fraction": np.where(inside & (size > 0), nvalid / np.maximum(size, 1), np.nan),
            "complete_mean": np.where(complete, mean, np.nan), "partial_mean": np.where(partial_ok, mean, np.nan)}


def peak_support(common):
    """Cells whose center and eight neighbors are all in the common support."""
    c = np.asarray(common, bool)
    ny, nx = c.shape
    pad = np.zeros((ny + 2, nx + 2), bool)
    pad[1:-1, 1:-1] = c
    ok = c.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ok &= pad[1 + di:ny + 1 + di, 1 + dj:nx + 1 + dj]
    return ok


def neighborhood(m, tol, lat_all, lon_all):
    """Every 1-degree grid point (on the wide grid's own coordinates) within `tol` of the
    read geometry."""
    pts = []
    for la in lat_all:
        for lo in lon_all:
            if EM.ref_distance(la, lo, m) <= tol:
                pts.append((float(la), float(lo)))
    return pts


def support_share(points, ok, lat1, lon1):
    """The share of the points that are peak-supported cells of the subgrid; points outside
    the subgrid count as unsupported."""
    idx_lat = {round(float(x), 6): i for i, x in enumerate(lat1)}
    idx_lon = {round(float(x), 6): j for j, x in enumerate(lon1)}
    hit = 0
    for la, lo in points:
        i, j = idx_lat.get(round(la, 6)), idx_lon.get(round(lo, 6))
        if i is not None and j is not None and ok[i, j]:
            hit += 1
    return hit, len(points)


def map_steps(times, dates):
    out = {}
    for d in dates:
        day = (dt.datetime.strptime(d[:13], "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0
        hit = [k for k in range(times.size) if abs(times[k] - day) < 1.0 / 1440.0]
        if len(hit) != 1:
            raise SystemExit(f"REFUSED: the map {d} is not exactly one step of the case")
        out[d] = hit[0]
    return out


def fields_at(wide, k, rows, cols, sp_k, radius_km):
    pa = FS.panel_arrays(wide, k, rows, cols, "zeta", sp_k)
    lat1 = np.asarray(wide["latgrid"], float)[rows, 0]; lon1 = np.asarray(wide["longrid"], float)[0, cols]
    valid_b = pa["classes"] == "valid"
    c = radial_mean(pa["shaded"], valid_b, lat1, lon1, radius_km)
    common = valid_b & c["complete"]
    return {"B": pa["shaded"], "valid_B": valid_b, "C": c, "common": common, "peak_ok": peak_support(common), "lat1": lat1, "lon1": lon1}


def load_inputs(args, sub, dates, steps, times, wide):
    """The winds on the declared subgrid from the original one-degree files, the surface
    pressure from the region file and, north of it, the bounded northern file. Each is
    read at exact coordinates and times. The winds must equal the wide case on their
    overlap, and the northern pressure must equal the region file on theirs, or the run
    refuses."""
    lat1 = np.arange(sub["lat"][1], sub["lat"][0] - 0.5, -1.0)          # rows running south, as the wide case
    lon1 = np.arange(sub["lon"][0], sub["lon"][1] + 0.5, 1.0)
    days = [float(times[steps[d]]) for d in dates]
    U = FS.field_on(os.path.join(args.winds_dir, f"era5_u700_{args.year}_6h_region.nc"), "u", "m s**-1", lat1, lon1, days, 700.0)
    V = FS.field_on(os.path.join(args.winds_dir, f"era5_v700_{args.year}_6h_region.nc"), "v", "m s**-1", lat1, lon1, days, 700.0)
    wlat, wlon = np.asarray(wide["latgrid"], float)[:, 0], np.asarray(wide["longrid"], float)[0, :]
    ri = [i for i, x in enumerate(lat1) if np.any(np.isclose(wlat, x))]; ci = [j for j, x in enumerate(lon1) if np.any(np.isclose(wlon, x))]
    wr = [int(np.flatnonzero(np.isclose(wlat, lat1[i]))[0]) for i in ri]; wc = [int(np.flatnonzero(np.isclose(wlon, lon1[j]))[0]) for j in ci]
    worst = 0.0
    for n, d in enumerate(dates):
        for name, arr in (("u", U), ("v", V)):
            mine = arr[n][np.ix_(ri, ci)]
            theirs = np.asarray(wide[name][steps[d]], float)[np.ix_(wr, wc)]
            if not np.array_equal(np.isnan(mine), np.isnan(theirs)):
                raise SystemExit(f"REFUSED: at {d} the original {name} winds and the wide case disagree on missing cells")
            worst = max(worst, float(np.nanmax(np.abs(mine - theirs))))
    if worst != 0.0:
        raise SystemExit(f"REFUSED: the original winds differ from the wide case on their overlap by up to {worst} m/s")
    south = lat1[lat1 <= 35.0]
    SP = np.full((len(dates), lat1.size, lon1.size), np.nan)
    SP[:, lat1 <= 35.0, :] = FS.surface_pressure_on(args.sp_file, south, lon1, days)
    north_used, overlap_worst = [], None
    if args.sp_north:
        import netCDF4
        nd = netCDF4.Dataset(args.sp_north)
        ntimes = np.asarray(nd.variables["valid_time"][:], float) / 86400.0 + 25567.0      # days since 1900
        nd.close()
        nrows = lat1[(lat1 > 35.0)]
        orows = lat1[(lat1 >= 33.0) & (lat1 <= 35.0)]
        for n, d in enumerate(dates):
            if not np.any(np.abs(ntimes - days[n]) < 1.0 / 1440.0):
                continue
            SP[n, lat1 > 35.0, :] = FS.surface_pressure_on(args.sp_north, nrows, lon1, [days[n]])[0]
            ov = FS.surface_pressure_on(args.sp_north, orows, lon1, [days[n]])[0]
            ref = SP[n][(lat1 >= 33.0) & (lat1 <= 35.0), :]
            diff = float(np.nanmax(np.abs(ov - ref)))
            if diff != 0.0 or not np.array_equal(np.isnan(ov), np.isnan(ref)):
                raise SystemExit(f"REFUSED: at {d} the northern surface pressure differs from the region file on their overlap by {diff} Pa")
            overlap_worst = diff if overlap_worst is None else max(overlap_worst, diff)
            north_used.append(d)
    LAT, LON = np.meshgrid(lat1, lon1, indexing="ij")
    grid = {"u": U, "v": V, "latgrid": LAT, "longrid": LON}
    check = {"winds_equal_wide_case_on_overlap": True, "overlap_cells_per_map": len(ri) * len(ci), "winds_max_abs_difference": worst,
             "northern_pressure_dates": north_used, "northern_pressure_overlap_max_abs_difference_pa": overlap_worst}
    return grid, lat1, lon1, SP, check


def outcome(peaks, m, t, assessable):
    """Hit, miss or not assessable at tolerance t, with the nearest peak."""
    d = [EM.ref_distance(p["lat"], p["lon"], m) for p in peaks]
    j = int(np.argmin(d)) if d else None
    near = None if j is None else {"lat": peaks[j]["lat"], "lon": peaks[j]["lon"], "strength_1e5_per_s": round(peaks[j]["value"], 3), "distance_deg": round(d[j], 3)}
    if d and min(d) <= t:
        return "hit", near
    return ("miss" if assessable else "not assessable"), near


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--declaration", "--wide-dir", "--winds-dir", "--sp-file", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--reference")
    ap.add_argument("--sp-north")
    ap.add_argument("--support-only", action="store_true")
    ap.add_argument("--finals-trace", action="append", default=[], help="path:digest12, the final candidates as context")
    ap.add_argument("--finals-baseline")
    ap.add_argument("--control-run"); ap.add_argument("--control-case"); ap.add_argument("--calibration"); ap.add_argument("--lineage-reading")
    ap.add_argument("--support-record", help="the passed support check this peak stage follows")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    decl = json.load(open(args.declaration))
    radius = float(decl["radius_km"])
    sub = decl["subgrid"]
    if not args.support_only:
        if not args.support_record:
            raise SystemExit("REFUSED: the peak stage needs the passed support record")
        srec = json.load(open(args.support_record))
        if srec.get("declaration", {}).get("sha256") != A.X.digest(args.declaration) or (srec.get("sufficient") is False):
            raise SystemExit("REFUSED: the support record is not this declaration's, or support was insufficient")
    scored, ref = [], None
    if args.reference:
        if A.X.digest(args.reference) != decl["scored_maps_from"]["sha256"]:
            raise SystemExit("REFUSED: the reference is not the declared one")
        ref = json.load(open(args.reference))["reference"]
        scored = [m for m in ref["maps"] if m["scored"]]
    controls = [c for c in decl["controls"] if c["year"] == args.year]
    control_dates = {c["name"]: EM.M.inspected_dates(c["window"]) for c in controls}
    dates = sorted({m["date"] for m in scored} | {d for ds in control_dates.values() for d in ds})
    wide = A.load_wide(args.wide_dir, args.year)
    times = np.asarray(wide["time"], float).ravel()
    steps = map_steps(times, dates)
    grid, lat1, lon1, SP, check = load_inputs(args, sub, dates, steps, times, wide)
    lat_all, lon_all = np.arange(-60.0, 61.0), np.arange(-30.0, 101.0)      # points outside the subgrid stay unsupported, never dropped
    tols = decl["tolerances_deg"]
    thr = float(decl["threshold_per_second"]) * 1e5
    rows, cols = np.arange(lat1.size), np.arange(lon1.size)
    finals = {}
    if not args.support_only:
        import pilot_eastern_maxima as MX
        for spec in args.finals_trace:
            path, dg = spec.rsplit(":", 1)
            finals.update(MX.finals_from_trace(path, dg))
        if args.finals_baseline:
            finals.update(MX.finals_from_baseline(args.finals_baseline))
    per_map, per_control, peaks_by_date = [], [], {}
    for n, d in enumerate(dates):
        fz = fields_at(grid, n, rows, cols, SP[n], radius)
        if not args.support_only:
            import pilot_eastern_maxima as MX
            b_pk, _ = MX.local_maxima(fz["B"], fz["common"], thr, grid["latgrid"], grid["longrid"])
            c_pk, _ = MX.local_maxima(fz["C"]["complete_mean"], fz["common"], thr, grid["latgrid"], grid["longrid"])
            part = np.isfinite(fz["C"]["partial_mean"])
            p_pk, _ = MX.local_maxima(np.where(part, fz["C"]["partial_mean"], np.nan), part, thr, grid["latgrid"], grid["longrid"])
            for p in p_pk:
                i, j = p["cells_ij"][0]
                p["valid_fraction"] = round(float(fz["C"]["valid_fraction"][i, j]), 3)
            peaks_by_date[d] = {"B": b_pk, "C": c_pk, "C_partial": p_pk, "F": [{"lat": a, "lon": b} for a, b, _ in finals.get(d, [])]}
        for m in [m for m in scored if m["date"] == d]:
            row = {"date": d, "segment": m["segment"], "west_of_40E": EM.west_of_40(m)}
            for t in tols:
                pts = neighborhood(m, t, lat_all, lon_all)
                got, tot = support_share(pts, fz["peak_ok"], fz["lat1"], fz["lon1"])
                row[f"supported_{t}"] = [got, tot]
                row[f"fully_assessable_{t}"] = bool(tot > 0 and got == tot)
            if not args.support_only:
                box = ref["segments"][m["segment"]]["case_box"]
                bpts = [(float(la), float(lo)) for la in lat_all for lo in lon_all if box["lat"][0] <= la <= box["lat"][1] and box["lon"][0] <= lo <= box["lon"][1]]
                bgot, btot = support_share(bpts, fz["peak_ok"], fz["lat1"], fz["lon1"])
                row["case_box_supported"] = [bgot, btot]
                for key in ("B", "C"):
                    pk = peaks_by_date[d][key]
                    res = {}
                    for t in tols:
                        o, near = outcome(pk, m, t, row[f"fully_assessable_{t}"])
                        res[str(t)] = o
                        res["nearest"] = near
                    res["in_case_box"] = [{"lat": p["lat"], "lon": p["lon"], "strength_1e5_per_s": round(p["value"], 3)} for p in pk if in_box(p, box)]
                    row[key] = res
                fpk = peaks_by_date[d]["F"]
                fd = [EM.ref_distance(p["lat"], p["lon"], m) for p in fpk]
                row["F"] = {**{str(t): ("hit" if fd and min(fd) <= t else "miss") for t in tols}, "nearest_deg": round(min(fd), 3) if fd else None,
                            "in_case_box": sum(1 for p in fpk if in_box(p, box))}
                row["C_partial_in_case_box"] = [{"lat": p["lat"], "lon": p["lon"], "valid_fraction": p["valid_fraction"]} for p in peaks_by_date[d]["C_partial"] if in_box(p, box)]
            per_map.append(row)
        for c in controls:
            if d in control_dates[c["name"]]:
                box = c["box"]
                pts = [(float(la), float(lo)) for la in lat_all for lo in lon_all if box["lat"][0] <= la <= box["lat"][1] and box["lon"][0] <= lo <= box["lon"][1]]
                got, tot = support_share(pts, fz["peak_ok"], fz["lat1"], fz["lon1"])
                rowc = {"control": c["name"], "date": d, "box_supported": [got, tot]}
                if not args.support_only:
                    rowc["counts_in_box"] = {key: sum(1 for p in peaks_by_date[d][key] if in_box(p, box)) for key in ("B", "C", "C_partial", "F")}
                per_control.append(rowc)
    segs = sorted({r["segment"] for r in per_map})
    by_seg = {s: {"maps": sum(1 for r in per_map if r["segment"] == s), "fully_assessable_3": sum(1 for r in per_map if r["segment"] == s and r["fully_assessable_3.0"]),
                  "fully_assessable_5": sum(1 for r in per_map if r["segment"] == s and r["fully_assessable_5.0"])} for s in segs}
    total3 = sum(v["fully_assessable_3"] for v in by_seg.values())
    rule = decl["sufficiency"]
    sufficient = bool(scored) and total3 >= int(rule["min_fully_assessable_scored_maps_at_3"]) and all(v["fully_assessable_3"] >= 1 for v in by_seg.values())
    out = {"generated_by": "scripts/pilot_eastern_scale.py", "script_sha256": A.X.digest(os.path.abspath(__file__)), "year": args.year,
           "stage": "support check, no peaks computed" if args.support_only else "peaks on the common support",
           "declaration": {"path": args.declaration, "sha256": A.X.digest(args.declaration)}, "radius_km": radius, "subgrid": sub, "input_check": check,
           "scored_maps": per_map, "controls": per_control, "by_segment": by_seg, "fully_assessable_scored_maps_at_3": total3,
           "sufficient": sufficient if scored else None, "rule": rule}
    if not args.support_only:
        out["summary"] = summarize(per_map, per_control, tols)
        if args.lineage_reading:
            out["shared_regions"] = shared_region_counts(args, peaks_by_date, finals, ref)
        out["peaks_by_date"] = {d: {k: [{kk: vv for kk, vv in p.items() if kk != "cells_ij"} for p in v] for k, v in pk.items()} for d, pk in peaks_by_date.items()}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"input_check": check, "by_segment": by_seg, "fully_assessable_at_3": total3, "sufficient": out["sufficient"],
                      **({"summary": out["summary"]} if "summary" in out else {})}, indent=1))
    for c in controls:
        rows_c = [x for x in per_control if x["control"] == c["name"]]
        print(c["name"], "box support per map", [f"{a}/{b}" for a, b in (x["box_supported"] for x in rows_c)])
    print(f"Record {args.out}.")
    return 0


def in_box(p, box):
    return box["lat"][0] <= p["lat"] <= box["lat"][1] and box["lon"][0] <= p["lon"] <= box["lon"][1]


def summarize(per_map, per_control, tols):
    """Counts of hits, misses and not-assessable maps, the maps C gains and loses against B
    and against F among maps where both outcomes are determined, and control totals."""
    out = {}
    for t in tols:
        k = str(t)
        cnt = {key: {o: sum(1 for r in per_map if r[key][k] == o) for o in ("hit", "miss", "not assessable")} for key in ("B", "C")}
        cnt["F"] = {o: sum(1 for r in per_map if r["F"][k] == o) for o in ("hit", "miss")}
        def gl(a, b):
            both = [r for r in per_map if r[a][k] != "not assessable" and r[b][k] != "not assessable"]
            return {"maps_determined": len(both),
                    "gained": [(r["date"], r["segment"]) for r in both if r[a][k] == "hit" and r[b][k] == "miss"],
                    "lost": [(r["date"], r["segment"]) for r in both if r[a][k] == "miss" and r[b][k] == "hit"]}
        out[k] = {"counts": cnt, "C_against_B": gl("C", "B"), "C_against_F": gl("C", "F"), "B_against_F": gl("B", "F")}
    out["peaks_in_case_boxes"] = {key: sum(len(r[key]["in_case_box"]) for r in per_map) for key in ("B", "C")}
    out["finals_in_case_boxes"] = sum(r["F"]["in_case_box"] for r in per_map)
    out["partial_support_C_peaks_in_case_boxes"] = sum(len(r["C_partial_in_case_box"]) for r in per_map)
    out["controls_totals"] = {}
    for r in per_control:
        tot = out["controls_totals"].setdefault(r["control"], {"maps": 0, "B": 0, "C": 0, "C_partial": 0, "F": 0, "box_cells_supported": 0, "box_cells": 0})
        tot["maps"] += 1
        for key in ("B", "C", "C_partial", "F"):
            tot[key] += r["counts_in_box"][key]
        tot["box_cells_supported"] += r["box_supported"][0]; tot["box_cells"] += r["box_supported"][1]
    return out


def shared_region_counts(args, peaks_by_date, finals, ref):
    """B and C peaks inside each shared region the baseline implicated, described and not
    scored. The regions are recovered by rerunning the frozen detector at those maps, bound
    to the retained finals and to each region's recorded digest."""
    import pilot_eastern_maxima as MX
    run = A.P.load_run(args.control_run, year=args.year)
    case_b = A.SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, args.year)
    ds = run["record"]["dataset_specific"]
    ct, ft, _thr = A.thresholds_for_run((float(ds["coarse_threshold"]), float(ds["fine_threshold"])), args.calibration, False, True, run["domain"])
    absorb = bool(ds["tracker_flags"]["absorb"])
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    lat_c, lon_c = np.asarray(grids["latgrid_c"], float)[:, 0], np.asarray(grids["longrid_c"], float)[0, :]
    times = np.asarray(case_b["time"], float).ravel()
    lr = json.load(open(args.lineage_reading))
    maps_by = {(m["date"], m["segment"]): m for m in ref["maps"]}
    out = []
    for name in ("tolerance", "alternative"):
        for row in lr["readings"][name]["uncovered_scored_maps"]:
            d = row["date"]
            k = map_steps(times, [d])[d]
            fields = A.prepared_fields(wide, k, *sel)
            lin = {}
            waves = A.detect_from_prepared(fields, float(times[k]), ct, ft, grids, absorb=absorb, lineage=lin)
            mine = [(float(w["lat_mean"]), float(w["lon_mean"]), A.R.region_digest(w["region"])) for w in waves]
            if not MX.same_finals(mine, finals[d]):
                raise SystemExit(f"REFUSED: at {d} the recomputed detector does not reproduce the retained finals")
            coarse = {e["input"]: e for e in lin["coarse"]}
            seen = set()
            for ax in row["relevant_axes"]:
                near = [x for x in (ax.get("dropped_near") or []) if x["arc_deg"] == 0.0]
                if not near:
                    continue
                want = next(p["region_sha256"] for p in ax["positions"] if p["stage"] == "coarse region")
                mask = np.asarray(coarse[ax["axis"]]["select"]["region"], bool)
                if A.R.region_digest(mask) != want:
                    raise SystemExit(f"REFUSED: at {d} the recomputed region of axis {ax['axis']} is not the recorded one")
                if want in seen:
                    continue
                seen.add(want)
                m = maps_by[(d, row["segment"])]
                rec = {"reading": name, "date": d, "segment": row["segment"], "region_sha256": want, "cells": int(mask.sum())}
                for key in ("B", "C"):
                    inside = []
                    for p in peaks_by_date[d][key]:
                        i, j = MX.coarse_cell_of(p["lat"], p["lon"], lat_c, lon_c)
                        if mask[i, j]:
                            inside.append({"lat": p["lat"], "lon": p["lon"], "distance_to_read_deg": round(EM.ref_distance(p["lat"], p["lon"], m), 3)})
                    rec[key] = {"count": len(inside), "peaks": inside}
                out.append(rec)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
