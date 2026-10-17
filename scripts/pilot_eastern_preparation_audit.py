#!/usr/bin/env python3
"""The preparation and peak-correspondence audit of
`EASTERN_PREPARATION_AUDIT_2026-10-11.md`, sections 2 to 10, one season per run.

It follows 700 hPa vorticity through eight stages, from the field the retained maxima
were taken from (S0) to the detector's masked prepared coarse curvature (S7). It first
reproduces the production fields exactly (gates G1 to G4, without terrain masks), then
reads every stage on terrain-valid support carried through each operation's footprint,
and reports map coverage, net near and away peak counts, field values and loss classes.
Peak loss here is not detector-feature loss or physical-wave loss. The production code is
called, never changed.

    python3 scripts/pilot_eastern_preparation_audit.py --year 2006 --declaration <maxima declaration> --maxima <maxima record> \\
        --reference <2006 reference> --control-run <dir> --control-case <mat> --wide-dir <dir> --inputs <75E dir> \\
        --climo <npz> --sp-file <sp> --out <json>
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pilot_alledge_replay as A  # noqa: E402
import pilot_eastern_maxima as MX  # noqa: E402
import pilot_eastern_measurement as EM  # noqa: E402
import pilot_field_sequence as FS  # noqa: E402

STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
TOLS = (3.0, 5.0)
ANOMALY_ABS_TOL = 1e-18                                                  # per second, relative tolerance zero
WEAK_WIND = 1.0                                                          # m/s, a reporting convention
MOVING = ("1137_moving", "R8_east", "R9_first")
STANDING = ("1137_standing",)


def components_oriented(latgrid, longrid, u, v):
    """The detector's relative, shear and curvature vorticity on a grid in either row order,
    oriented ascending around the call as `pipeline.curvature_from_winds` orients it."""
    from aew.v1port import vorticity as VO
    latgrid, longrid, u, v = (np.asarray(x, float) for x in (latgrid, longrid, u, v))
    flip = latgrid[0, 0] > latgrid[-1, 0]
    if flip:
        r, s, c = VO.component_vorticity(latgrid[::-1], longrid[::-1], u[::-1], v[::-1])
        return r[::-1], s[::-1], c[::-1]
    return VO.component_vorticity(latgrid, longrid, u, v)


def stencil_valid(ok):
    """A cell and its four stencil neighbors valid, the edge excluded."""
    ok = np.asarray(ok, bool)
    out = np.zeros_like(ok)
    out[1:-1, 1:-1] = ok[1:-1, 1:-1] & ok[:-2, 1:-1] & ok[2:, 1:-1] & ok[1:-1, :-2] & ok[1:-1, 2:]
    return out


def block_valid(ok):
    """The production's 2 by 2 kernel window: the value at (r, c) is the mean of rows r and
    r + 1 and columns c and c + 1, so all four must be valid. The last row and column reach
    past the grid and are invalid."""
    ok = np.asarray(ok, bool)
    out = np.zeros_like(ok)
    out[:-1, :-1] = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    return out


def nine_valid(ok):
    """The nine-point smoother's footprint: an interior cell needs all nine cells valid, and
    the outer ring, which the smoother leaves unchanged, keeps its own support."""
    ok = np.asarray(ok, bool)
    out = ok.copy()
    inner = np.ones_like(ok[1:-1, 1:-1])
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            inner &= ok[1 + di:ok.shape[0] - 1 + di, 1 + dj:ok.shape[1] - 1 + dj]
    out[1:-1, 1:-1] = inner
    return out


def shear_category(sv, cv, speed):
    """The signed category of a peak cell, from shear and curvature that sum to relative."""
    if not (np.isfinite(sv) and np.isfinite(cv)):
        return "unresolved"
    if speed < WEAK_WIND:
        return "weak wind"
    if sv > 0 and cv > 0:
        return "shear larger, same sign" if sv > cv else "curvature larger, same sign"
    if sv > 0 >= cv:
        return "cancelling, shear positive"
    if cv > 0 >= sv:
        return "cancelling, curvature positive"
    return "neither positive"


def near_away(peaks, m, box, t):
    """Peaks in the box within the tolerance of the read geometry, and the box's others."""
    inbox = [p for p in peaks if MX.in_box(p["lat"], p["lon"], box)]
    near = [p for p in inbox if EM.ref_distance(p["lat"], p["lon"], m) <= t]
    return near, [p for p in inbox if p not in near]


def loss_class(field, valid, latg, long_, m, t, threshold, prev_pos):
    """Why a map covered at the previous stage is not covered here."""
    d = np.vectorize(lambda la, lo: EM.ref_distance(la, lo, m))(latg, long_)
    within = (d <= t) & valid
    if not within.any():
        return "unavailable"
    if prev_pos is not None:
        i = int(np.argmin(np.abs(latg[:, 0] - prev_pos[0]))); j = int(np.argmin(np.abs(long_[0, :] - prev_pos[1])))
        if not valid[i, j]:
            return "unavailable"
    allmax, _ = MX.local_maxima(field, valid, -np.inf, latg, long_)
    near = [p for p in allmax if EM.ref_distance(p["lat"], p["lon"], m) <= t]
    return "threshold loss" if near else "changed geometry"


def distance_grid(latg, long_, m):
    return np.vectorize(lambda la, lo: EM.ref_distance(la, lo, m))(latg, long_)


def crop_by_coords(field, lat_src, lon_src, lat_dst, lon_dst):
    ri = [int(np.flatnonzero(np.abs(lat_src - x) < 1e-6)[0]) for x in lat_dst]
    ci = [int(np.flatnonzero(np.abs(lon_src - x) < 1e-6)[0]) for x in lon_dst]
    return np.asarray(field)[np.ix_(ri, ci)]


def main(argv=None):
    import pilot_eastern_support600 as SU
    import pilot_wide_case as W
    from aew.v1port import climatology as clim
    from aew.v1port import load as L
    from aew.v1port import pipeline as PL
    from aew.v1port.detection import MAX_ZONAL_WIND, _prepare
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--declaration", "--maxima", "--control-run", "--control-case", "--wide-dir", "--inputs", "--climo", "--sp-file", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--reference")
    args = ap.parse_args(argv)
    t0 = time.time()
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    md, mrec = json.load(open(args.declaration)), json.load(open(args.maxima))
    if mrec["declaration"]["sha256"] != A.X.digest(args.declaration):
        raise SystemExit("REFUSED: the maxima record was not made under this declaration")
    ct = float(mrec["coarse_threshold"])                                 # full precision, from the bound artifact
    readings = {"B": float(md["thresholds"]["B_per_second"]), "coarse": ct}
    dom = md["domain"]
    run = A.P.load_run(args.control_run, year=args.year)
    case_b = A.SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, args.year)
    if wide["record"]["control_run"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the wide case was not bound to this control case")
    if A.X.digest(args.climo) != wide["record"]["climo_cache_sha256"]:
        raise SystemExit("REFUSED: the climatology is not the one the wide case names")
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    latgC, longC = np.asarray(grids["latgrid_c"], float), np.asarray(grids["longrid_c"], float)
    dates = sorted(mrec["maxima_by_date"])
    wtimes = np.asarray(wide["time"], float).ravel()
    steps = {d: int(np.flatnonzero(np.abs(wtimes - FS.to_day(d)) < 1.0 / 1440.0)[0]) for d in dates}
    times_f, latf, lonf, uf, vf = L.load_year(args.year, args.inputs, "era5")
    climo = W.read_pinned_climatology(args.climo)
    wlat, wlon = np.asarray(wide["latgrid"], float)[:, 0], np.asarray(wide["longrid"], float)[0, :]
    rows = np.flatnonzero((wlat >= MX.VORT_SUBGRID["lat"][0]) & (wlat <= MX.VORT_SUBGRID["lat"][1]))
    cols = np.flatnonzero((wlon >= MX.VORT_SUBGRID["lon"][0]) & (wlon <= MX.VORT_SUBGRID["lon"][1]))
    latS, lonS = wlat[rows], wlon[cols]
    latgS, longS = np.meshgrid(latS, lonS, indexing="ij")
    lat_cw, lon_cw = np.asarray(wide["lat_c"], float).ravel(), np.asarray(wide["lon_c"], float).ravel()
    r0, c0 = int(np.flatnonzero(np.abs(wlat - lat_cw[0]) < 1e-6)[0]), int(np.flatnonzero(np.abs(wlon - lon_cw[0]) < 1e-6)[0])
    native, nominal = 1.0, PL.COARSE_RESOLUTION_DEG
    width, stride = clim.decimation_shape(native, nominal)
    days = [float(wtimes[steps[d]]) for d in dates]
    cov_r = np.flatnonzero((wlat >= -25.0) & (wlat <= 35.0)); cov_c = np.flatnonzero((wlon >= -45.0) & (wlon <= 75.0))
    sp_part = FS.surface_pressure_on(args.sp_file, wlat[cov_r], wlon[cov_c], days)
    sp_w = np.full((len(dates), wlat.size, wlon.size), np.nan)
    sp_w[:, cov_r[0]:cov_r[-1] + 1, cov_c[0]:cov_c[-1] + 1] = sp_part
    ref = maps_by = None
    if args.reference:
        ref = json.load(open(args.reference))["reference"]
        maps_by = {(m["date"], m["segment"]): m for m in ref["maps"] if m["scored"]}
    scored = [(r["date"], r["segment"]) for r in mrec["scored_maps"]]
    controls = {}
    for c in mrec["controls"]:
        controls.setdefault(c["control"], []).append((c["date"], c["box"]))
    gates = {"G1": 0, "G2": 0, "G3a": 0, "G3b": 0, "G3c": 0, "G4a": 0, "G4b": 0}
    g2_max = 0.0
    flags = {"threshold_decisions_differing": 0, "peak_sets_differing": 0}
    per_map, bridge = {}, []
    for n, d in enumerate(dates):
        k = steps[d]
        kf = int(np.flatnonzero(np.abs(times_f - FS.to_day(d)) < 1.0 / 1440.0)[0])
        u_w, v_w = np.asarray(wide["u"][k], float), np.asarray(wide["v"][k], float)
        # G1: S0, the retained B, exactly as the maxima comparison formed it
        pa = FS.panel_arrays(wide, k, rows, cols, "zeta", sp_w[n][np.ix_(rows, cols)])
        b_prod = [p for p in MX.local_maxima(pa["shaded"], pa["classes"] == "valid", readings["B"] * 1e5, latgS, longS)[0] if MX.in_domain(p, dom)]
        if not SU.same_maxima(b_prod, mrec["maxima_by_date"][d]["B"], 1e-9):
            raise SystemExit(f"STOPPED: G1 fails at {d}, S0 does not reproduce the retained B maxima")
        gates["G1"] += 1
        # S1 and S2, the detector's decomposition on the full retrieval grid
        rf, sf, cf = components_oriented(latf, lonf, uf[kf], vf[kf])
        S1, SH, S2 = (crop_by_coords(x, latf[:, 0], lonf[0, :], latS, lonS) for x in (rf, sf, cf))
        # G2: the recomputed anomaly against the retained native anomaly
        s3_full = PL.anomaly_from_climatology(cf[None], times_f[kf:kf + 1], climo)[0]
        s3_rec_w = crop_by_coords(s3_full, latf[:, 0], lonf[0, :], wlat, wlon)
        s3_ret_w = np.asarray(wide["currv_anom"][k], float)
        fin = np.isfinite(s3_ret_w)
        if not np.array_equal(fin, np.isfinite(s3_rec_w)):
            raise SystemExit(f"STOPPED: G2 fails at {d}, finite masks differ")
        dmax = float(np.max(np.abs(s3_rec_w[fin] - s3_ret_w[fin]))) if fin.any() else 0.0
        g2_max = max(g2_max, dmax)
        if dmax > ANOMALY_ABS_TOL:
            raise SystemExit(f"STOPPED: G2 fails at {d}, largest difference {dmax:.3e} per second")
        gates["G2"] += 1
        # G3: smoothing, subsampling and the coarse wind, from the retained anomaly at the coarse grid's alignment
        nc, mc = lat_cw.size, lon_cw.size
        s4_al = clim.gaussian_decimate(s3_ret_w[r0:, c0:], native, native)
        s5_w = clim.gaussian_decimate(s3_ret_w[r0:, c0:], native, nominal)[:nc, :mc]
        if not np.array_equal(s4_al[::stride, ::stride][:nc, :mc], s5_w, equal_nan=True):
            raise SystemExit(f"STOPPED: G3a fails at {d}")
        gates["G3a"] += 1
        if not np.array_equal(s5_w, np.asarray(wide["currv_anom_c"][k], float), equal_nan=True):
            raise SystemExit(f"STOPPED: G3b fails at {d}")
        gates["G3b"] += 1
        if not np.array_equal(clim.gaussian_decimate(u_w[r0:, c0:], native, nominal)[:nc, :mc], np.asarray(wide["u_c"][k], float), equal_nan=True):
            raise SystemExit(f"STOPPED: G3c fails at {d}")
        gates["G3c"] += 1
        # G4: the prepared field and the retained A maxima
        fields = A.prepared_fields(wide, k, *sel)
        S6 = _prepare(s5_w, lat_cw)[np.ix_(np.flatnonzero(sel[0]), np.flatnonzero(sel[1]))]
        if not np.array_equal(S6, fields["coarse_curvature"], equal_nan=True):
            raise SystemExit(f"STOPPED: G4a fails at {d}")
        gates["G4a"] += 1
        westerly = fields["coarse_zonal_wind_smoothed"] > MAX_ZONAL_WIND
        S7 = np.where(westerly, np.nan, S6)
        a_prod = [p for p in MX.local_maxima(S7, np.isfinite(S7), ct, latgC, longC)[0] if MX.in_domain(p, dom)]
        want = mrec["maxima_by_date"][d]["A"]
        key = lambda p: (p["lat"], p["lon"])  # noqa: E731
        if not (len(a_prod) == len(want) and all(abs(x["lat"] - y["lat"]) < 1e-9 and abs(x["lon"] - y["lon"]) < 1e-9 and x["value"] == y["value"]
                                                 for x, y in zip(sorted(a_prod, key=key), sorted(want, key=key)))):
            raise SystemExit(f"STOPPED: G4b fails at {d}, S7 peaks are not the retained A maxima")
        gates["G4b"] += 1
        # terrain-valid support, carried through each footprint
        wcls = FS.terrain_classes(sp_w[n], np.where(np.isfinite(u_w) & np.isfinite(v_w), u_w, np.nan), 700.0)
        valid_w = FS.relative_vorticity(u_w, v_w, wlat, wlon, wcls)[1] == "valid"
        vS = valid_w[np.ix_(rows, cols)]
        s3_ret = s3_ret_w[np.ix_(rows, cols)]
        lat_al, lon_al = wlat[r0:], wlon[c0:]
        S4 = crop_by_coords(s4_al, lat_al, lon_al, latS, lonS)
        v4 = crop_by_coords(block_valid(valid_w[r0:, c0:]), lat_al, lon_al, latS, lonS)
        v5w = block_valid(valid_w[r0:, c0:])[::stride, ::stride][:nc, :mc]
        ic, jc = np.flatnonzero(sel[0]), np.flatnonzero(sel[1])
        v5 = v5w[np.ix_(ic, jc)]
        v6 = nine_valid(v5w)[np.ix_(ic, jc)]
        S5 = s5_w[np.ix_(ic, jc)]
        stage = {"S0": (pa["shaded"], vS, 1e5, latgS, longS), "S1": (S1, vS, 1.0, latgS, longS), "S2": (S2, vS, 1.0, latgS, longS),
                 "S3": (s3_ret, vS, 1.0, latgS, longS), "S4": (S4, v4, 1.0, latgS, longS), "S5": (S5, v5, 1.0, latgC, longC),
                 "S6": (S6, v6, 1.0, latgC, longC), "S7": (S7, v6 & ~westerly, 1.0, latgC, longC)}
        peaks = {}
        for s, (f, v, scale, lg, og) in stage.items():
            ok = v & np.isfinite(f)
            peaks[s] = {rd: [p for p in MX.local_maxima(f, ok, thr * scale, lg, og)[0] if MX.in_domain(p, dom)] for rd, thr in readings.items()}
            stage[s] = (f, ok, scale, lg, og)
        # the anomaly discrepancy's reach on reported decisions
        s3_rec = s3_rec_w[np.ix_(rows, cols)]
        for thr in readings.values():
            flags["threshold_decisions_differing"] += int(((s3_rec > thr) != (s3_ret > thr))[vS].sum())
            pr = MX.local_maxima(s3_rec, vS & np.isfinite(s3_rec), thr, latgS, longS)[0]
            pt = MX.local_maxima(s3_ret, vS & np.isfinite(s3_ret), thr, latgS, longS)[0]
            flags["peak_sets_differing"] += int(sorted((p["lat"], p["lon"]) for p in pr) != sorted((p["lat"], p["lon"]) for p in pt))
        # the bridge: S1 against S0 with the metric term and the radius ratio taken out
        both = vS & np.isfinite(pa["shaded"]) & np.isfinite(S1)
        s0 = pa["shaded"] / 1e5
        metric = u_w[np.ix_(rows, cols)] * np.tan(np.deg2rad(latgS)) / FS.EARTH_RADIUS_M
        from aew.v1port.vorticity import EARTH_RADIUS_M as A1
        resid = S1 - (FS.EARTH_RADIUS_M / A1) * (s0 - metric)
        bridge.append({"date": d, "cells": int(both.sum()), "max_abs_S1_minus_S0": float(np.max(np.abs(S1 - s0)[both])),
                       "median_abs_S1_minus_S0": float(np.median(np.abs(S1 - s0)[both])), "max_abs_metric": float(np.max(np.abs(metric)[both])),
                       "max_abs_residual_after_metric_and_radius": float(np.max(np.abs(resid)[both])),
                       "peaks_S0": {rd: len(peaks["S0"][rd]) for rd in readings}, "peaks_S1": {rd: len(peaks["S1"][rd]) for rd in readings}})
        per_map[d] = {"stage": stage, "peaks": peaks, "SH": SH, "S2": S2, "speed": np.hypot(u_w, v_w)[np.ix_(rows, cols)]}
    out = {"generated_by": "scripts/pilot_eastern_preparation_audit.py", "script_sha256": A.X.digest(os.path.abspath(__file__)), "year": args.year,
           "inputs": {"declaration": {"path": args.declaration, "sha256": A.X.digest(args.declaration)}, "maxima": {"path": args.maxima, "sha256": A.X.digest(args.maxima)},
                      "climo_sha256": A.X.digest(args.climo), "sp_file": args.sp_file, "reference": args.reference},
           "thresholds_per_second": readings, "decimation": {"nominal_deg": nominal, "native_deg": native, "kernel_width": width, "stride": stride,
                                                                "coarse_spacing_deg": float(abs(lat_cw[1] - lat_cw[0])), "alignment_start": [float(wlat[r0]), float(wlon[c0])]},
           "gates_passed_maps": gates, "maps": len(dates), "G2_largest_difference_per_second": g2_max, "anomaly_discrepancy_flags": flags, "bridge": bridge}
    out.update(interpret(per_map, scored, maps_by, ref, controls, readings))
    out["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("gates_passed_maps", "maps", "G2_largest_difference_per_second", "anomaly_discrepancy_flags", "decimation", "seconds")}, indent=1))
    return 0


def interpret(per_map, scored, maps_by, ref, controls, readings):
    """Coverage, net near and away counts, field values and loss classes per stage, and the
    signed categories at S1 peaks. Maps are counted once, and peaks are never matched."""
    rows = []
    for d, seg in scored:
        m, box = maps_by[(d, seg)], ref["segments"][seg]["case_box"]
        pm = per_map[d]
        row = {"date": d, "segment": seg, "stages": {}}
        prev = {}
        for s in STAGES:
            f, ok, scale, lg, og = pm["stage"][s]
            dist = distance_grid(lg, og, m)
            inb = (lg >= box["lat"][0]) & (lg <= box["lat"][1]) & (og >= box["lon"][0]) & (og <= box["lon"][1])
            ent = {}
            for rd, thr in readings.items():
                for t in TOLS:
                    pk = pm["peaks"][s][rd]
                    near, away = near_away(pk, m, box, t)
                    within = [p for p in pk if EM.ref_distance(p["lat"], p["lon"], m) <= t]
                    covered = bool(within)
                    e = {"covered": covered, "near": len(near), "away": len(away),
                         "max_value_within": None if not (ok & (dist <= t)).any() else float(np.max(f[ok & (dist <= t)]) / scale),
                         "max_value_in_box": None if not (ok & inb).any() else float(np.max(f[ok & inb]) / scale)}
                    key = f"{rd}|{t}"
                    if key in prev and prev[key]["covered"] and not covered:
                        e["loss"] = loss_class(f, ok, lg, og, m, t, thr * scale, prev[key]["pos"])
                    e["pos"] = None if not within else min(((p["lat"], p["lon"]) for p in within), key=lambda q: EM.ref_distance(q[0], q[1], m))
                    ent[key] = e
            row["stages"][s] = ent
            prev = ent
        cats = {str(t): {"near": {}, "away": {}} for t in TOLS}
        for t in TOLS:
            near, away = near_away(pm["peaks"]["S1"]["B"], m, box, t)
            for side, ps in (("near", near), ("away", away)):
                for p in ps:
                    i, j = int(np.argmin(np.abs(pm["stage"]["S1"][3][:, 0] - p["lat"]))), int(np.argmin(np.abs(pm["stage"]["S1"][4][0, :] - p["lon"])))
                    c = shear_category(pm["SH"][i, j], pm["S2"][i, j], pm["speed"][i, j])
                    cats[str(t)][side][c] = cats[str(t)][side].get(c, 0) + 1
        row["S1_categories"] = cats
        rows.append(row)
    groups = {"moving": [r for r in rows if r["segment"] in MOVING], "1137_standing": [r for r in rows if r["segment"] in STANDING]}
    for sgm in MOVING:
        groups[sgm] = [r for r in rows if r["segment"] == sgm]
    summary = {}
    for g, rs in groups.items():
        if not rs:
            continue
        summary[g] = {}
        for s in STAGES:
            summary[g][s] = {}
            for key in rs[0]["stages"][s]:
                es = [r["stages"][s][key] for r in rs]
                vals = [e["max_value_within"] for e in es if e["max_value_within"] is not None]
                losses = {}
                for e in es:
                    if "loss" in e:
                        losses[e["loss"]] = losses.get(e["loss"], 0) + 1
                summary[g][s][key] = {"maps": len(es), "covered": sum(e["covered"] for e in es), "near": sum(e["near"] for e in es), "away": sum(e["away"] for e in es),
                                      "unavailable_within": sum(1 for e in es if e["max_value_within"] is None),
                                      "median_max_value_within": None if not vals else float(np.median(vals)), "losses": losses}
        cat = {str(t): {"near": {}, "away": {}} for t in TOLS}
        for r in rs:
            for t in cat:
                for side in ("near", "away"):
                    for c, n in r["S1_categories"][t][side].items():
                        cat[t][side][c] = cat[t][side].get(c, 0) + n
        summary[g]["S1_categories"] = cat
    ctrl = {}
    for name, items in controls.items():
        ctrl[name] = {}
        for s in STAGES:
            ctrl[name][s] = {}
            for rd in readings:
                cnt, vals = 0, []
                for d, box in items:
                    f, ok, scale, lg, og = per_map[d]["stage"][s]
                    cnt += sum(1 for p in per_map[d]["peaks"][s][rd] if MX.in_box(p["lat"], p["lon"], box))
                    inb = ok & (lg >= box["lat"][0]) & (lg <= box["lat"][1]) & (og >= box["lon"][0]) & (og <= box["lon"][1])
                    if inb.any():
                        vals.append(float(np.max(f[inb]) / scale))
                ctrl[name][s][rd] = {"maps": len(items), "peaks_in_box": cnt, "median_box_max_value": None if not vals else float(np.median(vals))}
    transitions = {}
    if "moving" in summary:
        for a, b in zip(STAGES[:-1], STAGES[1:]):
            transitions[f"{a}->{b}"] = {}
            for key in summary["moving"][a]:
                x, y = summary["moving"][a][key], summary["moving"][b][key]
                share = lambda before, after: None if before == 0 else (before - after) / before  # noqa: E731
                transitions[f"{a}->{b}"][key] = {"covered_change": y["covered"] - x["covered"], "near_change": y["near"] - x["near"], "away_change": y["away"] - x["away"],
                                                 "near_net_loss_share": share(x["near"], y["near"]), "away_net_loss_share": share(x["away"], y["away"])}
            for rd in readings:
                ks = [f"{rd}|{t}" for t in TOLS]
                tr = transitions[f"{a}->{b}"]
                transitions[f"{a}->{b}"][f"{rd} preferential"] = all(tr[k]["covered_change"] < 0 and tr[k]["near_net_loss_share"] is not None and tr[k]["away_net_loss_share"] is not None
                                                                    and tr[k]["near_net_loss_share"] > tr[k]["away_net_loss_share"] for k in ks)
    for r in rows:
        for s in r["stages"].values():
            for e in s.values():
                e.pop("pos", None)
    return {"scored_maps": rows, "summary": summary, "controls": ctrl, "moving_transitions": transitions}


if __name__ == "__main__":
    raise SystemExit(main())
