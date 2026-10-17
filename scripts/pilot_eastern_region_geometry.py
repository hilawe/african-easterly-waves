#!/usr/bin/env python3
"""The region geometry inspection of `EASTERN_REGION_GEOMETRY_2026-10-11.md`.

For each of the 12 shared regions at 3 degrees, it pairs every reference-near axis that
selected the region with the input recorded as keeping it (from the retained lineage
reading), reruns the detector's lineage on the wide case to recover the region and the
axes (bound to the retained axes and the recorded region digest), draws the raw winds, raw
relative vorticity, the native curvature anomaly and the prepared coarse curvature over
the region with its axes, centers and positions, and reads one center-to-center transect
per pair. Descriptive only, with no matching or segmentation.

    python3 scripts/pilot_eastern_region_geometry.py --maxima <record> --lineage-reading <record> --reference <declaration> \\
        --trace <trace.json.gz>:<digest12> --control-run <dir> --control-case <mat> --wide-dir <dir> --calibration <json> \\
        --sp-file <sp> --png-prefix <prefix> --out <json>
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
import pilot_field_sequence as FS  # noqa: E402

STEP_DEG = 0.5
MIN_SEPARATION_DEG = 2.0
MARGIN_DEG = 4.0


def transect(a, b, step=STEP_DEG):
    """Points from a to b (lat, lon) every `step` degrees, both ends included."""
    n = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step)))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


def sample(field, valid, lat1, lon1, pts):
    """Nearest-cell values along the points, None where the cell is invalid or missing."""
    out = []
    for la, lo in pts:
        i, j = int(np.argmin(np.abs(lat1 - la))), int(np.argmin(np.abs(lon1 - lo)))
        v = field[i, j]
        out.append(float(v) if valid[i, j] and np.isfinite(v) else None)
    return out


def interior_dip(profile):
    """Whether an interior sample lies below both endpoint values, and the depth below the
    lower endpoint. None when any sample is missing."""
    if any(x is None for x in profile) or len(profile) < 3:
        return None
    lo = min(profile[1:-1])
    edge = min(profile[0], profile[-1])
    return {"dip": bool(lo < edge), "depth": float(edge - lo) if lo < edge else 0.0}


def reversals(v, a, b):
    """Meridional-wind sign changes along the transect, typed going east: trough-type where
    northerly (v < 0) turns southerly (v >= 0), ridge-type the reverse. Interpreted only when
    the transect lies within 45 degrees of east to west."""
    dlon = (b[1] - a[1]) * math.cos(math.radians(0.5 * (a[0] + b[0])))
    zonal = abs(dlon) >= abs(b[0] - a[0])
    seq = [x for x in v if x is not None]
    if dlon < 0:
        seq = seq[::-1]
    trough = sum(1 for p, q in zip(seq[:-1], seq[1:]) if p < 0 <= q)
    ridge = sum(1 for p, q in zip(seq[:-1], seq[1:]) if p >= 0 > q)
    return {"within_45_deg_of_zonal": bool(zonal), "trough_type": trough, "ridge_type": ridge, "interpreted": bool(zonal)}


def outcome(native_dip, prepared_dip):
    if native_dip is None or prepared_dip is None:
        return "unresolved geometry"
    if native_dip["dip"] and prepared_dip["dip"]:
        return "distinguishable lobes"
    if not native_dip["dip"] and not prepared_dip["dip"]:
        return "extended structure"
    return "unresolved geometry"


def pairs_from_lineage(lr, regions):
    """The fixed pairing rule: each reference-near axis that selected a recorded shared
    region, with every input recorded as dropping it at an identical pass-one position."""
    want = {(g["date"], g["region_sha256"]) for g in regions}
    out = {}
    for row in lr["readings"]["tolerance"]["uncovered_scored_maps"]:
        for ax in row["relevant_axes"]:
            reg = next((p for p in ax["positions"] if p["stage"] == "coarse region"), None)
            if reg is None or (row["date"], reg["region_sha256"]) not in want:
                continue
            key = (row["date"], reg["region_sha256"])
            e = out.setdefault(key, {"date": row["date"], "segment": row["segment"], "region_sha256": reg["region_sha256"], "pass_one": [reg["lat"], reg["lon"]], "pairs": [], "unpaired": []})
            near = [x for x in (ax.get("dropped_near") or []) if x["arc_deg"] == 0.0]
            if not near:
                e["unpaired"].append(ax["axis"])
            for x in near:
                e["pairs"].append({"near_input": ax["axis"], "near_center": ax["center"], "near_vertex_distance_deg": ax["vertex_distance_deg"],
                                   "keeping_input": x["input"], "output_position": x["output_position"], "output_distance_deg": x["output_distance_deg"]})
    return out


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from aew.v1port.detection import MAX_ZONAL_WIND
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--maxima", "--lineage-reading", "--reference", "--trace", "--control-run", "--control-case", "--wide-dir", "--calibration", "--sp-file", "--png-prefix", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    mrec, lr = json.load(open(args.maxima)), json.load(open(args.lineage_reading))
    ref = json.load(open(args.reference))["reference"]
    maps_by = {(m["date"], m["segment"]): m for m in ref["maps"] if m["scored"]}
    regions = [g for g in mrec["shared_regions"] if g["reading"] == "tolerance"]
    cases = pairs_from_lineage(lr, regions)
    if len(cases) != len(regions):
        raise SystemExit(f"STOPPED: the lineage reading gives {len(cases)} of the {len(regions)} recorded regions")
    path, dg = args.trace.rsplit(":", 1)
    with gzip.open(path, "rb") as fh:
        blob = fh.read()
    import hashlib
    if not hashlib.sha256(blob).hexdigest().startswith(dg):
        raise SystemExit("REFUSED: the trace does not carry its digest")
    retained_axes = {s["date"][:13]: s["lineage"]["axes"] for s in json.loads(blob)["trace"]["steps"] if s.get("lineage")}
    year = 2006
    run = A.P.load_run(args.control_run, year=year)
    case_b = A.SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, year)
    ds = run["record"]["dataset_specific"]
    ct, ft, _ = A.thresholds_for_run((float(ds["coarse_threshold"]), float(ds["fine_threshold"])), args.calibration, False, True, run["domain"])
    if ct != float(mrec["coarse_threshold"]):
        raise SystemExit("REFUSED: the coarse threshold is not the bound record's")
    absorb = bool(ds["tracker_flags"]["absorb"])
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    lat_c, lon_c = np.asarray(grids["lat_c"], float).ravel(), np.asarray(grids["lon_c"], float).ravel()
    latgC, longC = np.asarray(grids["latgrid_c"], float), np.asarray(grids["longrid_c"], float)
    times = np.asarray(case_b["time"], float).ravel()
    dates = sorted({c["date"] for c in cases.values()})
    steps = MX.map_steps(times, dates)
    wlat, wlon = np.asarray(wide["latgrid"], float)[:, 0], np.asarray(wide["longrid"], float)[0, :]
    rows = np.flatnonzero((wlat >= MX.VORT_SUBGRID["lat"][0]) & (wlat <= MX.VORT_SUBGRID["lat"][1]))
    cols = np.flatnonzero((wlon >= MX.VORT_SUBGRID["lon"][0]) & (wlon <= MX.VORT_SUBGRID["lon"][1]))
    lat1, lon1 = wlat[rows], wlon[cols]
    sp = FS.surface_pressure_on(args.sp_file, lat1, lon1, [float(times[steps[d]]) for d in dates])
    records = []
    for key in sorted(cases):
        c = cases[key]
        d = c["date"]
        n = dates.index(d)
        k = steps[d]
        fields = A.prepared_fields(wide, k, *sel)
        lin = {}
        A.detect_from_prepared(fields, float(times[k]), ct, ft, grids, absorb=absorb, lineage=lin)
        want = retained_axes[d]
        if len(want) != len(lin["axes"]) or any(list(map(float, la)) != w["lat"] or list(map(float, lo)) != w["lon"] for (la, lo), w in zip(lin["axes"], want)):
            raise SystemExit(f"STOPPED: at {d} the axes do not reproduce the retained axes")
        coarse = {e["input"]: e for e in lin["coarse"]}
        first = (c["pairs"][0]["near_input"] if c["pairs"] else c["unpaired"][0])
        mask = np.asarray(coarse[first]["select"]["region"], bool)
        if A.R.region_digest(mask) != c["region_sha256"]:
            raise SystemExit(f"STOPPED: at {d} the recomputed region is not the recorded one")
        S6 = fields["coarse_curvature"]
        westerly = fields["coarse_zonal_wind_smoothed"] > MAX_ZONAL_WIND
        weak = (S6 < ct) & ~westerly
        valid6 = ~westerly & np.isfinite(S6)
        pv = FS.panel_arrays(wide, k, rows, cols, "v", sp[n])
        pz = FS.panel_arrays(wide, k, rows, cols, "zeta", sp[n])
        u_n, v_n = np.asarray(wide["u"][k], float)[np.ix_(rows, cols)], np.asarray(wide["v"][k], float)[np.ix_(rows, cols)]
        S3 = np.asarray(wide["currv_anom"][k], float)[np.ix_(rows, cols)]
        valid_n = pz["classes"] == "valid"
        terrain_ok = pv["classes"] == "valid"
        m = maps_by[(d, c["segment"])]
        allmax, _ = MX.local_maxima(S6, valid6, -np.inf, latgC, longC)
        eligible, _ = MX.local_maxima(S6, valid6, ct, latgC, longC)
        def in_region(p):
            i, j = MX.coarse_cell_of(p["lat"], p["lon"], lat_c, lon_c)
            return bool(mask[i, j])
        region_info = {"cells": int(mask.sum()), "westerly_cells": int((mask & westerly).sum()), "below_threshold_cells": int((mask & weak).sum()),
                       "local_maxima_any_value": sum(1 for p in allmax if in_region(p)), "eligible_peaks": sum(1 for p in eligible if in_region(p)),
                       "lat_range": [float(latgC[mask].min()), float(latgC[mask].max())], "lon_range": [float(longC[mask].min()), float(longC[mask].max())]}
        pair_rows = []
        for p in c["pairs"]:
            kc = [float(np.mean(lin["axes"][p["keeping_input"]][0])), float(np.mean(lin["axes"][p["keeping_input"]][1]))]
            a, b = tuple(p["near_center"]), tuple(kc)
            sep = math.hypot(b[0] - a[0], b[1] - a[1])
            e = {**p, "keeping_center": kc, "center_separation_deg": round(sep, 3),
                 "near_center_in_region": bool(mask[MX.coarse_cell_of(a[0], a[1], lat_c, lon_c)]), "keeping_center_in_region": bool(mask[MX.coarse_cell_of(b[0], b[1], lat_c, lon_c)])}
            if sep < MIN_SEPARATION_DEG:
                e.update({"transect": None, "outcome": "unresolved geometry", "why": "centers less than one coarse cell apart"})
            else:
                pts = transect(a, b)
                prof = {"v": sample(v_n, terrain_ok, lat1, lon1, pts), "u": sample(u_n, terrain_ok, lat1, lon1, pts),
                        "S0": [None if x is None else x / 1e5 for x in sample(pz["shaded"], valid_n, lat1, lon1, pts)],
                        "S3": sample(S3, valid_n, lat1, lon1, pts), "S6": sample(S6, valid6, lat_c, lon_c, pts)}
                dips = {f: interior_dip(prof[f]) for f in ("S0", "S3", "S6")}
                uu = [x for x in prof["u"] if x is not None]; vv = [x for x in prof["v"] if x is not None]
                e.update({"transect": {"points": len(pts), "length_deg": round(sep, 3), "bearing_deg_from_east": round(math.degrees(math.atan2(b[0] - a[0], (b[1] - a[1]) * math.cos(math.radians(0.5 * (a[0] + b[0]))))), 1),
                                       "mean_u": None if not uu else round(float(np.mean(uu)), 2), "mean_v": None if not vv else round(float(np.mean(vv)), 2),
                                       "invalid_samples": {f: sum(1 for x in prof[f] if x is None) for f in prof}, "dips": dips,
                                       "reversals": reversals(prof["v"], a, b), "profiles": prof},
                          "outcome": outcome(dips["S3"], dips["S6"]),
                          "raw_to_prepared_difference": None if dips["S3"] is None or dips["S6"] is None else bool(dips["S3"]["dip"] != dips["S6"]["dip"])})
            pair_rows.append(e)
        # the figure
        la0, la1 = region_info["lat_range"][0] - MARGIN_DEG, region_info["lat_range"][1] + MARGIN_DEG
        lo0, lo1 = region_info["lon_range"][0] - MARGIN_DEG, region_info["lon_range"][1] + MARGIN_DEG
        npair = sum(1 for e in pair_rows if e.get("transect"))
        fig = plt.figure(figsize=(13.5, 9.6 + 2.6 * npair))
        gs = GridSpec(2 + npair, 5, figure=fig, width_ratios=[1, 0.035, 0.16, 1, 0.035], height_ratios=[3.2, 3.2] + [2.0] * npair, hspace=0.45, wspace=0.06,
                      left=0.06, right=0.95, top=1 - 1.1 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
        panels = [(0, 0, "Raw 700 hPa v (m/s) and winds", "v"), (0, 3, "Raw 700 hPa relative vorticity (1e-5 per s)", "zeta"),
                  (1, 0, "Native curvature anomaly S3 (1e-5 per s)", "S3"), (1, 3, "Prepared coarse curvature S6 (1e-5 per s)", "S6")]
        for r, cc, title, what in panels:
            ax = fig.add_subplot(gs[r, cc])
            if what in ("v", "zeta"):
                mesh = FS.draw_panel(ax, lon1, lat1, pv if what == "v" else pz, what, True)
            elif what == "S3":
                import matplotlib as mpl
                cm = mpl.colormaps["RdBu_r"].copy(); cm.set_bad((0, 0, 0, 0))
                mesh = ax.pcolormesh(lon1, lat1, np.ma.masked_where(~valid_n | ~np.isfinite(S3), S3 * 1e5), cmap=cm, vmin=-4, vmax=4, shading="auto")
            else:
                import matplotlib as mpl
                from matplotlib.colors import ListedColormap
                cm = mpl.colormaps["RdBu_r"].copy(); cm.set_bad((0, 0, 0, 0))
                mesh = ax.pcolormesh(lon_c, lat_c, np.ma.masked_where(~valid6, S6 * 1e5), cmap=cm, vmin=-2, vmax=2, shading="auto")
                ax.pcolormesh(lon_c, lat_c, np.ma.masked_where(~westerly, np.ones_like(S6)), cmap=ListedColormap(["#c8c8c8"]), shading="auto")
                ax.contourf(lon_c, lat_c, (weak & valid6).astype(float), levels=[0.5, 1.5], colors="none", hatches=["..."])
            ax.contour(lon_c, lat_c, mask.astype(float), levels=[0.5], colors="k", linewidths=1.6)
            for idx, (la, lo) in enumerate(lin["axes"]):
                lw, col = (2.2, "#00a000") if idx in [p["near_input"] for p in c["pairs"]] + c["unpaired"] else ((2.2, "#e07000") if idx in [p["keeping_input"] for p in c["pairs"]] else (0.9, "0.35"))
                ax.plot(lo, la, color=col, lw=lw)
            for e in pair_rows:
                ax.plot(e["near_center"][1], e["near_center"][0], "o", color="#00a000", ms=7, mec="k")
                ax.plot(e["keeping_center"][1], e["keeping_center"][0], "s", color="#e07000", ms=7, mec="k")
                if e.get("transect"):
                    ax.plot([e["near_center"][1], e["keeping_center"][1]], [e["near_center"][0], e["keeping_center"][0]], "--", color="0.1", lw=1.2)
                ax.plot(e["output_position"][1], e["output_position"][0], "*", color="#e07000", ms=13, mec="k")
            ax.plot(c["pass_one"][1], c["pass_one"][0], "X", color="k", ms=9)
            ax.plot([m["axis_lon"], m["axis_lon"]], m["lat_range"], color="#00c8d8", lw=3.4, solid_capstyle="butt")
            ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1); ax.set_title(title, fontsize=10)
            ax.set_xlabel("longitude (degrees E)", fontsize=9); ax.set_ylabel("latitude (degrees N)", fontsize=9)
            fig.colorbar(mesh, cax=fig.add_subplot(gs[r, cc + 1]))
        row_i = 2
        for e in pair_rows:
            if not e.get("transect"):
                continue
            pr = e["transect"]["profiles"]
            x = np.arange(len(pr["v"])) * e["transect"]["length_deg"] / max(1, len(pr["v"]) - 1)
            ax1 = fig.add_subplot(gs[row_i, 0])
            ax1.plot(x, [np.nan if y is None else y for y in pr["v"]], color="k", label="v")
            ax1.plot(x, [np.nan if y is None else y for y in pr["u"]], color="0.5", ls=":", label="u")
            ax1.axhline(0, color="0.6", lw=0.8); ax1.set_ylabel("m/s", fontsize=9); ax1.legend(fontsize=8, loc="upper right")
            ax1.set_title(f"Transect A (input {e['near_input']}) to B (input {e['keeping_input']}): winds", fontsize=9)
            ax2 = fig.add_subplot(gs[row_i, 3])
            for f, col in (("S0", "0.4"), ("S3", "#1f77b4"), ("S6", "#d62728")):
                ax2.plot(x, [np.nan if y is None else y * 1e5 for y in pr[f]], color=col, label=f)
            ax2.axhline(0, color="0.6", lw=0.8); ax2.set_ylabel("1e-5 per s", fontsize=9); ax2.legend(fontsize=8, loc="upper right")
            ax2.set_title(f"Same transect: S0, S3, S6. Class {e['outcome']}", fontsize=9)
            for axx in (ax1, ax2):
                axx.set_xlabel("distance from A along the transect (degrees)", fontsize=9)
            row_i += 1
        fig.suptitle(f"{d} {c['segment']}: shared region (black outline through coarse cell centers, diagonal links drawn as diamonds), reference-near axis and center "
                     f"(green, circle), keeping input's axis and center (orange, square), its coarse output (star), pass-one position (X), read geometry (cyan bar), "
                     f"transect A to B (dashed). Top panels: magenta below and black above the color range. S6: gray westerly-masked, dots below the coarse threshold.",
                     fontsize=9, wrap=True, y=1 - 0.2 / fig.get_figheight(), va="top")
        png = f"{args.png_prefix}_{d}.png"
        fig.savefig(png, dpi=FS.SAVE_DPI, facecolor="white")
        plt.close(fig)
        records.append({"date": d, "segment": c["segment"], "region_sha256": c["region_sha256"], "region": region_info, "pass_one": c["pass_one"],
                        "unpaired_near_inputs": c["unpaired"], "pairs": pair_rows, "figure": {"path": png, "sha256": A.X.digest(png)}})
    counts = {}
    for r in records:
        for e in r["pairs"]:
            counts[e["outcome"]] = counts.get(e["outcome"], 0) + 1
    out = {"generated_by": "scripts/pilot_eastern_region_geometry.py", "script_sha256": A.X.digest(os.path.abspath(__file__)), "coarse_threshold": ct,
           "regions": len(records), "pairs": sum(len(r["pairs"]) for r in records), "outcome_counts": counts, "cases": records, "seconds": round(time.time() - t0, 1),
           "limits": "descriptive; the keeping input is a computational role, not a truth label; no matching, segmentation, identity or merge rule"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("regions", "pairs", "outcome_counts", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
