#!/usr/bin/env python3
"""The eastern pilot's stage diagnostic: for a few named event families, both runs'
detector stages at every step of a window, drawn side by side and compared stage by
stage, from retained case files and tracks alone.

For each run (the 40 E control B and the 60 E treatment C) and each step, the detector's
own stages are recomputed from the run's retained case file through the port's own
functions, exactly as `detection.detect_troughs` orders them:

1. PREPARATION: the coarse anomaly smoothed and made cyclonic-positive, the westerly mask
   on the smoothed zonal wind, the sub-threshold mask against the run's coarse threshold.
2. DETECTION: the trough axes, the zero contour of the smoothed advection inside the mask.
3. MERGE: the axis candidates (each axis's mean position) merged on the coarse grid, then
   on the fine grid, with the run's two thresholds, giving the final candidates the
   association stage receives.
4. ASSOCIATION is not recomputed here (it needs the live state of every track). What is
   read from the stored tracks is which track, if any, holds a position at each candidate
   and where the named tracks are.

The comparison at each step, inside the family's box and the common domain west of 40 E:
EVERY PREPARED INPUT cell for cell on its own grid (the masked coarse curvature, the
masked coarse advection the axes are contoured from, the smoothed coarse zonal wind the
westerly mask reads, the masked fine curvature the second merge pass reads, and the
smoothed fine winds the association's region medians read), the axes (each run's axis
vertices against the other's nearest vertex), the final candidates (each against the other
run's nearest candidate), and for each named track its position, the nearest final
candidate in its own run and in the other run, and the nearest stored position of any
track in the other run. The first step at which each stage differs is reported per
family, per prepared input. A first version compared the coarse curvature alone and an
independent check found the fine curvature differing at 40 E, the fine smoother's own
edge column, at steps where the coarse field differed only at 39 E. The record now
carries both grids. Association is replayed by `pilot_association_replay.py`.

FIGURES, one page per family with a row per step and two panels, B left and C right. Each
panel shows the prepared field shaded on one scale with a colorbar (s^-1), masked cells
gray, the region outside the run's detection domain hatched white (a different thing from
a masked cell), the axes in black, the coarse-pass candidates as hollow triangles, the
final candidates as crosses, the smoothed coarse wind as arrows, coastlines, the 40 E and
60 E meridians labeled, every track of the run with a stored observation at this step as a
small gray dot, and the named tracks with their stored history: consecutive observations
joined by a solid line, a gap of more than one step by a dotted one, a FILLED marker only
when the track has an observation at this step, a HOLLOW marker at its last earlier
position labeled with the age otherwise. Proximity is a distance, drawn and listed, never a
class.

    python3 scripts/pilot_stage_diagnostic.py --families <json> --control-run <B dir> --control-case <B tracker_case.mat> \\
        --treatment-run <C dir> --treatment-case <C tracker_case.mat> --year 1990 --out-dir <fresh dir>

The families file: {"families": [{"name": ..., "control_tracks": [i, ...], "treatment_tracks": [j, ...],
"window": ["YYYY-MM-DDTHH", "YYYY-MM-DDTHH"], "box": {"lat": [lo, hi], "lon": [lo, hi]}, "rule": "..."}]}.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

EPOCH = dt.datetime(1900, 1, 1)
ANOM_LIMIT = 1.5e-5
COMMON_EAST_DEG = 40.0
MATCH_DEG = 1.0                 # a candidate or vertex in one run "has a counterpart" in the other within this
STEP_DAYS = 0.25
PAGE_ROWS = 6                   # steps per figure page


def date_of(days):
    return EPOCH + dt.timedelta(days=float(days))


def load_case(case_path, record):
    """The case's bytes, hashed against the record, parsed once into the arrays the stages need."""
    from scipy.io import loadmat
    with open(case_path, "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != record["dataset_specific"].get("case_sha256"):
        raise SystemExit(f"REFUSED: {case_path} is not the case the record names")
    names = ["time", "lat_c", "lon_c", "latgrid", "longrid", "u_c", "v_c", "currv_anom_c", "advcurrv_anom_c", "currv_anom", "u", "v"]
    out = {}
    for name in names:
        out[name] = np.asarray(loadmat(io.BytesIO(blob), variable_names=[name])[name], float)
    out["time"], out["lat_c"], out["lon_c"] = out["time"].ravel(), out["lat_c"].ravel(), out["lon_c"].ravel()
    out["sha256"] = digest
    return out


def stages(case, k, coarse_threshold, fine_threshold):
    """Preparation, detection and merge at step k, the detector's own functions in its order."""
    from aew.v1port.climatology import smooth9
    from aew.v1port.contours import merge_contours
    from aew.v1port.detection import MAX_ZONAL_WIND, _prepare, trough_axes
    lat_c, lon_c = case["lat_c"], case["lon_c"]
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    latgrid_f, longrid_f = case["latgrid"], case["longrid"]
    wind = smooth9(case["u_c"][k])
    vwind = smooth9(case["v_c"][k])
    curvature = _prepare(case["currv_anom_c"][k], lat_c)
    advection = smooth9(case["advcurrv_anom_c"][k])
    masked = (wind > MAX_ZONAL_WIND) | (curvature < coarse_threshold)
    curvature = np.where(masked, np.nan, curvature)
    advection = np.where(masked, np.nan, advection)
    axes = trough_axes(latgrid_c, longrid_c, advection)
    candidates = [{"time": float(case["time"][k]), "lat_mean": float(np.mean(a)), "lon_mean": float(np.mean(o))} for a, o in axes]
    curvature_f = _prepare(case["currv_anom"][k], latgrid_f[:, 0])
    curvature_f = np.where(curvature_f < fine_threshold, np.nan, curvature_f)
    coarse = merge_contours(candidates, latgrid_c, longrid_c, curvature, coarse_threshold) if candidates else []
    final = merge_contours(coarse, latgrid_f, longrid_f, curvature_f, fine_threshold) if coarse else []
    # The association's own inputs at this step, the smoothed fine winds whose medians over
    # a candidate's region predict where it goes (pipeline.track_year and track_case).
    fine_u, fine_v = smooth9(case["u"][k]), smooth9(case["v"][k])
    return {"prepared": curvature, "wind": wind, "vwind": vwind, "advection": advection, "prepared_fine": curvature_f,
            "fine_u": fine_u, "fine_v": fine_v, "axes": axes,
            "axis_candidates": [(c["lat_mean"], c["lon_mean"]) for c in candidates],
            "coarse_merged": [(c["lat_mean"], c["lon_mean"]) for c in coarse],
            "final": [(c["lat_mean"], c["lon_mean"]) for c in final]}


# Every prepared input the detector and the association read at a step, each on its grid.
PREPARED_INPUTS = (("coarse_curvature", "prepared", "coarse"), ("coarse_advection", "advection", "coarse"),
                   ("coarse_zonal_wind_smoothed", "wind", "coarse"), ("fine_curvature", "prepared_fine", "fine"),
                   ("fine_zonal_wind_smoothed", "fine_u", "fine"), ("fine_meridional_wind_smoothed", "fine_v", "fine"))


def field_difference(fb, fc, common_lons):
    """Two prepared fields on their common columns, cell for cell: masked-ness and values."""
    nan_diff = int(np.count_nonzero(np.isnan(fb) != np.isnan(fc)))
    both = ~np.isnan(fb) & ~np.isnan(fc)
    val_diff = int(np.count_nonzero(fb[both] != fc[both]))
    max_abs = float(np.max(np.abs(fb[both] - fc[both]))) if val_diff else 0.0
    diff_cols = np.where(((np.isnan(fb) != np.isnan(fc)) | (both & (fb != fc))).any(axis=0))[0]
    return {"common_cells": int(fb.size), "cells_differing": val_diff + nan_diff, "mask_cells_differing": nan_diff,
            "max_abs_difference": max_abs,
            "westernmost_differing_lon": float(common_lons[diff_cols.min()]) if diff_cols.size else None,
            "easternmost_differing_lon": float(common_lons[diff_cols.max()]) if diff_cols.size else None}


def nearest(point, points):
    if not points:
        return None, None
    d = [float(np.hypot(p[0] - point[0], p[1] - point[1])) for p in points]
    i = int(np.argmin(d))
    return d[i], points[i]


def in_box(points, box):
    return [p for p in points if box["lat"][0] <= p[0] <= box["lat"][1] and box["lon"][0] <= p[1] <= box["lon"][1]]


def positions_at(tracks, t):
    out = []
    for j, tr in enumerate(tracks):
        at = np.where(np.abs(tr["time"] - t) < 1e-9)[0]
        if at.size:
            out.append((j, float(tr["lat"][at[0]]), float(tr["lon"][at[0]])))
    return out


def compare_step(sb, sc, case_b, case_c, box):
    """The stage-by-stage comparison inside the box and west of the common edge."""
    lat_c, lon_b, lon_c = case_b["lat_c"], case_b["lon_c"], case_c["lon_c"]
    rows = (lat_c >= box["lat"][0]) & (lat_c <= box["lat"][1])
    cols_b = (lon_b >= box["lon"][0]) & (lon_b <= min(box["lon"][1], COMMON_EAST_DEG))
    common = lon_c[np.isin(lon_c, lon_b[cols_b])]
    cols_c = np.isin(lon_c, common)
    # the same selection on the fine grid, whose columns are its own
    lat_f, flon_b, flon_c = case_b["latgrid"][:, 0], case_b["longrid"][0, :], case_c["longrid"][0, :]
    if not np.array_equal(lat_f, case_c["latgrid"][:, 0]) or not np.array_equal(lat_c, case_c["lat_c"]):
        raise SystemExit("REFUSED: the two cases do not share their latitude rows")
    rows_f = (lat_f >= box["lat"][0]) & (lat_f <= box["lat"][1])
    fcols_b = (flon_b >= box["lon"][0]) & (flon_b <= min(box["lon"][1], COMMON_EAST_DEG))
    common_f = flon_c[np.isin(flon_c, flon_b[fcols_b])]
    support = {"coarse": (rows, np.isin(lon_b, common), cols_c, common), "fine": (rows_f, np.isin(flon_b, common_f), np.isin(flon_c, common_f), common_f)}
    prepared_inputs = {}
    for name, key, grid in PREPARED_INPUTS:
        r, cb, cc, lons = support[grid]
        prepared_inputs[name] = field_difference(sb[key][np.ix_(r, cb)], sc[key][np.ix_(r, cc)], lons)
    coordinate_support = {"coarse_common_columns": int(common.size), "coarse_columns_b": int(lon_b.size), "coarse_columns_c": int(lon_c.size),
                          "fine_common_columns": int(common_f.size), "fine_columns_b": int(flon_b.size), "fine_columns_c": int(flon_c.size),
                          "coarse_easternmost_lon_b": float(lon_b.max()), "coarse_easternmost_lon_c": float(lon_c.max()),
                          "fine_easternmost_lon_b": float(flon_b.max()), "fine_easternmost_lon_c": float(flon_c.max())}

    def vertices(s):
        return in_box([(float(a), float(o)) for la, lo in s["axes"] for a, o in zip(la, lo) if o <= COMMON_EAST_DEG], box)
    def unmatched(points, others):
        # an exact match has distance 0.0, which is a match, so None is tested, never truthiness
        out = []
        for p in points:
            d, _ = nearest(p, others)
            if d is None or d > MATCH_DEG:
                out.append(p)
        return out
    vb, vc = vertices(sb), vertices(sc)
    unmatched_b, unmatched_c = unmatched(vb, vc), unmatched(vc, vb)
    exact = sorted(set(vb)) == sorted(set(vc))                                   # exact floats, no rounding, so identity means identity
    cand_b = [c for c in in_box(sb["final"], box) if c[1] <= COMMON_EAST_DEG]
    cand_c = [c for c in in_box(sc["final"], box) if c[1] <= COMMON_EAST_DEG]
    only_b, only_c = unmatched(cand_b, cand_c), unmatched(cand_c, cand_b)
    return {"prepared_field": prepared_inputs["coarse_curvature"], "prepared_inputs": prepared_inputs, "coordinate_support": coordinate_support,
            "axes": {"vertices_b": len(vb), "vertices_c": len(vc), "vertex_sets_identical": exact, "b_vertices_without_c_counterpart": len(unmatched_b),
                     "c_vertices_without_b_counterpart": len(unmatched_c), "farthest_unmatched_from_common_edge_deg":
                     (COMMON_EAST_DEG - min([v[1] for v in unmatched_b + unmatched_c])) if (unmatched_b or unmatched_c) else None},
            "final_candidates": {"b": [[round(p[0], 3), round(p[1], 3)] for p in cand_b], "c": [[round(p[0], 3), round(p[1], 3)] for p in cand_c],
                                 "only_b": [[round(p[0], 3), round(p[1], 3)] for p in only_b], "only_c": [[round(p[0], 3), round(p[1], 3)] for p in only_c]}}


def track_row(name, tr, t, own_stage, other_stage, other_tracks):
    """The named track's STORED position at time t, which is the finished output: a
    five-point moving average of the positions the track claimed, after the final speed
    filter. A distance from it to a candidate is a distance from the finished record, not
    evidence of which candidate the track claimed, and a track absent from the stored
    output may have been pruned in the loop or removed by the final speed filter."""
    at = np.where(np.abs(tr["time"] - t) < 1e-9)[0]
    if not at.size:
        return {"track": name, "position": None}
    p = (float(tr["lat"][at[0]]), float(tr["lon"][at[0]]))
    d_own, c_own = nearest(p, own_stage["final"])
    d_oth, c_oth = nearest(p, other_stage["final"])
    others = positions_at(other_tracks, t)
    d_trk, trk = nearest(p, [(la, lo) for _, la, lo in others]) if others else (None, None)
    which = next((j for j, la, lo in others if trk is not None and (la, lo) == trk), None)
    return {"track": name, "position": [round(p[0], 3), round(p[1], 3)],
            "nearest_final_candidate_own_run_deg": None if d_own is None else round(d_own, 3),
            "nearest_final_candidate_other_run_deg": None if d_oth is None else round(d_oth, 3),
            "nearest_final_candidate_other_run": None if c_oth is None else [round(c_oth[0], 3), round(c_oth[1], 3)],
            "nearest_stored_position_other_run_deg": None if d_trk is None else round(d_trk, 3),
            "nearest_stored_position_other_run_track": which}


def coastlines():
    """Natural Earth 50 m coastlines from cartopy's local cache, never downloaded here."""
    import cartopy
    from cartopy.io import shapereader
    path = os.path.join(cartopy.config["data_dir"], "shapefiles", "natural_earth", "physical", "ne_50m_coastline.shp")
    if not os.path.exists(path):
        raise SystemExit(f"REFUSED: no cached coastline at {path}; this script does not download")
    lines = []
    for geom in shapereader.Reader(path).geometries():
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for part in parts:
            xy = np.asarray(part.coords)
            lines.append((xy[:, 0], xy[:, 1]))
    return lines


def draw_history(ax, tr, t, color, lw, alpha):
    """The stored positions up to t, consecutive observations joined by a solid line and
    a gap of more than one step by a dotted one, the same rule for every overlay."""
    past = tr["time"] <= t + 1e-9
    tt, la, lo = tr["time"][past], tr["lat"][past], tr["lon"][past]
    for i in range(1, tt.size):
        style = "-" if (tt[i] - tt[i - 1]) <= STEP_DAYS + 1e-9 else ":"
        ax.plot(lo[i - 1:i + 1], la[i - 1:i + 1], style, color=color, lw=lw, alpha=alpha, zorder=8)


def draw_panel(ax, case, stage, run_tracks, named, t, box, domain_lon, coast, label, color):
    import matplotlib.pyplot as plt
    lat_c, lon_c = case["lat_c"], case["lon_c"]
    ax.set_facecolor("white")
    # outside the run's detection domain: hatched, distinct from a masked cell
    ax.fill_between([box["lon"][0], box["lon"][1]], box["lat"][0], box["lat"][1], facecolor="white", hatch="xx", edgecolor="0.7", linewidth=0, zorder=0)
    ax.fill_between([max(box["lon"][0], domain_lon[0]), min(box["lon"][1], domain_lon[1])], max(box["lat"][0], -35.0), min(box["lat"][1], 35.0),
                    facecolor="0.82", edgecolor="none", zorder=1)
    mesh = ax.pcolormesh(lon_c, lat_c, np.ma.masked_invalid(stage["prepared"]), cmap="YlOrRd", vmin=0.0, vmax=ANOM_LIMIT, shading="auto", zorder=2)
    for la, lo in stage["axes"]:
        ax.plot(lo, la, "-", color="k", lw=0.9, zorder=4)
    sub = slice(None, None, 1)
    ax.quiver(lon_c[sub], lat_c[sub], stage["wind"][sub, sub], stage["vwind"][sub, sub], color="0.35", scale=350, width=0.0022, zorder=3)
    for x, y in coast:
        ax.plot(x, y, "-", color="0.15", lw=0.6, zorder=5)
    if stage["coarse_merged"]:
        ax.plot([p[1] for p in stage["coarse_merged"]], [p[0] for p in stage["coarse_merged"]], "^", mfc="none", mec="k", ms=6, zorder=6)
    if stage["final"]:
        ax.plot([p[1] for p in stage["final"]], [p[0] for p in stage["final"]], "x", color="k", ms=7, mew=1.5, zorder=7)
    others = positions_at(run_tracks, t)
    if others:
        ax.plot([lo for _, la, lo in others], [la for _, la, lo in others], "o", color="0.45", ms=3, zorder=6)
    for name, tr in named:
        past = tr["time"] <= t + 1e-9
        tt, la, lo = tr["time"][past], tr["lat"][past], tr["lon"][past]
        draw_history(ax, tr, t, color, 1.8, 1.0)
        at = np.where(np.abs(tr["time"] - t) < 1e-9)[0]
        if at.size:
            ax.plot(tr["lon"][at[0]], tr["lat"][at[0]], "o", color=color, ms=8, zorder=9)
        elif past.any():
            age_h = (t - tt[-1]) * 24.0
            ax.plot(lo[-1], la[-1], "o", mfc="none", mec=color, ms=8, mew=1.5, zorder=9)
            ax.annotate(f"{name} last +{age_h:.0f} h", (lo[-1], la[-1]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
    for lon, text in ((40.0, "40 E"), (60.0, "60 E")):
        if box["lon"][0] <= lon <= box["lon"][1]:
            ax.axvline(lon, color="k", lw=0.8, ls="--", zorder=5)
            ax.text(lon, box["lat"][1] - 0.6, text, fontsize=7, ha="center", va="top", zorder=10)
    ax.set_xlim(*box["lon"]); ax.set_ylim(*box["lat"])
    ax.set_title(f"{label}: {date_of(t):%Y-%m-%d %HZ}, {len(stage['axes'])} axes, {len(stage['final'])} final candidates", fontsize=8)
    return mesh


def run_family(fam, runs, cases, coast, out_dir, year):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    case_b, case_c = cases["B"], cases["C"]
    thr = {r: (float(runs[r]["record"]["dataset_specific"]["coarse_threshold"]), float(runs[r]["record"]["dataset_specific"]["fine_threshold"])) for r in ("B", "C")}
    times = case_c["time"]
    t0 = (dt.datetime.strptime(fam["window"][0], "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0
    t1 = (dt.datetime.strptime(fam["window"][1], "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0
    steps = [k for k in range(times.size) if t0 - 1e-9 <= times[k] <= t1 + 1e-9]
    if not steps:
        raise SystemExit(f"REFUSED: the window {fam['window']} holds no step of the case")
    named_b = [(f"B{i}", runs["B"]["tracks"][i]) for i in fam.get("control_tracks", [])]
    named_c = [(f"C{j}", runs["C"]["tracks"][j]) for j in fam.get("treatment_tracks", [])]
    box = fam["box"]
    rows, pages = [], []
    for page, first in enumerate(range(0, len(steps), PAGE_ROWS), start=1):
        chunk = steps[first:first + PAGE_ROWS]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(15.0, 4.6 * len(chunk)), squeeze=False)
        mesh = None
        for r, k in enumerate(chunk):
            t = float(times[k])
            kb = int(np.argmin(np.abs(case_b["time"] - t)))
            if abs(case_b["time"][kb] - t) > 1e-6:
                raise SystemExit("REFUSED: the control case has no step at this time")
            sb = stages(case_b, kb, *thr["B"])
            sc = stages(case_c, k, *thr["C"])
            mesh = draw_panel(axes[r, 0], case_b, sb, runs["B"]["tracks"], named_b, t, box, runs["B"]["domain"]["lon"], coast, "B, 40 E control", "tab:blue")
            draw_panel(axes[r, 1], case_c, sc, runs["C"]["tracks"], named_c, t, box, runs["C"]["domain"]["lon"], coast, "C, 60 E treatment", "tab:red")
            # the other run's named tracks, drawn thin on each panel for reference
            for name, tr in named_c:
                draw_history(axes[r, 0], tr, t, "tab:red", 0.8, 0.6)
            for name, tr in named_b:
                draw_history(axes[r, 1], tr, t, "tab:blue", 0.8, 0.6)
            rows.append({"step": k, "date": f"{date_of(t):%Y-%m-%dT%H:00Z}", "page": page, "comparison": compare_step(sb, sc, case_b, case_c, box),
                         "tracks": [track_row(n, tr, t, sb, sc, runs["C"]["tracks"]) for n, tr in named_b] + [track_row(n, tr, t, sc, sb, runs["B"]["tracks"]) for n, tr in named_c],
                         "candidates_in_box": {"B": {"axes": len(in_box([(float(np.mean(a)), float(np.mean(o))) for a, o in sb["axes"]], box)), "coarse_merged": len(in_box(sb["coarse_merged"], box)), "final": len(in_box(sb["final"], box))},
                                               "C": {"axes": len(in_box([(float(np.mean(a)), float(np.mean(o))) for a, o in sc["axes"]], box)), "coarse_merged": len(in_box(sc["coarse_merged"], box)), "final": len(in_box(sc["final"], box))}}})
        pages.append((fig, axes, mesh, page))
    handles = [Line2D([], [], color="k", lw=0.9, label="trough axes (detector)"), Line2D([], [], marker="^", mfc="none", mec="k", ls="", label="coarse-pass merged candidate"),
               Line2D([], [], marker="x", color="k", ls="", label="final candidate (to association)"), Line2D([], [], marker="o", color="0.45", ls="", ms=3, label="any track's stored position at this step"),
               Line2D([], [], marker="o", color="tab:blue", ls="", label="named B track, observed at this step"), Line2D([], [], marker="o", mfc="none", mec="tab:red", ls="", label="named track, last earlier position (hollow, with age)"),
               Line2D([], [], color="k", ls=":", label="history across a gap"), Patch(facecolor="0.82", label="masked: westerly > 2.5 m/s or below the coarse threshold"),
               Patch(facecolor="white", hatch="xx", edgecolor="0.7", label="outside the run's detection domain")]
    figures = []
    for fig, axes, mesh, page in pages:
        axes[0, 0].legend(handles=handles, loc="upper left", fontsize=6.5, framealpha=0.9)
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.3, pad=0.01)
        cbar.set_label("prepared curvature-vorticity anomaly (s^-1), cyclonic positive", fontsize=8)
        fig.suptitle(f"{year}, {fam['name']}, page {page}: B tracks {fam.get('control_tracks', [])}, C tracks {fam.get('treatment_tracks', [])}, {fam['window'][0]} to {fam['window'][1]}", fontsize=10)
        path = os.path.join(out_dir, f"family_{fam['name']}_{year}_p{page}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
        figures.append({"page": page, "figure": os.path.relpath(path), "figure_sha256": X.digest(path)})

    def first(pred):
        return next((r["date"] for r in rows if pred(r)), None)
    divergence = {"prepared_field_first_differs": first(lambda r: r["comparison"]["prepared_field"]["cells_differing"] > 0),
                  "prepared_inputs_first_differ": {name: first(lambda r, n=name: r["comparison"]["prepared_inputs"][n]["cells_differing"] > 0)
                                                   for name, _k, _g in PREPARED_INPUTS},
                  "prepared_inputs_westernmost_differing_lon": {name: min((r["comparison"]["prepared_inputs"][name]["westernmost_differing_lon"] for r in rows
                                                                            if r["comparison"]["prepared_inputs"][name]["westernmost_differing_lon"] is not None), default=None)
                                                                for name, _k, _g in PREPARED_INPUTS},
                  "axes_first_unmatched_within_match_deg": first(lambda r: r["comparison"]["axes"]["b_vertices_without_c_counterpart"] + r["comparison"]["axes"]["c_vertices_without_b_counterpart"] > 0),
                  "final_candidates_first_unmatched_within_match_deg": first(lambda r: r["comparison"]["final_candidates"]["only_b"] or r["comparison"]["final_candidates"]["only_c"]),
                  "note": "axes and candidates are compared by counterparts within match_deg, not by exact equality, so a vertex moved by less than that is not a difference here; association is not recomputed, and the track rows read finished (smoothed, speed-filtered) positions"}
    return {"name": fam["name"], "rule": fam.get("rule"), "window": fam["window"], "box": box, "steps": steps, "figures": figures,
            "first_divergence": divergence, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--families", "--control-run", "--control-case", "--treatment-run", "--treatment-case", "--out-dir"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    try:
        os.mkdir(args.out_dir)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out_dir} exists and figure sets are never overwritten")
    fams = json.load(open(args.families))
    runs = {"B": P.load_run(args.control_run, year=args.year), "C": P.load_run(args.treatment_run, year=args.year)}
    cases = {"B": load_case(args.control_case, runs["B"]["record"]), "C": load_case(args.treatment_case, runs["C"]["record"])}
    coast = coastlines()
    results = [run_family(f, runs, cases, coast, args.out_dir, args.year) for f in fams["families"]]
    out = {"generated_by": "scripts/pilot_stage_diagnostic.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
           "families_file": os.path.relpath(args.families), "families_sha256": X.digest(args.families),
           "runs": {r: {"dir": d, "record_sha256": runs[r]["record_sha256"], "tracks_sha256": runs[r]["tracks_sha256"], "case_sha256": cases[r]["sha256"],
                        "coarse_threshold": runs[r]["record"]["dataset_specific"]["coarse_threshold"], "fine_threshold": runs[r]["record"]["dataset_specific"]["fine_threshold"]}
                    for r, d in (("B", args.control_run), ("C", args.treatment_run))},
           "thresholds_equal": {k: runs["B"]["record"]["dataset_specific"][k] == runs["C"]["record"]["dataset_specific"][k] for k in ("coarse_threshold", "fine_threshold")},
           "prepared_inputs_compared": [name for name, _k, _g in PREPARED_INPUTS],
           "match_deg": MATCH_DEG, "common_east_deg": COMMON_EAST_DEG, "families": results}
    X.publish_json(os.path.join(args.out_dir, f"stage_diagnostic_{args.year}.json"), out, exclusive=True)
    for f in results:
        print(f"{f['name']}: {len(f['steps'])} steps, {len(f['figures'])} page(s); first divergence {f['first_divergence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
