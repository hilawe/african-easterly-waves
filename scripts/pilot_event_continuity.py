#!/usr/bin/env python3
"""The bounded continuity assessment of the documented 2006 event under the frozen 60 E
candidate (evaluation document, section 14). At each of a few steps the detector's
stages, preparation, detection and the two merges, are recomputed on the frozen
candidate's full grids and full axis population from the retained case files, through
the same functions the all-edge replay binds to, under the candidate's preparation and
pair. The final candidates at every step must reproduce the retained trace record's
candidates, position for position, point count for point count and region for region,
or the run refuses, so the intermediate stages read are the frozen candidate's. Only the
display and the readings are cropped to a box.

Beside the stages, the two-dimensional 700 hPa wind the tracker reads is drawn in the
box, and the raw band-mean meridional wind and the cold-cloud fraction are recomputed by
the context instrument from the existing wind file and imagery in a declared band. The
hypothesis examined is declared as a path (a start position, a speed and a direction),
and the record carries, at each step, the stages' content near that path and anywhere
in the box, every competing candidate within a radius of the path, the coarse merge's
region selection for each near-path axis level by level on the threshold ladder with
whether the ladder ran out, and the presence of a feature at a declared longitude before
a declared time. The reader's categories are recorded in the evaluation document, not
here.

    python3 scripts/pilot_event_continuity.py --control-run <dir> --control-case <mat> --wide-dir <dir> --year 2006 \
        --calibration <audit> --domain-calibration --trace <trace.json.gz> --steps 987 991 \
        --box 5 25 10 35 --band 9.5 19.5 --strip 20 28 --hypothesis 14.5 27.0 987 -5.0 \
        --v-file <nc> --imagery <dir> --out <record.json> --png-prefix <path>
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import exact_tracks as X  # noqa: E402
import pilot_alledge_replay as A  # noqa: E402
import pilot_association_replay as R  # noqa: E402
import pilot_case_wind_context as W  # noqa: E402
import pilot_margin_replay as MR  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

EPOCH = dt.datetime(1900, 1, 1)


def stages_from_prepared(fields, t, ct, ft, grids, absorb=False, coarse_clip_radius_deg=None):
    """`detect_from_prepared` with its intermediates kept: the westerly and weak masks,
    the masked coarse fields, the axes, the axis candidates, the coarse merge and the
    final merge. The final list is the same object the replay would produce.
    `coarse_clip_radius_deg` mirrors the replay's experimental clip, coarse merge only."""
    from aew.v1port.contours import merge_contours
    from aew.v1port.detection import MAX_ZONAL_WIND, trough_axes
    wind, curvature_c, advection_c, curvature_f = fields["coarse_zonal_wind_smoothed"], fields["coarse_curvature"], fields["coarse_advection"], fields["fine_curvature"]
    westerly = wind > MAX_ZONAL_WIND
    advection_c = np.where(westerly, np.nan, advection_c)
    curvature_c = np.where(westerly, np.nan, curvature_c)
    weak_c = curvature_c < ct
    advection_c = np.where(weak_c, np.nan, advection_c)
    curvature_c = np.where(weak_c, np.nan, curvature_c)
    curvature_f_m = np.where(curvature_f < ft, np.nan, curvature_f)
    axes = trough_axes(grids["latgrid_c"], grids["longrid_c"], advection_c)
    candidates = [{"time": t, "lat_mean": float(np.mean(la)), "lon_mean": float(np.mean(lo))} for la, lo in axes] if axes else []
    coarse = merge_contours(candidates, grids["latgrid_c"], grids["longrid_c"], curvature_c, ct, absorb=absorb, clip_radius_deg=coarse_clip_radius_deg) if candidates else []
    final = merge_contours(coarse, grids["latgrid_f"], grids["longrid_f"], curvature_f_m, ft, absorb=absorb) if coarse else []
    return {"westerly": westerly, "weak": weak_c, "curvature_masked": curvature_c, "advection_masked": advection_c, "curvature_unmasked": fields["coarse_curvature"],
            "fine_curvature_masked": curvature_f_m, "axes": axes, "axis_candidates": candidates, "coarse": coarse, "final": final}


def signature_of(waves):
    return [(float(w["lat_mean"]), float(w["lon_mean"]), int(np.asarray(w["lat_wave"]).size), R.region_digest(w["region"])) for w in waves]


def trace_signature(step_entry):
    return [(float(c["lat_mean"]), float(c["lon_mean"]), int(c["n_points"]), c["region_sha256"]) for c in step_entry["candidates"]]


def hypothesis_position(h, t):
    """The path's position at time t (days since 1900), from its start and its speed in
    degrees of longitude per day, latitude held."""
    return float(h["lat"]), float(h["lon"] + h["deg_per_day"] * (t - h["t0"]))


def in_box(lat, lon, box):
    return bool(box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1])


def strip_reading(field, latgrid, longrid, box, strip, lat_band):
    """The field's maximum and its position inside the box's latitude band and a
    longitude strip, with the count of finite cells there."""
    sel = (latgrid >= lat_band[0]) & (latgrid <= lat_band[1]) & (longrid >= strip[0]) & (longrid <= strip[1])
    vals = np.where(sel, field, np.nan)
    if not np.isfinite(vals).any():
        return {"finite_cells": 0, "max": None, "max_lat": None, "max_lon": None}
    i = np.nanargmax(vals)
    r, c = np.unravel_index(i, vals.shape)
    return {"finite_cells": int(np.isfinite(vals).sum()), "max": float(vals[r, c]), "max_lat": float(latgrid[r, c]), "max_lon": float(longrid[r, c])}


def holder_of(waves, latgrid, longrid, lat, lon):
    """The merged candidate whose region holds the grid cell nearest (lat, lon), with its
    center, its size, its region's digest and its distance from that position, or None.
    Holding a cell is overlap, not lineage. `ladder_detail` records the lineage."""
    r = int(np.argmin(np.abs(latgrid[:, 0] - lat))); c = int(np.argmin(np.abs(longrid[0, :] - lon)))
    for w in waves:
        if np.asarray(w["region"], bool)[r, c]:
            return {"center_lat": float(w["lat_mean"]), "center_lon": float(w["lon_mean"]), "n_points": int(np.asarray(w["lat_wave"]).size),
                    "center_distance_deg": round(float(np.hypot(float(w["lat_mean"]) - lat, float(w["lon_mean"]) - lon)), 3),
                    "cell": {"lat": float(latgrid[r, 0]), "lon": float(longrid[0, c])}, "region_sha256": R.region_digest(w["region"])}
    return None


def ladder_detail(masked_curvature, ct, lat, lon, latgrid, longrid):
    """What the coarse merge's region selection does for an input candidate at (lat, lon).

    The seed is the above-threshold cell nearest the position, enumerated column first as
    `merge_contours` enumerates it. The connected region seeded there is reported at each
    level of the threshold ladder with its cell count and its latitude and longitude
    extents, the level the cascade lands on is replicated from `_select_region`'s loop and
    checked against the region `_select_region` itself returns, and `exhausted` says
    whether that selected region's extent still reaches the trigger, which is the ladder
    run out with the oversized region retained. The trigger steps the cascade up a level
    and is not enforced on the last level."""
    from aew.v1port.contours import (MAX_LAT_EXTENT_DEG, MAX_LON_EXTENT_DEG, THRESHOLD_LADDER, _binary_masks, _extent, _select_region,
                                     connected_region)
    masks = _binary_masks(masked_curvature, ct)
    cols, rows = np.nonzero(masks[0].T)
    if rows.size == 0:
        return None
    d2 = (lat - latgrid[rows, cols]) ** 2 + (lon - longrid[rows, cols]) ** 2
    i = int(np.argmin(d2)); seed = (int(rows[i]), int(cols[i]))
    regions = [connected_region(m, seed) for m in masks]
    extents = [(_extent(latgrid[r]), _extent(longrid[r])) for r in regions]
    landed = 0
    for level in range(len(regions) - 1):
        if extents[level][0] >= MAX_LAT_EXTENT_DEG or extents[level][1] >= MAX_LON_EXTENT_DEG:
            landed = level + 1
    chosen = _select_region(masks, seed, latgrid, longrid)
    if not np.array_equal(chosen, regions[landed]):
        raise RuntimeError("the replicated cascade does not land on the region _select_region returns")
    sel_lat, sel_lon = _extent(latgrid[chosen]), _extent(longrid[chosen])
    return {"seed_cell": {"lat": float(latgrid[seed]), "lon": float(longrid[seed])},
            "levels": [{"multiple": float(m), "cells": int(r.sum()), "lat_extent": round(e[0], 2), "lon_extent": round(e[1], 2)}
                       for m, r, e in zip(THRESHOLD_LADDER, regions, extents)],
            "level_landed": landed, "last_level": len(regions) - 1, "cascade_reached_last_level": landed == len(regions) - 1,
            "extent_trigger_deg": {"lat": MAX_LAT_EXTENT_DEG, "lon": MAX_LON_EXTENT_DEG},
            "selected": {"cells": int(chosen.sum()), "lat_extent": round(sel_lat, 2), "lon_extent": round(sel_lon, 2), "region_sha256": R.region_digest(chosen)},
            "exhausted": bool(sel_lat >= MAX_LAT_EXTENT_DEG or sel_lon >= MAX_LON_EXTENT_DEG)}


def axis_detail(la, lo, masked_curvature, latgrid, longrid):
    """An axis's vertex count, its latitude and longitude spans, and how many of its
    vertices lie on cells the coarse masks leave above threshold."""
    la, lo = np.asarray(la, float).ravel(), np.asarray(lo, float).ravel()
    on = 0
    for a, o in zip(la, lo):
        r = int(np.argmin(np.abs(latgrid[:, 0] - a))); c = int(np.argmin(np.abs(longrid[0, :] - o)))
        on += bool(np.isfinite(masked_curvature[r, c]))
    return {"n_vertices": int(la.size), "lat_span": [round(float(la.min()), 2), round(float(la.max()), 2)], "lon_span": [round(float(lo.min()), 2), round(float(lo.max()), 2)],
            "vertices_on_above_threshold_cells": int(on)}


def near(points, lat, lon, radius):
    out = []
    for p in points:
        d = float(np.hypot(p[0] - lat, p[1] - lon))
        if d <= radius:
            out.append({"lat": p[0], "lon": p[1], "distance_deg": round(d, 3), **({"n_points": p[2]} if len(p) > 2 else {})})
    return sorted(out, key=lambda x: x["distance_deg"])


def ladder_at_each(st, latgrid, longrid, ct, axes_near):
    """The ladder detail for each near-path axis, with the lineage link: whether the region
    selected for the axis's seed is the region of the surviving coarse candidate that holds
    the axis's cell. Equal digests make the holder the axis's own selection rather than an
    overlap inference."""
    out = []
    for a in axes_near:
        detail = ladder_detail(st["curvature_masked"], ct, a["lat"], a["lon"], latgrid, longrid)
        holder = holder_of(st["coarse"], latgrid, longrid, a["lat"], a["lon"])
        out.append({"axis": a, "ladder": detail,
                    "selected_region_equals_holder_region": None if (detail is None or holder is None) else detail["selected"]["region_sha256"] == holder["region_sha256"]})
    return out


def read_step(st, grids, fine_grids, box, strip, lat_band, hyp, t, ct, ft, radius):
    """What the stages hold at this step, near the hypothesis path and in the box."""
    hlat, hlon = hypothesis_position(hyp, t)
    latgrid_c, longrid_c = grids["latgrid_c"], grids["longrid_c"]
    axis_pts = [(float(np.mean(la)), float(np.mean(lo))) for la, lo in st["axes"]]
    coarse_pts = [(float(w["lat_mean"]), float(w["lon_mean"]), int(np.asarray(w["lat_wave"]).size)) for w in st["coarse"]]
    final_pts = [(float(w["lat_mean"]), float(w["lon_mean"]), int(np.asarray(w["lat_wave"]).size)) for w in st["final"]]
    axes_in_box = [{"lat_mean": a[0], "lon_mean": a[1]} for a in axis_pts if in_box(a[0], a[1], box)]
    return {"hypothesis_position": {"lat": hlat, "lon": hlon},
            "preparation": {"strip_unmasked_curvature": strip_reading(st["curvature_unmasked"], latgrid_c, longrid_c, box, strip, lat_band),
                            "strip_masked_curvature": strip_reading(st["curvature_masked"], latgrid_c, longrid_c, box, strip, lat_band),
                            "strip_westerly_cells": int(((latgrid_c >= lat_band[0]) & (latgrid_c <= lat_band[1]) & (longrid_c >= strip[0]) & (longrid_c <= strip[1]) & st["westerly"]).sum()),
                            "strip_cells": int(((latgrid_c >= lat_band[0]) & (latgrid_c <= lat_band[1]) & (longrid_c >= strip[0]) & (longrid_c <= strip[1])).sum()),
                            "coarse_threshold": ct},
            "detection": {"axes_total": len(st["axes"]), "axes_in_box": axes_in_box, "axes_near_path": near(axis_pts, hlat, hlon, radius),
                          "axes_near_path_detail": [{"lat_mean": a[0], "lon_mean": a[1], "distance_deg": round(float(np.hypot(a[0] - hlat, a[1] - hlon)), 3), **axis_detail(la, lo, st["curvature_masked"], latgrid_c, longrid_c)}
                                                    for (la, lo), a in zip(st["axes"], axis_pts) if np.hypot(a[0] - hlat, a[1] - hlon) <= radius]},
            "merge": {"coarse_total": len(st["coarse"]), "coarse_in_box": [{"lat_mean": c[0], "lon_mean": c[1], "n_points": c[2]} for c in coarse_pts if in_box(c[0], c[1], box)],
                      "coarse_near_path": near(coarse_pts, hlat, hlon, radius),
                      "coarse_region_holding_path_cell": holder_of(st["coarse"], latgrid_c, longrid_c, hlat, hlon),
                      "coarse_region_holding_each_near_path_axis": [{"axis": a, "holder": holder_of(st["coarse"], latgrid_c, longrid_c, a["lat"], a["lon"])} for a in near(axis_pts, hlat, hlon, radius)],
                      "ladder_at_each_near_path_axis": ladder_at_each(st, latgrid_c, longrid_c, ct, near(axis_pts, hlat, hlon, radius)),
                      "final_total": len(st["final"]), "final_in_box": [{"lat_mean": c[0], "lon_mean": c[1], "n_points": c[2]} for c in final_pts if in_box(c[0], c[1], box)],
                      "final_near_path": near(final_pts, hlat, hlon, radius),
                      "fine_strip_masked_curvature": strip_reading(st["fine_curvature_masked"], fine_grids["latgrid_f"], fine_grids["longrid_f"], box, strip, lat_band), "fine_threshold": ft}}


def panel_axes(fig):
    """The step figure's layout: two map panels on top, each with its colorbar in its own
    narrow column, and the two profile panels beneath them spanning exactly the map
    panels' horizontal extent and sharing their x axis, so a longitude on a profile
    lies directly under the same longitude on the map above it (Hilawe, 2026-10-08).
    Returns the four panel axes as a 2 by 2 array and the two colorbar axes."""
    # columns: left map, its colorbar, a spacer for the colorbar's labels, right map, its colorbar
    gs = fig.add_gridspec(2, 5, width_ratios=[1.0, 0.035, 0.26, 1.0, 0.035], wspace=0.08, hspace=0.3)
    top_left, top_right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 3])
    axes = np.array([[top_left, top_right],
                     [fig.add_subplot(gs[1, 0], sharex=top_left), fig.add_subplot(gs[1, 3], sharex=top_right)]])
    caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    return axes, caxes


def draw_step(st, case_fields, grids, box, strip, lat_band, hyp, t, reading, v_profile, cloud_profile, lons_profile, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    latgrid_c, longrid_c = grids["latgrid_c"], grids["longrid_c"]
    lat_c, lon_c = latgrid_c[:, 0], longrid_c[0, :]
    fig = plt.figure(figsize=(14, 10))
    axes, caxes = panel_axes(fig)
    ax = axes[0, 0]
    shown = np.where(st["westerly"], np.nan, st["curvature_unmasked"])
    m = ax.pcolormesh(lon_c, lat_c, np.ma.masked_invalid(shown), cmap="YlOrRd", vmin=0.0, vmax=max(float(np.nanmax(np.where(np.isfinite(shown), shown, 0))) * 0.5, 1e-7), shading="auto")
    fig.colorbar(m, cax=caxes[0], label="coarse curvature anomaly, cyclonic-positive, westerly cells blank (s^-1)")
    weak = np.where(~st["westerly"] & st["weak"], 1.0, np.nan)
    ax.pcolormesh(lon_c, lat_c, np.ma.masked_invalid(weak), cmap="Greys", vmin=0, vmax=3, shading="auto", alpha=0.35)
    for la, lo in st["axes"]:
        ax.plot(lo, la, "-", color="k", lw=0.9)
    for w in st["coarse"]:
        ax.plot(w["lon_mean"], w["lat_mean"], "^", mfc="none", mec="k", ms=7)
    for w in st["final"]:
        ax.plot(w["lon_mean"], w["lat_mean"], "x", color="k", ms=9, mew=1.6)
    hp = reading["hypothesis_position"]
    ax.plot(hp["lon"], hp["lat"], "o", mfc="none", mec="tab:blue", ms=12, mew=2)
    ax.axvspan(strip[0], strip[1], color="tab:blue", alpha=0.06)
    ax.axhline(lat_band[0], color="tab:blue", lw=0.6, ls=":"); ax.axhline(lat_band[1], color="tab:blue", lw=0.6, ls=":")
    ax.set_xlim(*box["lon"]); ax.set_ylim(*box["lat"])
    ax.set_title(f"stages, {len(st['axes'])} axes, {len(st['coarse'])} coarse, {len(st['final'])} final (gray, below the coarse threshold)", fontsize=9)
    ax = axes[0, 1]
    u, v = case_fields["u"], case_fields["v"]
    latgrid_f, longrid_f = case_fields["latgrid"], case_fields["longrid"]
    m2 = ax.pcolormesh(longrid_f[0, :], latgrid_f[:, 0], v, cmap="RdBu_r", vmin=-10, vmax=10, shading="auto")
    fig.colorbar(m2, cax=caxes[1], label="700 hPa meridional wind (m/s)")
    step_q = 2
    ax.quiver(longrid_f[::step_q, ::step_q], latgrid_f[::step_q, ::step_q], u[::step_q, ::step_q], v[::step_q, ::step_q], color="0.25", scale=300, width=0.002)
    for w in st["final"]:
        ax.plot(w["lon_mean"], w["lat_mean"], "x", color="k", ms=9, mew=1.6)
    ax.plot(hp["lon"], hp["lat"], "o", mfc="none", mec="tab:blue", ms=12, mew=2)
    ax.axhline(lat_band[0], color="tab:blue", lw=0.6, ls=":"); ax.axhline(lat_band[1], color="tab:blue", lw=0.6, ls=":")
    ax.set_xlim(*box["lon"]); ax.set_ylim(*box["lat"])
    ax.set_title("two-dimensional 700 hPa wind the tracker reads (fine grid), final candidates as crosses", fontsize=9)
    ax = axes[1, 0]
    ax.plot(lons_profile, v_profile, "-", color="k")
    ax.axhline(0, color="0.5", lw=0.6); ax.axvspan(strip[0], strip[1], color="tab:blue", alpha=0.06); ax.axvline(hp["lon"], color="tab:blue", lw=1.0)
    ax.set_xlim(*box["lon"]); ax.set_xlabel("longitude (degrees east)"); ax.set_ylabel("band-mean v (m/s)")
    ax.set_title(f"raw ERA5 v700, band mean {lat_band[0]:g} to {lat_band[1]:g} N (declared before reading)", fontsize=9)
    ax = axes[1, 1]
    if cloud_profile is None:
        ax.text(0.5, 0.5, "imagery MISSING at this time", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(lons_profile, cloud_profile, "-", color="tab:purple")
        ax.set_ylim(0, 1)
    ax.axvspan(strip[0], strip[1], color="tab:blue", alpha=0.06); ax.axvline(hp["lon"], color="tab:blue", lw=1.0)
    ax.set_xlim(*box["lon"]); ax.set_xlabel("longitude (degrees east)"); ax.set_ylabel("fraction colder than 240 K")
    ax.set_title("GridSat-B1 cold-cloud fraction in the same band (nearest 3-hourly file)", fontsize=9)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.07, wspace=0.08, hspace=0.3)
    alignment = panel_alignment(axes)
    if not alignment["aligned"]:
        plt.close(fig)
        raise RuntimeError(f"REFUSED: a profile panel does not sit under its map's horizontal extent: {alignment}")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return alignment


def panel_alignment(axes, tol=1e-6):
    """Whether each profile panel spans exactly its map's horizontal extent and x range.
    A figure that fails this is refused rather than saved."""
    cols = []
    for j in (0, 1):
        top, bottom = axes[0, j].get_position(), axes[1, j].get_position()
        cols.append({"map_x0": round(top.x0, 4), "map_x1": round(top.x1, 4), "profile_x0": round(bottom.x0, 4), "profile_x1": round(bottom.x1, 4),
                     "map_xlim": [float(x) for x in axes[0, j].get_xlim()], "profile_xlim": [float(x) for x in axes[1, j].get_xlim()]})
    aligned = all(abs(c["map_x0"] - c["profile_x0"]) <= tol and abs(c["map_x1"] - c["profile_x1"]) <= tol and c["map_xlim"] == c["profile_xlim"] for c in cols)
    return {"aligned": bool(aligned), "columns": cols}


def draw_hovmoller(hov_v, hov_c, strip, hyp, steps_t, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    m = axes[0].pcolormesh(hov_v["lons"], hov_v["times"], hov_v["v"], cmap="RdBu_r", vmin=-8, vmax=8, shading="nearest")
    fig.colorbar(m, ax=axes[0], label="band-mean v700 (m/s)")
    axes[0].set_title("raw ERA5 v700, band mean in the declared band", fontsize=9)
    if hov_c["times"].size:
        m2 = axes[1].pcolormesh(hov_c["lons"], hov_c["times"], hov_c["cold_fraction"], cmap="Blues", vmin=0, vmax=1, shading="nearest")
        fig.colorbar(m2, ax=axes[1], label="fraction colder than 240 K")
        for t, cov in zip(hov_c["times"], hov_c["coverage"]):
            if cov != "retained":
                axes[1].axhspan(t - 0.0625, t + 0.0625, color=W.COVERAGE_SHADE.get(cov, "0.85"), lw=0)
    axes[1].set_title("GridSat-B1 cold cloud, band mean (gray rows missing)", fontsize=9)
    for ax in axes:
        ax.axvspan(strip[0], strip[1], color="tab:blue", alpha=0.06)
        path_t = np.array(steps_t); path_lon = [hypothesis_position(hyp, t)[1] for t in path_t]
        ax.plot(path_lon, path_t, "o--", color="tab:blue", ms=5, lw=1.2, label="hypothesis path")
        ax.set_xlabel("longitude (degrees east)")
        ticks = np.arange(np.ceil(hov_v["times"].min()), np.floor(hov_v["times"].max()) + 1)
        ax.set_yticks(ticks); ax.set_yticklabels([(EPOCH + dt.timedelta(days=float(t))).strftime("%m-%d") for t in ticks])
    axes[0].set_ylabel("date (month-day)"); axes[0].legend(loc="lower left")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--control-run", "--control-case", "--wide-dir", "--calibration", "--trace", "--v-file", "--imagery", "--out", "--png-prefix"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--domain-calibration", action="store_true")
    ap.add_argument("--steps", nargs=2, type=int, required=True, metavar=("FIRST", "LAST"))
    ap.add_argument("--box", nargs=4, type=float, required=True, metavar=("LAT0", "LAT1", "LON0", "LON1"))
    ap.add_argument("--band", nargs=2, type=float, required=True, metavar=("LAT0", "LAT1"), help="the declared latitude band of the band means")
    ap.add_argument("--strip", nargs=2, type=float, required=True, metavar=("LON0", "LON1"), help="the longitude strip the stage readings summarize")
    ap.add_argument("--hypothesis", nargs=4, type=float, required=True, metavar=("LAT", "LON", "STEP", "DEG_PER_DAY"), help="the path examined, a start position at a step and a speed")
    ap.add_argument("--radius", type=float, default=10.0, help="degrees within which a candidate or axis counts as competing with the path")
    ap.add_argument("--stationary-lon", type=float, default=22.0); ap.add_argument("--stationary-before-step", type=int, default=990)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    run = P.load_run(args.control_run, year=args.year)
    case_b = SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, args.year)
    if wide["record"]["control_run"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the wide case was not bound to this control case")
    flags = run["record"]["dataset_specific"]["tracker_flags"]; ds = run["record"]["dataset_specific"]
    ct, ft, thresholds = A.thresholds_for_run((float(ds["coarse_threshold"]), float(ds["fine_threshold"])), args.calibration, False, args.domain_calibration, run["domain"])
    grids = MR.control_grids(case_b)
    sel = A.selection(case_b, wide)
    raw = A.raw_equality(case_b, wide, sel)
    if any(raw.values()):
        raise SystemExit(f"REFUSED: the wide case differs from the control case on the control's own cells: {raw}")
    with gzip.open(args.trace, "rb") as fh:
        blob = fh.read()
    trace_rec = json.loads(blob); trace_sha = hashlib.sha256(blob).hexdigest()
    if (trace_rec["thresholds"]["coarse"], trace_rec["thresholds"]["fine"]) != (ct, ft) or trace_rec["runs"]["B"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the trace record is not of this control case under this pair")
    by_step = {s["step"]: s for s in trace_rec["trace"]["steps"]}
    first, last = args.steps
    missing = [k for k in range(first, last + 1) if k not in by_step]
    if missing:
        raise SystemExit(f"REFUSED: the trace record holds no logged step {missing}, so the final candidates there cannot be reproduced against it")
    box = {"lat": [args.box[0], args.box[1]], "lon": [args.box[2], args.box[3]]}
    band, strip = [args.band[0], args.band[1]], [args.strip[0], args.strip[1]]
    times = np.asarray(case_b["time"], float).ravel()
    hyp = {"lat": args.hypothesis[0], "lon": args.hypothesis[1], "t0": float(times[int(args.hypothesis[2])]), "step0": int(args.hypothesis[2]), "deg_per_day": args.hypothesis[3]}
    absorb = bool(flags["absorb"])
    fine_grids = {"latgrid_f": grids["latgrid_f"], "longrid_f": grids["longrid_f"]}
    # the band means and the cloud fraction over the window, from the existing wind file and imagery
    t_first, t_last = float(times[first]), float(times[last])
    window = {"lon": [box["lon"][0], box["lon"][1]], "time": [t_first - 0.75, t_last + 0.75]}
    hov_v = W.v_hovmoller(args.v_file, band, window)
    hov_c = W.cloud_hovmoller(args.imagery, band, window)
    steps_out, gates = [], []
    for k in range(first, last + 1):
        t = float(times[k])
        fields = A.prepared_fields(wide, k, *sel)
        st = stages_from_prepared(fields, t, ct, ft, grids, absorb=absorb)
        check = A.detect_from_prepared(fields, t, ct, ft, grids, absorb=absorb)
        if MR.signature(st["final"]) != MR.signature(check):
            raise SystemExit(f"REFUSED: at step {k} the staged recomputation does not reproduce detect_from_prepared")
        mine, theirs = signature_of(st["final"]), trace_signature(by_step[k])
        gates.append({"step": k, "final_candidates": len(mine), "reproduces_trace": mine == theirs})
        if mine != theirs:
            raise SystemExit(f"REFUSED: at step {k} the final candidates do not reproduce the retained trace ({len(mine)} against {len(theirs)}), so the stages are not the frozen candidate's")
        reading = read_step(st, grids, fine_grids, box, strip, band, hyp, t, ct, ft, args.radius)
        i_t = int(np.argmin(np.abs(hov_v["times"] - t)))
        v_profile = hov_v["v"][i_t]
        j_t = int(np.argmin(np.abs(hov_c["times"] - t))) if hov_c["times"].size else None
        cloud_profile = hov_c["cold_fraction"][j_t] if (j_t is not None and hov_c["coverage"][j_t] == "retained" and abs(float(hov_c["times"][j_t]) - t) <= 0.0626) else None
        lons = np.asarray(hov_v["lons"], float)
        in_strip = (lons >= strip[0]) & (lons <= strip[1])
        vs = np.asarray(v_profile, float)
        reading["band_mean_v"] = {"band": band, "strip_min": float(np.nanmin(vs[in_strip])) if np.isfinite(vs[in_strip]).any() else None,
                                  "strip_min_lon": float(lons[in_strip][int(np.nanargmin(vs[in_strip]))]) if np.isfinite(vs[in_strip]).any() else None,
                                  "strip_max": float(np.nanmax(vs[in_strip])) if np.isfinite(vs[in_strip]).any() else None,
                                  "sign_changes_in_strip": [float(lons[in_strip][i]) for i in range(1, int(in_strip.sum())) if np.sign(vs[in_strip][i - 1]) != np.sign(vs[in_strip][i])],
                                  "profile": [None if not np.isfinite(x) else round(float(x), 2) for x in vs], "lons": lons.tolist()}
        reading["cloud"] = None if cloud_profile is None else {"strip_mean": float(np.nanmean(np.asarray(cloud_profile, float)[(np.asarray(hov_c["lons"], float) >= strip[0]) & (np.asarray(hov_c["lons"], float) <= strip[1])])),
                                                              "profile": [None if not np.isfinite(x) else round(float(x), 3) for x in cloud_profile], "lons": np.asarray(hov_c["lons"], float).tolist()}
        case_fields = {"u": np.asarray(wide["u"][k], float)[sel[2], :][:, sel[3]], "v": np.asarray(wide["v"][k], float)[sel[2], :][:, sel[3]], "latgrid": grids["latgrid_f"], "longrid": grids["longrid_f"]}
        png = f"{args.png_prefix}_step{k}.png"
        date = f"{SD.date_of(t):%Y-%m-%d %HZ}"
        draw_step(st, case_fields, grids, box, strip, band, hyp, t, reading, vs, cloud_profile, lons, png, f"{args.year} continuity assessment, step {k}, {date}, frozen 60 E candidate")
        stationary = {"axes_within_2deg_of_stationary_lon": near([(float(np.mean(la)), float(np.mean(lo))) for la, lo in st["axes"]], hyp["lat"], args.stationary_lon, 2.0),
                      "final_within_2deg_of_stationary_lon": near([(float(w["lat_mean"]), float(w["lon_mean"]), int(np.asarray(w["lat_wave"]).size)) for w in st["final"]], hyp["lat"], args.stationary_lon, 2.0),
                      "band_mean_v_at_stationary_lon": None if not np.isfinite(vs[int(np.argmin(np.abs(lons - args.stationary_lon)))]) else round(float(vs[int(np.argmin(np.abs(lons - args.stationary_lon)))]), 2),
                      "before_declared_step": k < args.stationary_before_step}
        steps_out.append({"step": k, "time": t, "date": date, "figure": {"path": png, "sha256": X.digest(png)}, **reading, "stationary_feature_check": stationary})
    hov_png = f"{args.png_prefix}_hovmoller.png"
    draw_hovmoller(hov_v, hov_c, strip, hyp, [float(times[k]) for k in range(first, last + 1)], hov_png, f"{args.year} band means over the assessment window, band {band[0]:g} to {band[1]:g} N")
    out = {"generated_by": "scripts/pilot_event_continuity.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
           "runs": {"B": {"dir": args.control_run, "record_sha256": run["record_sha256"], "tracks_sha256": run["tracks_sha256"], "case_sha256": case_b["sha256"], "control_domain": run["domain"]},
                    "wide": {"dir": args.wide_dir, "record_sha256": wide["record_sha256"], "case_sha256": wide["sha256"], "box": wide["record"]["box"]}},
           "thresholds": thresholds, "tracker_flags": flags, "trace": {"path": args.trace, "sha256": trace_sha},
           "declared": {"steps": [first, last], "box": box, "band": band, "band_note": "declared before any panel was read, the track's last latitude plus and minus 5 degrees",
                        "strip": strip, "hypothesis": hyp, "radius_deg": args.radius, "stationary_lon": args.stationary_lon, "stationary_before_step": args.stationary_before_step,
                        "reading": "the hypothesis is what is examined, not the rule by which a feature is selected. A different continuation, several candidates or an indeterminate result are possible outcomes. Absence in a band mean alone does not establish physical discontinuity. The reader's categories are recorded in the evaluation document."},
           "gates": gates, "wind_file": {"path": args.v_file, "sha256": hov_v["file_sha256"]}, "imagery": {"dir": args.imagery, "files": hov_c["files"], "summary": hov_c["summary"]},
           "hovmoller": {"figure": {"path": hov_png, "sha256": X.digest(hov_png)}, "times": hov_v["times"].tolist(), "lons": hov_v["lons"].tolist(), "v_band_mean": np.round(hov_v["v"], 3).tolist(),
                         "cloud_times": hov_c["times"].tolist(), "cloud_lons": hov_c["lons"].tolist(), "cloud_coverage": hov_c["coverage"],
                         "cold_fraction": [[None if not np.isfinite(x) else round(float(x), 3) for x in row] for row in hov_c["cold_fraction"]]},
           "steps": steps_out}
    X.publish_json(args.out, out, exclusive=True)
    for s in steps_out:
        hp = s["hypothesis_position"]
        print(f"{s['date']}: path at {hp['lat']:.1f}N {hp['lon']:.1f}E. Axes in box {len(s['detection']['axes_in_box'])}, near path {len(s['detection']['axes_near_path'])}. "
              f"Final in box {len(s['merge']['final_in_box'])}, near path {len(s['merge']['final_near_path'])}. Strip unmasked curvature max {s['preparation']['strip_unmasked_curvature']['max']} "
              f"at {s['preparation']['strip_unmasked_curvature']['max_lon']}E, masked finite cells {s['preparation']['strip_masked_curvature']['finite_cells']} of {s['preparation']['strip_cells']}. "
              f"Band v min {s['band_mean_v']['strip_min']} at {s['band_mean_v']['strip_min_lon']}E. Cloud {'missing' if s['cloud'] is None else round(s['cloud']['strip_mean'], 2)}. "
              f"Ladder at near-path axes: {[(e['ladder']['level_landed'], e['ladder']['selected']['cells'], e['ladder']['selected']['lat_extent'], e['ladder']['selected']['lon_extent'], e['ladder']['exhausted'], e['selected_region_equals_holder_region']) for e in s['merge']['ladder_at_each_near_path_axis'] if e['ladder']]}.")
    print(f"Gates {gates}. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
