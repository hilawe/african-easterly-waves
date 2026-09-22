#!/usr/bin/env python3
"""Trace pair 63's missing candidate through the port's real merge passes.

Pair 63 differs from its version 1 counterpart at one step, 33030.50, where version 1
holds a candidate at 13.0S 19.0W and the port does not. An earlier probe fed single axes
through the merges and concluded the merge rule was not at fault. That conclusion was
unsound. A lone candidate hits the early return in merge_contours,

    if len(merged) <= 1:
        return merged

so an axis-alone run never reaches duplicate removal or the extent test, and it cannot
say anything about what the full population does. This probe runs the real passes over
the real population instead.

Two guards keep it honest. The masking is asserted to reproduce detect_troughs exactly,
because an earlier version of this probe omitted the below-threshold mask on the
advection field and produced 65 axes where the port produces 77. And the replicated
passes are bound to contours.merge_contours, so a divergence stops the run rather than
being reported as a measurement.

Run it with the project virtualenv from the repository root:

    .venv/bin/python3 scripts/trace_pair63_merge.py
"""

import argparse
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aew.v1port import contours as CT  # noqa: E402
from aew.v1port import detection as D  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port.geometry import great_circle_distance  # noqa: E402

CASE = "docs/aewc_v2/evidence/tracker_case.mat"
STEP = 33030.5
# The port's own axis for the feature, as its centroid before any merging.
FEATURE = (-14.16, -20.97)
# What version 1 publishes here, from its dumped candidates.
V1_COARSE = (-14.0, -19.0)
V1_FINE = (-13.0, -19.0)


def masked_fields(case, index):
    """Reproduce the masking detect_troughs applies, coarse grid and fine grid."""
    lat_c = case["lat_c"].ravel()
    lon_c = case["lon_c"].ravel()
    latgrid, longrid = np.meshgrid(lat_c, lon_c, indexing="ij")
    lat_f = np.asarray(case["latgrid"], dtype=float)[:, 0]
    coarse_threshold, fine_threshold = P.thresholds_for("ERA-Int", 700)

    wind = D.smooth9(np.asarray(case["u_c"][index], dtype=float))
    curvature = D._prepare(np.asarray(case["currv_anom_c"][index], dtype=float), lat_c)
    advection = D.smooth9(np.asarray(case["advcurrv_anom_c"][index], dtype=float))
    fine = D._prepare(np.asarray(case["currv_anom"][index], dtype=float), lat_f)

    westerly = wind > D.MAX_ZONAL_WIND
    advection = np.where(westerly, np.nan, advection)
    curvature = np.where(westerly, np.nan, curvature)
    weak = curvature < coarse_threshold
    advection = np.where(weak, np.nan, advection)
    curvature = np.where(weak, np.nan, curvature)
    fine = np.where(fine < fine_threshold, np.nan, fine)
    return {
        "latgrid": latgrid, "longrid": longrid,
        "curvature": curvature, "advection": advection,
        "fine_lat": np.asarray(case["latgrid"]), "fine_lon": np.asarray(case["longrid"]),
        "fine": fine,
        "coarse_threshold": coarse_threshold, "fine_threshold": fine_threshold,
    }


def assert_baseline(case, index, field, candidates):
    """The untouched reconstruction must equal what detect_troughs returns."""
    real = D.detect_troughs(
        STEP, field["latgrid"], field["longrid"], case["u_c"][index],
        case["currv_anom_c"][index], case["advcurrv_anom_c"][index],
        case["latgrid"], case["longrid"], case["currv_anom"][index],
        coarse_threshold=field["coarse_threshold"],
        fine_threshold=field["fine_threshold"], absorb=False)
    mine = CT.merge_contours(candidates, field["latgrid"], field["longrid"],
                             field["curvature"], field["coarse_threshold"])
    mine = CT.merge_contours(mine, field["fine_lat"], field["fine_lon"],
                             field["fine"], field["fine_threshold"])
    if len(real) != len(mine):
        raise AssertionError(
            f"the reconstruction gives {len(mine)} waves, detect_troughs gives {len(real)}")
    for a, b in zip(real, mine):
        if abs(a["lat_mean"] - b["lat_mean"]) > 1e-12 or abs(a["lon_mean"] - b["lon_mean"]) > 1e-12:
            raise AssertionError("the reconstruction disagrees with detect_troughs")
    return len(real)


def grow(candidates, field):
    """Pass one, which absorbs nothing and replaces each center with a region median."""
    masks = CT._binary_masks(field["curvature"], field["coarse_threshold"])
    rows, cols = np.nonzero(masks[0])
    base_lat, base_lon = field["latgrid"][rows, cols], field["longrid"][rows, cols]
    grown = []
    for here in candidates:
        d2 = (here["lat_mean"] - base_lat) ** 2 + (here["lon_mean"] - base_lon) ** 2
        seed = int(np.argmin(d2))
        region = CT._select_region(masks, (rows[seed], cols[seed]),
                                   field["latgrid"], field["longrid"])
        lats, lons = field["latgrid"][region], field["longrid"][region]
        grown.append({
            "time": here["time"],
            "lat_mean": float(np.median(lats)) if lats.size else here["lat_mean"],
            "lon_mean": float(np.median(lons)) if lons.size else here["lon_mean"],
            "region": region, "lat_wave": lats, "lon_wave": lons,
        })
    return grown


def duplicate_removal(grown):
    """Pass two, returning the kept indices and who dropped each removed wave."""
    kept, pending, dropped_by = [], list(range(len(grown))), {}
    while pending:
        first, others = pending[0], pending[1:]
        kept.append(first)
        close = set()
        if others:
            separation = great_circle_distance(
                grown[first]["lat_mean"], grown[first]["lon_mean"],
                np.array([grown[k]["lat_mean"] for k in others]),
                np.array([grown[k]["lon_mean"] for k in others]), "nm") / 60.0
            close = {others[k] for k in np.flatnonzero(separation <= CT.MERGE_DISTANCE_DEG)}
            for k in close:
                dropped_by.setdefault(k, first)
        pending = [k for k in pending if k != first and k not in close]
    return kept, dropped_by


def extent(wave):
    """The north to south span pass three tests, in degrees."""
    lats, lons = wave["lat_wave"], wave["lon_wave"]
    if lats.size == 0:
        return 0.0
    top, bottom = lats == lats.max(), lats == lats.min()
    top_lon = (lons[top].min() + lons[top].max()) / 2.0
    bottom_lon = (lons[bottom].min() + lons[bottom].max()) / 2.0
    return float(great_circle_distance(lats.min(), bottom_lon,
                                       lats.max(), top_lon, "nm") / 60.0)


def refined(wave, field):
    """The pass three center refinement, which pulls the center toward the peak anomaly."""
    values = field["curvature"][wave["region"]]
    peak = values == np.nanmax(values)
    if not peak.any():
        return wave["lat_mean"], wave["lon_mean"]
    lats, lons = wave["lat_wave"], wave["lon_wave"]
    return ((float(np.mean(lats[peak])) + wave["lat_mean"]) / 2.0,
            (float(np.mean(lons[peak])) + wave["lon_mean"]) / 2.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default=CASE)
    args = ap.parse_args(argv)

    raw = sio.loadmat(args.case)
    case = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    times = np.asarray(raw["time"]).ravel()
    index = int(np.argmin(np.abs(times - STEP)))

    field = masked_fields(case, index)
    axes = D.trough_axes(field["latgrid"], field["longrid"], field["advection"])
    candidates = [{"time": STEP, "lat_mean": float(np.mean(a)), "lon_mean": float(np.mean(b))}
                  for a, b in axes]

    waves = assert_baseline(case, index, field, candidates)
    print(f"baseline verified against detect_troughs: {len(candidates)} axes, {waves} waves")

    watch = next(n for n, c in enumerate(candidates)
                 if abs(c["lat_mean"] - FEATURE[0]) < 0.05
                 and abs(c["lon_mean"] - FEATURE[1]) < 0.05)
    grown = grow(candidates, field)
    print(f"\npass 1 absorbs nothing, leaving {len(grown)} waves from {len(candidates)} axes.")
    print(f"  the feature's axis centroid is "
          f"({candidates[watch]['lat_mean']:.2f}, {candidates[watch]['lon_mean']:.2f})")
    print(f"  its region of {grown[watch]['lat_wave'].size} points has median "
          f"({grown[watch]['lat_mean']:.2f}, {grown[watch]['lon_mean']:.2f})")

    kept, dropped_by = duplicate_removal(grown)
    if watch in kept:
        print("\npass 2 keeps the feature. Nothing further to report here.")
        return 0
    killer = dropped_by[watch]
    separation = float(great_circle_distance(
        grown[killer]["lat_mean"], grown[killer]["lon_mean"],
        grown[watch]["lat_mean"], grown[watch]["lon_mean"], "nm") / 60.0)
    raw_separation = float(great_circle_distance(
        grown[killer]["lat_mean"], grown[killer]["lon_mean"],
        candidates[watch]["lat_mean"], candidates[watch]["lon_mean"], "nm") / 60.0)

    print(f"\npass 2 keeps {len(kept)} of {len(grown)}, at a merge distance of "
          f"{CT.MERGE_DISTANCE_DEG} degrees.")
    print(f"  THE FEATURE IS DROPPED AS A DUPLICATE of the wave at "
          f"({grown[killer]['lat_mean']:.2f}, {grown[killer]['lon_mean']:.2f}), "
          f"separation {separation:.2f} degrees")
    print(f"  that keeper grew from a region of {grown[killer]['lat_wave'].size} point(s)")
    print(f"  had pass 1 not moved the center, the separation would be "
          f"{raw_separation:.2f} degrees, outside the merge distance")

    print("\npass 3 would then test both for extent, minimum "
          f"{CT.MIN_EXTENT_DEG} degrees north to south.")
    for name, wave in (("the keeper", grown[killer]), ("the feature", grown[watch])):
        span = extent(wave)
        verdict = "passes" if span >= CT.MIN_EXTENT_DEG else "REJECTED"
        print(f"  {name:12s}: {wave['lat_wave'].size:2d} point(s), span {span:.2f} degrees, {verdict}")

    lat, lon = refined(grown[watch], field)
    print(f"\nthe feature's coarse center, had it survived, would refine to "
          f"({lat:.2f}, {lon:.2f})")
    print(f"version 1 publishes coarse {V1_COARSE} and fine {V1_FINE}")
    print("the fine merge has not been run on the lifted wave, so whether lifting the drop "
          "reproduces\nversion 1's observation is NOT established here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
