#!/usr/bin/env python3
"""Pair 63, replacing the feature's axis at 33030.50, with an unchanged-replacement control.

THE EXPERIMENT CONTRACT, stated here rather than cited. Measurement is the candidate after
the COARSE MERGE and nothing later. One candidate is replaced in place, at the same index,
with the rest of the population and the ordering untouched. The control replaces that
candidate BY ITSELF, so any difference is the geometry rather than the act of substituting,
and it must reproduce the baseline exactly.

THE MEASUREMENT IS STAGE 2, the candidate after the coarse merge. Not an observation, not a
track. The feature dies in the coarse merge and that is what this runs.

Three runs, each differing only in candidate 65 of 77:

    baseline      untouched, the port's own axis centroid
    control       REPLACED BY ITSELF, the same value at the same index
    replacement   version 1's axis centroid for the same feature

The control exists so that a difference between baseline and replacement can be attributed
to the geometry rather than to the act of substituting. It must reproduce the baseline
exactly, and the script refuses if it does not.

It writes `merge_cases.mat` for `scripts/octave/run_merge.m`, so version 1's own merge can
be run on the identical three inputs.

    .venv/bin/python3 scripts/pair63_axis_replacement.py --out <dir>
    AEW_REPO=<repo> AEW_ORACLE_DIR=<dir> octave --no-gui --quiet scripts/octave/run_merge.m
    .venv/bin/python3 scripts/pair63_axis_replacement.py --out <dir> --report
"""

import argparse
import glob
import gzip
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
LOGS = "docs/aewc_v2/evidence/reference_log_*.log.gz"
STEP = 33030.5
FEATURE = (-14.16, -20.97)
# Version 1's published coarse candidate for this feature, and the tolerance the brief sets.
TARGET = (-14.0, -19.0)
TOLERANCE = 0.6


def masked_fields(case, index):
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid, longrid = np.meshgrid(lat_c, lon_c, indexing="ij")
    coarse_threshold, _ = P.thresholds_for("ERA-Int", 700)
    wind = D.smooth9(np.asarray(case["u_c"][index], dtype=float))
    curvature = D._prepare(np.asarray(case["currv_anom_c"][index], dtype=float), lat_c)
    advection = D.smooth9(np.asarray(case["advcurrv_anom_c"][index], dtype=float))
    westerly = wind > D.MAX_ZONAL_WIND
    advection = np.where(westerly, np.nan, advection)
    curvature = np.where(westerly, np.nan, curvature)
    weak = curvature < coarse_threshold
    advection = np.where(weak, np.nan, advection)
    curvature = np.where(weak, np.nan, curvature)
    return latgrid, longrid, curvature, advection, coarse_threshold


def version1_axis():
    """Version 1's dumped axis for this feature, the AXISPTS centroid nearest the port's."""
    best = None
    for path in sorted(glob.glob(LOGS)):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                parts = line.split()
                if not parts or parts[0] != "AXISPTS":
                    continue
                if abs(float(parts[1]) - STEP) > 1e-6:
                    continue
                vals = [float(x) for x in parts[3:]]
                lat, lon = np.array(vals[0::2]), np.array(vals[1::2])
                gap = float(np.hypot(lat.mean() - FEATURE[0], lon.mean() - FEATURE[1]))
                if best is None or gap < best[0]:
                    best = (gap, path, lat, lon)
    if best is None:
        raise SystemExit("no version 1 axis dump at this step in the retained logs")
    return best


def stages(candidates, watch, latgrid, longrid, curvature, threshold):
    """Pass by pass, what becomes of candidate `watch`. Bound to merge_contours at the end."""
    masks = CT._binary_masks(curvature, threshold)
    rows, cols = np.nonzero(masks[0])
    base_lat, base_lon = latgrid[rows, cols], longrid[rows, cols]
    grown = []
    for here in candidates:
        d2 = (here["lat_mean"] - base_lat) ** 2 + (here["lon_mean"] - base_lon) ** 2
        seed = int(np.argmin(d2))
        region = CT._select_region(masks, (rows[seed], cols[seed]), latgrid, longrid)
        lats, lons = latgrid[region], longrid[region]
        grown.append({"time": here["time"],
                      "lat_mean": float(np.median(lats)) if lats.size else here["lat_mean"],
                      "lon_mean": float(np.median(lons)) if lons.size else here["lon_mean"],
                      "region": region, "lat_wave": lats, "lon_wave": lons})

    kept, pending, dropped_by = [], list(range(len(grown))), {}
    while pending:
        first, others = pending[0], pending[1:]
        kept.append(first)
        close = set()
        if others:
            sep = great_circle_distance(
                grown[first]["lat_mean"], grown[first]["lon_mean"],
                np.array([grown[k]["lat_mean"] for k in others]),
                np.array([grown[k]["lon_mean"] for k in others]), "nm") / 60.0
            close = {others[k] for k in np.flatnonzero(sep <= CT.MERGE_DISTANCE_DEG)}
            for k in close:
                dropped_by.setdefault(k, first)
        pending = [k for k in pending if k != first and k not in close]

    report = {
        "pass1": (grown[watch]["lat_mean"], grown[watch]["lon_mean"]),
        "region_points": int(grown[watch]["lat_wave"].size),
        "pass2_kept": watch in kept,
    }
    if watch in dropped_by:
        other = dropped_by[watch]
        report["pass2_remover"] = (grown[other]["lat_mean"], grown[other]["lon_mean"])
        report["pass2_remover_points"] = int(grown[other]["lat_wave"].size)
        report["pass2_separation"] = float(great_circle_distance(
            grown[other]["lat_mean"], grown[other]["lon_mean"],
            grown[watch]["lat_mean"], grown[watch]["lon_mean"], "nm") / 60.0)
    out = CT.merge_contours(candidates, latgrid, longrid, curvature, threshold)
    report["output"] = [(w["lat_mean"], w["lon_mean"]) for w in out]
    return report


def found(points):
    """The nearest output wave to version 1's published coarse candidate, if within tolerance."""
    if not points:
        return None, float("nan")
    d = [float(np.hypot(a - TARGET[0], b - TARGET[1])) for a, b in points]
    k = int(np.argmin(d))
    return (points[k] if d[k] <= TOLERANCE else None), d[k]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory for the Octave exchange")
    ap.add_argument("--report", action="store_true",
                    help="read version 1's Octave output and report both sides")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    raw = sio.loadmat(CASE)
    case = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    times = np.asarray(raw["time"]).ravel()
    index = int(np.argmin(np.abs(times - STEP)))
    latgrid, longrid, curvature, advection, threshold = masked_fields(case, index)

    axes = D.trough_axes(latgrid, longrid, advection)
    base = [{"time": STEP, "lat_mean": float(np.mean(a)), "lon_mean": float(np.mean(b))}
            for a, b in axes]
    watch = next(n for n, c in enumerate(base)
                 if abs(c["lat_mean"] - FEATURE[0]) < 0.05
                 and abs(c["lon_mean"] - FEATURE[1]) < 0.05)

    gap, path, v1_lat, v1_lon = version1_axis()
    v1 = (float(v1_lat.mean()), float(v1_lon.mean()))
    print(f"step {STEP}, {len(base)} candidates, feature at index {watch}")
    print(f"  port's axis      : {len(axes[watch][0])} vertices, centroid "
          f"({base[watch]['lat_mean']:.3f}, {base[watch]['lon_mean']:.3f})")
    print(f"  version 1's axis : {len(v1_lat)} vertices, centroid ({v1[0]:.3f}, {v1[1]:.3f}), "
          f"{gap:.3f} deg away, from {os.path.basename(path)}")

    runs = {}
    runs["baseline"] = [dict(c) for c in base]
    runs["control"] = [dict(c) for c in base]
    runs["control"][watch] = dict(base[watch])            # replaced by itself, same index
    runs["replacement"] = [dict(c) for c in base]
    runs["replacement"][watch] = {"time": STEP, "lat_mean": v1[0], "lon_mean": v1[1]}

    if not args.report:
        cases = {}
        for name, cands in runs.items():
            cases[name] = {
                "time": STEP, "latgrid": latgrid, "longrid": longrid,
                "cRVt": curvature, "thr": float(threshold),
                "cand_lat": np.array([c["lat_mean"] for c in cands], dtype=float),
                "cand_lon": np.array([c["lon_mean"] for c in cands], dtype=float),
                "cand_time": np.array([c["time"] for c in cands], dtype=float),
            }
        sio.savemat(os.path.join(args.out, "merge_cases.mat"), cases, do_compression=True)
        print(f"\nwrote the three runs to {args.out}/merge_cases.mat for Octave")

    print("\nTHE PORT'S MERGE, stage by stage for the feature's candidate")
    port = {}
    for name in ("baseline", "control", "replacement"):
        r = stages(runs[name], watch, latgrid, longrid, curvature, threshold)
        port[name] = r
        hit, nearest = found(r["output"])
        print(f"\n  {name}")
        print(f"    stage 1, candidate handed in : "
              f"({runs[name][watch]['lat_mean']:.3f}, {runs[name][watch]['lon_mean']:.3f})")
        print(f"    pass 1, grown region         : {r['region_points']} points, "
              f"center ({r['pass1'][0]:.2f}, {r['pass1'][1]:.2f})")
        if r["pass2_kept"]:
            print(f"    pass 2, duplicate removal    : KEPT")
        else:
            print(f"    pass 2, duplicate removal    : DROPPED by the wave at "
                  f"({r['pass2_remover'][0]:.2f}, {r['pass2_remover'][1]:.2f}), "
                  f"{r['pass2_remover_points']} point(s), "
                  f"{r['pass2_separation']:.2f} deg away")
        print(f"    stage 2, coarse merge output : {len(r['output'])} waves, "
              f"feature {'PRESENT at (%.2f, %.2f)' % hit if hit else 'ABSENT'}"
              f" (nearest {nearest:.2f} deg)")

    if port["control"]["output"] != port["baseline"]["output"]:
        raise SystemExit("\nCONTROL FAILED. The unchanged replacement changed the output, so "
                         "the instrument is at fault and the replacement result means nothing.")
    print("\n  CONTROL PASSES. The unchanged replacement reproduces the baseline exactly, "
          f"{len(port['baseline']['output'])} waves.")

    if args.report:
        oct_path = os.path.join(args.out, "merge_octave_out.mat")
        if not os.path.exists(oct_path):
            raise SystemExit(f"no Octave output at {oct_path}, run run_merge.m first")
        o = sio.loadmat(oct_path, squeeze_me=True, struct_as_record=False)
        print("\nVERSION 1'S OWN MERGE, same three inputs, final output only")
        v1out = {}
        for name in ("baseline", "control", "replacement"):
            e = o[name]
            pts = list(zip(np.atleast_1d(e.lat).tolist(), np.atleast_1d(e.lon).tolist()))
            v1out[name] = pts
            hit, nearest = found(pts)
            print(f"  {name:12s}: {len(pts)} waves, feature "
                  f"{'PRESENT at (%.2f, %.2f)' % hit if hit else 'ABSENT'} "
                  f"(nearest {nearest:.2f} deg)")
        if v1out["control"] != v1out["baseline"]:
            raise SystemExit("\nCONTROL FAILED on version 1's side.")
        print("  CONTROL PASSES on version 1's side too.")
        for name in ("baseline", "control", "replacement"):
            a = sorted(port[name]["output"])
            b = sorted(v1out[name])
            agree = len(a) == len(b) and np.allclose(a, b)
            print(f"  {name:12s}: port and version 1 agree exactly: {agree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
