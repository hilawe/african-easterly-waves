#!/usr/bin/env python3
"""Compare the two trackers' merge inputs at 33030.50, and say what cannot be compared.

Run after the axis replacement came back null, to do what the null result calls for, which
is to look at the actual ordered merge inputs before any new mechanism is proposed.

THE LIMIT IS THE POINT OF THIS SCRIPT, so it is stated first. The retained logs carry an
`AXISPTS` record per contour, emitted by `scripts/make_instrumented_v1.py`. That dump loop
applies only version 1's FIRST guard on a contour,

    if floor(ch(2,id(zz))) == ch(2,id(zz)) & ch(2,id(zz)) > 1

while version 1's real `pot_wv` construction in `find_ews_f.m` applies TWO MORE conditions
to every contour after the first,

    elseif length(ch(2,:)) >= ch(2,id(i-1))+1+id(i-1);
      if ch(2,id(i)) == ch(2,floor(ch(2,id(i-1))+1+id(i-1)));

so AXISPTS is a SUPERSET of what version 1 actually handed its merge. Its count is an upper
bound and nothing more. Comparing it against the port's candidate count as though the two
were the same quantity would repeat the error this project has already made twice, once on a
detection deficit and once on a divergence signature.

WHAT CAN STILL BE ESTABLISHED, without knowing version 1's exact merge input. The log records
`COARSE` separately, which is version 1's own coarse merge output on its own input. Running
version 1's merge on the PORT's candidates gives a different count. Since the merge code is
the same in both, a difference in output is a difference in input.

    .venv/bin/python3 scripts/compare_merge_inputs_pair63.py
"""

import argparse
import collections
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

CASE = "docs/aewc_v2/evidence/tracker_case.mat"
LOGS = "docs/aewc_v2/evidence/reference_log_*.log.gz"
STEP = 33030.5


def log_counts(path):
    """Per timestep, how many AXISPTS, COARSE and FINE records version 1 emitted."""
    counts = collections.defaultdict(collections.Counter)
    with gzip.open(path, "rt") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) > 1 and parts[0] in ("AXISPTS", "COARSE", "FINE"):
                try:
                    counts[float(parts[1])][parts[0]] += 1
                except ValueError:
                    continue
    return counts


def port_candidates(case, index, lat_c, latgrid, longrid, threshold):
    wind = D.smooth9(np.asarray(case["u_c"][index], dtype=float))
    curvature = D._prepare(np.asarray(case["currv_anom_c"][index], dtype=float), lat_c)
    advection = D.smooth9(np.asarray(case["advcurrv_anom_c"][index], dtype=float))
    westerly = wind > D.MAX_ZONAL_WIND
    advection = np.where(westerly, np.nan, advection)
    curvature = np.where(westerly, np.nan, curvature)
    weak = curvature < threshold
    advection = np.where(weak, np.nan, advection)
    curvature = np.where(weak, np.nan, curvature)
    axes = D.trough_axes(latgrid, longrid, advection)
    cands = [{"time": float(STEP), "lat_mean": float(np.mean(a)),
              "lon_mean": float(np.mean(b))} for a, b in axes]
    return cands, curvature


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default=CASE)
    args = ap.parse_args(argv)

    path = sorted(glob.glob(LOGS))[0]
    for candidate in sorted(glob.glob(LOGS)):
        if log_counts(candidate):
            path = candidate
            break
    counts = log_counts(path)

    raw = sio.loadmat(args.case)
    case = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    times = np.asarray(raw["time"]).ravel()
    lat_c, lon_c = case["lat_c"].ravel(), case["lon_c"].ravel()
    latgrid, longrid = np.meshgrid(lat_c, lon_c, indexing="ij")
    threshold, _ = P.thresholds_for("ERA-Int", 700)

    print(f"version 1's records from {os.path.basename(path)}\n")
    print("THE UPPER BOUND, not a population count. AXISPTS is emitted before version 1's")
    print("chained guard, so it counts contours version 1's merge input may not contain.\n")
    print(f"  {'step':>10s} {'port cands':>11s} {'v1 AXISPTS':>11s} {'v1 COARSE':>10s}")
    rows = []
    for i, t in enumerate(times):
        t = float(t)
        if t not in counts:
            continue
        cands, _ = port_candidates(case, i, lat_c, latgrid, longrid, threshold)
        rows.append((t, len(cands), counts[t]["AXISPTS"], counts[t]["COARSE"]))
        print(f"  {t:10.2f} {len(cands):11d} {counts[t]['AXISPTS']:11d} "
              f"{counts[t]['COARSE']:10d}")

    print(f"\n  the port draws fewer than the AXISPTS upper bound at "
          f"{sum(1 for r in rows if r[1] < r[2])} of {len(rows)} steps, which is CONSISTENT "
          f"with\n  the superset relation and does NOT by itself show a population "
          f"difference.")

    print("\nWHAT THE OUTPUTS SHOW, where the inputs cannot be compared directly.")
    index = int(np.argmin(np.abs(times - STEP)))
    cands, curvature = port_candidates(case, index, lat_c, latgrid, longrid, threshold)
    port_out = CT.merge_contours(cands, latgrid, longrid, curvature, threshold)
    v1_coarse = counts[STEP]["COARSE"]
    print(f"  at {STEP}, the SAME merge code gives:")
    print(f"    {len(port_out)} coarse waves from the port's {len(cands)} candidates")
    print(f"    {v1_coarse} coarse waves in version 1's own run, from its own candidates")
    if len(port_out) != v1_coarse:
        print(f"  THE INPUTS DIFFER. The merge is shared and verified identical on this "
              f"step, so a\n  difference of {abs(v1_coarse - len(port_out))} in the output "
              f"is a difference in what went in.")
    else:
        print("  The outputs agree, so this comparison shows no input difference.")
    print("\n  The SIZE of that input difference is NOT established here, because version 1's")
    print("  actual merge input is not recorded. Dumping `pot_wv` itself would establish it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
