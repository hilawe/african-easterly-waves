#!/usr/bin/env python3
"""What version 1's captured merge input explains at pair 63's step.

THE EXPERIMENT CONTRACT, stated here rather than cited. Measurement is the candidate after
the COARSE MERGE. Two checks run before any comparison: the added recording must leave version
1's finished tracks unchanged, and the standalone merge must reproduce the captured coarse
output from the capture alone. Four runs then differ only as described below, and the control
must reproduce the baseline exactly.

THE INPUT IS CAPTURED, NOT INFERRED. `scripts/make_instrumented_v1.py` now emits `POTWV`,
`MERGETHR`, `CRVT` and `CRVTSHAPE`, which are the actual arguments of version 1's coarse
`merge_contours_f` call at the dumped step, in order and at full precision. The `AXIS`
records that existed before are emitted under version 1's first guard only and are a
SUPERSET of the merge input, and reading them as the input produced a wrong answer.

Two checks come before any comparison, both run by `--verify`:

    the added recording leaves version 1's finished tracks unchanged
    the standalone merge_contours_f reproduces the captured COARSE output from the capture

Then four runs through the PORT's merge, differing only as the brief sets out.

    .venv/bin/python3 scripts/pair63_merge_input.py --log <dir>/potwv_run.log
"""

import argparse
import glob
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aew.v1port import contours as CT  # noqa: E402
from aew.v1port import detection as D  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402

CASE = "docs/aewc_v2/evidence/tracker_case.mat"
STEP = 33030.5
TARGET = (-14.0, -19.0)
TOLERANCE = 0.6
# The port's own indices for the two candidates this step turns on, by centroid.
FEATURE = (-14.158, -20.973)
REMOVER = (-12.990, -15.240)


def read_capture(path):
    """The captured merge input, in order, plus the field, threshold and COARSE records."""
    pot, crvt, coarse, thr, shape = [], [], [], None, None
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "POTWV":
                pot.append((int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
            elif parts[0] == "CRVT":
                crvt.append((int(parts[2]), int(parts[3]), float(parts[6])))
            elif parts[0] == "MERGETHR":
                thr = float(parts[2])
            elif parts[0] == "CRVTSHAPE":
                shape = (int(parts[2]), int(parts[3]))
            elif parts[0] == "COARSE":
                coarse.append((float(parts[2]), float(parts[3])))
    pot.sort()
    if [r[0] for r in pot] != list(range(1, len(pot) + 1)):
        raise SystemExit("the POTWV indices are not 1..n in order, so the capture's order "
                         "cannot be trusted and pass 2 depends on it")
    field = np.full(shape, np.nan)
    for row, col, value in crvt:
        field[row - 1, col - 1] = value
    return pot, field, thr, coarse


def port_side(case, index):
    lat_c = case["lat_c"].ravel()
    latgrid, longrid = np.meshgrid(lat_c, case["lon_c"].ravel(), indexing="ij")
    threshold, _ = P.thresholds_for("ERA-Int", 700)
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
    cands = [{"time": STEP, "lat_mean": float(a.mean()), "lon_mean": float(b.mean())}
             for a, b in axes]
    return cands, curvature, latgrid, longrid, threshold


def verify_tracks(instrumented):
    """The added recording must leave version 1's finished tracks unchanged."""
    if not os.path.exists(instrumented):
        return None
    new = sio.loadmat(instrumented)
    n_new = int(np.asarray(new["n"]).ravel()[0])
    verdicts = []
    for path in sorted(glob.glob("docs/aewc_v2/evidence/reference_output_*.mat")):
        ref = sio.loadmat(path)
        same = int(np.asarray(ref["n"]).ravel()[0]) == n_new and all(
            np.array_equal(np.asarray(new[f"{f}{i}"]).ravel(),
                           np.asarray(ref[f"{f}{i}"]).ravel())
            for i in range(n_new) for f in ("lat", "lon", "time"))
        verdicts.append((os.path.basename(path), same))
    return n_new, verdicts


def index_of(cands, target):
    d = [np.hypot(c["lat_mean"] - target[0], c["lon_mean"] - target[1]) for c in cands]
    return int(np.argmin(d))


def run(cands, field, threshold, latgrid, longrid, label):
    out = CT.merge_contours(cands, latgrid, longrid, field, threshold)
    pts = [(w["lat_mean"], w["lon_mean"]) for w in out]
    d = [float(np.hypot(a - TARGET[0], b - TARGET[1])) for a, b in pts]
    k = int(np.argmin(d)) if d else -1
    hit = pts[k] if d and d[k] <= TOLERANCE else None
    print(f"  {label:44s} {len(out):3d} waves   feature "
          f"{'PRESENT at (%.3f, %.3f)' % hit if hit else 'ABSENT'}"
          f"  (nearest {d[k]:.2f} deg)" if d else "")
    return sorted(pts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True, help="the instrumented run's log")
    ap.add_argument("--instrumented-output", default=None,
                    help="tracker_octave_instrumented.mat, for the unchanged-output check")
    args = ap.parse_args(argv)

    pot, v1field, v1thr, coarse = read_capture(args.log)
    raw = sio.loadmat(CASE)
    case = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    times = np.asarray(raw["time"]).ravel()
    index = int(np.argmin(np.abs(times - STEP)))
    cands, curvature, latgrid, longrid, threshold = port_side(case, index)

    checked = verify_tracks(args.instrumented_output) if args.instrumented_output else None
    if checked:
        n_new, verdicts = checked
        ok = all(v for _, v in verdicts)
        print(f"THE ADDED RECORDING LEAVES THE OUTPUT UNCHANGED: {ok} "
              f"({n_new} tracks, against {len(verdicts)} retained references)")

    port_mask, v1_mask = ~np.isnan(curvature), ~np.isnan(v1field)
    same_cells = bool(np.array_equal(port_mask, v1_mask))
    both = port_mask & v1_mask
    same_values = bool(np.array_equal(curvature[both], v1field[both]))
    print(f"THE MASKED CURVATURE FIELDS ARE IDENTICAL: "
          f"{same_cells and same_values} "
          f"({int(port_mask.sum())} cells on both sides, values equal: {same_values})")
    print(f"THE THRESHOLDS AGREE: {v1thr == threshold} "
          f"(version 1 {v1thr:.6e}, port {threshold:.6e})")

    feature, remover = index_of(cands, FEATURE), index_of(cands, REMOVER)
    print(f"\nversion 1 merges {len(pot)} candidates, the port merges {len(cands)}")
    print(f"  in the PORT's list the remover is index {remover} and the feature is "
          f"index {feature}")
    V = np.array([[r[1], r[2]] for r in pot])
    for name, tgt in (("feature", FEATURE), ("remover", REMOVER)):
        d = np.hypot(V[:, 0] - tgt[0], V[:, 1] - tgt[1])
        k = int(np.argmin(d))
        print(f"  the {name}'s nearest version 1 candidate is index {pot[k][0]}, "
              f"{d[k]:.3f} deg away")

    print("\nfour runs through THE PORT's merge, per the brief")
    v1cands = [{"time": r[3], "lat_mean": r[1], "lon_mean": r[2]} for r in pot]
    a = run(cands, curvature, threshold, latgrid, longrid, "A baseline, port's 77, own order")
    run(v1cands, v1field, v1thr, latgrid, longrid, "B version 1's captured input, own order")
    reordered = [dict(c) for c in cands]
    reordered.insert(remover, reordered.pop(feature))
    run(reordered, curvature, threshold, latgrid, longrid,
        "C order only, feature moved before remover")
    control = [dict(c) for c in cands]
    spare = 20 if 20 not in (feature, remover) else 21
    control.insert(5, control.pop(spare))
    d = run(control, curvature, threshold, latgrid, longrid,
            "D control, unrelated move, order preserved")
    print(f"\n  D REPRODUCES THE BASELINE EXACTLY: {d == a}")
    if d != a:
        raise SystemExit("the control changed the output, so run C means nothing")
    print("\n  STAGE 2 ONLY. A restored candidate here is not an observation and not a "
          "finished\n  track, and pair 63 is not closed on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
