#!/usr/bin/env python3
"""Whether two coarse-threshold numbers make the same mask decisions on a run's retained
detector inputs, at the base threshold and at every level of the merge ladder.

The eastern pilot's input control (B) and treatment (C) carry coarse thresholds that differ
by two units in the last place (B's own calibration against the baseline artifact's number
transferred into C). The coarse threshold enters the detector in two places: the base mask
that discards cells below it (`detection.detect_troughs`) and the ladder of multiples that
`contours.merge_contours` thresholds the field at. This check prepares each timestep of the
case exactly as the detector does (nine-point smoother, cyclonic positive in both
hemispheres, the westerly mask on the smoothed zonal wind) and counts the cells whose
decision differs between the two numbers at the base and at each ladder level, with the
smallest distance of any finite prepared value from any ladder level of the second number.
Zero differences everywhere means the two numbers would not change a single threshold
decision on these fields. It does not make a failed exact gate pass, and it says nothing
about decisions on other fields.

The case file is read once, hashed, and refused unless its digest is the record's, and the
variables are parsed from the hashed bytes.

    python3 scripts/check_threshold_mask_equivalence.py --run <run dir> --case <tracker_case.mat> --year 1990 \\
        --threshold-a <coarse> --threshold-b <coarse> --out <json>
"""
import argparse
import hashlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402


def prepared_coarse(u, anom, lat_c):
    """The coarse anomaly as the detector thresholds it: smoothed, cyclonic positive in
    both hemispheres, NaN where the smoothed zonal wind exceeds the westerly limit."""
    from aew.v1port.climatology import smooth9
    from aew.v1port.detection import MAX_ZONAL_WIND, _prepare
    field = _prepare(np.asarray(anom, float), np.asarray(lat_c, float))
    return np.where(smooth9(np.asarray(u, float)) > MAX_ZONAL_WIND, np.nan, field)


def check(case_blob, lat_c_name, thr_a, thr_b):
    from scipy.io import loadmat
    from aew.v1port.contours import THRESHOLD_LADDER
    ladder = [float(m) for m in THRESHOLD_LADDER]
    head = loadmat(io.BytesIO(case_blob), variable_names=["time", lat_c_name])
    times = np.asarray(head["time"], float).ravel()
    lat_c = np.asarray(head[lat_c_name], float).ravel()
    u_all = loadmat(io.BytesIO(case_blob), variable_names=["u_c"])["u_c"]
    anom_all = loadmat(io.BytesIO(case_blob), variable_names=["currv_anom_c"])["currv_anom_c"]
    base_diff, finite, ladder_diff, min_dist = 0, 0, [0] * len(ladder), float("inf")
    for k in range(times.size):
        f = prepared_coarse(u_all[k], anom_all[k], lat_c)
        ok = np.isfinite(f)
        finite += int(ok.sum())
        # the base mask discards cells BELOW the threshold (detection.py: curvature_c < coarse_threshold)
        base_diff += int(np.count_nonzero((f[ok] < thr_a) != (f[ok] < thr_b)))
        for j, m in enumerate(ladder):
            # the ladder keeps cells AT OR ABOVE each multiple (contours._binary_masks)
            ladder_diff[j] += int(np.count_nonzero((f[ok] >= thr_a * m) != (f[ok] >= thr_b * m)))
            if ok.any():
                min_dist = min(min_dist, float(np.min(np.abs(f[ok] - thr_b * m))))
    return {"timesteps": int(times.size), "finite_prepared_coarse_values": finite, "threshold_a": thr_a, "threshold_b": thr_b,
            "ladder_multipliers": ladder, "base_mask_cells_differing": base_diff, "ladder_mask_cells_differing": ladder_diff,
            "min_abs_distance_to_any_level_of_b": None if not np.isfinite(min_dist) else min_dist,
            "passed": base_diff == 0 and all(d == 0 for d in ladder_diff)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="the run directory whose record names the case")
    ap.add_argument("--case", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--threshold-a", type=float, required=True)
    ap.add_argument("--threshold-b", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run = P.load_run(args.run, year=args.year)
    with open(args.case, "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != run["record"]["dataset_specific"].get("case_sha256"):
        raise SystemExit(f"REFUSED: {args.case} is not the case the record in {args.run} names")
    result = check(blob, "lat_c", args.threshold_a, args.threshold_b)
    out = {"generated_by": "scripts/check_threshold_mask_equivalence.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
           "run": {"dir": args.run, "record_sha256": run["record_sha256"], "tracks_sha256": run["tracks_sha256"]}, "case_sha256": digest,
           "fine_threshold_of_run": run["record"]["dataset_specific"].get("fine_threshold"),
           "what": "base mask (below the threshold is discarded) and the merge ladder's masks (at or above each multiple) on every timestep's prepared coarse anomaly",
           **result}
    try:
        X.publish_json(args.out, out, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    print(f"{args.year}: {'PASSED' if out['passed'] else 'DIFFERENT'}; base {out['base_mask_cells_differing']}, ladder {out['ladder_mask_cells_differing']} "
          f"over {out['finite_prepared_coarse_values']} finite values in {out['timesteps']} steps; min distance to a level {out['min_abs_distance_to_any_level_of_b']:.3e}; wrote {args.out}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
