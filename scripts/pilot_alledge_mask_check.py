#!/usr/bin/env python3
"""The calibration audit's second half, which asks whether two threshold pairs make the
same mask decisions on the all-edge margin-prepared fields of one season, on BOTH grids,
at the base level and at every level of the merge's threshold ladder, at every step.

The first half is `pilot_alledge_calibration.py`, which recomputes the two percentiles
over the reference period with the smoother applied before the crop, beside the archived
order, and must reproduce the released pair under the archived order before the other
order is read. This half takes the two pairs from that audit's record, the released run's
and the recomputed, after checking the audit's bind and that the pair it audited is this
control run's, and the fields the margin
replay prepares (`pilot_alledge_replay.prepared_fields` on the wide case cropped to the
control grids, and the native preparation beside it), and counts the cells where the
coarse base mask (`curvature < coarse threshold`, as `detect_troughs` masks), the fine base
mask (`curvature < fine threshold`) or any ladder mask (`curvature >= level * threshold`,
as `merge_contours` builds them) differs between the two pairs. The retained coarse-only
checker (`check_threshold_mask_equivalence.py`) cannot do this, because it takes two
coarse thresholds, reads the coarse fields from a case file and prepares them itself on
the cropped grid, so it tests neither the fine threshold nor a margin preparation.

Calibration effects are kept apart from preparation effects. The comparison is always
between two threshold pairs on ONE preparation (the margin), and the same comparison on
the native preparation is recorded beside it, so a reader sees what the recomputed pair
changes on each preparation separately.

    python3 scripts/pilot_alledge_mask_check.py --control-run <B dir> --control-case <B case> --wide-dir <wide dir> --year 1990 \\
        --recomputed <calibration audit json> --out <fresh json>
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_alledge_replay as A  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402


def mask_decisions(curvature, threshold):
    """The base mask the detector applies and the six ladder masks the merge builds."""
    from aew.v1port.contours import THRESHOLD_LADDER, _binary_masks
    base = curvature < threshold                       # NaN compares False, as in detect_troughs
    ladder = _binary_masks(curvature, threshold)
    return base, ladder, THRESHOLD_LADDER


def compare_pairs(fields, pair_a, pair_b):
    """Cells whose decisions differ between the two pairs, per grid, base and per level."""
    out = {}
    for grid, key, ia in (("coarse", "coarse_curvature", 0), ("fine", "fine_curvature", 1)):
        base_a, ladder_a, levels = mask_decisions(fields[key], pair_a[ia])
        base_b, ladder_b, _ = mask_decisions(fields[key], pair_b[ia])
        out[grid] = {"base": int(np.count_nonzero(base_a != base_b)),
                     "ladder": {f"{lv:g}": int(np.count_nonzero(la != lb)) for lv, la, lb in zip(levels, ladder_a, ladder_b)},
                     "cells": int(fields[key].size)}
    return out


def _pair(entry):
    """A (coarse, fine) pair of floats from an audit order entry, a bind list or an artifact."""
    if isinstance(entry, dict) and "coarse" in entry and "fine" in entry:
        c, f = entry["coarse"], entry["fine"]
        return (float(c["threshold"]) if isinstance(c, dict) else float(c), float(f["threshold"]) if isinstance(f, dict) else float(f))
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        return float(entry[0]), float(entry[1])
    return None


def recomputed_pair(rec, released):
    """The smooth-then-crop pair from the calibration audit's record, read only when the
    record is the audit's with both orders, its flag is exactly true, and the archived
    order's pair, the bind's got and released values and the audited artifact's pair all
    equal this control run's pair to the last bit. A flag alone is not evidence."""
    orders = rec.get("orders") if isinstance(rec.get("orders"), dict) else {}
    if rec.get("generated_by") != "scripts/pilot_alledge_calibration.py" or not {"crop_then_smooth", "smooth_then_crop"} <= set(orders):
        raise SystemExit("REFUSED: the recomputed artifact is not the calibration audit's record with both orders")
    bind = rec.get("bind") if isinstance(rec.get("bind"), dict) else {}
    if bind.get("archived_order_reproduces_released_pair") is not True:
        raise SystemExit("REFUSED: the audit's archived order did not reproduce the released pair, so its loop is not the producer's")
    control = (float(released[0]), float(released[1]))
    for name, pair in (("the audit's archived order", _pair(orders["crop_then_smooth"])), ("the bind's got value", _pair(bind.get("got"))),
                       ("the bind's released value", _pair(bind.get("released"))), ("the audited released artifact", _pair(rec.get("released_artifact")))):
        if pair != control:
            raise SystemExit(f"REFUSED: {name} gives {pair}, which is not this control run's pair {control}")
    recomputed = _pair(orders["smooth_then_crop"])
    if recomputed is None:
        raise SystemExit("REFUSED: the audit's smooth_then_crop order carries no pair")
    return recomputed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--control-run", "--control-case", "--wide-dir", "--recomputed", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    run = P.load_run(args.control_run, year=args.year)
    case_b = SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, args.year)
    if wide["record"]["control_run"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the wide case was not bound to this control case")
    with open(args.recomputed, "rb") as fh:
        rec_blob = fh.read()
    rec = json.loads(rec_blob)
    ds = run["record"]["dataset_specific"]
    released = (float(ds["coarse_threshold"]), float(ds["fine_threshold"]))
    recomputed = recomputed_pair(rec, released)
    sel = A.selection(case_b, wide)
    all_c = (np.ones(case_b["lat_c"].size, bool), np.ones(case_b["lon_c"].size, bool))
    all_f = (np.ones(case_b["latgrid"].shape[0], bool), np.ones(case_b["longrid"].shape[1], bool))
    t0 = time.perf_counter()
    totals = {"margin": None, "native": None}
    steps_with_difference = {"margin": {"coarse": 0, "fine": 0}, "native": {"coarse": 0, "fine": 0}}
    for step in range(case_b["time"].size):
        prepared = {"margin": A.prepared_fields(wide, step, *sel), "native": A.prepared_fields(case_b, step, all_c[0], all_c[1], all_f[0], all_f[1])}
        for prep, fields in prepared.items():
            c = compare_pairs(fields, released, recomputed)
            if totals[prep] is None:
                totals[prep] = {g: {"base": 0, "ladder": {k: 0 for k in c[g]["ladder"]}, "cells_per_step": c[g]["cells"]} for g in c}
            for g in c:
                totals[prep][g]["base"] += c[g]["base"]
                for k, v in c[g]["ladder"].items():
                    totals[prep][g]["ladder"][k] += v
                if c[g]["base"] or any(c[g]["ladder"].values()):
                    steps_with_difference[prep][g] += 1
    elapsed = time.perf_counter() - t0
    identical = {prep: all(totals[prep][g]["base"] == 0 and all(v == 0 for v in totals[prep][g]["ladder"].values()) for g in totals[prep]) for prep in totals}
    out = {"generated_by": "scripts/pilot_alledge_mask_check.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
           "runs": {"B": {"dir": args.control_run, "record_sha256": run["record_sha256"], "case_sha256": case_b["sha256"]},
                    "wide": {"dir": args.wide_dir, "record_sha256": wide["record_sha256"], "case_sha256": wide["sha256"]}},
           "recomputed_artifact": {"path": args.recomputed, "sha256": hashlib.sha256(rec_blob).hexdigest(), "order": "smooth_then_crop",
                                   "audited_released_case_id": rec["released_artifact"].get("case_id"), "bind": rec["bind"]},
           "thresholds": {"released": {"coarse": released[0], "fine": released[1]}, "recomputed": {"coarse": recomputed[0], "fine": recomputed[1]},
                          "relative_difference": {"coarse": (recomputed[0] - released[0]) / released[0], "fine": (recomputed[1] - released[1]) / released[1]}},
           "steps": int(case_b["time"].size), "cells_differing_total": totals, "steps_with_any_difference": steps_with_difference,
           "decisions_identical_at_every_step": identical, "elapsed_seconds": round(elapsed, 1),
           "reading": "identical decisions on the margin preparation would mean the recomputed calibration is unchanged in effect on this season's margin-prepared fields. The native column shows the same two pairs on the native preparation, so calibration and preparation effects stay apart."}
    X.publish_json(args.out, out, exclusive=True)
    print(f"{args.year}: thresholds released {released} recomputed {recomputed}; decisions identical at every step: {identical}; "
          f"margin totals {totals['margin']}; native totals {totals['native']}; {elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
