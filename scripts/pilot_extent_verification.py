#!/usr/bin/env python3
"""The verification that must pass before a track comparison between two domains is read
as an EXTENT effect and nothing else. Two wide cases, the narrower domain's and the
wider's, built from the same inputs and the same climatology, with the wider's grids
containing the narrower's in order. At every step the raw fields must be equal on every
shared cell of both grids, and the two all-edge margin preparations (each smoothed on its
own wide case and cropped to its own control grids) must be equal on every cell of the
narrower control's grids, so that the narrower domain's prepared fields are the wider
domain's prepared fields cut at the narrower edge. The two replay records must apply one
threshold pair from one calibration audit, and the record says whether that pair was
transferred to the wider domain.

THE READING. `extent_is_the_only_difference` is true only when every count is zero and
every declared equality holds. It is published either way, and the command refuses
(after publishing) when it is false, so that a document cannot read an extent effect from
a comparison this record does not support. The verification says nothing about the
tracks themselves.

    python3 scripts/pilot_extent_verification.py --year 1990 \\
        --wide-a <40 E wide dir> --control-run-a <B dir> --control-case-a <B case> --replay-a <40 E margin replay json.gz> \\
        --wide-b <60 E wide dir> --control-run-b <C dir> --control-case-b <C case> --replay-b <60 E margin replay json.gz> --out <fresh json>
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_alledge_replay as A  # noqa: E402
import pilot_margin_compare as MC  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

RAW = ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c", "u", "v", "currv_anom")


def contains_in_order(outer_lat, outer_lon, inner_lat, inner_lon):
    """Whether the inner coordinates are a subset of the outer ones in order, with the
    outer rows and columns that are the inner ones."""
    rows, cols = np.isin(outer_lat, inner_lat), np.isin(outer_lon, inner_lon)
    ok = bool(np.array_equal(outer_lat[rows], inner_lat) and np.array_equal(outer_lon[cols], inner_lon))
    return ok, rows, cols


def differing(a, b):
    """Cells that differ, a NaN against a NaN not counted."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return int(np.count_nonzero((np.isnan(a) != np.isnan(b)) | (~np.isnan(a) & ~np.isnan(b) & (a != b))))


def verify(wide_a, case_a, wide_b, case_b):
    """The grids, the raw fields on the shared wide cells, and the two margin preparations
    on the narrower control's cells, at every step."""
    ok_c, rows_c, cols_c = contains_in_order(wide_b["lat_c"], wide_b["lon_c"], wide_a["lat_c"], wide_a["lon_c"])
    ok_f, rows_f, cols_f = contains_in_order(wide_b["latgrid"][:, 0], wide_b["longrid"][0], wide_a["latgrid"][:, 0], wide_a["longrid"][0])
    if not (ok_c and ok_f):
        raise SystemExit("REFUSED: the narrower wide case's grids are not a subset of the wider's in order")
    ok_cc, rows_cc, cols_cc = contains_in_order(case_b["lat_c"], case_b["lon_c"], case_a["lat_c"], case_a["lon_c"])
    ok_cf, rows_cf, cols_cf = contains_in_order(case_b["latgrid"][:, 0], case_b["longrid"][0], case_a["latgrid"][:, 0], case_a["longrid"][0])
    if not (ok_cc and ok_cf):
        raise SystemExit("REFUSED: the narrower control's grids are not a subset of the wider control's in order")
    if not np.array_equal(np.asarray(wide_a["time"]).ravel(), np.asarray(wide_b["time"]).ravel()):
        raise SystemExit("REFUSED: the two wide cases do not share their steps")
    raw = {}
    for name in RAW:
        b = wide_b[name][:, rows_c, :][:, :, cols_c] if name.endswith("_c") else wide_b[name][:, rows_f, :][:, :, cols_f]
        raw[name] = differing(wide_a[name], b)
    sel_a, sel_b = A.selection(case_a, wide_a), A.selection(case_b, wide_b)
    prepared = {name: 0 for name, _grid in A.PREPARED}
    steps = int(np.asarray(wide_a["time"]).size)
    for k in range(steps):
        fa, fb = A.prepared_fields(wide_a, k, *sel_a), A.prepared_fields(wide_b, k, *sel_b)
        for name, grid in A.PREPARED:
            bb = fb[name][rows_cc, :][:, cols_cc] if grid == "coarse" else fb[name][rows_cf, :][:, cols_cf]
            prepared[name] += differing(fa[name], bb)
    return {"grids": {"wide_b_contains_wide_a_in_order": True, "control_b_contains_control_a_in_order": True,
                      "shared_wide_cells": {"coarse": [int(rows_c.sum()), int(cols_c.sum())], "fine": [int(rows_f.sum()), int(cols_f.sum())]},
                      "control_a_cells": {"coarse": [int(rows_cc.sum()), int(cols_cc.sum())], "fine": [int(rows_cf.sum()), int(cols_cf.sum())]}},
            "steps": steps, "raw_shared_cells_differing": raw, "prepared_control_a_cells_differing": prepared,
            "extent_is_the_only_difference": all(v == 0 for v in raw.values()) and all(v == 0 for v in prepared.values())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--wide-a", "--control-run-a", "--control-case-a", "--replay-a", "--wide-b", "--control-run-b", "--control-case-b", "--replay-b", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    replay_a, sha_ra = MC.load_gz(args.replay_a)
    replay_b, sha_rb = MC.load_gz(args.replay_b)
    declared = {}
    declared["both_margin"] = replay_a["preparation"] == "margin" and replay_b["preparation"] == "margin"
    declared["one_applied_pair"] = MC.applied(replay_a) == MC.applied(replay_b)
    ca, cb = replay_a["thresholds"].get("calibration_audit"), replay_b["thresholds"].get("calibration_audit")
    declared["one_calibration_audit"] = bool(ca and cb and ca["sha256"] == cb["sha256"])
    declared["transfer"] = {"a": replay_a["thresholds"].get("transfer"), "b": replay_b["thresholds"].get("transfer")}
    declared["same_tracker_flags"] = replay_a["tracker_flags"] == replay_b["tracker_flags"]
    declared["same_year"] = replay_a["year"] == replay_b["year"] == args.year
    run_a, run_b = P.load_run(args.control_run_a, year=args.year), P.load_run(args.control_run_b, year=args.year)
    case_a, case_b = SD.load_case(args.control_case_a, run_a["record"]), SD.load_case(args.control_case_b, run_b["record"])
    wide_a, wide_b = A.load_wide(args.wide_a, args.year), A.load_wide(args.wide_b, args.year)
    for label, replay, run, case, wide in (("a", replay_a, run_a, case_a, wide_a), ("b", replay_b, run_b, case_b, wide_b)):
        if replay["runs"]["B"]["case_sha256"] != case["sha256"] or replay["runs"]["wide"]["case_sha256"] != wide["sha256"] or replay["runs"]["B"]["record_sha256"] != run["record_sha256"]:
            raise SystemExit(f"REFUSED: replay {label} was not made from control run {label}, its case and its wide case")
    ra, rb = wide_a["record"], wide_b["record"]
    declared["same_inputs"] = ra["inputs_sha256"] == rb["inputs_sha256"]
    declared["same_climatology"] = ra["climo_cache_sha256"] == rb["climo_cache_sha256"]
    declared["boxes"] = {"a": ra["box"], "b": rb["box"], "same_latitudes_and_western_edge": ra["box"]["lat"] == rb["box"]["lat"] and ra["box"]["lon"][0] == rb["box"]["lon"][0]}
    declared["control_domains"] = {"a": run_a["domain"], "b": run_b["domain"]}
    t0 = time.perf_counter()
    result = verify(wide_a, case_a, wide_b, case_b)
    elapsed = time.perf_counter() - t0
    holds = result["extent_is_the_only_difference"] and all(declared[k] for k in ("both_margin", "one_applied_pair", "one_calibration_audit", "same_tracker_flags", "same_year", "same_inputs", "same_climatology")) and declared["boxes"]["same_latitudes_and_western_edge"]
    result["extent_is_the_only_difference"] = bool(holds)
    out = {"generated_by": "scripts/pilot_extent_verification.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
           "inputs": {"replay_a": {"path": args.replay_a, "sha256": sha_ra}, "replay_b": {"path": args.replay_b, "sha256": sha_rb},
                      "wide_a": {"dir": args.wide_a, "record_sha256": wide_a["record_sha256"], "case_sha256": wide_a["sha256"]},
                      "wide_b": {"dir": args.wide_b, "record_sha256": wide_b["record_sha256"], "case_sha256": wide_b["sha256"]},
                      "control_a": {"dir": args.control_run_a, "record_sha256": run_a["record_sha256"], "case_sha256": case_a["sha256"]},
                      "control_b": {"dir": args.control_run_b, "record_sha256": run_b["record_sha256"], "case_sha256": case_b["sha256"]}},
           "applied_thresholds": {"a": replay_a["thresholds"], "b": replay_b["thresholds"]}, "declared": declared, **result, "elapsed_seconds": round(elapsed, 1)}
    X.publish_json(args.out, out, exclusive=True)
    print(f"{args.year}: extent is the only difference {holds}; raw shared cells differing {result['raw_shared_cells_differing']}; "
          f"prepared cells differing on control a {result['prepared_control_a_cells_differing']}; declared {declared}; {elapsed:.0f} s")
    if not holds:
        raise SystemExit("REFUSED: the two margin replays differ in more than the extent, so no extent effect can be read from their comparison (record published)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
