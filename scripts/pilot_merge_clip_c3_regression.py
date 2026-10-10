#!/usr/bin/env python3
"""Criterion E for C3, resolved from retained records after the fact.

The contract (`MERGE_EXHAUSTION_PROPOSAL_2026-10-08.md`, section 5, E) asks, for the
converging eastern family, whether the frozen candidate's final candidates AT THE STORED
POSITIONS OF THE FOUR HISTORIES (B1052, B1053, C1207, C1208) remain within 2 degrees under
the variant. The experiment runner measured every baseline candidate in the family box
instead, which is the C4 form of the criterion. This script resolves the C3 form from the
retained second-version records and the two runs' stored tracks, without rerunning any
detection. At each step of the window and for each history with an observation at that
time, it counts the baseline and variant final candidates within 2 coordinate degrees of
the stored position. A position is assessable when the baseline has at least one
candidate within 2 degrees, and it remains when the variant has at least one too.

    python3 scripts/pilot_merge_clip_c3_regression.py --baseline <r2.json.gz> --variant <r2.json.gz> \
        --control-run <dir> --control-case <mat> --year 2006 --history B:data/protocol_runs/pilot_B/era5_2006:1052 \
        ... --steps 889 898 --out <record.json>
"""
import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import exact_tracks as X  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

MATCH_DEG = 2.0


def within_count(finals, lat, lon, radius=MATCH_DEG):
    return int(sum(1 for f in finals if float(np.hypot(float(f[0]) - lat, float(f[1]) - lon)) <= radius))


def remained_at(lat, lon, finals_b, finals_v, radius=MATCH_DEG):
    """E at one stored position: assessable when the baseline has a candidate within the
    radius, remained when the variant has one too. An unassessable position is neither
    a pass nor a failure."""
    nb, nv = within_count(finals_b, lat, lon, radius), within_count(finals_v, lat, lon, radius)
    return {"baseline_within_2": nb, "variant_within_2": nv, "assessable": nb > 0, "remained": (nb > 0 and nv > 0)}


def observation_at(track, t, tol=1e-6):
    times = np.asarray(track["time"], float).ravel()
    hit = np.flatnonzero(np.abs(times - t) < tol)
    if hit.size == 0:
        return None
    i = int(hit[0])
    return float(np.asarray(track["lat"], float).ravel()[i]), float(np.asarray(track["lon"], float).ravel()[i])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--baseline", "--variant", "--control-run", "--control-case", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--history", action="append", required=True, metavar="LABEL:RUN_DIR:INDEX", help="a stored history, label, run directory and index")
    ap.add_argument("--steps", nargs=2, type=int, required=True, metavar=("FIRST", "LAST"))
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    recs, shas = {}, {}
    for key, path in (("baseline", args.baseline), ("variant", args.variant)):
        with gzip.open(path, "rb") as fh:
            blob = fh.read()
        recs[key] = json.loads(blob); shas[key] = hashlib.sha256(blob).hexdigest()
    if recs["variant"]["baseline"]["sha256"] != shas["baseline"]:
        raise SystemExit("REFUSED: the variant record is not bound to this baseline record")
    run = P.load_run(args.control_run, year=args.year)
    case_b = SD.load_case(args.control_case, run["record"])
    if recs["baseline"]["runs"]["B"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the baseline record is not of this control case")
    times = np.asarray(case_b["time"], float).ravel()
    histories = {}
    for spec in args.history:
        label, run_dir, index = spec.rsplit(":", 2)
        r = P.load_run(run_dir, year=args.year)
        histories[label] = {"run_dir": run_dir, "tracks_sha256": r["tracks_sha256"], "index": int(index), "track": r["tracks"][int(index)]}
    first, last = args.steps
    rows = []
    for k in range(first, last + 1):
        t = float(times[k]); sk = str(k)
        if sk not in recs["baseline"]["steps"] or sk not in recs["variant"]["steps"]:
            raise SystemExit(f"REFUSED: step {k} is not in both records")
        fb, fv = recs["baseline"]["steps"][sk]["final"], recs["variant"]["steps"][sk]["final"]
        for label, h in histories.items():
            obs = observation_at(h["track"], t)
            if obs is None:
                continue
            rows.append({"step": k, "date": recs["baseline"]["steps"][sk]["date"], "history": label, "stored": [obs[0], obs[1]], **remained_at(obs[0], obs[1], fb, fv)})
    assessable = [r for r in rows if r["assessable"]]
    failures = [r for r in assessable if not r["remained"]]
    out = {"generated_by": "scripts/pilot_merge_clip_c3_regression.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
           "criterion": "E for C3 as the contract states it, the frozen candidate's final candidates at the stored positions of the four histories remaining within 2 degrees under the variant",
           "records": {"baseline": {"path": args.baseline, "sha256": shas["baseline"]}, "variant": {"path": args.variant, "sha256": shas["variant"]}},
           "histories": {label: {k2: v for k2, v in h.items() if k2 != "track"} for label, h in histories.items()},
           "steps": [first, last], "match_deg": MATCH_DEG, "rows": rows,
           "summary": {"stored_positions_in_window": len(rows), "assessable": len(assessable), "remained": len(assessable) - len(failures), "failures": failures,
                       "unassessable_positions": [{"step": r["step"], "history": r["history"], "stored": r["stored"]} for r in rows if not r["assessable"]]}}
    X.publish_json(args.out, out, exclusive=True)
    s = out["summary"]
    print(f"Stored positions in the window: {s['stored_positions_in_window']}, assessable (baseline candidate within 2 degrees): {s['assessable']}, remained under the variant: {s['remained']}, failures: {len(s['failures'])}, unassessable: {len(s['unassessable_positions'])}. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
