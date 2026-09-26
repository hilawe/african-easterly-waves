#!/usr/bin/env python3
"""Retain what the season pilot's runs actually did: timings from the runner's own log
lines, the digests of the inputs and outputs, and whether two runs agree exactly.

A review found the pilot page stating two Octave timings and a second run's equality with
nothing retained behind either. This reads each run's log for the runner's own line
`find_ews_f returned N tracks in S s`, digests the case file and the output on each side,
compares the two outputs' complete tracks under the exact comparison, and publishes the
record exclusively. It computes nothing about the science.

    .venv/bin/python3 scripts/season_run_record.py --run <dir>:<log> --run <dir>:<log> --out <json>
"""

import argparse
import hashlib
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import season_metrics as M  # noqa: E402

OUTPUT = "tracker_octave_instrumented.mat"
CASE = "tracker_case.mat"
LINE = re.compile(r"find_ews_f returned (\d+) tracks in (\d+) s")


def _digest_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_run(directory, log_path):
    """One run: its runner line, its case and output digests, and its tracks."""
    with open(log_path, "rb") as fh:
        log = fh.read()
    hits = LINE.findall(log.decode(errors="replace"))
    if len(hits) != 1:
        raise SystemExit(f"REFUSED: {log_path} carries {len(hits)} runner lines, not one")
    with open(os.path.join(directory, OUTPUT), "rb") as fh:
        out_bytes = fh.read()
    tracks, case = M.read_mat_tracks(out_bytes)
    if int(hits[0][0]) != len(tracks):
        raise SystemExit(f"REFUSED: the log says {hits[0][0]} tracks and the output holds {len(tracks)}")
    return {"directory": os.path.abspath(directory), "log": os.path.abspath(log_path),
            "log_sha256": hashlib.sha256(log).hexdigest(),
            "case_sha256": _digest_file(os.path.join(directory, CASE)),
            "output_sha256": hashlib.sha256(out_bytes).hexdigest(),
            "case_id": case, "tracks": len(tracks), "runner_seconds": int(hits[0][1]),
            "_canonical": X.canonical([{"time": t["time"], "meanlat": t["lat"], "meanlon": t["lon"]} for t in tracks])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="<directory>:<log path>")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    runs = []
    for spec in args.run:
        directory, log_path = spec.split(":", 1)
        runs.append(read_run(directory, log_path))
    same_case = len({r["case_sha256"] for r in runs}) == 1
    same_tracks = len({str(r["_canonical"]) for r in runs}) == 1
    payload = {"generated_by": "scripts/season_run_record.py",
               "script_sha256": X.digest(__file__),
               "git_head_at_launch": X.repository_head(
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
               "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
               "same_case_bytes": same_case,
               "outputs_byte_identical": len({r["output_sha256"] for r in runs}) == 1,
               "tracks_identical_under_exact_comparison": same_tracks,
               "comparison": X.COMPARISON}
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    for r in payload["runs"]:
        print(f"  {r['tracks']} tracks in {r['runner_seconds']} s, case {r['case_sha256'][:12]}, output {r['output_sha256'][:12]}")
    print(f"same case {same_case}, tracks identical {same_tracks}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
