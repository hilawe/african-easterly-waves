#!/usr/bin/env python3
"""Retain the six validation windows' reference and port runs, with their provenance.

WHAT IS RETAINED AND WHAT IS NOT. The two finished-track outputs per window are about 36 kB
each and are the evidence every later phase is judged against, so they are kept. The input
field file is about 15 MB per window and is regenerable from the exporter and the archived
data, so it is NOT kept, but its digest is, which is what binds a run to the input it came
from. Keeping the runs without that digest would retain twelve files nobody could tie to
anything.

THE ORACLE MODE IS RECORDED BECAUSE IT DECIDES COMPARABILITY. Version 1 under Octave refuses
find_ews_f's convhull(...,'simplify',true) inside an empty catch, so the archived source
builds every association polygon from a much larger fallback hexagon and produces a
different answer. Every run here used the REPAIRED copy, which is what the 1990 baseline
used. A run of the unrepaired source gave 117 tracks for window A against the repaired
copy's 117 on a different day and is not interchangeable with it; the mode is therefore
stated per run rather than assumed.

    .venv/bin/python3 scripts/retain_validation_runs.py <work-dir>
"""

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

WINDOWS = [("A", 1983, "jul", 724), ("B", 1983, "sep", 972),
           ("C", 1995, "jul", 724), ("D", 1995, "sep", 972),
           ("E", 2007, "jul", 724), ("F", 2007, "sep", 972)]
DEST = "docs/aewc_v2/evidence/validation"
RUNS = {"reference": "tracker_octave_instrumented.mat", "port": "tracker_port.mat"}


def track_count(path):
    from scipy.io import loadmat
    return int(loadmat(path)["n"].ravel()[0])


def octave_facts(log):
    """What the run's own log says about itself, rather than what the caller believes."""
    facts = {}
    if not os.path.exists(log):
        return facts
    text = open(log, errors="replace").read()
    for key, pattern in (("tracks_returned", r"find_ews_f returned (\d+) tracks"),
                         ("seconds", r"returned \d+ tracks in (\d+) s"),
                         ("dump_schedule", r"dumping intermediates at (\d+) timesteps")):
        found = re.search(pattern, text)
        if found:
            facts[key] = int(found.group(1))
    facts["run_digest_line"] = bool(re.search(r"^RUN [0-9a-f]{64}$", text, re.M))
    return facts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work")
    ap.add_argument("--dest", default=DEST)
    args = ap.parse_args(argv)
    os.makedirs(args.dest, exist_ok=True)

    manifest, missing = {}, []
    for name, year, month, start in WINDOWS:
        source = os.path.join(args.work, f"{name}_{year}_{month}")
        entry = {"year": year, "month": month, "start_index": start,
                 "steps": 60, "oracle_mode": "convhull-repaired instrumented copy, "
                                             "inert dump schedule",
                 "octave": octave_facts(os.path.join(source, "octave_instrumented.log"))}
        case = os.path.join(source, "tracker_case.mat")
        if os.path.exists(case):
            # THE INPUT IS NOT RETAINED, ONLY ITS DIGEST. It is 15 MB per window and
            # regenerable; the digest is what binds these runs to it.
            entry["input_not_retained"] = os.path.basename(case)
            entry["input_sha256"] = X.digest(case)
        for role, filename in RUNS.items():
            path = os.path.join(source, filename)
            if not os.path.exists(path):
                missing.append(f"{name}: {filename}")
                continue
            out_name = f"{name}_{year}_{month}_{role}.mat"
            shutil.copy2(path, os.path.join(args.dest, out_name))
            entry[role] = {"file": out_name, "sha256": X.digest(path),
                           "tracks": track_count(path)}
        residual = os.path.join(source, "residuals.json")
        if os.path.exists(residual):
            out_name = f"{name}_{year}_{month}_residuals.json"
            shutil.copy2(residual, os.path.join(args.dest, out_name))
            entry["residuals"] = {"file": out_name, "sha256": X.digest(residual)}
        manifest[name] = entry

    payload = {
        "generated_by": "scripts/retain_validation_runs.py",
        "what_these_are": "the finished-track outputs of both trackers in the six "
                          "preselected validation windows, phase A",
        "oracle_mode_note": "every reference run used the CONVHULL-REPAIRED copy, which is "
                            "what the 1990 baseline used. The archived source under Octave "
                            "builds every association polygon from a fallback hexagon and "
                            "is not interchangeable with it.",
        "input_fields_not_retained": "about 15 MB per window, regenerable from "
                                     "scripts/export_tracker_case.py and the archived data; "
                                     "each window's digest is recorded instead",
        "windows": manifest,
        "missing": missing,
    }
    path = os.path.join(args.dest, "MANIFEST.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    total = sum(1 for e in manifest.values() for r in RUNS if r in e)
    print(f"retained {total} runs across {len(manifest)} windows into {args.dest}")
    for name in sorted(manifest):
        e = manifest[name]
        print(f"  {name}: reference {e.get('reference', {}).get('tracks', '?')} tracks, "
              f"port {e.get('port', {}).get('tracks', '?')} tracks, "
              f"input {e.get('input_sha256', '')[:12]}")
    if missing:
        print(f"  MISSING: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
