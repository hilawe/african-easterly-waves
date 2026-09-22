#!/usr/bin/env python3
"""Exact whole-trajectory comparison and run retention for the final-pair diagnostics.

WHY THIS EXISTS. The first versions of `pair63_finished_track.py` and
`pair30_merge_input.py` compared runs with tolerances: `np.allclose` on timestamps,
`np.round(t, 4)` before intersecting, a 0.6 degree radius for "the same observation", and a
control check that compared only each track's length and first time. Tolerances are how a
diagnostic reports agreement it has not established. Every comparison here is EXACT float
equality over COMPLETE arrays, and a control that does not match REFUSES rather than
reporting a verdict beside it.

Exactness is attainable and was checked before it was required. Version 1's finished track
times are float64 values that appear exactly in the case's own time array, so nothing here
depends on a tolerance to line two runs up.

WHAT A RETAINED RUN CARRIES. A result nobody can bind to its inputs is a number in a
document. Each retained run records the digest of the case file, the digest and basename of
the reference output it was compared against, the reference track's index, the repository
head, and the digest of the script that produced it.
"""

import glob
import hashlib
import json
import os
import subprocess

import numpy as np
from scipy.io import loadmat


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(final):
    """Every track as exact tuples, in a canonical order, for whole-set comparison."""
    out = []
    for tr in final:
        t = tuple(float(x) for x in np.asarray(tr["time"], dtype=float).ravel())
        la = tuple(float(x) for x in np.asarray(tr["meanlat"], dtype=float).ravel())
        lo = tuple(float(x) for x in np.asarray(tr["meanlon"], dtype=float).ravel())
        out.append((t, la, lo))
    return sorted(out)


def describe_difference(a, b, label_a, label_b):
    """Why two run outputs differ, specifically enough to act on."""
    ca, cb = canonical(a), canonical(b)
    if len(ca) != len(cb):
        return (f"{label_a} holds {len(ca)} finished tracks and {label_b} holds {len(cb)}")
    for k, (x, y) in enumerate(zip(ca, cb)):
        if x == y:
            continue
        if len(x[0]) != len(y[0]):
            return (f"track {k} in canonical order has {len(x[0])} observations in "
                    f"{label_a} and {len(y[0])} in {label_b}")
        for j, (ta, tb) in enumerate(zip(x[0], y[0])):
            if ta != tb:
                return (f"track {k}, observation {j}: time {ta!r} in {label_a} and "
                        f"{tb!r} in {label_b}")
        for j in range(len(x[1])):
            if x[1][j] != y[1][j] or x[2][j] != y[2][j]:
                return (f"track {k}, observation {j} at time {x[0][j]}: "
                        f"({x[1][j]!r}, {x[2][j]!r}) in {label_a} and "
                        f"({y[1][j]!r}, {y[2][j]!r}) in {label_b}")
    return None


def require_identical(a, b, label_a, label_b):
    """Refuse unless two runs produced the same complete trajectories, exactly.

    A CONTROL THAT DOES NOT MATCH IS A REFUSAL, NOT A LINE IN A TABLE. If the operation a
    control performs changes the output on its own, nothing the intervention shows can be
    attributed to what the intervention changed, so there is no result to report.
    """
    why = describe_difference(a, b, label_a, label_b)
    if why is not None:
        raise SystemExit(
            f"REFUSED: {label_b} does not reproduce {label_a} exactly. {why}. The "
            f"intervention result is therefore unattributable and is not reported.")
    return True


def reference_output(pattern="docs/aewc_v2/evidence/reference_output_*.mat"):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no reference output matches {pattern}")
    return paths[0]


def reference_track(path, first_time, near_lat, near_lon, span=1.5):
    """Version 1's finished track starting at `first_time` near a position, read exactly."""
    ref = loadmat(path)
    n = int(np.asarray(ref["n"]).ravel()[0])
    for i in range(n):
        t = np.asarray(ref[f"time{i}"], dtype=float).ravel()
        la = np.asarray(ref[f"lat{i}"], dtype=float).ravel()
        lo = np.asarray(ref[f"lon{i}"], dtype=float).ravel()
        if t.size and t[0] == first_time and abs(la[0] - near_lat) <= span \
                and abs(lo[0] - near_lon) <= span * 2:
            return i, t, la, lo
    raise SystemExit(f"{os.path.basename(path)} holds no track starting exactly at "
                     f"{first_time} near ({near_lat}, {near_lon})")


def holds_exactly(final, t, la, lo):
    """Whether some finished track equals the reference track EXACTLY, whole arrays."""
    want = (tuple(float(x) for x in t), tuple(float(x) for x in la),
            tuple(float(x) for x in lo))
    return want in set(canonical(final))


def repository_head(repo="."):
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def save_run(path, runs, case_path, reference_path, reference_index, script_path, extra=None):
    """Retain each run's complete trajectories beside the identities they were judged against."""
    payload = {
        "identity": {
            "case_file": os.path.basename(case_path),
            "case_sha256": digest(case_path),
            "reference_output": os.path.basename(reference_path),
            "reference_output_sha256": digest(reference_path),
            "reference_track_index": reference_index,
            "script": os.path.basename(script_path),
            "script_sha256": digest(script_path),
            "git_head": repository_head(os.path.dirname(os.path.dirname(
                os.path.abspath(script_path)))),
            "comparison": "exact float equality over complete time, meanlat and meanlon "
                          "arrays; no tolerance is applied anywhere in the verdict",
        },
        "runs": {name: [{"time": list(t), "meanlat": list(la), "meanlon": list(lo)}
                        for t, la, lo in canonical(final)]
                 for name, final in runs.items()},
    }
    if extra:
        payload["identity"].update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    return payload["identity"]
