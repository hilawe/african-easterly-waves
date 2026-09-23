#!/usr/bin/env python3
"""Register Phase A: the residue statistics for all six preselected windows.

WHAT PHASE A IS FOR. The case investigations are adaptive, so the baseline must not be.
These numbers are produced and written down for all six windows BEFORE any item is
investigated, which is what stops the method being tuned on one window and applied to the
rest. They also need no intervention at all, so they are the one comparison that carries
none of the machinery the later phases depend on.

IT REPORTS, IT DOES NOT JUDGE. There is no pass mark here and the proposal declined to
invent one. The quantity of interest is whether the residue DISTRIBUTION resembles the 1990
window's, and that is a comparison a reader makes from the table rather than a verdict this
script returns.

    .venv/bin/python3 scripts/validation_phase_a_summary.py <work-dir> --out <artifact>
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

# The six, fixed in the proposal before anything ran. The 1990 window is carried beside them
# as the reference distribution, and is NOT one of the six.
WINDOWS = [("A", 1983, "jul"), ("B", 1983, "sep"), ("C", 1995, "jul"),
           ("D", 1995, "sep"), ("E", 2007, "jul"), ("F", 2007, "sep")]
BASELINE_1990 = "docs/aewc_v2/artifacts/tracker_oracle_residuals_2026-09-18.json"


def summarize(residuals):
    """The registered statistics, all of them counts over the residual record."""
    pairs = residuals["pairs"]
    extra = collections.Counter(p["extra_kind"] for p in pairs)
    steps = collections.Counter()
    for p in pairs:
        n = int((p.get("v1_extra") or {}).get("n") or 0)
        if n:
            steps[n] += 1
    return {
        "nonidentical_pairs": len(pairs),
        # EVERY KIND THE RECORD CARRIES, not a chosen subset. A first version named three
        # and silently dropped `extra_port_only`, which does not occur in the 1990 window at
        # all and occurs in five of the six validation windows. Hardcoding the categories the
        # training window happens to contain is how a validation misses what it was for.
        "pairs_by_extra_kind": {k: extra[k] for k in sorted(extra)},
        "pairs_with_v1_extra_observations": extra.get("extra_v1_only", 0),
        "pairs_with_port_extra_observations": extra.get("extra_port_only", 0),
        "pairs_with_extra_on_both_sides": extra.get("extra_both_sides", 0),
        "pairs_displaced_only": extra.get("no_extra", 0),
        "v1_unmatched_tracks": len(residuals.get("v1_unmatched") or []),
        # AN INT IN ONE ARTIFACT AND A LIST IN ANOTHER. The 1990 residual record carries this
        # as a count and the new ones as a count too; an earlier version here assumed a list
        # and crashed on the first window rather than reporting a wrong number, which is the
        # better of the two failures.
        "v1_unmatched_with_eligible_counterpart":
            (lambda v: len(v) if isinstance(v, list) else v)(
                residuals.get("v1_unmatched_with_eligible_counterpart")),
        "port_unmatched_tracks": len(residuals.get("port_unmatched") or []),
        "pairs_with_a_step_over_one_degree":
            residuals.get("pairs_with_a_step_over_one_degree"),
        "worst_individual_step_deg": residuals.get("worst_individual_step_deg"),
        "divergence_step_counts": {str(k): steps[k] for k in sorted(steps)},
        "total_v1_extra_observations":
            sum(int((p.get("v1_extra") or {}).get("n") or 0) for p in pairs),
    }


def track_counts(path):
    """How many tracks each side produced, read from the residual record's own inputs."""
    from scipy.io import loadmat
    out = {}
    for side, name in (("port", "tracker_port.mat"),
                       ("version1", "tracker_octave_instrumented.mat")):
        full = os.path.join(path, name)
        if os.path.exists(full):
            out[side] = int(loadmat(full)["n"].ravel()[0])
    return out


def findings(rows, reference):
    """What the rows show about the 1990 window, stated as consequences of the counts."""
    ref = summarize(reference)
    with_port_extra = sorted(k for k, r in rows.items()
                             if r["pairs_with_port_extra_observations"])
    with_both = sorted(k for k, r in rows.items() if r["pairs_with_extra_on_both_sides"])
    like_1990 = sorted(k for k, r in rows.items()
                       if not r["pairs_with_port_extra_observations"]
                       and not r["pairs_with_extra_on_both_sides"])
    out = []
    if ref["pairs_with_port_extra_observations"] == 0 and with_port_extra:
        out.append(
            f"THE 1990 WINDOW HAS NO PAIR WHOSE EXTRA OBSERVATIONS ARE ON THE PORT'S SIDE, "
            f"and {len(with_port_extra)} of {len(rows)} validation windows do "
            f"({', '.join(with_port_extra)}).")
    if ref["pairs_with_extra_on_both_sides"] == 0 and with_both:
        out.append(
            f"THE 1990 WINDOW HAS NO PAIR WITH EXTRAS ON BOTH SIDES, and "
            f"{len(with_both)} of {len(rows)} validation windows do "
            f"({', '.join(with_both)}).")
    if like_1990:
        out.append(f"Windows resembling 1990's category mix: {', '.join(like_1990)}.")
    if with_port_extra:
        out.append(
            "A STRUCTURAL CONSEQUENCE, verified against the residual records. For a "
            "port-only pair the reference holds NO extra observation, so v1_extra.times is "
            "empty, so the divergence-step set the accounting contract defines for a pair "
            "is EMPTY, and experiment_problems refuses with 'the case declares no "
            "divergence step'. THE CONTRACT AS IT STANDS CANNOT EXPRESS SUCH A PAIR AT ALL, "
            "whatever intervention might explain it. Injection is not merely expected to "
            "fail on these, it cannot be constructed, since there is no divergence step to "
            "inject at.")
        out.append(
            "SO THE RESIDUE WALK ADDRESSED ONE SHAPE OF DIFFERENCE. Every credited case in "
            "1990 was a pair where the reference held observations the port lacked, or an "
            "unmatched reference track. Two categories that appear in most validation "
            "windows were absent from the window every instrument was built against.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows, missing = {}, []
    for name, year, month in WINDOWS:
        directory = os.path.join(args.work, f"{name}_{year}_{month}")
        residual = os.path.join(directory, "residuals.json")
        if not os.path.exists(residual):
            missing.append(f"{name} ({year} {month})")
            continue
        data = json.load(open(residual))
        rows[name] = {
            "year": year, "month": month,
            "case_id": data.get("case_id"),
            "oracle_file": data.get("oracle_file"),
            "tracks": track_counts(directory),
            "residuals_sha256": X.digest(residual),
            **summarize(data),
        }

    reference = json.load(open(BASELINE_1990))
    payload = {
        "generated_by": "scripts/validation_phase_a_summary.py",
        "phase": "A, the baseline comparison, registered before any case investigation",
        "windows_declared": [f"{n} ({y} {m})" for n, y, m in WINDOWS],
        "windows_not_run": missing,
        "reference_window_1990": {
            "note": "NOT one of the six. The window every instrument was built against, "
                    "carried here as the distribution the six are compared with.",
            "source": os.path.basename(BASELINE_1990),
            "oracle_file": reference.get("oracle_file"),
            **summarize(reference),
        },
        "windows": rows,
        # THE FINDING, computed from the rows rather than narrated, so it regenerates with
        # them and cannot drift from the numbers it rests on.
        "findings": findings(rows, reference),
        "what_this_does_not_do": [
            "it does not judge; no pass mark was declared, because inventing one now would "
            "be a number chosen to be met",
            "it involves no intervention and no control, so nothing here credits or refuses "
            "any mechanism",
            "a window that could not be run is listed as not run and is never replaced by a "
            "neighbor",
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)

    ref = payload["reference_window_1990"]
    print(f"PHASE A, {len(rows)} of {len(WINDOWS)} windows"
          + (f", NOT RUN: {', '.join(missing)}" if missing else ""))
    print(f"\n  {'window':>8} {'v1':>4} {'port':>5} | {'pairs':>5} {'v1only':>6} "
          f"{'PORTonly':>8} {'both':>4} {'displ':>5} | {'v1unm':>5} {'portunm':>7}")
    def line(label, row):
        t = row.get("tracks", {})
        print(f"  {label:>8} {t.get('version1', '?'):>4} {t.get('port', '?'):>5} | "
              f"{row['nonidentical_pairs']:>5} {row['pairs_with_v1_extra_observations']:>6} "
              f"{row['pairs_with_port_extra_observations']:>8} "
              f"{row['pairs_with_extra_on_both_sides']:>4} {row['pairs_displaced_only']:>5} | "
              f"{row['v1_unmatched_tracks']:>5} {row['port_unmatched_tracks']:>7}")
    line("1990 ref", {**ref, "tracks": {"version1": 123, "port": 114}})
    for name, _, _ in WINDOWS:
        if name in rows:
            line(f"{name}", rows[name])
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
