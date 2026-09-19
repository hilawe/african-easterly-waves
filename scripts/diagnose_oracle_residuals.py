#!/usr/bin/env python3
"""Classify what separates version 1's tracks from the port's on the identical-input
window, pair by pair, so the tracing of the residue starts from a classification.

For each matched pair that is not identical: are the shared steps at zero separation
(so the difference is an extra step at one end, a lifetime or prune difference), or are
the positions themselves displaced, and by how much, and on which steps. For each
unmatched track on either side: the nearest track on the other side in time-overlap
terms, its separation, and whether a same-side track duplicates it (which would make
the miss an artifact of one-to-one matching rather than a detection difference).

    AEW_ORACLE_DIR=<exchange dir> .venv/bin/python scripts/diagnose_oracle_residuals.py \\
        --oracle tracker_octave_instrumented.mat --out <artifact.json>
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_tracker_oracle as C  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def step_separations(a, b):
    """Per shared timestep, the separation in degrees (cos-latitude weighted in lon)."""
    shared = np.intersect1d(a["time"], b["time"])
    out = []
    for t in shared:
        i = int(np.where(a["time"] == t)[0][0])
        j = int(np.where(b["time"] == t)[0][0])
        dlat = a["lat"][i] - b["lat"][j]
        dlon = (a["lon"][i] - b["lon"][j]) * np.cos(np.radians(0.5 * (a["lat"][i] + b["lat"][j])))
        out.append((float(t), float(np.hypot(dlat, dlon))))
    return out


def extras(own, other):
    """The timesteps one track has and the other does not, classed by where they sit
    relative to the OTHER track's span: before it, inside it, or after it. Both sides
    are reported independently, because a review found the first version dropping the
    port's interior extra whenever version 1 also had extras (the real pair 94/89)."""
    only = sorted(set(own["time"].tolist()) - set(other["time"].tolist()))
    lo, hi = float(other["time"].min()), float(other["time"].max())
    return {"n": len(only), "times": only,
            "before": sum(1 for t in only if t < lo),
            "interior": sum(1 for t in only if lo < t < hi),
            "after": sum(1 for t in only if t > hi)}


def classify_pair(v, p):
    seps = step_separations(v, p)
    shared = len(seps)
    displaced = any(s > 0.0 for _, s in seps)
    v_extra, p_extra = extras(v, p), extras(p, v)
    if v_extra["n"] and p_extra["n"]:
        extra_kind = "extra_both_sides"
    elif v_extra["n"]:
        extra_kind = "extra_v1_only"
    elif p_extra["n"]:
        extra_kind = "extra_port_only"
    else:
        extra_kind = "no_extra"
    kind = extra_kind + ("_displaced" if displaced else "")
    if kind == "no_extra_displaced":
        kind = "displaced_only"
    return {"kind": kind, "extra_kind": extra_kind, "displaced": displaced,
            "v1_steps": int(v["time"].size), "port_steps": int(p["time"].size),
            "shared_steps": shared, "max_sep_deg": max((s for _, s in seps), default=0.0),
            "n_displaced_steps": sum(1 for _, s in seps if s > 0.0),
            "v1_extra": v_extra, "port_extra": p_extra,
            "mean_lat": float(v["lat"].mean()), "mean_lon": float(v["lon"].mean()),
            "first_time": float(v["time"].min())}


def nearest(track, others):
    best = None
    for k, o in enumerate(others):
        n, sep = C.overlap_separation(track, o)
        if n and (best is None or sep < best[2]):
            best = (k, n, sep)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--oracle-dir", default=os.environ.get("AEW_ORACLE_DIR"))
    ap.add_argument("--oracle", default=C.FAITHFUL_ORACLE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not args.oracle_dir:
        ap.error("set AEW_ORACLE_DIR or pass --oracle-dir")
    v1, port, case, _ = C.load_case(args.oracle_dir, oracle_name=args.oracle)
    assigned, _ = C.match(v1, port)
    matched_v1 = {i for i, _, _, _ in assigned}
    matched_port = {j for _, j, _, _ in assigned}
    pairs = []
    for i, j, _, _ in assigned:
        v, p = v1[i], port[j]
        identical = (v["time"].shape == p["time"].shape and np.array_equal(v["time"], p["time"])
                     and np.array_equal(v["lat"], p["lat"]) and np.array_equal(v["lon"], p["lon"]))
        if not identical:
            pairs.append({"v1_index": int(i), "port_index": int(j), **classify_pair(v, p)})
    kinds = {}
    for q in pairs:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
    extra_kinds = {}
    for q in pairs:
        extra_kinds[q["extra_kind"]] = extra_kinds.get(q["extra_kind"], 0) + 1
    # THE WORST INDIVIDUAL STEP, separately from any pair mean: a review found the
    # write-up quoting the worst pair MEAN (0.54) as if it bounded the largest single
    # displacement, which was 4.70.
    worst_step = max((q["max_sep_deg"] for q in pairs), default=0.0)
    pairs_over_one_degree = sum(1 for q in pairs if q["max_sep_deg"] > 1.0)

    def unmatched_report(side_tracks, side_matched, other_tracks, label):
        out = []
        for k, t in enumerate(side_tracks):
            if k in side_matched:
                continue
            near_other = nearest(t, other_tracks)
            same = [(m, o) for m, o in enumerate(side_tracks) if m != k]
            near_same = nearest(t, [o for _, o in same])
            out.append({"index": int(k), "steps": int(t["time"].size),
                        "mean_lat": float(t["lat"].mean()), "mean_lon": float(t["lon"].mean()),
                        "first_time": float(t["time"].min()), "last_time": float(t["time"].max()),
                        "nearest_other_side": None if near_other is None else
                        {"index": int(near_other[0]), "shared_steps": int(near_other[1]),
                         "sep_deg": float(near_other[2])},
                        "nearest_same_side": None if near_same is None else
                        {"index": int(same[near_same[0]][0]), "shared_steps": int(near_same[1]),
                         "sep_deg": float(near_same[2]),
                         "that_track_matched": same[near_same[0]][0] in side_matched}})
        return out
    v1_un = unmatched_report(v1, matched_v1, port, "v1")
    port_un = unmatched_report(port, matched_port, v1, "port")
    # AN ELIGIBLE CANDIDATE IS NOT A DUPLICATE. For each unmatched track the artifact
    # says whether ANY track on the other side is eligible under the matching rule
    # (two shared steps within five degrees); if one is, the track was left over by the
    # one-to-one assignment, which is a statement about the assignment and not a
    # classification of the track as a physical duplicate.
    for u in v1_un:
        near = u["nearest_other_side"]
        u["eligible_counterpart_exists"] = bool(near and near["shared_steps"] >= C.MIN_OVERLAP
                                                and near["sep_deg"] <= C.TOLERANCE_DEG)
    for u in port_un:
        near = u["nearest_other_side"]
        u["eligible_counterpart_exists"] = bool(near and near["shared_steps"] >= C.MIN_OVERLAP
                                                and near["sep_deg"] <= C.TOLERANCE_DEG)
    res = {"case_id": case, "oracle_file": args.oracle,
           "nonidentical_pairs": len(pairs), "nonidentical_kinds": kinds,
           "nonidentical_extra_kinds": extra_kinds,
           "worst_individual_step_deg": worst_step,
           "pairs_with_a_step_over_one_degree": pairs_over_one_degree,
           "v1_unmatched_with_eligible_counterpart": sum(
               1 for u in v1_un if u["eligible_counterpart_exists"]),
           "pairs": pairs, "v1_unmatched": v1_un, "port_unmatched": port_un}
    print(f"case {case}: {len(pairs)} nonidentical pairs {kinds}; by extras {extra_kinds}; "
          f"worst individual step {worst_step:.2f} deg; {pairs_over_one_degree} pairs with a "
          f"step over one degree")
    for q in pairs:
        print(f"  pair v1 {q['v1_index']:3d} port {q['port_index']:3d} {q['kind']:26s} "
              f"steps {q['v1_steps']:2d}/{q['port_steps']:2d} shared {q['shared_steps']:2d} "
              f"max {q['max_sep_deg']:.2f} displaced {q['n_displaced_steps']:2d} "
              f"v1 extra {q['v1_extra']['n']} (before {q['v1_extra']['before']}, interior "
              f"{q['v1_extra']['interior']}, after {q['v1_extra']['after']}) port extra "
              f"{q['port_extra']['n']} (before {q['port_extra']['before']}, interior "
              f"{q['port_extra']['interior']}, after {q['port_extra']['after']})")
    print(f"version 1 unmatched {len(v1_un)}:")
    for u in v1_un:
        print(f"  v1 {u['index']:3d} {u['steps']:2d} steps lat {u['mean_lat']:+6.2f} lon "
              f"{u['mean_lon']:+7.2f} nearest port {u['nearest_other_side']} nearest v1 "
              f"{u['nearest_same_side']}")
    print(f"port unmatched {len(port_un)}:")
    for u in port_un:
        print(f"  port {u['index']:3d} {u['steps']:2d} steps lat {u['mean_lat']:+6.2f} lon "
              f"{u['mean_lon']:+7.2f} nearest v1 {u['nearest_other_side']} nearest port "
              f"{u['nearest_same_side']}")
    if args.out:
        res.update({"generated_by": "scripts/diagnose_oracle_residuals.py",
                    "input_sha256": {n: _sha256(os.path.join(args.oracle_dir, n)) for n in
                                     ("tracker_case.mat", "tracker_port.mat", args.oracle)},
                    "source_sha256": {"scripts/diagnose_oracle_residuals.py":
                                      _sha256(os.path.abspath(__file__)),
                                      "scripts/compare_tracker_oracle.py":
                                      _sha256(os.path.abspath(C.__file__))}})
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
