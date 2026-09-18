#!/usr/bin/env python3
"""Join two storm-link artifacts, one per best-track basin file, into one resolution per
storm key, so a development statistic reads a single validated link per key.

WHAT IT SETTLES. The linker (scripts/link_qtrack_storms.py) runs once per IBTrACS basin
file over the same population, so a key has one state per file. This script joins the
two by key (year, tag, QTrack genesis time) and declares, per key:

  - `validated` with `from` = the file whose state is `matched`, when exactly one file
    matched AND the other file's state is NEGATIVE EVIDENCE (`no_candidate`,
    `outside_window`, `rejected_distance`), or when both matched THE SAME STORM with THE
    SAME EVENT (SID, event time, event basin and event subbasin all equal), which is a
    storm present in both files (a basin crosser);
  - `unresolved` otherwise, with both states and the reason kept: neither matched; one
    matched while the other holds an identity conflict (`ambiguous`,
    `shared_identifier`), which a unique candidate in one file cannot settle for their
    union; or both matched the same storm with disagreeing event metadata.

Every validated result carries `event_basin` AND `event_subbasin`, because the genesis
basin is the field a development statistic reads, and the subbasin alone cannot stand
in for it (the eastern Pacific file records "MM", no designated subbasin, for nearly
every event).

WHAT IT REFUSES, before any row is resolved. Two artifacts are joinable only if they are
the SAME RUN over the SAME POPULATION with a different best-track file: their `rule`
blocks (tolerances, genesis definition, candidate and choice rules) must be equal, their
recorded source hashes for the matcher, the filter module and the link script must be
equal, their per-year population hashes must be equal, their key sets must be equal,
and each key's track membership (`systems`) must be equal. Equal names and genesis
times alone do not establish that "matched" meant the same thing in both files. The
two best-track hashes are expected to differ and are not compared. After resolution it
also refuses a key matched in both files to DIFFERENT storms, and two validated keys
that name the same storm identifier, because either would be two links for one storm
and no rule here chooses between them.

Every key whose state was a rejection in one file and a match in the other is listed
with both files' evidence (offset and distance on each side), since those are the keys
where the join changes a conclusion.

    .venv/bin/python scripts/join_qtrack_storm_links.py \\
        --first docs/aewc_v2/artifacts/qtrack_storm_link_2026-09-17.json \\
        --second docs/aewc_v2/artifacts/qtrack_storm_link_EP_2026-09-17.json \\
        --out <artifact.json>
"""
import argparse
import collections
import hashlib
import json
import os
import sys

NEGATIVE_STATES = ("no_candidate", "outside_window", "rejected_distance")
CONFLICT_STATES = ("ambiguous", "shared_identifier")
EVENT_FIELDS = ("sid", "event_time", "event_basin", "event_subbasin")
COMPARED_SOURCES = ("src/aew/qtrack_storms.py", "src/aew/qtrack.py",
                    "scripts/link_qtrack_storms.py")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def rows_by_key(artifact):
    out = {}
    for year, v in artifact["years"].items():
        for r in v["rows"]:
            key = (year, r["name"], str(r["qtrack_genesis"]))
            if key in out:
                raise ValueError(f"key appears twice in one artifact: {key}")
            out[key] = r
    return out


def check_compatible(first, second, labels):
    """Raise unless the two artifacts are the same run over the same population."""
    for field in ("rule", "source_sha256", "years"):
        for art, lab in ((first, labels[0]), (second, labels[1])):
            if field not in art:
                raise ValueError(f"{lab} artifact records no `{field}`; cannot establish "
                                 f"that its matches mean the same thing")
    if first["rule"] != second["rule"]:
        diff = sorted(k for k in set(first["rule"]) | set(second["rule"])
                      if first["rule"].get(k) != second["rule"].get(k))
        raise ValueError(f"matching rules differ between {labels[0]} and {labels[1]} "
                         f"on {diff}")
    for src in COMPARED_SOURCES:
        ha, hb = first["source_sha256"].get(src), second["source_sha256"].get(src)
        if ha is None or hb is None or ha != hb:
            raise ValueError(f"source hash for {src} differs or is missing "
                             f"({labels[0]} {ha}, {labels[1]} {hb})")
    if set(first["years"]) != set(second["years"]):
        raise ValueError("the two artifacts cover different years")
    for year in sorted(first["years"]):
        ha = first["years"][year].get("input_sha256")
        hb = second["years"][year].get("input_sha256")
        if ha is None or ha != hb:
            raise ValueError(f"population hash for {year} differs or is missing "
                             f"({labels[0]} {ha}, {labels[1]} {hb})")


def join_links(first, second, labels=("first", "second")):
    """The per-key resolution of two link artifacts; raises on incompatible runs, a
    key-set or membership mismatch, a key matched to different storms, or two
    validated keys naming one storm."""
    check_compatible(first, second, labels)
    a, b = rows_by_key(first), rows_by_key(second)
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise ValueError(f"key sets differ: {len(only_a)} only in {labels[0]}, "
                         f"{len(only_b)} only in {labels[1]}")
    resolutions, flips, unresolved = {}, [], []
    state_pairs = collections.Counter()
    for key in sorted(a):
        ra, rb = a[key], b[key]
        if list(ra["systems"]) != list(rb["systems"]):
            raise ValueError(f"{key} has different track membership in the two artifacts: "
                             f"{ra['systems']} and {rb['systems']}")
        states = {labels[0]: ra["status"], labels[1]: rb["status"]}
        state_pairs[(ra["status"], rb["status"])] += 1
        a_m, b_m = ra["status"] == "matched", rb["status"] == "matched"
        res = None
        if a_m and b_m:
            if ra["sid"] != rb["sid"]:
                raise ValueError(f"{key} is matched to different storms: {labels[0]} "
                                 f"{ra['sid']} at {ra['event_time']}, {labels[1]} "
                                 f"{rb['sid']} at {rb['event_time']}")
            disagree = [f for f in EVENT_FIELDS if str(ra[f]) != str(rb[f])]
            if disagree:
                res = {"state": "unresolved",
                       "reason": f"matched to the same storm with disagreeing "
                                 f"{', '.join(disagree)}",
                       "disagreement": {f: {labels[0]: ra[f], labels[1]: rb[f]}
                                        for f in disagree}}
            else:
                res = {"state": "validated", "from": "both",
                       **{f: (str(ra[f]) if f == "event_time" else ra[f])
                          for f in EVENT_FIELDS}}
        elif a_m or b_m:
            src, r, other = (labels[0], ra, rb) if a_m else (labels[1], rb, ra)
            if other["status"] in NEGATIVE_STATES:
                res = {"state": "validated", "from": src,
                       **{f: (str(r[f]) if f == "event_time" else r[f])
                          for f in EVENT_FIELDS}}
                if other["status"] in ("rejected_distance", "outside_window"):
                    flips.append({"key": list(key), "validated_in": src,
                                  "offset_hours": r["offset_hours"],
                                  "wave_to_event_km": r["wave_to_event_km"],
                                  "other_state": other["status"],
                                  "other_nearest_sid": other.get("nearest_sid"),
                                  "other_offset_hours": other.get("offset_hours"),
                                  "other_wave_to_event_km": other.get("wave_to_event_km")})
            else:
                other_label = labels[1] if a_m else labels[0]
                res = {"state": "unresolved",
                       "reason": f"matched in {src} while {other_label} holds an identity "
                                 f"conflict ({other['status']}), which one file's unique "
                                 f"candidate cannot settle for the union"}
        else:
            res = {"state": "unresolved", "reason": "matched in neither file"}
        res["states"] = states
        if res["state"] == "unresolved":
            unresolved.append({"key": list(key), "systems": ra["systems"],
                               "states": states, "reason": res["reason"]})
        resolutions["|".join(key)] = res
    chosen = collections.Counter(r["sid"] for r in resolutions.values()
                                 if r["state"] == "validated")
    repeated = sorted(s for s, n in chosen.items() if n > 1)
    if repeated:
        raise ValueError(f"{len(repeated)} storm identifier(s) validated for more than "
                         f"one key: {repeated[:5]}")
    validated = sum(1 for r in resolutions.values() if r["state"] == "validated")
    by_source = collections.Counter(r["from"] for r in resolutions.values()
                                    if r["state"] == "validated")
    by_basin = collections.Counter(r["event_basin"] for r in resolutions.values()
                                   if r["state"] == "validated")
    return {"keys": len(resolutions), "validated": validated,
            "validated_by_source": dict(by_source),
            "validated_by_event_basin": dict(by_basin),
            "distinct_storms_validated": len(chosen),
            "unresolved": unresolved,
            "state_pairs": {f"{x}|{y}": n for (x, y), n in sorted(state_pairs.items())},
            "flips": flips, "resolutions": resolutions}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--first", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--labels", nargs=2, default=["NA", "EP"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    with open(args.first) as fh:
        first = json.load(fh)
    with open(args.second) as fh:
        second = json.load(fh)
    try:
        joined = join_links(first, second, tuple(args.labels))
    except ValueError as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    print(f"keys {joined['keys']}, validated {joined['validated']} "
          f"{joined['validated_by_source']}, by event basin "
          f"{joined['validated_by_event_basin']}, distinct storms "
          f"{joined['distinct_storms_validated']}, unresolved {len(joined['unresolved'])}, "
          f"flips {len(joined['flips'])}", flush=True)
    for pair, n in joined["state_pairs"].items():
        print(f"  {pair:40s} {n}")
    out = dict(joined)
    out.update({"generated_by": "scripts/join_qtrack_storm_links.py",
                "inputs": {args.labels[0]: {"path": args.first, "sha256": _sha256(args.first),
                                            "ibtracs_sha256": first.get("ibtracs_sha256")},
                           args.labels[1]: {"path": args.second, "sha256": _sha256(args.second),
                                            "ibtracs_sha256": second.get("ibtracs_sha256")}},
                "compatibility_checked": {"rule": "equal", "source_sha256": list(COMPARED_SOURCES),
                                          "years_input_sha256": "equal per year",
                                          "key_sets": "equal", "systems_per_key": "equal",
                                          "ibtracs_sha256": "NOT compared, expected to differ"},
                "rule": {"validated": "exactly one file matched and the other's state is "
                                      "negative evidence (no_candidate, outside_window, "
                                      "rejected_distance), or both matched the same storm "
                                      "with equal SID, event time, event basin and event "
                                      "subbasin",
                         "unresolved": "neither matched; one matched while the other holds "
                                       "ambiguous or shared_identifier; or both matched "
                                       "the same storm with disagreeing event metadata",
                         "refused": "incompatible runs, differing key sets or membership, "
                                    "a key matched to different storms, or one storm "
                                    "identifier validated for more than one key",
                         "flip": "one file matched and the other rejected on distance or "
                                 "window; both sides' evidence recorded",
                         "event_basin": "the IBTrACS BASIN at the genesis event, carried "
                                        "on every validated key; the field a development "
                                        "statistic reads"},
                "source_sha256": {"scripts/join_qtrack_storm_links.py":
                                  _sha256(os.path.abspath(__file__))},
                "what_this_does_not_establish": "anything about an unresolved key, or the "
                                                "identity of a key in any state but "
                                                "matched in its source file"})
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True, default=str)
        print(f"written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
