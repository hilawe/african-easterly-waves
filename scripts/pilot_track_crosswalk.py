#!/usr/bin/env python3
"""The eastern pilot's assessment instrument: two runs of one year compared track by track.

Two modes, both predeclared in EASTERN_EXTENSION_PILOT_BRIEF_2026-10-05.md section 7.

IDENTITY (run B against run A, the input control). The finished track arrays must be the
same MULTISET of complete track arrays, with identical multiplicities: a track that appears
twice on one side appears exactly twice on the other, since version 1's duplication is
retained. Anything else is reported and fails.

CROSSWALK (run C against run B, the treatment against the control). Tracks are paired by
the comparator's rule, at least PAIR_STEPS shared time steps at mean separation at most
PAIR_SEPARATION_DEG degrees, with every pairing kept: a track paired with more than one
partner is listed as such rather than resolved. Each paired track is classed IDENTICAL,
PREFIX CHANGED (the treatment's track starts earlier and then continues through exactly the
control's observations), ENDPOINT CHANGED, or OTHERWISE CHANGED, and a prefix is a
CANDIDATE recovered prefix, never proof of one disturbance. Unpaired tracks on either side are listed. Every track carries:
its first detection's region by the retained classifier, the index, position and time
of its first observation inside the Africa polygon by the same classifier (or none), whether its
start or end lies within BOUNDARY_DEG of the run's own domain edges (a start at the east
edge is POTENTIALLY CENSORED, meaning its earlier history may lie outside the domain, not
that a disturbance existed there), its first observation's month, and its ten-degree start
band. For paired tracks the signed start-time shift and the cohort transitions in both
directions are reported. For the Africa-origin June to September cohort of each side the
count, the duplication fraction and the group-collapsed distinct-wave count are reported by
the instrument's own functions, and the distinct-wave count is never called a count of
physical waves.

    python3 scripts/pilot_track_crosswalk.py identity --a <run dir A> --b <run dir B> --year 1990 --out <json>
    python3 scripts/pilot_track_crosswalk.py crosswalk --a <run dir B> --b <run dir C> --year 1990 \\
        --regions-dir data/aewc_v2_pilot/v1_src --out <json>
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import season_metrics as S  # noqa: E402
from compare_tracker_oracle import overlap_separation  # noqa: E402

PAIR_STEPS = S.WAVE_SHARED_STEPS
PAIR_SEPARATION_DEG = S.WAVE_SEPARATION_DEG
BOUNDARY_DEG = 2.0
SEASON = tuple(S.SEASON_MONTHS)


def load_run(run_dir, dataset="era5", year=None):
    """A run's tracks, its record and the domain the record declares."""
    names = [n for n in os.listdir(run_dir) if n.startswith("tracking_") and n.endswith(".json")]
    if len(names) != 1:
        raise SystemExit(f"REFUSED: {run_dir} holds {len(names)} tracking records, not one")
    with open(os.path.join(run_dir, names[0]), "rb") as fh:
        record_blob = fh.read()
    record = json.loads(record_blob)
    if year is not None and record["dataset_specific"].get("year") != year:
        raise SystemExit(f"REFUSED: the record in {run_dir} is for {record['dataset_specific'].get('year')}, not {year}")
    tracks_path = os.path.join(run_dir, "tracker_port.mat")
    with open(tracks_path, "rb") as fh:
        blob = fh.read()
    import hashlib
    if hashlib.sha256(blob).hexdigest() != record["dataset_specific"].get("tracks_sha256"):
        raise SystemExit(f"REFUSED: the tracks in {run_dir} do not have the digest the record names")
    tracks, case = S.read_mat_tracks(blob)
    dom = record["protocol_settings"]["domain"]
    return {"tracks": tracks, "case_id": case, "record": record, "domain": dom,
            "record_file": names[0], "record_sha256": hashlib.sha256(record_blob).hexdigest(),
            "tracks_sha256": record["dataset_specific"]["tracks_sha256"],
            "manifest_sha256": record["protocol_settings"]["manifest_sha256"],
            "variant": record["protocol_settings"].get("domain_variant")}


def canonical_set(tracks):
    return X.canonical([{"time": t["time"], "meanlat": t["lat"], "meanlon": t["lon"]} for t in tracks])


def identity(a, b):
    """The same multiset of track arrays: version 1's retained duplicates mean a track can
    legitimately appear twice on both sides, so multiplicities are compared, not a set."""
    ca, cb = collections.Counter(canonical_set(a["tracks"])), collections.Counter(canonical_set(b["tracks"]))
    only_a, only_b = ca - cb, cb - ca
    return {"mode": "identity", "tracks_a": sum(ca.values()), "tracks_b": sum(cb.values()),
            "only_in_a": sum(only_a.values()), "only_in_b": sum(only_b.values()), "shared": sum((ca & cb).values()),
            "passed": ca == cb}


def first_entry_into_africa(track, regions):
    """The first observation the retained classifier assigns to AFR when tested as a first
    observation, or None."""
    for k in range(track["time"].size):
        one = {"time": track["time"][k:k + 1], "lat": track["lat"][k:k + 1], "lon": track["lon"][k:k + 1]}
        if S.source_region(one, regions) == S.RECORD_REGION:
            return k
    return None


def attributes(track, domain, regions):
    lon0, lon1 = (float(x) for x in domain["lon"])
    first_lon, last_lon = float(track["lon"][0]), float(track["lon"][-1])
    k = first_entry_into_africa(track, regions)
    return {"observations": int(track["time"].size), "start_time": float(track["time"][0]), "end_time": float(track["time"][-1]),
            "start_lat": float(track["lat"][0]), "start_lon": first_lon, "end_lon": last_lon,
            "first_detection_region": S.source_region(track, regions),
            "first_entry_into_africa_index": k,
            "first_entry_into_africa_lon": None if k is None else float(track["lon"][k]),
            "first_entry_into_africa_lat": None if k is None else float(track["lat"][k]),
            "first_entry_into_africa_time": None if k is None else float(track["time"][k]),
            "start_month": S.date_of(track["time"][0]).month, "start_band": S.band_of(first_lon),
            "start_within_boundary_of_east_edge": bool(first_lon >= lon1 - BOUNDARY_DEG),
            "start_within_boundary_of_west_edge": bool(first_lon <= lon0 + BOUNDARY_DEG),
            "end_within_boundary_of_east_edge": bool(last_lon >= lon1 - BOUNDARY_DEG),
            "end_within_boundary_of_west_edge": bool(last_lon <= lon0 + BOUNDARY_DEG),
            "potentially_censored_start": bool(first_lon >= lon1 - BOUNDARY_DEG)}


def pair_class(ta, tb):
    """How track tb (the treatment's) relates to ta (the control's)."""
    same_end = ta["time"][-1] == tb["time"][-1] and ta["lat"][-1] == tb["lat"][-1] and ta["lon"][-1] == tb["lon"][-1]
    if ta["time"].size == tb["time"].size and np.array_equal(ta["time"], tb["time"]) \
            and np.array_equal(ta["lat"], tb["lat"]) and np.array_equal(ta["lon"], tb["lon"]):
        return "identical"
    if same_end and tb["time"][0] < ta["time"][0]:
        n = int(np.sum(tb["time"] < ta["time"][0]))
        if np.array_equal(tb["time"][n:], ta["time"]) and np.array_equal(tb["lat"][n:], ta["lat"]) and np.array_equal(tb["lon"][n:], ta["lon"]):
            return "prefix changed"
    if not same_end:
        return "endpoint changed"
    return "otherwise changed"


PAIR_CLASSES = ("identical", "prefix changed", "endpoint changed", "otherwise changed")
TRANSITIONS = ("in season to in season", "in season to out of season", "out of season to in season", "out of season to out of season")


def band_table(attrs):
    out = {f"{lo:+d}..{lo + 10:+d}": 0 for lo in S.BAND_EDGES[:-1]}
    out["outside"] = 0
    for x in attrs:
        out[x["start_band"]] += 1
    return out


def region_table(attrs):
    out = {code: 0 for code, _ in S.REGION_FILES}
    out["OTH"] = 0
    for x in attrs:
        out[x["first_detection_region"]] += 1
    return out


def boundary_table(attrs, dom):
    n = len(attrs)
    out = {"tracks": n, "domain_lon": [float(v) for v in dom["lon"]], "boundary_deg": BOUNDARY_DEG}
    for key in ("start_within_boundary_of_east_edge", "start_within_boundary_of_west_edge",
                "end_within_boundary_of_east_edge", "end_within_boundary_of_west_edge"):
        count = sum(1 for x in attrs if x[key])
        out[key] = {"count": count, "share": round(count / n, 4) if n else None}
    return out


def cohort(tracks, year, regions):
    season = [t for t in tracks if S.in_season(t, year)]
    africa = [t for t in season if S.in_record_domain(t, regions)]
    dup = S.duplication(africa) if africa else {"fraction": None}
    return {"season_whole_domain": len(season), "africa_origin_season": len(africa),
            "duplication_fraction": dup["fraction"],
            "distinct_waves_group_collapsed": S.distinct_waves(africa) if africa else 0,
            "note": "the distinct-wave count is group-collapsed under the comparator's rule, not a count of physical waves"}


def common_domain(a, b, year, regions, pairs, attrs_a, attrs_b):
    """Section 7 item 4: the assessment scoped to tracks with any observation west of the
    control's own eastern edge, so a wholly eastern track enters none of it. The paired
    classes among those tracks, the start bands of those starting west of the edge, the
    lifetimes of each side's Africa-origin June to September cohort, and the Atlantic-side
    coast-crossing fraction with both denominators as the retained coast-crossing
    instrument reports them."""
    import coast_crossing_measurement as CC
    edge = float(a["domain"]["lon"][1])
    inside = {side: [i for i, t in enumerate(run["tracks"]) if bool(np.any(np.asarray(t["lon"]) < edge))]
              for side, run in (("a", a), ("b", b))}
    in_a, in_b = set(inside["a"]), set(inside["b"])
    classes = {c: 0 for c in PAIR_CLASSES}
    classes.update(collections.Counter(q["class"] for q in pairs if q["a"] in in_a and q["b"] in in_b))
    out = {"common_edge_lon": edge, "tracks_with_an_observation_west_of_the_edge": {"a": len(in_a), "b": len(in_b)},
           "pair_classes": classes,
           "start_bands_of_starts_west_of_the_edge": {side: band_table([x for i, x in enumerate(attrs) if i in idx and x["start_lon"] < edge])
                                                      for side, attrs, idx in (("a", attrs_a, in_a), ("b", attrs_b, in_b))}}
    for side, run in (("a", a), ("b", b)):
        tracks = [dict(t, id=i) for i, t in enumerate(run["tracks"]) if i in set(inside[side])]
        africa = [t for t in tracks if S.in_season(t, year) and S.in_record_domain(t, regions)]
        lifetimes = np.asarray([t["time"].size for t in africa], float)
        measured = CC.measure(tracks, year, regions, S._day(year, 1, 1), CC.groups_ours(tracks))
        out[side] = {"africa_origin_season_tracks": len(africa),
                     "lifetime_observations_percentiles": ({str(q): float(v) for q, v in zip(S.PERCENTILES, np.percentile(lifetimes, S.PERCENTILES))}
                                                           if lifetimes.size else None),
                     "coast_crossing": {"instrument": "scripts/coast_crossing_measurement.py",
                                        "cohort_n": measured["cohort_n"], "endpoint": measured["endpoint"],
                                        "stored_tracks": measured["by_band"]["total"],
                                        "grouping_sensitivity_units": measured["grouping_sensitivity"]["by_band"]["total"]}}
    return out


def crosswalk(a, b, year, regions):
    ta, tb = a["tracks"], b["tracks"]
    attrs_a = [attributes(t, a["domain"], regions) for t in ta]
    attrs_b = [attributes(t, b["domain"], regions) for t in tb]
    pairs = []
    for i, x in enumerate(ta):
        for j, y in enumerate(tb):
            n, sep = overlap_separation(x, y)
            if n >= PAIR_STEPS and sep <= PAIR_SEPARATION_DEG:
                prefix_steps = int(np.sum(y["time"] < x["time"][0]))
                pairs.append({"a": i, "b": j, "shared_steps": int(n), "mean_separation_deg": round(float(sep), 3),
                              "class": pair_class(x, y),
                              "start_shift_days": round(float(y["time"][0] - x["time"][0]), 3),
                              "candidate_prefix_steps": prefix_steps,
                              # eastward extent of the candidate prefix, treatment start minus control start
                              "candidate_prefix_degrees": round(float(y["lon"][0] - x["lon"][0]), 2) if prefix_steps else 0.0,
                              "season_a": attrs_a[i]["start_month"] in SEASON, "season_b": attrs_b[j]["start_month"] in SEASON})
    partners_a = collections.Counter(p["a"] for p in pairs)
    partners_b = collections.Counter(p["b"] for p in pairs)
    # EVERY CATEGORY IS SEEDED WITH ZERO, so an empty cell is reported as empty, never absent
    classes = {c: 0 for c in PAIR_CLASSES}
    classes.update(collections.Counter(p["class"] for p in pairs))
    transitions = {t: 0 for t in TRANSITIONS}
    transitions.update(collections.Counter(("in season" if p["season_a"] else "out of season") + " to " +
                                           ("in season" if p["season_b"] else "out of season") for p in pairs))
    bands = {side: band_table(attrs) for side, attrs in (("a", attrs_a), ("b", attrs_b))}
    return {"mode": "crosswalk", "year": year, "pairing_rule": {"shared_steps_at_least": PAIR_STEPS, "mean_separation_at_most_deg": PAIR_SEPARATION_DEG},
            "boundary_deg": BOUNDARY_DEG,
            "tracks_a": len(ta), "tracks_b": len(tb), "pairs": pairs,
            "unpaired_a": [i for i in range(len(ta)) if i not in partners_a],
            "unpaired_b": [j for j in range(len(tb)) if j not in partners_b],
            "multiply_paired_a": sorted(i for i, c in partners_a.items() if c > 1),
            "multiply_paired_b": sorted(j for j, c in partners_b.items() if c > 1),
            "pair_classes": classes, "season_transitions_of_pairs": transitions,
            "boundary_counts": {side: boundary_table(attrs, dom) for side, attrs, dom in (("a", attrs_a, a["domain"]), ("b", attrs_b, b["domain"]))},
            "first_detection_regions": {side: region_table(attrs) for side, attrs in (("a", attrs_a), ("b", attrs_b))},
            "starts_outside_africa_with_later_entry": {side: sum(1 for x in attrs if x["first_detection_region"] != S.RECORD_REGION and x["first_entry_into_africa_index"] is not None)
                                                      for side, attrs in (("a", attrs_a), ("b", attrs_b))},
            "start_bands": bands,
            "common_domain": common_domain(a, b, year, regions, pairs, attrs_a, attrs_b),
            "cohort": {"a": cohort(ta, year, regions), "b": cohort(tb, year, regions)},
            "attributes_a": attrs_a, "attributes_b": attrs_b}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("identity", "crosswalk"))
    ap.add_argument("--a", required=True, help="the control's run directory (A for identity, B for crosswalk)")
    ap.add_argument("--b", required=True, help="the other run directory (B for identity, C for crosswalk)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--regions-dir", default="data/aewc_v2_pilot/v1_src")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    a, b = load_run(args.a, year=args.year), load_run(args.b, year=args.year)
    if args.mode == "identity":
        result = identity(a, b)
    else:
        regions, region_digests = S.load_regions(args.regions_dir)
        result = crosswalk(a, b, args.year, regions)
        result["region_polygons_sha256"] = region_digests
    result.update({"generated_by": "scripts/pilot_track_crosswalk.py", "script_sha256": X.digest(os.path.abspath(__file__)),
                   "runs": {"a": {"dir": args.a, "record_file": a["record_file"], "record_sha256": a["record_sha256"], "tracks_sha256": a["tracks_sha256"], "manifest_sha256": a["manifest_sha256"], "domain": a["domain"], "variant": a["variant"]},
                            "b": {"dir": args.b, "record_file": b["record_file"], "record_sha256": b["record_sha256"], "tracks_sha256": b["tracks_sha256"], "manifest_sha256": b["manifest_sha256"], "domain": b["domain"], "variant": b["variant"]}}})
    try:
        X.publish_json(args.out, result, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    if args.mode == "identity":
        print(f"identity {args.year}: {result['tracks_a']} against {result['tracks_b']} tracks, shared {result['shared']}, "
              f"only in A {result['only_in_a']}, only in B {result['only_in_b']}: {'PASSED' if result['passed'] else 'NOT PASSED'}")
        return 0 if result["passed"] else 1
    print(f"crosswalk {args.year}: {result['tracks_a']} against {result['tracks_b']} tracks, {len(result['pairs'])} pairs {result['pair_classes']}, "
          f"unpaired {len(result['unpaired_a'])} and {len(result['unpaired_b'])}, multiply paired {len(result['multiply_paired_a'])} and {len(result['multiply_paired_b'])}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
