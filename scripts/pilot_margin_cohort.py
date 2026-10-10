#!/usr/bin/env python3
"""The margin replay read through the paper's own definitions, one season: the summer
Africa-origin cohort of the native and the margin finished tracks, measured by the
retained coast-crossing instrument unchanged, with the cohort's correspondence across the
two runs read from the margin comparison's pairing and kept in its classes.

WHAT IS REUSED, and not restated. `coast_crossing_measurement.measure` gives the cohort
(first observation in June to September, in the Africa polygon under the archive's rule,
first latitude 0 to 25 N, not starting at the record's first step), the first-latitude
bands, the Atlantic-side outcome on observations at or before October 31 18Z (North
Atlantic under the archive's rule with raw longitude at or west of 7.5 W), the
follow-up-incomplete status, the denominators, and the duplicate-grouping sensitivity
under the instrument's whole-record rule. `season_metrics.band_of` gives the ten-degree
genesis-longitude bands and `season_metrics.spacing` the duration percentiles. The
tracks are the finished smoothed arrays, as the archive holds them and as the paper reads
them. Nothing is tracked, and no case is opened.

THE CORRESPONDENCE is the margin comparison's pairing (`pilot_margin_compare.pair`, the
declared rules), recomputed from the two retained replay records: identical raw
histories paired by count, unique mutual best matches, ambiguous and unmatched kept apart.
A cohort member's correspondence class is reported, and a change is read only across an
identical or a mutual pair. An identical pair is an EXACT history, so none of its
summaries can differ. For a mutual pair the summaries that differ are counted separately
(cohort membership, first-latitude band, genesis-longitude band, outcome, follow-up,
single-observation entrant, observation count), so an exact-history difference is never
equated with a changed scientific summary. An identical pair is required to agree in
cohort membership, in every one of those summaries and in its finished arrays, and the
read refuses otherwise.

    python3 scripts/pilot_margin_cohort.py --native <native json.gz> --margin <margin json.gz> --compare <compare r2 json> \\
        --regions-dir data/aewc_v2_pilot/v1_src --out <fresh json>
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import coast_crossing_measurement as CC  # noqa: E402
import exact_tracks as X  # noqa: E402
import pilot_margin_compare as MC  # noqa: E402
import qtrack_pairing_pilot as M  # noqa: E402
import season_metrics as S  # noqa: E402

PERCENTILES = (10, 50, 90)


def as_tracks(finished):
    """The replay record's finished tracks in the measurement's layout, ids being the
    record's own finished indices (the archive's for a native run)."""
    return [{"id": i, "time": np.asarray(f["time"], float), "lat": np.asarray(f["lat"], float), "lon": np.asarray(f["lon"], float), "n_valid": len(f["time"])}
            for i, f in enumerate(finished)]


def correspondence(native, margin):
    """Each finished index's class and counterpart, from the comparison's pairing."""
    p = MC.pair(native, margin)
    cls_n, cls_m = {}, {}
    for i, j in p["identical"]:
        cls_n[i] = ("identical", j)
        cls_m[j] = ("identical", i)
    for i, j, _o in p["mutual"]:
        kind = MC.divergence(native[i], margin[j])["kind"]
        cls_n[i] = ("mutual_" + kind, j)
        cls_m[j] = ("mutual_" + kind, i)
    for a in p["ambiguous_native"]:
        cls_n[a["native"]] = ("ambiguous", None)
    for a in p["ambiguous_margin"]:
        cls_m[a["margin"]] = ("ambiguous", None)
    for i in p["unmatched_native"]:
        cls_n[i] = ("unmatched", None)
    for j in p["unmatched_margin"]:
        cls_m[j] = ("unmatched", None)
    assert len(cls_n) == len(native) and len(cls_m) == len(margin)
    return cls_n, cls_m


def genesis_band(track):
    return S.band_of(float(track["lon"][0]))


def percentiles(values):
    return {str(p): float(v) for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))} if len(values) else None


def describe(measurement, tracks_by_id):
    """A cohort's genesis-longitude bands, lifetimes and start locations, from its records."""
    recs = measurement["tracks"]
    members = [tracks_by_id[r["id"]] for r in recs]
    bands = {}
    for t in members:
        bands[genesis_band(t)] = bands.get(genesis_band(t), 0) + 1
    n_obs = [r["n_observations"] for r in recs]
    return {"cohort_n": len(recs), "genesis_longitude_bands_10deg": dict(sorted(bands.items())),
            "first_latitude_bands": {lab: measurement["by_band"][lab]["cohort"] for lab in CC.band_labels()},
            "observations_per_track": {"mean": float(np.mean(n_obs)) if n_obs else None, "percentiles": percentiles(n_obs), "total": int(sum(n_obs))},
            "duration_days": S.spacing(members)["duration_days_percentiles"] if members else None,
            "start_longitude_percentiles": percentiles([float(t["lon"][0]) for t in members]),
            "start_latitude_percentiles": percentiles([float(t["lat"][0]) for t in members]),
            "start_east_of_30E": sum(1 for t in members if float(t["lon"][0]) >= 30.0),
            "start_10E_to_30E": sum(1 for t in members if 10.0 <= float(t["lon"][0]) < 30.0)}


def outcomes_table(measurement):
    out = {}
    for lab, row in measurement["by_band"].items():
        out[lab] = {k: row[k] for k in ("cohort", "atlantic_side", "gulf_only", "none", "follow_up_incomplete", "single_observation_entrants",
                                         "fraction_atlantic_side_over_cohort", "fraction_atlantic_side_over_complete_follow_up", "denominators")}
    return out


def cohort_correspondence(meas_n, meas_m, cls_n, cls_m, native, margin):
    """Rule-level reading of the cohort across the runs: classes, membership changes, and
    the summaries that differ across identical and mutual pairs."""
    rec_n = {r["id"]: r for r in meas_n["tracks"]}
    rec_m = {r["id"]: r for r in meas_m["tracks"]}
    classes_n = {}
    for i in rec_n:
        k = cls_n[i][0]
        classes_n[k] = classes_n.get(k, 0) + 1
    classes_m = {}
    for j in rec_m:
        k = cls_m[j][0]
        classes_m[k] = classes_m.get(k, 0) + 1
    # membership across a counterpart: a native member whose counterpart is not a margin member left the cohort
    left, entered, both = [], [], []
    for i in rec_n:
        kind, j = cls_n[i]
        if j is None:
            continue
        (both if j in rec_m else left).append((i, j, kind))
    for j in rec_m:
        kind, i = cls_m[j]
        if i is None:
            continue
        if i not in rec_n:
            entered.append((i, j, kind))

    def summaries(i, j):
        a, b = rec_n[i], rec_m[j]
        ta, tb = native[i], margin[j]
        return {"outcome": a["outcome"] != b["outcome"], "follow_up_incomplete": a["follow_up_incomplete"] != b["follow_up_incomplete"],
                "single_observation_entrant": bool(a.get("single_observation_entrant")) != bool(b.get("single_observation_entrant")),
                "first_latitude_band": a["band"] != b["band"], "genesis_longitude_band": genesis_band(ta) != genesis_band(tb),
                "observations_difference": b["n_observations"] - a["n_observations"]}
    # AN EXACT HISTORY IS EXACT ON BOTH SIDES. An identical raw history gives identical finished
    # arrays, so its cohort membership, its measurement row and its arrays must agree, and any
    # disagreement is a contradiction in the inputs that this read refuses rather than reports.
    for i, (kind, j) in cls_n.items():
        if kind == "identical" and (i in rec_n) != (j in rec_m):
            raise SystemExit(f"REFUSED: the identical pair native {i} and margin {j} is in one cohort and not the other")
    exact = [(i, j) for i, j, kind in both if kind == "identical"]
    mutual = [(i, j, kind) for i, j, kind in both if kind != "identical"]
    for i, j in exact:
        s = summaries(i, j)
        if any(v for k, v in s.items() if k != "observations_difference") or s["observations_difference"] != 0:
            raise SystemExit(f"REFUSED: the identical pair native {i} and margin {j} differs in a summary: {s}")
        for key in ("time", "lat", "lon"):
            if not np.array_equal(native[i][key], margin[j][key]):
                raise SystemExit(f"REFUSED: the identical pair native {i} and margin {j} differs in its finished {key} array")
    diffs = {"outcome": 0, "follow_up_incomplete": 0, "single_observation_entrant": 0, "first_latitude_band": 0, "genesis_longitude_band": 0, "any_summary": 0}
    obs_diff = []
    changed_rows = []
    for i, j, kind in mutual:
        s = summaries(i, j)
        obs_diff.append(s["observations_difference"])
        flags = {k: s[k] for k in ("outcome", "follow_up_incomplete", "single_observation_entrant", "first_latitude_band", "genesis_longitude_band")}
        for k, v in flags.items():
            diffs[k] += int(v)
        diffs["any_summary"] += int(any(flags.values()))
        changed_rows.append({"native": i, "margin": j, "kind": kind, **flags, "observations": [rec_n[i]["n_observations"], rec_m[j]["n_observations"]],
                             "outcomes": [rec_n[i]["outcome"], rec_m[j]["outcome"]], "bands": [rec_n[i]["band"], rec_m[j]["band"]]})
    bins = {"0": 0, "1 to 3": 0, "4 to 7": 0, "8 or more": 0}
    for d in obs_diff:
        a = abs(d)
        bins["0" if a == 0 else "1 to 3" if a <= 3 else "4 to 7" if a <= 7 else "8 or more"] += 1
    return {"native_cohort_by_class": classes_n, "margin_cohort_by_class": classes_m,
            "members_in_both_cohorts_through_a_counterpart": {"exact_histories": len(exact), "mutual_pairs": len(mutual)},
            "left_the_cohort_through_a_counterpart": {"count": len(left), "by_kind": {k: sum(1 for *_x, kk in left if kk == k) for k in set(kk for *_x, kk in left)},
                                                      "rows": [{"native": i, "margin": j, "kind": k, "native_outcome": rec_n[i]["outcome"], "margin_first": [float(margin[j]["lat"][0]), float(margin[j]["lon"][0])],
                                                                "margin_in_season": bool(S.in_season(margin[j])), "margin_region": S.source_region(margin[j], REGIONS)} for i, j, k in left]},
            "entered_the_cohort_through_a_counterpart": {"count": len(entered), "by_kind": {k: sum(1 for *_x, kk in entered if kk == k) for k in set(kk for *_x, kk in entered)},
                                                         "rows": [{"native": i, "margin": j, "kind": k, "margin_outcome": rec_m[j]["outcome"], "native_first": [float(native[i]["lat"][0]), float(native[i]["lon"][0])],
                                                                   "native_in_season": bool(S.in_season(native[i])), "native_region": S.source_region(native[i], REGIONS)} for i, j, k in entered]},
            "without_a_counterpart": {"native": {k: classes_n.get(k, 0) for k in ("ambiguous", "unmatched")}, "margin": {k: classes_m.get(k, 0) for k in ("ambiguous", "unmatched")}},
            "mutual_pairs_in_both_cohorts": {"summaries_differing": diffs, "observations_difference_margin_minus_native": {"bins_by_absolute_value": bins, "percentiles": percentiles(obs_diff)},
                                             "rows": changed_rows}}


REGIONS = None


def verification_identity(cmp, contrast):
    """The extent comparison's verification identity, carried into the cohort record so
    that an extent cohort names the verification its comparison rested on. An extent
    comparison without one is refused, and the other contrasts carry none."""
    if contrast != "extent":
        return None
    v = cmp.get("verification")
    if not isinstance(v, dict) or not all(isinstance(v.get(k), str) for k in ("path", "sha256", "script_sha256")) or not isinstance(v.get("replays"), dict):
        raise SystemExit("REFUSED: the extent comparison carries no verification identity, so no extent cohort is read from it")
    return v


def admit(contrast, native_rec, margin_rec):
    """What the comparison's declared contrast requires of the two replays here. The
    candidate is always a margin run. The reference is a native run that reproduced the
    archive under the preparation contrasts, and a margin run under the calibration and
    extent contrasts. The record keys stay native (reference) and margin (candidate)."""
    if contrast not in MC.CONTRASTS:
        raise SystemExit(f"REFUSED: unknown contrast {contrast!r}")
    if margin_rec["preparation"] != "margin":
        raise SystemExit("REFUSED: the candidate is not a margin replay")
    if contrast in ("preparation", "preparation_and_calibration"):
        if native_rec["preparation"] != "native":
            raise SystemExit(f"REFUSED: the {contrast} contrast's reference is a native replay")
        if not native_rec["archive_comparison"]["passed"]:
            raise SystemExit("REFUSED: the native replay did not reproduce the archived tracks")
    elif native_rec["preparation"] != "margin":
        raise SystemExit(f"REFUSED: the {contrast} contrast's reference is a margin replay")
    return contrast


def main(argv=None):
    global REGIONS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--native", "--margin", "--compare", "--regions-dir", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    native_rec, sha_n = MC.load_gz(args.native)
    margin_rec, sha_m = MC.load_gz(args.margin)
    with open(args.compare, "rb") as fh:
        cmp_blob = fh.read()
    cmp = json.loads(cmp_blob)
    if cmp["inputs"]["native"]["sha256"] != sha_n or cmp["inputs"]["margin"]["sha256"] != sha_m:
        raise SystemExit("REFUSED: the comparison record was not made from these replay records")
    contrast = admit(cmp.get("contrast", "preparation"), native_rec, margin_rec)
    verification = verification_identity(cmp, contrast)
    if verification is not None and verification["replays"] != {"a": sha_n, "b": sha_m}:
        raise SystemExit("REFUSED: the comparison's verification is not bound to these two replays")
    year = native_rec["year"]
    REGIONS, region_digests = S.load_regions(args.regions_dir)
    native, margin = as_tracks(native_rec["finished"]), as_tracks(margin_rec["finished"])
    first_day = M._day(year, 1, 1)
    meas = {}
    for name, tracks in (("native", native), ("margin", margin)):
        meas[name] = CC.measure(tracks, year, REGIONS, first_day, CC.groups_ours(tracks))
    cls_n, cls_m = correspondence(native_rec["finished"], margin_rec["finished"])
    by_id = {"native": {t["id"]: t for t in native}, "margin": {t["id"]: t for t in margin}}
    out = {"generated_by": "scripts/pilot_margin_cohort.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": year,
           "contrast": contrast, "roles": cmp.get("roles", MC.CONTRASTS[contrast]), "verification": verification,
           "inputs": {"native": {"path": args.native, "sha256": sha_n}, "margin": {"path": args.margin, "sha256": sha_m},
                      "compare": {"path": args.compare, "sha256": hashlib.sha256(cmp_blob).hexdigest()},
                      "region_polygons": {"dir": args.regions_dir, "sha256": region_digests}},
           "definitions": {"tool": "scripts/coast_crossing_measurement.py, measure(), unchanged", "endpoint": meas["native"]["endpoint"],
                           "cohort": "first observation in June to September, in the Africa polygon under the archive's rule, first latitude 0 to 25 N inclusive, not starting at the record's first step",
                           "event": "any observation at or before the endpoint classified North Atlantic under the archive's rule with raw longitude at or west of 7.5 W",
                           "grouping": "the instrument's whole-record duplicate rule, a sensitivity", "genesis_bands": "season_metrics.band_of, ten degrees of first longitude",
                           "tracks": "the finished smoothed arrays as the archive holds them"},
           "runs": {name: {"archive_rule_n": meas[name]["archive_rule_n"], "cohort_n": meas[name]["cohort_n"],
                           "outside_cohort_latitude": len(meas[name]["outside_cohort_latitude_ids"]), "left_censored_excluded": len(meas[name]["left_censored_excluded_ids"]),
                           "outcomes_by_first_latitude_band": outcomes_table(meas[name]),
                           "grouping_sensitivity": {"units": meas[name]["grouping_sensitivity"]["units"], "grouped_units": meas[name]["grouping_sensitivity"]["grouped_units"],
                                                    "total_row": {k: meas[name]["grouping_sensitivity"]["by_band"]["total"][k] for k in ("cohort", "atlantic_side", "gulf_only", "none", "follow_up_incomplete", "fraction_atlantic_side_over_cohort", "fraction_atlantic_side_over_complete_follow_up", "denominators")}},
                           "description": describe(meas[name], by_id[name]), "finished_tracks": len(by_id[name])} for name in ("native", "margin")},
           "correspondence": cohort_correspondence(meas["native"], meas["margin"], cls_n, cls_m, native, margin),
           "cohort_tracks": {name: meas[name]["tracks"] for name in ("native", "margin")}}
    X.publish_json(args.out, out, exclusive=True)
    rn, rm = out["runs"]["native"], out["runs"]["margin"]
    tn, tm = rn["outcomes_by_first_latitude_band"]["total"], rm["outcomes_by_first_latitude_band"]["total"]
    c = out["correspondence"]
    print(f"{year}: cohort native {rn['cohort_n']} (archive rule {rn['archive_rule_n']}), margin {rm['cohort_n']} ({rm['archive_rule_n']}). "
          f"Atlantic-side {tn['atlantic_side']}/{tn['denominators']['cohort']} against {tm['atlantic_side']}/{tm['denominators']['cohort']} over the cohort, "
          f"{tn['atlantic_side']}/{tn['denominators']['complete_follow_up']} against {tm['atlantic_side']}/{tm['denominators']['complete_follow_up']} over complete follow-up. "
          f"Native cohort by class {c['native_cohort_by_class']}. Left {c['left_the_cohort_through_a_counterpart']['count']}, entered {c['entered_the_cohort_through_a_counterpart']['count']}. "
          f"Mutual pairs in both {c['members_in_both_cohorts_through_a_counterpart']['mutual_pairs']} with summaries differing {c['mutual_pairs_in_both_cohorts']['summaries_differing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
