#!/usr/bin/env python3
"""The margin replay's comparison, one season: the native and the margin finished tracks
under the rules declared in `EASTERN_MARGIN_REPLAY_2026-10-06.md` before the runs. It
quantifies sensitivity to the boundary preparation. It does not say which record is
better, and it attributes no cause.

THE RULES, as declared.

1. CONTENT EQUALITY, with duplicate multiplicities, separately from ORDER. The finished
   smoothed arrays (time, latitude, longitude) are compared as a multiset, the way the
   archive gate compares them, and the raw histories (steps, claimed positions) as a
   multiset too. Whether the finished lists are also in the same order is reported on its
   own and never folded into equality.
2. IDENTITY IS CONTENT. A native track and a margin track are identical when their raw
   histories are identical (the same steps and the same claimed positions). Identical
   histories pair by count, so a history present twice on one side and once on the other
   leaves one copy unpaired. Finished indices and the runs' private birth labels are never
   equated across runs.
3. PAIRING THE REST BY SHARED OBSERVATIONS. For the tracks left after step 2, the overlap
   of two histories is the number of (step, latitude, longitude) observations they share.
   Each native track's best match is the margin track with the largest positive overlap,
   and each margin track's likewise. A pair is formed only when the two are each other's
   unique best match (mutual, no tie). A track whose best match is not mutual, or is tied,
   is AMBIGUOUS and is retained with its candidates and their overlaps. A track with no
   shared observation is UNMATCHED and is retained with its birth position and length.
   Nothing is forced onto one counterpart.
4. FIRST DIVERGENCE FROM RAW HISTORIES. For a mutual pair, the two raw histories, each
   ordered by step, are walked together. If they begin alike, the divergence is the first
   observation at which they differ, and the divergence longitude is the longitude of the
   last common observation, with both runs' first differing observations recorded. If they
   begin apart and share observations later (converging histories), the pair is recorded
   as converging, with each run's first observation and the first shared observation, and
   the divergence longitude is each run's birth longitude. Smoothed positions are never
   used for this.
5. THE BAND, beside the longitudes. Every divergence longitude and every unmatched track's
   birth longitude is reported as it is, and also classified as within 2 degrees of 40 E
   (at or east of 38 E) or farther west, with 5-degree bins over the domain, the converging
   pairs' birth longitudes binned on their own for each run.

    python3 scripts/pilot_margin_compare.py --native <reference json.gz> --margin <candidate json.gz> --out <fresh json> [--contrast preparation|calibration|extent|preparation_and_calibration] [--band-west-edge 38]
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

BAND_WEST_EDGE = 38.0            # within 2 degrees of 40 E
BIN_DEG = 5.0


def load_gz(path):
    with gzip.open(path, "rb") as fh:
        blob = fh.read()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def raw_key(track):
    return tuple(zip(track["steps"], track["raw_lat"], track["raw_lon"]))


def smoothed_key(track):
    return (tuple(track["time"]), tuple(track["lat"]), tuple(track["lon"]))


def content_and_order(native, margin):
    """Rule 1."""
    out = {}
    for name, key in (("smoothed_arrays", smoothed_key), ("raw_histories", raw_key)):
        cn, cm = collections.Counter(key(t) for t in native), collections.Counter(key(t) for t in margin)
        out[name] = {"native_tracks": sum(cn.values()), "margin_tracks": sum(cm.values()), "shared_with_multiplicity": sum((cn & cm).values()),
                     "only_native": sum((cn - cm).values()), "only_margin": sum((cm - cn).values()), "equal_as_multiset": cn == cm,
                     "same_order": [key(t) for t in native] == [key(t) for t in margin]}
    return out


def pair(native, margin):
    """Rules 2 and 3. Returns identical pairs (by index), mutual pairs, ambiguous and
    unmatched tracks on each side."""
    by_key_n, by_key_m = collections.defaultdict(list), collections.defaultdict(list)
    for i, t in enumerate(native):
        by_key_n[raw_key(t)].append(i)
    for j, t in enumerate(margin):
        by_key_m[raw_key(t)].append(j)
    identical, rest_n, rest_m = [], [], []
    for key, idx_n in by_key_n.items():
        idx_m = by_key_m.get(key, [])
        n_pairs = min(len(idx_n), len(idx_m))
        identical.extend(zip(idx_n[:n_pairs], idx_m[:n_pairs]))
        rest_n.extend(idx_n[n_pairs:])
    for key, idx_m in by_key_m.items():
        n_pairs = min(len(by_key_n.get(key, [])), len(idx_m))
        rest_m.extend(idx_m[n_pairs:])
    obs_n = {i: set(raw_key(native[i])) for i in rest_n}
    obs_m = {j: set(raw_key(margin[j])) for j in rest_m}
    overlap = {(i, j): len(obs_n[i] & obs_m[j]) for i in rest_n for j in rest_m}

    def best(i, side):
        cands = [(overlap[(i, j)] if side == "n" else overlap[(j, i)], j) for j in (rest_m if side == "n" else rest_n)]
        cands = [(o, j) for o, j in cands if o > 0]
        if not cands:
            return None, []
        top = max(o for o, _ in cands)
        winners = sorted(j for o, j in cands if o == top)
        return top, winners
    mutual, ambiguous_n, ambiguous_m, unmatched_n, unmatched_m = [], [], [], [], []
    best_n = {i: best(i, "n") for i in rest_n}
    best_m = {j: best(j, "m") for j in rest_m}
    for i in rest_n:
        top, winners = best_n[i]
        if top is None:
            unmatched_n.append(i)
        elif len(winners) == 1 and best_m[winners[0]][1] == [i]:
            mutual.append((i, winners[0], top))
        else:
            ambiguous_n.append({"native": i, "overlap": top, "candidates": winners, "their_best": {str(j): best_m[j][1] for j in winners}})
    paired_m = {j for _i, j, _o in mutual}
    for j in rest_m:
        top, winners = best_m[j]
        if top is None:
            unmatched_m.append(j)
        elif j not in paired_m:
            ambiguous_m.append({"margin": j, "overlap": top, "candidates": winners, "their_best": {str(i): best_n[i][1] for i in winners}})
    return {"identical": identical, "mutual": mutual, "ambiguous_native": ambiguous_n, "ambiguous_margin": ambiguous_m,
            "unmatched_native": unmatched_n, "unmatched_margin": unmatched_m}


def divergence(tn, tm, band=BAND_WEST_EDGE):
    """Rule 4, for one mutual pair, the band edge a parameter (38 E for the 40 E edge, 58 E for the 60 E edge)."""
    kn, km = raw_key(tn), raw_key(tm)
    common = 0
    while common < min(len(kn), len(km)) and kn[common] == km[common]:
        common += 1
    if common > 0:
        last = kn[common - 1]
        first_n = kn[common] if common < len(kn) else None
        first_m = km[common] if common < len(km) else None
        return {"kind": "diverging", "common_prefix_observations": common, "last_common": {"step": last[0], "lat": last[1], "lon": last[2]},
                "first_differing_native": None if first_n is None else {"step": first_n[0], "lat": first_n[1], "lon": first_n[2]},
                "first_differing_margin": None if first_m is None else {"step": first_m[0], "lat": first_m[1], "lon": first_m[2]},
                "divergence_lon": float(last[2]), "within_band": bool(last[2] >= band)}
    shared = sorted(set(kn) & set(km))
    first_shared = shared[0] if shared else None
    return {"kind": "converging", "common_prefix_observations": 0, "shared_observations": len(shared),
            "first_native": {"step": kn[0][0], "lat": kn[0][1], "lon": kn[0][2]}, "first_margin": {"step": km[0][0], "lat": km[0][1], "lon": km[0][2]},
            "first_shared": None if first_shared is None else {"step": first_shared[0], "lat": first_shared[1], "lon": first_shared[2]},
            "divergence_lon": {"native_birth": float(kn[0][2]), "margin_birth": float(km[0][2])},
            "within_band": {"native_birth": bool(kn[0][2] >= band), "margin_birth": bool(km[0][2] >= band)}}


def bins(lons):
    edges = np.arange(-140.0, 65.0, BIN_DEG)             # to the 60 E domain's edge. A 40 E record never reaches a bin beyond 45
    counts = {}
    for lo in lons:
        k = int(np.floor((lo + 140.0) / BIN_DEG))
        k = max(0, min(k, edges.size - 1))
        label = f"[{edges[k]:.0f}, {edges[k] + BIN_DEG:.0f})"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: float(kv[0].split(",")[0][1:])))


def summarize(native, margin, band=BAND_WEST_EDGE):
    content = content_and_order(native, margin)
    pairs = pair(native, margin)
    diverging, converging = [], []
    for i, j, overlap in pairs["mutual"]:
        d = divergence(native[i], margin[j], band)
        d.update({"native": i, "margin": j, "overlap": overlap, "native_observations": len(native[i]["steps"]), "margin_observations": len(margin[j]["steps"]),
                  "native_birth": native[i]["birth"], "margin_birth": margin[j]["birth"]})
        (diverging if d["kind"] == "diverging" else converging).append(d)

    def unmatched_rows(tracks, idx):
        return [{"index": i, "birth": tracks[i]["birth"], "observations": len(tracks[i]["steps"]), "birth_lat": tracks[i]["raw_lat"][0], "birth_lon": tracks[i]["raw_lon"][0],
                 "birth_within_band": bool(tracks[i]["raw_lon"][0] >= band), "last_lon": tracks[i]["raw_lon"][-1]} for i in idx]
    un_n, un_m = unmatched_rows(native, pairs["unmatched_native"]), unmatched_rows(margin, pairs["unmatched_margin"])
    div_lons = [d["divergence_lon"] for d in diverging]
    band_counts = {"diverging_pairs_within_band": sum(1 for d in diverging if d["within_band"]), "diverging_pairs_west_of_band": sum(1 for d in diverging if not d["within_band"]),
            "converging_pairs": len(converging),
            "converging_native_births_within_band": sum(1 for d in converging if d["within_band"]["native_birth"]),
            "converging_margin_births_within_band": sum(1 for d in converging if d["within_band"]["margin_birth"]),
            "unmatched_native_births_within_band": sum(1 for u in un_n if u["birth_within_band"]), "unmatched_native_births_west": sum(1 for u in un_n if not u["birth_within_band"]),
            "unmatched_margin_births_within_band": sum(1 for u in un_m if u["birth_within_band"]), "unmatched_margin_births_west": sum(1 for u in un_m if not u["birth_within_band"])}
    return {"content_and_order": content,
            "counts": {"native_finished": len(native), "margin_finished": len(margin), "identical_pairs": len(pairs["identical"]), "mutual_changed_pairs": len(pairs["mutual"]),
                       "ambiguous_native": len(pairs["ambiguous_native"]), "ambiguous_margin": len(pairs["ambiguous_margin"]),
                       "unmatched_native": len(pairs["unmatched_native"]), "unmatched_margin": len(pairs["unmatched_margin"])},
            "band_west_edge_lon": band, "band": band_counts,
            "divergence_longitudes_sorted": sorted(round(x, 3) for x in div_lons), "divergence_longitude_bins": bins(div_lons),
            "converging_birth_longitude_bins": {"native": bins([d["divergence_lon"]["native_birth"] for d in converging]),
                                                "margin": bins([d["divergence_lon"]["margin_birth"] for d in converging])},
            "unmatched_birth_longitude_bins": {"native": bins([u["birth_lon"] for u in un_n]), "margin": bins([u["birth_lon"] for u in un_m])},
            "diverging_pairs": diverging, "converging_pairs": converging,
            "ambiguous_native": pairs["ambiguous_native"], "ambiguous_margin": pairs["ambiguous_margin"],
            "unmatched_native": un_n, "unmatched_margin": un_m}


CONTRASTS = {
    "preparation": {"native": "the native preparation under the released configuration, which reproduces the archived tracks",
                    "margin": "the all-edge margin preparation on the same control run and wide case under the same thresholds"},
    "calibration": {"native": "the all-edge margin preparation under the released thresholds",
                    "margin": "the all-edge margin preparation under the recomputed thresholds, on the same control run and wide case"},
    "extent": {"native": "the narrower domain, all-edge margin preparation, recomputed thresholds",
               "margin": "the wider domain, all-edge margin preparation, the same recomputed thresholds as a declared transfer"},
    "preparation_and_calibration": {"native": "the native preparation under the released configuration, which reproduces the archived tracks",
                                    "margin": "the all-edge margin preparation under the recomputed thresholds on the same control run, so preparation and calibration change together"},
    "calibration_audits": {"native": "the all-edge margin preparation under a pair transferred from another domain's audit",
                           "margin": "the all-edge margin preparation under the domain's own audit pair, on the same control run and wide case"},
}


def audit_domain_is(thresholds, control_domain):
    """Whether a replay's audit was calibrated on the control's own domain."""
    dom = (thresholds.get("calibration_audit") or {}).get("domain") or {}
    return control_domain is not None and list(dom.get("lat_range", [])) == [float(x) for x in control_domain["lat"]] and list(dom.get("lon_range", [])) == [float(x) for x in control_domain["lon"]]


def applied(rec):
    return float(rec["thresholds"]["coarse"]), float(rec["thresholds"]["fine"])


def audit_check(rec):
    """What a replay's calibration audit file says about the replay, when the file can be
    found at the recorded path. The file's digest must equal the recorded one and the
    replay's applied pair must equal the audit's smooth_then_crop pair, refused otherwise.
    A replay under the control's pair has no audit to check. A file that cannot be found
    is recorded as not found, and nothing is claimed for it."""
    t = rec["thresholds"]
    audit = t.get("calibration_audit")
    if not isinstance(audit, dict):
        return {"applies": False}
    path = audit.get("path")
    if not path or not os.path.exists(path):
        return {"applies": True, "path": path, "found": False}
    with open(path, "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != audit.get("sha256"):
        raise SystemExit(f"REFUSED: the audit at {path} has digest {digest[:12]}, not the {str(audit.get('sha256'))[:12]} the replay recorded")
    stc = (json.loads(blob).get("orders") or {}).get("smooth_then_crop") or {}
    try:
        pair = (float(stc["coarse"]["threshold"]), float(stc["fine"]["threshold"]))
    except (KeyError, TypeError, ValueError):
        raise SystemExit(f"REFUSED: the audit at {path} carries no smooth_then_crop pair")
    if pair != applied(rec):
        raise SystemExit(f"REFUSED: the replay applies {applied(rec)}, not the audit's smooth_then_crop pair {pair}")
    return {"applies": True, "path": path, "found": True, "sha256": digest, "applied_pair_is_the_audits_smooth_then_crop_pair": True}


def same_control(a, b):
    """One control run (record, tracks, case) and one wide case on both sides."""
    return (all(a["runs"]["B"].get(k) == b["runs"]["B"].get(k) for k in ("dir", "record_sha256", "tracks_sha256", "case_sha256"))
            and a["runs"]["wide"].get("case_sha256") == b["runs"]["wide"].get("case_sha256"))


def provenance(rec):
    """Where a replay's applied pair came from, the control run's record or a calibration
    audit identified by its digest. An audit entry without a digest, or of the wrong
    shape, is refused rather than read as a control pair, since a provenance without an
    identity cannot be compared with another."""
    thresholds = rec["thresholds"]
    if "calibration_audit" not in thresholds:
        return ("control", None)
    audit = thresholds["calibration_audit"]
    digest = audit.get("sha256") if isinstance(audit, dict) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise SystemExit("REFUSED: the replay's calibration provenance carries no audit digest, so it cannot be compared")
    return ("audit", digest)


def gate(contrast, native, margin):
    """What the declared contrast requires of its two inputs, refused otherwise. The roles
    are assigned by argument position, the first input the reference and the second the
    candidate, and each input's recorded provenance must match its role, so a pair given
    in the wrong order is refused rather than read the other way round. The record keys
    stay native (the reference) and margin (the candidate) in every contrast."""
    if contrast not in CONTRASTS:
        raise SystemExit(f"REFUSED: unknown contrast {contrast!r}")
    preparations = (native["preparation"], margin["preparation"])
    source_n, source_m = provenance(native), provenance(margin)
    if contrast in ("preparation", "preparation_and_calibration"):
        if preparations != ("native", "margin") or native["year"] != margin["year"]:
            raise SystemExit("REFUSED: the inputs are not a native and a margin replay of one season")
        if not native["archive_comparison"]["passed"]:
            raise SystemExit("REFUSED: the native replay did not reproduce the archived tracks")
        if not same_control(native, margin) or (contrast == "preparation" and (applied(native) != applied(margin) or source_m[0] != "control")):
            raise SystemExit("REFUSED: the two replays do not share their runs and thresholds")
        if source_n[0] != "control":
            raise SystemExit(f"REFUSED: the {contrast} contrast's reference applies the control run's own pair, this one applies an audit's")
        if contrast == "preparation_and_calibration" and (source_m[0] != "audit" or applied(native) == applied(margin)):
            raise SystemExit("REFUSED: the preparation_and_calibration contrast's candidate applies the audit's pair, which must differ from the reference's")
    elif contrast == "calibration":
        if preparations != ("margin", "margin") or native["year"] != margin["year"] or not same_control(native, margin):
            raise SystemExit("REFUSED: the calibration contrast needs two margin replays of one season on one control run and wide case")
        if source_n[0] != "control" or source_m[0] != "audit":
            raise SystemExit("REFUSED: the calibration contrast's reference applies the control run's own pair and its candidate the audit's, these do not")
        if applied(native) == applied(margin):
            raise SystemExit("REFUSED: the calibration contrast needs two different threshold pairs, these are the same")
    elif contrast == "calibration_audits":
        if preparations != ("margin", "margin") or native["year"] != margin["year"] or not same_control(native, margin):
            raise SystemExit("REFUSED: the calibration_audits contrast needs two margin replays of one season on one control run and wide case")
        if source_n[0] != "audit" or source_m[0] != "audit" or source_n[1] == source_m[1]:
            raise SystemExit("REFUSED: the calibration_audits contrast needs two different audit provenances, one on each side")
        control_domain = native["runs"]["B"].get("control_domain")
        if not (native["thresholds"].get("calibration_audit") or {}).get("domain"):                # unchecked never means passed
            raise SystemExit("REFUSED: the calibration_audits contrast's reference records no audit domain, so a transfer cannot be told from a domain calibration")
        if not native["thresholds"].get("transfer") or audit_domain_is(native["thresholds"], control_domain):
            raise SystemExit("REFUSED: the calibration_audits contrast's reference applies a pair transferred from another domain's audit, this one does not")
        if not margin["thresholds"].get("domain_calibration") or not audit_domain_is(margin["thresholds"], control_domain):
            raise SystemExit("REFUSED: the calibration_audits contrast's candidate applies the domain's own audit pair, this one does not")
        if applied(native) == applied(margin):
            raise SystemExit("REFUSED: the calibration_audits contrast needs two different threshold pairs, these are the same")
    elif contrast == "extent":
        if preparations != ("margin", "margin") or native["year"] != margin["year"]:
            raise SystemExit("REFUSED: the extent contrast needs two margin replays of one season")
        if same_control(native, margin):
            raise SystemExit("REFUSED: the extent contrast needs two different control runs, these share one")
        if applied(native) != applied(margin):
            raise SystemExit("REFUSED: the extent contrast needs one threshold pair on both domains, these differ")
        if source_n[0] != "audit" or source_n != source_m:
            raise SystemExit("REFUSED: the extent contrast needs one threshold pair from one audit on both domains, these do not come from one audit")
        da, db = native["runs"]["B"].get("control_domain"), margin["runs"]["B"].get("control_domain")
        if not (da and db) or list(da["lat"]) != list(db["lat"]) or da["lon"][0] != db["lon"][0] or not da["lon"][1] < db["lon"][1]:
            raise SystemExit("REFUSED: the extent contrast needs two control domains that differ in the eastern edge only, the candidate's farther east")


VERIFICATION_DECLARED = ("both_margin", "one_applied_pair", "one_calibration_audit", "same_tracker_flags", "same_year", "same_inputs", "same_climatology")
VERIFICATION_RAW_FIELDS = ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c", "u", "v", "currv_anom")                     # the instrument's seven raw fields
VERIFICATION_PREPARED_FIELDS = ("coarse_curvature", "coarse_advection", "coarse_zonal_wind_smoothed",                     # and its six prepared fields
                                "fine_curvature", "fine_zonal_wind_smoothed", "fine_meridional_wind_smoothed")


def bound_verification(path, sha_a, sha_b, year):
    """The extent verification record read for an extent comparison. It must be the
    verification instrument's record, bound to these two replays (the reference as a,
    the candidate as b) and this season, and its counts and declared equalities, not its
    flag, must say the extent is the only difference. Returns the identity carried into
    the comparison record."""
    with open(path, "rb") as fh:
        blob = fh.read()
    v = json.loads(blob)
    if v.get("generated_by") != "scripts/pilot_extent_verification.py":
        raise SystemExit("REFUSED: the verification is not the verification instrument's record")
    inputs = v.get("inputs", {})
    if (inputs.get("replay_a", {}).get("sha256"), inputs.get("replay_b", {}).get("sha256"), v.get("year")) != (sha_a, sha_b, year):
        raise SystemExit("REFUSED: the verification record is not bound to these two replays (the reference as a, the candidate as b) and this season")
    raw, prepared = v.get("raw_shared_cells_differing", {}), v.get("prepared_control_a_cells_differing", {})
    if not isinstance(raw, dict) or not isinstance(prepared, dict) or set(raw) != set(VERIFICATION_RAW_FIELDS) or set(prepared) != set(VERIFICATION_PREPARED_FIELDS):
        raise SystemExit("REFUSED: the verification record does not carry the seven raw and the six prepared field counts by name")
    counts = [raw[k] for k in VERIFICATION_RAW_FIELDS] + [prepared[k] for k in VERIFICATION_PREPARED_FIELDS]
    declared = v.get("declared", {})
    declared_hold = all(declared.get(k) is True for k in VERIFICATION_DECLARED) and declared.get("boxes", {}).get("same_latitudes_and_western_edge") is True
    if any(c != 0 for c in counts) or not declared_hold:
        raise SystemExit("REFUSED: the verification record did not find the extent the only difference between these replays (its counts and declared equalities are read, not its flag)")
    return {"path": path, "sha256": hashlib.sha256(blob).hexdigest(), "script_sha256": v.get("script_sha256"), "replays": {"a": sha_a, "b": sha_b},
            "steps": v.get("steps"), "extent_is_the_only_difference": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--native", "--margin", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--contrast", choices=tuple(CONTRASTS), default="preparation", help="what the two inputs differ in, declared. The record keys stay native (reference) and margin (candidate)")
    ap.add_argument("--band-west-edge", type=float, default=BAND_WEST_EDGE, help="the band's western edge in degrees east, 38 for the 40 E edge and 58 for the 60 E edge")
    ap.add_argument("--verification", help="the extent verification record made from these two replays, required for the extent contrast and refused for the others")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    native, sha_n = load_gz(args.native)
    margin, sha_m = load_gz(args.margin)
    if native.get("tracker_flags") != margin.get("tracker_flags"):
        raise SystemExit(f"REFUSED: the two replays do not share their tracker flags ({native.get('tracker_flags')} against {margin.get('tracker_flags')})")
    gate(args.contrast, native, margin)
    audits = {"native": audit_check(native), "margin": audit_check(margin)}
    verification = None
    if args.contrast == "extent":
        if not args.verification:
            raise SystemExit("REFUSED: the extent contrast needs its verification record (--verification), made from these two replays")
        verification = bound_verification(args.verification, sha_n, sha_m, native["year"])
    elif args.verification:
        raise SystemExit("REFUSED: a verification record belongs to the extent contrast only")
    s = summarize(native["finished"], margin["finished"], args.band_west_edge)
    out = {"generated_by": "scripts/pilot_margin_compare.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": native["year"],
           "contrast": args.contrast, "roles": CONTRASTS[args.contrast], "tracker_flags": native.get("tracker_flags"), "verification": verification, "audits": audits,
           "inputs": {"native": {"path": args.native, "sha256": sha_n, "script_sha256": native["script_sha256"], "preparation": native["preparation"],
                                 "archive_comparison": native["archive_comparison"], "native_bind": native["native_bind"]},
                      "margin": {"path": args.margin, "sha256": sha_m, "script_sha256": margin["script_sha256"], "preparation": margin["preparation"],
                                 "archive_comparison": margin["archive_comparison"]}},
           "runs": {"native": native["runs"], "margin": margin["runs"]}, "thresholds": {"native": native["thresholds"], "margin": margin["thresholds"]},
           "rules": __doc__.split("THE RULES, as declared.")[1].split("    python3")[0].strip(), **s}
    X.publish_json(args.out, out, exclusive=True)
    c, b = s["counts"], s["band"]
    print(f"{native['year']} {args.contrast}: native {c['native_finished']} finished, margin {c['margin_finished']}. Smoothed multiset equal {s['content_and_order']['smoothed_arrays']['equal_as_multiset']}, "
          f"raw multiset equal {s['content_and_order']['raw_histories']['equal_as_multiset']}. Identical {c['identical_pairs']}, changed mutual {c['mutual_changed_pairs']} "
          f"(diverging within band {b['diverging_pairs_within_band']}, west {b['diverging_pairs_west_of_band']}, converging {b['converging_pairs']}), "
          f"ambiguous {c['ambiguous_native']}/{c['ambiguous_margin']}, unmatched {c['unmatched_native']}/{c['unmatched_margin']}. "
          f"Audit files found, reference {audits['native'].get('found')}, candidate {audits['margin'].get('found')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
