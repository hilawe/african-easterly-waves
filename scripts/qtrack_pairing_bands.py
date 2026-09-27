#!/usr/bin/env python3
"""The pairing pilot's three seasons tabulated by EXPLORATORY latitude bands of the first
observation in coverage, on both sides, with the matching rules and the assignments taken
FIXED from the retained pairing artifacts. Nothing is re-paired. Each eligible track is
counted once per tolerance as paired, without a candidate, or having lost the one-to-one
assignment, with its duplicate-group (this record) or shared-tail (QTrack) membership,
and the same partition is repeated by classes of in-coverage length, because a pairing
that needs three shared observations can select on duration and reach.

The eligible populations are rebuilt through the pilot's own loaders from the inputs the
artifacts name, after verifying their digests, and are then BOUND to the artifacts: the
counts must equal the artifacts' and every identifier must appear in exactly one of the
artifact's three status lists, or the run refuses. Separately, and independently of any
matching, the script counts the two records' Africa-origin populations under the
archive's membership rule (first observation, quarter-degree rounding, first containing
polygon in priority order, season months), which is the eligibility unit proposed for a
later coast-crossing statistic. It computes NO coast-crossing statistic.

    .venv/bin/python3 scripts/qtrack_pairing_bands.py --artifacts <dir> --years 1990 2002 2007 --out <json>
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import exact_tracks as X  # noqa: E402

# EXPLORATORY bands of first-observation latitude in coverage, [lo, hi), the last one closed
# at the region's northern edge. The pilot's 5 N and 20 N splits are edges here, so its
# section 6 counts are sums of these.
BANDS = ((-20.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0), (25.0, 35.0))
# EXPLORATORY classes of in-coverage observations (six-hourly), [lo, hi]
LENGTH_CLASSES = ((3, 7), (8, 15), (16, 31), (32, 10 ** 9))
STATUSES = ("paired", "no_candidate", "lost_assignment")


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def band_of(lat):
    """The band holding a latitude, refusing one outside the bands' span."""
    for lo, hi in BANDS:
        if lo <= lat < hi:
            return f"{lo:g} to {hi:g}"
    if lat == BANDS[-1][1]:
        lo, hi = BANDS[-1]
        return f"{lo:g} to {hi:g}"
    raise SystemExit(f"REFUSED: first latitude {lat} lies outside the bands")


def band_label(lo, hi):
    return f"{lo:g} to {hi:g}"


def length_class(n):
    for lo, hi in LENGTH_CLASSES:
        if lo <= n <= hi:
            return f"{lo} to {hi}" if hi < 10 ** 9 else f"{lo} or more"
    raise SystemExit(f"REFUSED: {n} observations in coverage is below the pilot's minimum")


def length_label(lo, hi):
    return f"{lo} to {hi}" if hi < 10 ** 9 else f"{lo} or more"


def verified_inputs(art):
    """The two records and the region polygons the artifact names, refused unless every
    digest is the artifact's, and the eligible populations under the artifact's own window
    and region, through the pilot's loaders."""
    import qtrack_pairing_pilot as M
    import season_metrics as S
    for key in ("this_record", "qtrack"):
        path = art["inputs"][key]["path"]
        if _sha256(path) != art["inputs"][key]["sha256"]:
            raise SystemExit(f"REFUSED: {path} is not the file the artifact names")
    rdir = art["inputs"]["region_polygons"]["dir"]
    recorded = art["inputs"]["region_polygons"]["sha256"]
    # the digests are compared BEFORE the polygons are parsed, so a changed file is refused as
    # not the artifact's rather than failing inside the reader
    if set(recorded) != {name for _, name in S.REGION_FILES}:
        raise SystemExit("REFUSED: the artifact's region polygon digests are incomplete")
    if {name: _sha256(os.path.join(rdir, f"{name}.mat")) for name in recorded} != recorded:
        raise SystemExit("REFUSED: the region polygons are not the ones the artifact names")
    regions, digests = S.load_regions(rdir)
    if digests != recorded:
        raise SystemExit("REFUSED: the region polygons are not the ones the artifact names")
    window = tuple(art["rules"]["window_days_since_1900"])
    region = tuple(art["rules"]["region_lat_lon"])
    ours, _ = M.load_ours(art["inputs"]["this_record"]["path"])
    q, _ = M.load_qtrack(art["inputs"]["qtrack"]["path"])
    a_el, _, _ = M.populations(ours, window, region)
    b_el, _, _ = M.populations(q, window, region)
    return ours, q, a_el, b_el, regions


def statuses(art, tol):
    """{id: status} per side from the artifact at one tolerance: the three lists must
    partition the eligible population exactly, or the artifact is refused."""
    r = art["results"][tol]
    out = {}
    for side, key in (("a", "a"), ("b", "b")):
        m = {}
        for p in r["pairs"]:
            if p[key] in m:
                raise SystemExit(f"REFUSED: {side} {p[key]} appears twice among the pairs")
            m[p[key]] = "paired"
        for u in r[side]["no_candidate"]:
            if u["id"] in m:
                raise SystemExit(f"REFUSED: {side} {u['id']} appears in two status lists")
            m[u["id"]] = "no_candidate"
        for u in r[side]["lost_assignment"]:
            if u["id"] in m:
                raise SystemExit(f"REFUSED: {side} {u['id']} appears in two status lists")
            m[u["id"]] = "lost_assignment"
        if len(m) != r[side]["n"]:
            raise SystemExit(f"REFUSED: side {side} at {tol} km lists {len(m)} statuses for {r[side]['n']} eligible tracks")
        out[side] = m
    return out


def bind(art, a_el, b_el, tol):
    """The rebuilt eligible populations are the artifact's: same size, same identifiers
    as the status lists, and every pair's first observation equal to the rebuilt one."""
    st = statuses(art, tol)
    for side, el in (("a", a_el), ("b", b_el)):
        ids = {t["id"] for t in el}
        if len(el) != art["populations"][side]["eligible"] or ids != set(st[side]):
            raise SystemExit(f"REFUSED: the rebuilt population {side} is not the artifact's")
    first = {"a": {t["id"]: t for t in a_el}, "b": {t["id"]: t for t in b_el}}
    for p in art["results"][tol]["pairs"]:
        for side in ("a", "b"):
            t = first[side][p[side]]
            f = p[f"{side}_first"]
            if not (abs(float(t["lat"][0]) - f["lat"]) < 1e-9 and abs(float(t["lon"][0]) - f["lon"]) < 1e-9 and abs(float(t["time"][0]) - f["time"]) < 1e-9):
                raise SystemExit(f"REFUSED: pair {p['a']},{p['b']} does not start where the artifact says on side {side}")
    return st


def _table(el, status, members, subset, keyfn, labels):
    """Counts by a class of each eligible track: eligible, the three statuses, the members
    of the involvement set (duplicate groups or shared tails) overall and per status, and
    the subset (Africa-origin or Atlantic filter)."""
    rows = {lab: {"eligible": 0, "paired": 0, "no_candidate": 0, "lost_assignment": 0, "members": 0,
                  "members_paired": 0, "members_no_candidate": 0, "members_lost_assignment": 0, "subset": 0, "subset_paired": 0}
            for lab in labels}
    for t in el:
        lab = keyfn(t)
        row = rows[lab]
        s = status[t["id"]]
        row["eligible"] += 1
        row[s] += 1
        if t["id"] in members:
            row["members"] += 1
            row[f"members_{s}"] += 1
        if t["id"] in subset:
            row["subset"] += 1
            if s == "paired":
                row["subset_paired"] += 1
    return rows


def full_band_count(lats):
    """Every latitude counted once: the exploratory bands plus one overflow row on each
    side, so the total equals the population and nothing outside the bands' span is
    dropped silently (the Africa polygon reaches 34.75 S, the bands start at 20 S)."""
    labels = [band_label(lo, hi) for lo, hi in BANDS]
    out = {f"south of {BANDS[0][0]:g}": 0, **{lab: 0 for lab in labels}, f"north of {BANDS[-1][1]:g}": 0}
    for lat in lats:
        if lat < BANDS[0][0]:
            out[f"south of {BANDS[0][0]:g}"] += 1
        elif lat > BANDS[-1][1]:
            out[f"north of {BANDS[-1][1]:g}"] += 1
        else:
            out[band_of(lat)] += 1
    if sum(out.values()) != len(lats):
        raise SystemExit("REFUSED: the band count does not add up to the population")
    return out


def africa_origin_full(tracks, year, regions):
    """The archive's membership rule on the FULL tracks of either record: first observation
    in the season months of the year, and in the Africa polygon under the archive's
    rounding and priority order. Independent of any matching."""
    import season_metrics as S
    return sorted(t["id"] for t in tracks if S.in_season(t, year) and S.in_record_domain(t, regions))


def tabulate(year, art):
    ours, q, a_el, b_el, regions = verified_inputs(art)
    a_members = set(art["populations"]["a"]["duplicate_group_members"])
    b_members = set(art["populations"]["b"]["shared_tail_members"])
    a_subset = set(art["populations"]["a"]["africa_origin_in_season_ids"])
    b_subset = set(art["populations"]["b"]["atlantic_filter_ids"])
    band_labels = [band_label(lo, hi) for lo, hi in BANDS]
    length_labels = [length_label(lo, hi) for lo, hi in LENGTH_CLASSES]
    by_lat = lambda t: band_of(float(t["lat"][0]))
    by_len = lambda t: length_class(int(t["in_coverage"]))
    out = {"year": year, "pairing_artifact_sha256": None, "by_tolerance": {}}
    for tol in art["results"]:
        st = bind(art, a_el, b_el, tol)
        out["by_tolerance"][tol] = {
            "a_by_band": _table(a_el, st["a"], a_members, a_subset, by_lat, band_labels),
            "b_by_band": _table(b_el, st["b"], b_members, b_subset, by_lat, band_labels),
            "a_by_length": _table(a_el, st["a"], a_members, a_subset, by_len, length_labels),
            "b_by_length": _table(b_el, st["b"], b_members, b_subset, by_len, length_labels)}
    # the eligibility unit proposed for a later statistic, on both records, independent of matching
    a_afr = africa_origin_full(ours, year, regions)
    if a_afr != sorted(a_subset):
        raise SystemExit("REFUSED: the archive rule on this record does not reproduce the artifact's Africa-origin subset")
    b_afr = africa_origin_full(q, year, regions)
    first_lat = lambda tracks, ids: [float(t["lat"][0]) for t in tracks if t["id"] in set(ids)]
    band_count = full_band_count
    out["archive_rule_populations"] = {
        "rule": "first observation in June to September of the year, rounded to the quarter degree as MATLAB rounds, first containing "
                "polygon in the archive's priority order is Africa; on the full tracks, no coverage restriction, no matching",
        "this_record": {"n": len(a_afr), "ids": a_afr, "first_latitude_by_band": band_count(first_lat(ours, a_afr))},
        "qtrack": {"n": len(b_afr), "ids": b_afr, "first_latitude_by_band": band_count(first_lat(q, b_afr))},
        "what_this_is_not": "no coast-crossing statistic was computed; these are the two denominators the proposed unit would use"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    seasons = []
    for y in args.years:
        path = os.path.join(args.artifacts, f"pairing_{y}_2026-09-27.json")
        art = json.load(open(path))
        if art.get("year") != y:
            raise SystemExit(f"REFUSED: {path} is not the artifact for {y}")
        season = tabulate(y, art)
        season["pairing_artifact_sha256"] = _sha256(path)
        seasons.append(season)
    payload = {
        "generated_by": "scripts/qtrack_pairing_bands.py", "script_sha256": X.digest(__file__),
        "pilot_script_sha256": X.digest(os.path.join(HERE, "qtrack_pairing_pilot.py")),
        "what_this_is": "the pilot's fixed assignments counted by exploratory first-latitude bands and in-coverage length classes, "
                        "and the archive-rule Africa-origin populations of both records; not re-paired, no coast-crossing statistic",
        "bands_lat": [list(b) for b in BANDS], "bands_status": "exploratory, chosen for this tabulation, not a domain claim",
        "length_classes_observations": [[lo, (hi if hi < 10 ** 9 else None)] for lo, hi in LENGTH_CLASSES],
        "seasons": seasons}
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    for s in seasons:
        t = s["by_tolerance"]["500"]
        print(f"{s['year']}: A {sum(r['eligible'] for r in t['a_by_band'].values())} eligible, B {sum(r['eligible'] for r in t['b_by_band'].values())}, "
              f"archive-rule Africa-origin {s['archive_rule_populations']['this_record']['n']} here and {s['archive_rule_populations']['qtrack']['n']} in QTrack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
