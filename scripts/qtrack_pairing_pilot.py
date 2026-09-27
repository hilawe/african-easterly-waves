#!/usr/bin/env python3
"""One season of the pairing pilot: this record's campaign ERA5 tracks against QTrack's
published ERA5 record, under the rules predeclared in the pilot's brief of 2026-09-27
(both populations, both tolerances and the three seasons fixed before any result was seen).

The result is a set of STORED-TRACK CORRESPONDENCES under predeclared rules, never a
count of recovered physical waves. Both records are validated onto the six-hour grid,
observations are restricted to the common window (QTrack's June to October season) and
the common region (20 S to 35 N, 140 W to 40 E), a track is eligible with at least three
observations in coverage, and the pairing is one-to-one maximum-cardinality on the
median great-circle separation, at 500 km and at 350 km, with three shared observations.
Tracks with no observation in coverage, or one or two, are reported apart, never dropped
silently. Two labeled subsets ride beside the whole populations: this record's
Africa-origin tracks in season under the archive's rule, and the systems the retained
Atlantic filter kept. Neither changes the pairing.

    .venv/bin/python3 scripts/qtrack_pairing_pilot.py --year 1990 --out <json>
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import exact_tracks as X  # noqa: E402

QTRACK_FILE = re.compile(r"ERA5_AEW_tracks_with_basins_(\d{4})\.nc\Z")
ATLANTIC_FILE = re.compile(r"ERA5_AEW_tracks_atlantic_(\d{4})\.nc\Z")
REGION = (-20.0, 35.0, -140.0, 40.0)         # lat_min, lat_max, lon_min, lon_max, the two domains' intersection
TOLERANCES = (500.0, 350.0)                  # primary, then the one sensitivity the brief allows
MIN_SHARED = 3
MIN_COVERAGE = 3


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _day(year, month, day):
    import datetime as dt
    return float((dt.date(year, month, day) - dt.date(1900, 1, 1)).days)


def strict_path(directory, pattern, year):
    """The one file in `directory` whose name matches `pattern` for `year`, by the strict
    pattern and nothing looser: companions and derivatives never qualify."""
    hits = [n for n in sorted(os.listdir(directory)) if (m := pattern.match(n)) and int(m.group(1)) == year]
    if len(hits) != 1:
        raise SystemExit(f"REFUSED: {len(hits)} files in {directory} match the strict pattern for {year}: {hits}")
    return os.path.join(directory, hits[0])


def load_qtrack(path):
    """QTrack's systems as tracks in days since 1900, through the file's own time units,
    raw positions (not the smoothed pair), one-based system identifiers."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    try:
        tvar = d.variables["time"]
        units = str(getattr(tvar, "units", ""))
        raw_t = np.ma.filled(tvar[:], np.nan).astype(float)
        if units.startswith("hours since 1900-01-01"):
            days = raw_t / 24.0
        elif units.startswith("seconds since 1970-01-01"):
            days = raw_t / 86400.0 + _day(1970, 1, 1)
        else:
            raise SystemExit(f"REFUSED: unknown QTrack time units {units!r}")
        lon = np.ma.filled(d.variables["AEW_lon"][:], np.nan).astype(float)
        lat = np.ma.filled(d.variables["AEW_lat"][:], np.nan).astype(float)
        system = np.asarray(d.variables["system"][:]).astype(int)
    finally:
        d.close()
    from aew import pairing as P
    if not P.on_grid(days):                                      # the whole decoded axis, before any position masks a step away
        raise SystemExit(f"REFUSED: QTrack's time axis in {path} does not sit on the six-hour grid")
    tracks = []
    for k in range(lon.shape[0]):
        ok = np.isfinite(lon[k]) & np.isfinite(lat[k])
        tracks.append({"id": int(system[k]), "time": days[ok], "lat": lat[k][ok], "lon": lon[k][ok], "n_valid": int(ok.sum())})
    return tracks, {"time_units": units, "steps": int(raw_t.size), "first_day": float(days[0]), "last_day": float(days[-1])}


def load_ours(path):
    import season_metrics as S
    with open(path, "rb") as fh:
        blob = fh.read()
    tracks, case = S.read_mat_tracks(blob)
    return [{"id": k, "time": np.asarray(t["time"], float), "lat": np.asarray(t["lat"], float),
             "lon": np.asarray(t["lon"], float), "n_valid": int(np.asarray(t["time"]).size)} for k, t in enumerate(tracks)], case


def populations(tracks, window, region, min_coverage=MIN_COVERAGE):
    """Split a record into the eligible population (restricted to coverage) and the two
    excluded groups, by identifier, so nothing is dropped silently."""
    from aew import pairing as P
    eligible, none, few = [], [], []
    for t in tracks:
        mask = P.coverage(t, window, region)
        n = int(mask.sum())
        if n == 0:
            none.append(t["id"])
        elif n < min_coverage:
            few.append({"id": t["id"], "in_coverage": n})
        else:
            r = P.restrict(t, mask)
            r["n_valid"], r["in_coverage"] = t["n_valid"], n
            eligible.append(r)
    return eligible, none, few


def duplicate_groups_ours(tracks):
    """This record's duplicate groups under the instrument's rule: connected components
    of tracks sharing at least three timesteps within one degree (cosine-weighted).
    Returns {id: group index} for members of groups of size two or more."""
    import season_metrics as S
    parent = list(range(len(tracks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            n, sep = S.overlap_separation(tracks[i], tracks[j])
            if n >= S.WAVE_SHARED_STEPS and sep <= S.WAVE_SEPARATION_DEG:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    roots = {}
    for i, t in enumerate(tracks):
        roots.setdefault(find(i), []).append(t["id"])
    return {tid: g for g, members in enumerate(roots.values()) if len(members) > 1 for tid in members}


def shared_tail_members_qtrack(path):
    """QTrack systems in a shared-tail pair under the retained filter's rule, by system id."""
    from aew import qtrack as Q
    data = Q.read_year(path)
    pairs = Q.shared_tail_pairs(data["lon"], data["lat"])
    members = set()
    for p in pairs:
        members.add(int(data["system"][p["a"]]))
        members.add(int(data["system"][p["b"]]))
    return sorted(members), len(pairs)


def atlantic_systems(path):
    import netCDF4 as nc
    d = nc.Dataset(path)
    try:
        return sorted(int(s) for s in np.asarray(d.variables["system"][:]))
    finally:
        d.close()


def africa_origin_ids(tracks, year, regions_dir):
    """This record's Africa-origin tracks in season under the archive's rule, from the
    instrument's own functions, on the full tracks (not the coverage restriction), with
    the polygon digests the instrument records."""
    import season_metrics as S
    regions, digests = S.load_regions(regions_dir)
    return sorted(t["id"] for t in tracks if S.in_season(t, year) and S.in_record_domain(t, regions)), digests


def involvement(ids, assigned_ids):
    ids = set(ids)
    return {"members": len(ids), "assigned": len(ids & assigned_ids), "unassigned": len(ids - assigned_ids)}


def run(year, campaign_evidence, qtrack_dir, atlantic_dir, regions_dir, out):
    from aew import pairing as P
    q_path = strict_path(qtrack_dir, QTRACK_FILE, year)
    a_path = strict_path(atlantic_dir, ATLANTIC_FILE, year)
    ours_path = os.path.join(campaign_evidence, f"era5_{year}", "tracker_port.mat")
    record_path = os.path.join(campaign_evidence, f"era5_{year}", f"tracking_era5_{year}.json")
    record = json.load(open(record_path))["dataset_specific"]
    record_sha256 = _sha256(record_path)
    if record.get("tracks_sha256") != _sha256(ours_path) or record.get("year") != year or record.get("dataset") != "era5":
        raise SystemExit("REFUSED: the campaign's ERA5 tracks file is not the one its record names")
    q_tracks, q_meta = load_qtrack(q_path)
    ours, case = load_ours(ours_path)
    for name, tracks in (("QTrack", q_tracks), ("this record", ours)):
        bad = [t["id"] for t in tracks if not P.on_grid(t["time"])]
        if bad:
            raise SystemExit(f"REFUSED: {len(bad)} tracks of {name} do not sit on the six-hour grid, first {bad[:5]}")
    window = (q_meta["first_day"], q_meta["last_day"])
    expected = (_day(year, 6, 1), _day(year, 10, 31) + 0.75)
    if any(abs(w - e) > 1e-6 for w, e in zip(window, expected)):
        raise SystemExit(f"REFUSED: QTrack's window for {year} is {window}, not June 1 00Z to October 31 18Z")
    window = expected
    a_el, a_none, a_few = populations(ours, window, REGION)
    b_el, b_none, b_few = populations(q_tracks, window, REGION)
    africa, region_digests = africa_origin_ids(ours, year, regions_dir)
    atlantic = atlantic_systems(a_path)
    # duplicate groups under the instrument's rule are a property of the FULL tracks, as the
    # instrument computes them; clipping to coverage would change both the overlaps and the
    # separations and so the membership. The eligible population's members are then the
    # intersection, and the members outside it are counted apart.
    dup_full = duplicate_groups_ours(ours)
    eligible_ids_a = {t["id"] for t in a_el}
    dup_a = {tid: g for tid, g in dup_full.items() if tid in eligible_ids_a}
    tails_b, n_tail_pairs = shared_tail_members_qtrack(q_path)
    first_a = {t["id"]: {"time": float(t["time"][0]), "lat": float(t["lat"][0]), "lon": float(t["lon"][0])} for t in a_el}
    first_b = {t["id"]: {"time": float(t["time"][0]), "lat": float(t["lat"][0]), "lon": float(t["lon"][0])} for t in b_el}

    def subset(ids, side_ids, assigned, r_side):
        ids = set(ids) & set(side_ids)
        lost = {u["id"] for u in r_side["lost_assignment"]} & ids
        none = {u["id"] for u in r_side["no_candidate"]} & ids
        return {"eligible": len(ids), "assigned": len(ids & assigned), "lost_assignment": len(lost), "no_candidate": len(none),
                "fraction": (len(ids & assigned) / len(ids)) if ids else None}
    results = {}
    for tol in TOLERANCES:
        r = P.pair(a_el, b_el, tol, MIN_SHARED)
        a_assigned = {p["a"] for p in r["pairs"]}
        b_assigned = {p["b"] for p in r["pairs"]}
        a_ids, b_ids = [t["id"] for t in a_el], [t["id"] for t in b_el]
        for p in r["pairs"]:                                   # the first observation in coverage of each side, so a pair can be placed
            p["a_first"], p["b_first"] = first_a[p["a"]], first_b[p["b"]]
            p["a_africa_origin"], p["b_atlantic_filter"] = p["a"] in set(africa), p["b"] in set(atlantic)
        r["fractions"] = {
            "a_assigned_over_population_a": r["a"]["assigned"] / len(a_el) if a_el else None,
            "b_assigned_over_population_b": r["b"]["assigned"] / len(b_el) if b_el else None,
            "africa_origin_subset": subset(africa, a_ids, a_assigned, r["a"]),
            "atlantic_filter_subset": subset(atlantic, b_ids, b_assigned, r["b"])}
        r["distributions"] = P.distributions(r["pairs"])
        r["duplicate_involvement"] = {"a_duplicate_groups": involvement(dup_a, a_assigned),
                                      "b_shared_tail_members": involvement(tails_b, b_assigned)}
        results[str(int(tol))] = r
    out_payload = {
        "generated_by": "scripts/qtrack_pairing_pilot.py", "script_sha256": X.digest(__file__),
        "pairing_module_sha256": X.digest(os.path.join(HERE, "..", "src", "aew", "pairing.py")),
        "year": year, "what_this_is": "stored-track correspondences between two records under predeclared rules, not recovered physical waves",
        "inputs": {"this_record": {"path": ours_path, "sha256": _sha256(ours_path), "case_id": case, "record_tracks_sha256": record.get("tracks_sha256")},
                   "qtrack": {"path": q_path, "sha256": _sha256(q_path), **q_meta},
                   "atlantic_derivative": {"path": a_path, "sha256": _sha256(a_path)},
                   "campaign_record": {"path": record_path, "sha256": record_sha256},
                   "region_polygons": {"dir": regions_dir, "sha256": region_digests}},
        "rules": {"coordinates": "QTrack raw AEW_lon and AEW_lat, this record's mean positions", "grid": "quarter days since 1900, matched on equality",
                  "window_days_since_1900": list(window), "region_lat_lon": list(REGION), "min_coverage": MIN_COVERAGE, "min_shared": MIN_SHARED,
                  "separation": "median great-circle kilometers over shared observations in coverage",
                  "assignment": "one-to-one, maximum cardinality then minimum total median separation, Hungarian method, ties to the solver "
                                "with this record's tracks in file order and QTrack's systems in ascending system order",
                  "tolerances_km": list(TOLERANCES)},
        "populations": {"a": {"total": len(ours), "eligible": len(a_el), "no_coverage": a_none, "under_three": a_few,
                              "africa_origin_in_season_ids": africa, "duplicate_group_members": sorted(dup_a),
                              "duplicate_group_members_whole_record": len(dup_full),
                              "duplicate_group_members_outside_eligible": len(set(dup_full) - eligible_ids_a)},
                        "b": {"total": len(q_tracks), "eligible": len(b_el), "no_coverage": b_none, "under_three": b_few,
                              "atlantic_filter_ids": atlantic, "shared_tail_members": tails_b, "shared_tail_pairs": n_tail_pairs}},
        "results": results}
    try:
        X.publish_json(out, out_payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {out} exists and artifacts are never overwritten")
    for tol, r in results.items():
        print(f"{year} at {tol} km: {r['a']['assigned']} of {r['a']['n']} of this record assigned, {r['b']['assigned']} of {r['b']['n']} of QTrack, "
              f"{len(r['a']['lost_assignment'])} and {len(r['b']['lost_assignment'])} lost one-to-one with candidates")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--campaign-evidence", default="docs/aewc_v2/evidence/validation/protocol_campaign")
    ap.add_argument("--qtrack-dir", default="data/aewc_v2_pilot/ERA5_WITH_EPAC")
    ap.add_argument("--atlantic-dir", default="data/aewc_v2_pilot/ERA5_ATLANTIC")
    ap.add_argument("--regions-dir", default="data/aewc_v2_pilot/v1_src")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return run(args.year, args.campaign_evidence, args.qtrack_dir, args.atlantic_dir, args.regions_dir, args.out)


if __name__ == "__main__":
    sys.exit(main())
