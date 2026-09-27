#!/usr/bin/env python3
"""The three-season descriptive coast-crossing measurement, one season per run, both
records, under the coast-crossing measurement brief of 2026-09-27 as amended before the
run (its rules are restated below). Every number is a count of STORED TRACKS.

THE COHORT (each record, its full tracks): first observation in June to September of the
year and in the Africa polygon under the archive's rule (quarter-degree rounding as MATLAB
rounds, polygons in priority order, boundary inside), first latitude between 0 and 25 N
inclusive (exploratory), and not left-censored (a first observation at the record's first
time step is excluded and listed). Archive-rule tracks outside 0 to 25 N are listed as
outside the exploratory cohort so the full denominators stay visible.

THE OUTCOME, on observations at or before the FOLLOW-UP ENDPOINT, October 31 18Z of the
season for both records: ATLANTIC-SIDE if any observation is classified North Atlantic
under the archive's rule and its raw longitude is at or west of the 7.5 W cutoff
(lon <= -7.5, the meridian itself counting as west), otherwise GULF-ONLY if any North
Atlantic observation lies east of the cutoff, otherwise NONE. A track without an
Atlantic-side event whose last observation is at or after the endpoint is FOLLOW-UP
INCOMPLETE. Observations after the endpoint are counted but take no part. The sectors a
track visited are recorded, so a Gulf entry followed by an Atlantic-side observation is
Atlantic-side with both sectors recorded. Single-observation entrants (exactly one
qualifying Atlantic-side observation) are counted separately.

THE GROUPING SENSITIVITY: group membership from the full records (this record's duplicate
groups under the instrument's whole-record rule, QTrack's shared-tail pairs), retained
with the members outside the cohort listed, but a group's outcome, band and first
position come only from its cohort members. Follow-up completeness is kept separate
from the observed sector: the group's sector is Atlantic-side if any cohort member's
is, else Gulf-only if any is, else none, and its follow-up is incomplete when no cohort
member has an Atlantic-side event and any cohort member's follow-up is incomplete,
including a Gulf-only group. A group with one cohort member keeps that member's status.
It is a sensitivity to the grouping rule.

    .venv/bin/python3 scripts/coast_crossing_measurement.py --year 1990 --campaign-evidence <dir> \\
        --qtrack-dir <dir> --regions-dir <dir> --out <json>
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

CUTOFF_LON = -7.5                                      # raw longitude, at or west counts as Atlantic side
COHORT_LAT = (0.0, 25.0)                               # first latitude, inclusive both ends, exploratory
BANDS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0))   # [lo, hi), the last closed at 25
OUTCOMES = ("atlantic_side", "gulf_only", "none")


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def endpoint_day(year):
    import qtrack_pairing_pilot as M
    return M._day(year, 10, 31) + 0.75


def band_of(lat):
    for lo, hi in BANDS:
        if lo <= lat < hi:
            return f"{lo:g} to {hi:g}"
    if lat == BANDS[-1][1]:
        return f"{BANDS[-1][0]:g} to {BANDS[-1][1]:g}"
    raise SystemExit(f"REFUSED: first latitude {lat} is outside the cohort's bands")


def band_labels():
    return [f"{lo:g} to {hi:g}" for lo, hi in BANDS]


def region_sequence(track, regions, upto):
    import season_metrics as S
    return [S.source_region({"lon": [lo], "lat": [la]}, regions) for lo, la in zip(track["lon"][:upto], track["lat"][:upto])]


def classify(track, regions, endpoint):
    """One track's outcome on the observations at or before the endpoint."""
    import season_metrics as S
    t = np.asarray(track["time"], float)
    upto = int(np.sum(t <= endpoint + 1e-9))
    seq = region_sequence(track, regions, upto)
    lon = np.asarray(track["lon"], float)
    atl = [k for k in range(upto) if seq[k] == "NAL" and lon[k] <= CUTOFF_LON]
    gulf = [k for k in range(upto) if seq[k] == "NAL" and lon[k] > CUTOFF_LON]
    alive = bool(t[-1] >= endpoint - 1e-9)
    outcome = "atlantic_side" if atl else ("gulf_only" if gulf else "none")
    sectors = ("both" if (atl and gulf) else "atlantic_side" if atl else "gulf" if gulf else "none")
    obs = lambda k: {"lon": float(lon[k]), "lat": float(track["lat"][k]), "time": str(S.date_of(t[k])), "index": int(k)}
    return {"outcome": outcome, "sectors_visited": sectors, "follow_up_incomplete": bool(outcome != "atlantic_side" and alive),
            "alive_at_endpoint": alive, "n_observations_used": upto, "n_observations_after_endpoint": int(t.size - upto),
            "n_atlantic_side_observations": len(atl), "single_observation_entrant": bool(len(atl) == 1),
            "first_atlantic_side": obs(atl[0]) if atl else None, "first_gulf": obs(gulf[0]) if gulf else None,
            "last_used": obs(upto - 1)}


def cohort(tracks, year, regions, record_first_day):
    """The archive's rule, the exploratory latitude range, the left-censor exclusion.
    Returns (cohort tracks, outside-latitude ids, left-censored ids)."""
    import season_metrics as S
    keep, outside, left = [], [], []
    for t in tracks:
        if not (t["time"].size and S.in_season(t, year) and S.in_record_domain(t, regions)):
            continue
        if not (COHORT_LAT[0] <= float(t["lat"][0]) <= COHORT_LAT[1]):
            outside.append(t["id"])
        elif abs(float(t["time"][0]) - record_first_day) < 1e-9:
            left.append(t["id"])
        else:
            keep.append(t)
    return keep, outside, left


def groups_ours(tracks):
    import qtrack_pairing_pilot as M
    return M.duplicate_groups_ours(tracks)


def groups_qtrack(path):
    """QTrack's shared-tail pairs as groups (connected components), {system id: group}."""
    from aew import qtrack as Q
    data = Q.read_year(path)
    pairs = Q.shared_tail_pairs(data["lon"], data["lat"])
    ids = [int(s) for s in np.asarray(data["system"])]
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for p in pairs:
        a, b = find(ids[p["a"]]), find(ids[p["b"]])
        if a != b:
            parent[a] = b
    roots = {}
    for i in ids:
        roots.setdefault(find(i), []).append(i)
    return {i: g for g, members in enumerate(v for v in roots.values() if len(v) > 1) for i in members}


def tabulate(records):
    """Counts by band and in total from a list of per-unit records (tracks or groups)."""
    labels = band_labels() + ["total"]
    rows = {lab: {"cohort": 0, "atlantic_side": 0, "gulf_only": 0, "none": 0, "follow_up_incomplete": 0, "single_observation_entrants": 0} for lab in labels}
    for r in records:
        for lab in (r["band"], "total"):
            row = rows[lab]
            row["cohort"] += 1
            if r["follow_up_incomplete"]:
                row["follow_up_incomplete"] += 1
            else:
                row[r["outcome"]] += 1
            if r.get("single_observation_entrant"):
                row["single_observation_entrants"] += 1
    for row in rows.values():
        complete = row["cohort"] - row["follow_up_incomplete"]
        row["fraction_atlantic_side_over_cohort"] = (row["atlantic_side"] / row["cohort"]) if row["cohort"] else None
        row["fraction_atlantic_side_over_complete_follow_up"] = (row["atlantic_side"] / complete) if complete else None
        row["denominators"] = {"cohort": row["cohort"], "complete_follow_up": complete}
    return rows


def group_units(track_records, membership):
    """Units under the grouping rule: each group with at least one cohort member is one
    unit whose outcome, band and first position come from its cohort members only;
    every ungrouped cohort track is its own unit."""
    by_group, units = {}, []
    for r in track_records:
        g = membership.get(r["id"])
        if g is None:
            units.append({**{k: r[k] for k in ("band", "outcome", "follow_up_incomplete", "single_observation_entrant", "first")}, "members": [r["id"]], "grouped": False})
        else:
            by_group.setdefault(g, []).append(r)
    for g, members in sorted(by_group.items()):
        members = sorted(members, key=lambda r: r["first"]["time"])
        # the brief's section 0a: sector and follow-up kept separate. The sector is Atlantic-side if any
        # cohort member's is, else Gulf-only if any is, else none; follow-up is incomplete when no cohort
        # member has an Atlantic-side event and any cohort member's follow-up is incomplete, including a
        # Gulf-only group. A group with one cohort member keeps that member's status exactly.
        outcomes = [m["outcome"] for m in members]
        outcome = "atlantic_side" if "atlantic_side" in outcomes else ("gulf_only" if "gulf_only" in outcomes else "none")
        incomplete = bool(outcome != "atlantic_side" and any(m["follow_up_incomplete"] for m in members))
        units.append({"band": members[0]["band"], "outcome": outcome, "follow_up_incomplete": incomplete,
                      "single_observation_entrant": bool(outcome == "atlantic_side" and all(m["single_observation_entrant"] for m in members if m["outcome"] == "atlantic_side")),
                      "first": members[0]["first"], "members": [m["id"] for m in members], "grouped": True, "group": g})
    return units


def measure(tracks, year, regions, record_first_day, membership):
    endpoint = endpoint_day(year)
    keep, outside, left = cohort(tracks, year, regions, record_first_day)
    import season_metrics as S
    records = []
    for t in keep:
        c = classify(t, regions, endpoint)
        records.append({"id": t["id"], "first": {"lon": float(t["lon"][0]), "lat": float(t["lat"][0]), "time": str(S.date_of(t["time"][0]))},
                        "band": band_of(float(t["lat"][0])), "n_observations": int(t["time"].size), "group": membership.get(t["id"]), **c})
    cohort_ids = {r["id"] for r in records}
    groups = {}
    for tid, g in membership.items():
        groups.setdefault(g, {"members_in_cohort": [], "members_outside_cohort": []})
        groups[g]["members_in_cohort" if tid in cohort_ids else "members_outside_cohort"].append(tid)
    groups = {str(g): v for g, v in groups.items() if v["members_in_cohort"]}
    units = group_units(records, membership)
    return {"endpoint": str(S.date_of(endpoint)), "cohort_n": len(records), "outside_cohort_latitude_ids": outside, "left_censored_excluded_ids": left,
            "archive_rule_n": len(records) + len(outside) + len(left),
            "by_band": tabulate(records), "grouping_sensitivity": {"units": len(units), "grouped_units": sum(1 for u in units if u["grouped"]),
                                                                   "groups_with_cohort_members": groups, "by_band": tabulate(units),
                                                                   "group_units": [u for u in units if u["grouped"]]},
            "tracks": records}


def run(year, campaign_evidence, qtrack_dir, regions_dir, out):
    import qtrack_pairing_pilot as M
    import season_metrics as S
    ours_path = os.path.join(campaign_evidence, f"era5_{year}", "tracker_port.mat")
    record_path = os.path.join(campaign_evidence, f"era5_{year}", f"tracking_era5_{year}.json")
    record = json.load(open(record_path))["dataset_specific"]
    if record.get("tracks_sha256") != _sha256(ours_path) or record.get("year") != year or record.get("dataset") != "era5":
        raise SystemExit("REFUSED: the campaign's ERA5 tracks file is not the one its record names")
    q_path = M.strict_path(qtrack_dir, M.QTRACK_FILE, year)
    ours, case = M.load_ours(ours_path)
    q, q_meta = M.load_qtrack(q_path)
    regions, digests = S.load_regions(regions_dir)
    endpoint = endpoint_day(year)
    if abs(q_meta["last_day"] - endpoint) > 1e-9:
        raise SystemExit(f"REFUSED: QTrack's file for {year} does not end at the follow-up endpoint")
    sides = {"this_record": measure(ours, year, regions, M._day(year, 1, 1), groups_ours(ours)),
             "qtrack": measure(q, year, regions, q_meta["first_day"], groups_qtrack(q_path))}
    payload = {
        "generated_by": "scripts/coast_crossing_measurement.py", "script_sha256": X.digest(__file__), "year": year,
        "what_this_is": "a descriptive measurement on stored tracks: whether each cohort track is recorded on the Atlantic side of the "
                        "archive's West African coast at or before the follow-up endpoint; not a physical-wave probability, not a "
                        "replication of the 2021 study",
        "rules": {"cohort": "first observation in June to September, in the Africa polygon under the archive's rule, first latitude "
                            f"{COHORT_LAT[0]:g} to {COHORT_LAT[1]:g} N inclusive (exploratory), not starting at the record's first time step",
                  "event": f"any observation at or before the endpoint classified North Atlantic under the archive's rule with raw longitude <= {CUTOFF_LON} "
                           "(the meridian counts as west); North Atlantic observations east of it are Gulf sector visits",
                  "endpoint": f"October 31 18Z of the season for both records ({str(S.date_of(endpoint))}); later observations preserved, not used",
                  "follow_up_incomplete": "no Atlantic-side event and the last observation at or after the endpoint",
                  "bands": [list(b) for b in BANDS], "bands_status": "exploratory, first latitude",
                  "grouping": "this record's duplicate groups (instrument's whole-record rule) and QTrack's shared-tail pairs, outcomes and bands "
                              "from cohort members only, the sector (Atlantic-side, else Gulf-only, else none) kept separate from follow-up, "
                              "which is incomplete when no cohort member has an Atlantic-side event and any cohort member is incomplete; "
                              "a sensitivity to the grouping rule"},
        "inputs": {"this_record": {"path": ours_path, "sha256": _sha256(ours_path), "case_id": case, "first_day": str(S.date_of(M._day(year, 1, 1)))},
                   "qtrack": {"path": q_path, "sha256": _sha256(q_path), **q_meta},
                   "campaign_record": {"path": record_path, "sha256": _sha256(record_path)},
                   "region_polygons": {"dir": regions_dir, "sha256": digests}},
        "sides": sides}
    try:
        X.publish_json(out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {out} exists and artifacts are never overwritten")
    for name, s in sides.items():
        tot = s["by_band"]["total"]
        print(f"{year} {name}: cohort {tot['cohort']} (archive rule {s['archive_rule_n']}, outside 0-25 N {len(s['outside_cohort_latitude_ids'])}, "
              f"left-censored {len(s['left_censored_excluded_ids'])}), atlantic side {tot['atlantic_side']}, gulf only {tot['gulf_only']}, "
              f"none {tot['none']}, incomplete {tot['follow_up_incomplete']}, single-observation {tot['single_observation_entrants']}; "
              f"grouping units {s['grouping_sensitivity']['units']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--campaign-evidence", required=True)
    ap.add_argument("--qtrack-dir", required=True)
    ap.add_argument("--regions-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return run(a.year, a.campaign_evidence, a.qtrack_dir, a.regions_dir, a.out)


if __name__ == "__main__":
    sys.exit(main())
