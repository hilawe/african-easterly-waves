#!/usr/bin/env python3
"""Boundary examples for the coast-crossing measurement definition, one season, both
records, NO statistic. For every stored track in the exploratory cohort (the archive's
membership rule, first observation in the Africa polygon in June to September, first
latitude between 0 and 25 N), the script walks the track's observations through the
archive's region rule and records the FIRST observation classified North Atlantic, if
any: where it is, which stretch of the polygon boundary it lies on (an exploratory
segment rule recorded in the artifact), how far it is from the last African observation
before it, whether the track later returns to the Africa polygon, and whether the track
touches either end of its record (a censoring flag). The entries are listed so the
definition can be checked against real boundary cases before anything is counted as a
crossing. The counts by segment describe the ENTRANTS' geometry, and no fraction of the
cohort is computed or printed, because the measurement is not yet defined.

    .venv/bin/python3 scripts/coast_entry_examples.py --year 1990 --campaign-evidence <dir> \\
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

COHORT_LAT = (0.0, 25.0)        # exploratory comparison cohort, first latitude inclusive
# EXPLORATORY stretches of the Africa / North Atlantic polygon boundary, by the entry point.
# The boundary runs from the Gulf of Guinea's north shore (about 5 N, 9 E to 7.5 W) round
# Cape Palmas (7.5 W, 4.4 N) and up the Atlantic-facing coast to Cap Blanc (17 W, 21 N)
# and on to Morocco. These are labels for reading the entries, not a domain claim.
SEGMENTS = (("west_coast_4_to_22N", "lon <= -7.5 and 4 <= lat <= 22"),
            ("gulf_of_guinea", "lon > -7.5 and lat < 6.5"),
            ("north_of_22N", "lat > 22"),
            ("other", "anything else"))


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def coast_segment(lon, lat):
    if lon <= -7.5 and 4.0 <= lat <= 22.0:
        return "west_coast_4_to_22N"
    if lon > -7.5 and lat < 6.5:
        return "gulf_of_guinea"
    if lat > 22.0:
        return "north_of_22N"
    return "other"


def region_sequence(track, regions):
    import season_metrics as S
    return [S.source_region({"lon": [lo], "lat": [la]}, regions) for lo, la in zip(track["lon"], track["lat"])]


def cohort(tracks, year, regions, record_first_day):
    """The archive's membership rule, then the exploratory first-latitude cohort, then
    the exclusion of LEFT-CENSORED tracks, whose first observation is the record's first
    time step and so is the window's edge rather than the tracker's start. Returns the
    cohort and the excluded left-censored tracks' identifiers."""
    import season_metrics as S
    out, left = [], []
    for t in tracks:
        if t["time"].size and S.in_season(t, year) and S.in_record_domain(t, regions) and COHORT_LAT[0] <= float(t["lat"][0]) <= COHORT_LAT[1]:
            if abs(float(t["time"][0]) - record_first_day) < 1e-9:
                left.append(t["id"])
            else:
                out.append(t)
    return out, left


def describe(track, regions, record_first_day, record_last_day):
    """One cohort track: its first North Atlantic observation if any, and its edges."""
    from aew.v1port.geometry import great_circle_distance
    import season_metrics as S
    seq = region_sequence(track, regions)
    entry = None
    for k, code in enumerate(seq):
        if code == "NAL":
            entry = k
            break
    d = {"id": track["id"], "n": int(track["time"].size),
         "first": {"lon": float(track["lon"][0]), "lat": float(track["lat"][0]), "time": str(S.date_of(track["time"][0])), "region": seq[0]},
         "last": {"lon": float(track["lon"][-1]), "lat": float(track["lat"][-1]), "time": str(S.date_of(track["time"][-1])), "region": seq[-1]},
         "starts_at_record_start": bool(abs(float(track["time"][0]) - record_first_day) < 1e-9),
         "ends_at_record_end": bool(abs(float(track["time"][-1]) - record_last_day) < 1e-9),
         "regions_visited": sorted(set(seq)), "north_atlantic_entry": None}
    if entry is not None:
        lo, la = float(track["lon"][entry]), float(track["lat"][entry])
        prev = entry - 1
        d["north_atlantic_entry"] = {
            "index": entry, "lon": lo, "lat": la, "time": str(S.date_of(track["time"][entry])),
            "segment": coast_segment(lo, la),
            "previous_region": seq[prev] if prev >= 0 else None,
            "step_from_previous_km": float(great_circle_distance(track["lat"][prev], track["lon"][prev], la, lo, "km")) if prev >= 0 else None,
            "steps_from_previous": int(round((float(track["time"][entry]) - float(track["time"][prev])) / 0.25)) if prev >= 0 else None,
            "later_africa_observations": int(sum(1 for c in seq[entry + 1:] if c == "AFR")),
            "north_atlantic_observations": int(sum(1 for c in seq if c == "NAL")),
            "entry_is_last_observation": bool(entry == len(seq) - 1)}
    return d


def summarize(entries):
    ent = [e for e in entries if e["north_atlantic_entry"] is not None]
    by_segment = {name: 0 for name, _ in SEGMENTS}
    for e in ent:
        by_segment[e["north_atlantic_entry"]["segment"]] += 1
    return {"cohort": len(entries), "with_north_atlantic_entry": len(ent), "entries_by_segment": by_segment,
            "entrants_returning_to_africa": sum(1 for e in ent if e["north_atlantic_entry"]["later_africa_observations"] > 0),
            "entrants_whose_entry_is_their_last_observation": sum(1 for e in ent if e["north_atlantic_entry"]["entry_is_last_observation"]),
            "entrants_with_a_gap_before_entry": sum(1 for e in ent if (e["north_atlantic_entry"]["steps_from_previous"] or 1) > 1),
            "cohort_starting_at_record_start": sum(1 for e in entries if e["starts_at_record_start"]),
            "cohort_ending_at_record_end": sum(1 for e in entries if e["ends_at_record_end"]),
            "cohort_ending_at_record_end_without_entry": sum(1 for e in entries if e["ends_at_record_end"] and e["north_atlantic_entry"] is None)}


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
    ours_first, ours_last = M._day(year, 1, 1), M._day(year, 12, 31) + 0.75
    seasons = {}
    for name, tracks, first_day, last_day in (("this_record", ours, ours_first, ours_last), ("qtrack", q, q_meta["first_day"], q_meta["last_day"])):
        c, left = cohort(tracks, year, regions, first_day)
        entries = [describe(t, regions, first_day, last_day) for t in c]
        if any(e["starts_at_record_start"] for e in entries):
            raise SystemExit("REFUSED: a left-censored track remained in the cohort")
        seasons[name] = {"record_first_day": str(S.date_of(first_day)), "record_last_day": str(S.date_of(last_day)),
                         "excluded_left_censored": left, "summary": summarize(entries), "tracks": entries}
    payload = {
        "generated_by": "scripts/coast_entry_examples.py", "script_sha256": X.digest(__file__), "year": year,
        "what_this_is": "boundary examples for a coast-crossing measurement definition, one season: each cohort track's first "
                        "North Atlantic observation under the archive's region rule, with its boundary segment and its record edges; "
                        "no fraction of the cohort is computed",
        "cohort_rule": "first observation in June to September of the year and in the Africa polygon under the archive's rule "
                       f"(quarter-degree rounding, priority order), first latitude between {COHORT_LAT[0]:g} and {COHORT_LAT[1]:g} N inclusive, "
                       "EXPLORATORY comparison cohort, not a confirmed common domain; a track whose first observation is the record's "
                       "first time step is left-censored and excluded, listed under excluded_left_censored",
        "entry_rule": "the first observation whose rounded position falls in the North Atlantic polygon under the same rule",
        "segments": {name: rule for name, rule in SEGMENTS}, "segments_status": "exploratory labels for reading entries",
        "inputs": {"this_record": {"path": ours_path, "sha256": _sha256(ours_path), "case_id": case},
                   "qtrack": {"path": q_path, "sha256": _sha256(q_path), **q_meta},
                   "campaign_record": {"path": record_path, "sha256": _sha256(record_path)},
                   "region_polygons": {"dir": regions_dir, "sha256": digests}},
        "seasons": seasons}
    try:
        X.publish_json(out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {out} exists and artifacts are never overwritten")
    for name, s in seasons.items():
        sm = s["summary"]
        print(f"{year} {name}: cohort {sm['cohort']}, entries by segment {sm['entries_by_segment']}, returning {sm['entrants_returning_to_africa']}, "
              f"ending at record end {sm['cohort_ending_at_record_end']}")
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
