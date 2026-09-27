#!/usr/bin/env python3
"""The origin-dependent comparison on the final three-season coast-crossing artifacts:
the same cohort, event and endpoint, re-tabulated by four PREDECLARED bands of first
recorded longitude (edges 10 W, 10 E and 30 E, fixed from geography before any outcome
fraction by longitude was inspected), with the latitude composition and the season kept
visible inside every band, empty cells retained, every fraction with both denominators,
the stored-track count primary and the grouping sensitivity secondary. Nothing is
re-classified: every outcome, band and first position is read from the season artifacts,
whose digests the output records.

    .venv/bin/python3 scripts/coast_crossing_origin.py --artifacts <dir> --years 1990 2002 2007 --out <json>

With --pilot-years, the pooled tables are also given for those years alone and for the
remaining years alone, so an extension can show its pilot seasons apart from the rest.
--pattern names the season artifact files (a {year} placeholder).
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import exact_tracks as X  # noqa: E402

LON_EDGES = (-10.0, 10.0, 30.0)
LON_BANDS = ("west of 10 W", "10 W to 10 E", "10 E to 30 E", "east of 30 E")
LAT_BANDS = ("0 to 5", "5 to 10", "10 to 15", "15 to 20", "20 to 25")
STATUSES = ("atlantic_side", "gulf_only", "none", "follow_up_incomplete")
ARTIFACT = "coast_crossing_{year}_v3_2026-09-27.json"


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def lon_band(lon):
    """[lower, upper) with edges at 10 W, 10 E and 30 E: a start exactly on an edge belongs
    to the band east of it."""
    if lon < LON_EDGES[0]:
        return LON_BANDS[0]
    if lon < LON_EDGES[1]:
        return LON_BANDS[1]
    if lon < LON_EDGES[2]:
        return LON_BANDS[2]
    return LON_BANDS[3]


def status(unit):
    return "follow_up_incomplete" if unit["follow_up_incomplete"] else unit["outcome"]


def _empty_row():
    return {"cohort": 0, **{s: 0 for s in STATUSES}, "single_observation_entrants": 0,
            "latitude_composition": {lab: {"cohort": 0, "atlantic_side": 0} for lab in LAT_BANDS}}


def tabulate(units):
    """Counts by longitude band (every band present, empty or not) and in total."""
    rows = {lab: _empty_row() for lab in LON_BANDS}
    rows["total"] = _empty_row()
    for u in units:
        lab = lon_band(float(u["first"]["lon"]))
        for row in (rows[lab], rows["total"]):
            row["cohort"] += 1
            row[status(u)] += 1
            if u.get("single_observation_entrant"):
                row["single_observation_entrants"] += 1
            comp = row["latitude_composition"][u["band"]]
            comp["cohort"] += 1
            if status(u) == "atlantic_side":
                comp["atlantic_side"] += 1
    for row in rows.values():
        complete = row["cohort"] - row["follow_up_incomplete"]
        row["denominators"] = {"cohort": row["cohort"], "complete_follow_up": complete}
        row["fraction_atlantic_side_over_cohort"] = (row["atlantic_side"] / row["cohort"]) if row["cohort"] else None
        row["fraction_atlantic_side_over_complete_follow_up"] = (row["atlantic_side"] / complete) if complete else None
    return rows


def units_of(side):
    """The grouping sensitivity's units, as the season artifact defines them: every
    ungrouped cohort track plus one unit per group with a cohort member."""
    ungrouped = [t for t in side["tracks"] if t.get("group") is None]
    return ungrouped + side["grouping_sensitivity"]["group_units"]


def bind(side, tracks_rows, units_rows):
    """The re-tabulation must reproduce the season artifact's totals exactly."""
    for name, rows, ref in (("tracks", tracks_rows, side["by_band"]["total"]), ("units", units_rows, side["grouping_sensitivity"]["by_band"]["total"])):
        got = rows["total"]
        for key in ("cohort", "atlantic_side", "gulf_only", "none", "follow_up_incomplete", "single_observation_entrants"):
            if got[key] != ref[key]:
                raise SystemExit(f"REFUSED: the {name} total for {key} is {got[key]} where the season artifact says {ref[key]}")
    if units_rows["total"]["cohort"] != side["grouping_sensitivity"]["units"]:
        raise SystemExit("REFUSED: the unit count does not equal the season artifact's")


def run(artifacts_dir, years, out, pattern=ARTIFACT, pilot_years=()):
    seasons, pooled = {}, {"this_record": {"tracks": [], "units": []}, "qtrack": {"tracks": [], "units": []}}
    inputs = {}
    subsets = {"pilot_years": [y for y in years if y in pilot_years], "remaining_years": [y for y in years if y not in pilot_years]}
    sub_pool = {k: {"this_record": {"tracks": [], "units": []}, "qtrack": {"tracks": [], "units": []}} for k in subsets}
    for y in years:
        path = os.path.join(artifacts_dir, pattern.format(year=y))
        art = json.load(open(path))
        if art.get("year") != y:
            raise SystemExit(f"REFUSED: {path} is not the artifact for {y}")
        inputs[str(y)] = {"path": path, "sha256": _sha256(path), "script_sha256": art["script_sha256"]}
        seasons[str(y)] = {}
        for name, side in art["sides"].items():
            units = units_of(side)
            tr, un = tabulate(side["tracks"]), tabulate(units)
            bind(side, tr, un)
            seasons[str(y)][name] = {"by_longitude_band": tr, "grouping_sensitivity_by_longitude_band": un,
                                     "first_positions": [{"id": t["id"], "lon": t["first"]["lon"], "lat": t["first"]["lat"], "status": status(t)} for t in side["tracks"]]}
            pooled[name]["tracks"].extend(side["tracks"])
            pooled[name]["units"].extend(units)
            for k, ys in subsets.items():
                if y in ys:
                    sub_pool[k][name]["tracks"].extend(side["tracks"])
                    sub_pool[k][name]["units"].extend(units)
    pool_rows = lambda pool: {name: {"by_longitude_band": tabulate(v["tracks"]), "grouping_sensitivity_by_longitude_band": tabulate(v["units"])} for name, v in pool.items()}
    pooled_rows = pool_rows(pooled)
    subset_rows = {k: {"years": ys, **pool_rows(sub_pool[k])} for k, ys in subsets.items()} if pilot_years else {}
    payload = {
        "generated_by": "scripts/coast_crossing_origin.py", "script_sha256": X.digest(__file__),
        "what_this_is": "the three-season coast-crossing measurement re-tabulated by predeclared bands of first recorded longitude, with "
                        "latitude composition and season visible; a descriptive comparison of stored tracks, not a separation of causes",
        "longitude_bands": {"edges": list(LON_EDGES), "labels": list(LON_BANDS), "convention": "[lower, upper), a start on an edge belongs to the band east of it",
                            "status": "predeclared from geography before any outcome fraction by longitude was inspected"},
        "years": list(years), "season_artifact_pattern": pattern, "inputs": inputs, "seasons": seasons, "pooled_three_seasons": pooled_rows,
        "pooled_all_years": pooled_rows, "pooled_subsets": subset_rows}
    try:
        X.publish_json(out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {out} exists and artifacts are never overwritten")
    for name, rows in pooled_rows.items():
        print(name + ": " + ", ".join(f"{lab} {r['atlantic_side']} of {r['cohort']}" for lab, r in rows["by_longitude_band"].items()))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default=ARTIFACT, help="season artifact file name with a {year} placeholder")
    ap.add_argument("--pilot-years", nargs="*", type=int, default=[])
    a = ap.parse_args(argv)
    return run(a.artifacts, a.years, a.out, a.pattern, tuple(a.pilot_years))


if __name__ == "__main__":
    sys.exit(main())
