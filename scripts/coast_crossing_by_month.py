#!/usr/bin/env python3
"""The stratum 10 E to 30 E, 5 to 15 N of the coast-crossing measurement, tabulated by
start month over the season artifacts given: Atlantic-side over cohort for each record
and each month of first observation, with incomplete follow-up counted apart. Nothing is
re-classified, since outcomes, first positions and first times are read from the season
artifacts, and the stratum totals must equal the origin re-tabulation's latitude
composition for the band (5 to 10 N plus 10 to 15 N) or the run refuses.

    .venv/bin/python3 scripts/coast_crossing_by_month.py --artifacts <dir> --origin <json> --years 1981 ... 2010 --out <json>
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

LON = (10.0, 30.0)                    # [lower, upper)
LAT = (5.0, 15.0)                     # [lower, upper)
MONTHS = (6, 7, 8, 9)
PATTERN = "coast_crossing_{year}_2026-09-27.json"


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def in_stratum(t):
    return LON[0] <= t["first"]["lon"] < LON[1] and LAT[0] <= t["first"]["lat"] < LAT[1]


def tabulate(seasons):
    rows = {n: {str(m): {"cohort": 0, "atlantic_side": 0, "gulf_only": 0, "none": 0, "follow_up_incomplete": 0} for m in MONTHS} for n in ("this_record", "qtrack")}
    for art in seasons:
        for n in rows:
            for t in art["sides"][n]["tracks"]:
                if not in_stratum(t):
                    continue
                m = str(int(t["first"]["time"][5:7]))
                if m not in rows[n]:
                    raise SystemExit(f"REFUSED: a cohort track starts in month {m}")
                r = rows[n][m]
                r["cohort"] += 1
                r["follow_up_incomplete" if t["follow_up_incomplete"] else t["outcome"]] += 1
    for n in rows:
        for r in rows[n].values():
            r["fraction_atlantic_side_over_cohort"] = (r["atlantic_side"] / r["cohort"]) if r["cohort"] else None
        rows[n]["total"] = {k: sum(rows[n][str(m)][k] for m in MONTHS) for k in ("cohort", "atlantic_side", "gulf_only", "none", "follow_up_incomplete")}
        rows[n]["total"]["fraction_atlantic_side_over_cohort"] = (rows[n]["total"]["atlantic_side"] / rows[n]["total"]["cohort"]) if rows[n]["total"]["cohort"] else None
    return rows


def bind(rows, origin):
    """The stratum totals equal the origin re-tabulation's 5 to 10 N plus 10 to 15 N
    composition of the 10 E to 30 E band, pooled over all years."""
    for n in rows:
        comp = origin["pooled_all_years"][n]["by_longitude_band"]["10 E to 30 E"]["latitude_composition"]
        cohort = comp["5 to 10"]["cohort"] + comp["10 to 15"]["cohort"]
        atl = comp["5 to 10"]["atlantic_side"] + comp["10 to 15"]["atlantic_side"]
        if rows[n]["total"]["cohort"] != cohort or rows[n]["total"]["atlantic_side"] != atl:
            raise SystemExit(f"REFUSED: {n} stratum totals {rows[n]['total']['cohort']}, {rows[n]['total']['atlantic_side']} do not equal the origin artifact's {cohort}, {atl}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--origin", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    origin = json.load(open(a.origin))
    if origin.get("years") != a.years:
        raise SystemExit("REFUSED: the origin artifact covers different years")
    seasons, inputs = [], {}
    for y in a.years:
        path = os.path.join(a.artifacts, PATTERN.format(year=y))
        art = json.load(open(path))
        if art.get("year") != y:
            raise SystemExit(f"REFUSED: {path} is not the artifact for {y}")
        if origin["inputs"][str(y)]["sha256"] != _sha256(path):
            raise SystemExit(f"REFUSED: {path} is not the season artifact the origin artifact names")
        seasons.append(art)
        inputs[str(y)] = {"path": path, "sha256": _sha256(path)}
    rows = tabulate(seasons)
    bind(rows, origin)
    payload = {"generated_by": "scripts/coast_crossing_by_month.py", "script_sha256": X.digest(__file__),
               "what_this_is": "the stratum 10 E to 30 E, 5 to 15 N by month of first observation, both records, stored-track counts, bound to the origin re-tabulation",
               "stratum": {"lon": list(LON), "lat": list(LAT), "convention": "[lower, upper)"}, "years": a.years,
               "inputs": {"origin": {"path": a.origin, "sha256": _sha256(a.origin)}, "seasons": inputs}, "by_month": rows}
    try:
        X.publish_json(a.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {a.out} exists and artifacts are never overwritten")
    for n in rows:
        print(n + ": " + ", ".join(f"month {m} {rows[n][str(m)]['atlantic_side']} of {rows[n][str(m)]['cohort']}" for m in MONTHS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
