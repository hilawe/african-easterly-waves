#!/usr/bin/env python3
"""Run the ported tracker on ERA-Interim and compare it against version 1's own record.

THIS IS THE MEASUREMENT THE PORT EXISTS TO SURVIVE. Version 1's published record is what
cannot be executed on a test case, and repeated independent checking found twenty-three places the
port's reading of its source was wrong. Every other check in this repository compares the
port against that same reading. This one compares it against what version 1 actually
produced: NCEI C00784, in data/aewc for 1983 to 2007.

It is also the only thing that can settle the convex hull's starting vertex, recorded in
association._hull_polygon as a divergence that is NOT ESTABLISHED and worth up to about a
degree of search-polygon offset.

WHAT A DIFFERENCE MEANS, and naming this before seeing the numbers is the point. A perfect
match would be suspicious, because three divergences are already recorded and two are
deliberate. Each has a signature and `aew.v1port.compare` reports all three:

  the port following a wave version 1 lost      -> region truncation, deliberate
  version 1 holding waves near zero longitude   -> contour parsing, deliberate
  tracks agreeing then drifting apart gradually -> the hull vertex, NOT established

A difference fitting none of those is a porting defect and cannot be explained away after
the fact.

WHAT THIS COMPARES. The published files held here are the AFRICA region only, so the port's
tracks are filed by basin and only the African ones are compared. The basin rule is itself
checked against all 12,163 published tracks elsewhere and agrees on every one, so filtering
this way is not adding an untested step.

    .venv/bin/python scripts/run_validation.py --years 2005
    .venv/bin/python scripts/run_validation.py --years 2005 --exclusive
    .venv/bin/python scripts/run_validation.py --years 2003 2004 2005 --out out.json

BUDGET about three minutes a year to track plus six minutes for the climatology, and note
the climatology needs ALL THIRTY YEARS retrieved before the anomalies are version 1's. The
script refuses to run on fewer rather than quietly producing a different anomaly.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import compare as C  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port import regions as G  # noqa: E402
from aew.v1port import validate as V  # noqa: E402

CLIMATOLOGY_YEARS = tuple(range(1981, 2011))
LEVEL = 700
REANALYSIS = "ERA-Int"
PUBLISHED_DIR = "data/aewc"


def published_path(year, directory=PUBLISHED_DIR):
    return os.path.join(directory, f"ERA-Int_ew_{LEVEL}hPa_{year}_AFR.nc")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", type=int, required=True,
                    help="years to track and compare; the published record covers 1983-2007")
    # THE BUFFERED ONE-DEGREE TREE IS THE DEFAULT, and the old one has to be asked for by
    # name. `data/eraint/v1port` is the original unbuffered 0.75 degree retrieval, which
    # this project has since MEASURED to be the wrong configuration: version 1 ran at one
    # degree with a buffered domain, and the coarse tracking grid follows from the input
    # spacing through `decimate_f`'s floor, so 0.75 gives a 2.25 degree mesh where version
    # 1 had 2.0. A review found this default able to recreate a disproven configuration in
    # silence, which is the failure mode a default is worst at announcing.
    ap.add_argument("--directory", default="data/eraint/v1port_buffered")
    ap.add_argument("--published", default=PUBLISHED_DIR)
    ap.add_argument("--exclusive", action="store_true",
                    help="apply the duplication repair; the default reproduces version 1")
    ap.add_argument("--tolerance-km", type=float, default=500.0)
    ap.add_argument("--allow-partial-climatology", action="store_true",
                    help="run with fewer than thirty years, which changes every anomaly")
    ap.add_argument("--climatology-years", nargs="+", type=int, default=None,
                    help="use exactly these years for the climatology. Only for measuring "
                         "how much the climatology window itself moves the result; a "
                         "validation run uses all thirty.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    have = set(L.available_years(args.directory, "eraint"))
    missing = sorted(set(CLIMATOLOGY_YEARS) - have)
    if missing and not args.allow_partial_climatology:
        print(f"the climatology needs {len(CLIMATOLOGY_YEARS)} years and {len(missing)} are "
              f"absent ({missing[0]}..{missing[-1]}).\nVersion 1's anomaly is a departure "
              f"from a mean over exactly 1981-2010, so a shorter window is a DIFFERENT "
              f"anomaly and the comparison would not be the one intended.\nLet the "
              f"retrieval finish, or pass --allow-partial-climatology to run anyway and "
              f"read the result as a smoke test.", file=sys.stderr)
        return 2
    if args.climatology_years:
        climatology_years = sorted(set(args.climatology_years) & have)
        missing = missing or ["(a deliberate subset was requested)"]
    else:
        climatology_years = sorted(have & set(CLIMATOLOGY_YEARS))

    for year in args.years:
        if year not in have:
            print(f"{year} has not been retrieved", file=sys.stderr)
            return 2
        if not os.path.exists(published_path(year, args.published)):
            print(f"no published file for {year} at {published_path(year, args.published)}",
                  file=sys.stderr)
            return 2

    started = time.time()
    print(f"climatology over {len(climatology_years)} years "
          f"({climatology_years[0]}-{climatology_years[-1]})"
          f"{'  PARTIAL' if missing else ''}", flush=True)
    climatology = L.build_climatology(
        climatology_years, args.directory, "eraint",
        progress=lambda y, n, t: print(f"   {y} ({n}/{t})", flush=True))
    print(f"   {len(climatology['keys'])} calendar steps, {time.time()-started:.0f}s",
          flush=True)

    coarse, fine = P.thresholds_for(REANALYSIS, LEVEL)
    print(f"thresholds: coarse {coarse:.3e}, fine {fine:.3e} "
          f"(version 1's own, hardcoded in find_ews_f.m)", flush=True)

    results = {"exclusive": args.exclusive, "years": {},
               "climatology_years": len(climatology_years),
               "partial_climatology": bool(missing)}
    for year in args.years:
        t0 = time.time()
        print(f"\n{year}: tracking", flush=True)
        times, latgrid, longrid, u, v = L.load_year(year, args.directory, "eraint")
        curvature = P.curvature_from_winds(latgrid, longrid, u, v)
        anomaly = P.anomaly_from_climatology(curvature, times, climatology)
        tracks = P.track_year(times, latgrid, longrid, u, v, anomaly,
                              coarse_threshold=coarse, fine_threshold=fine,
                              exclusive=args.exclusive)
        del curvature, anomaly
        _, unassigned = G.assign_regions(tracks)
        african = [t for t in tracks if t.get("region_name") == "AFR"]
        published = V.read_tracks(published_path(year, args.published))
        outcome = C.compare(african, published, tolerance_km=args.tolerance_km)
        outcome["all_tracks"] = len(tracks)
        outcome["unassigned"] = unassigned
        outcome["seconds"] = round(time.time() - t0, 1)
        results["years"][str(year)] = outcome

        s = outcome["signatures"]
        print(f"   {len(tracks)} tracks, {len(african)} African, "
              f"{unassigned} in no basin, {outcome['seconds']:.0f}s")
        print(f"   published {outcome['published_tracks']}, "
              f"matched {outcome['matched']} ({outcome['match_rate']:.1%}), "
              f"median separation {outcome['median_separation_km']:.0f} km")
        print(f"   signatures: gradual drift {s['gradual_drift_fraction']:.1%} of pairs "
              f"| port longer {s['port_tracks_longer_fraction']:.1%} (null 50%) "
              f"| meridian enrichment {s['meridian_enrichment']:.2f} (null 1.00, "
              f"base rate {s['near_meridian_base_rate']:.1%})")

    print(f"\ntotal {time.time()-started:.0f}s")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
