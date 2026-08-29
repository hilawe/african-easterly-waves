#!/usr/bin/env python3
"""Find out WHERE the port and version 1 diverge, before asking why.

The first comparison against version 1's published record matched 9.8 percent of its
tracks, and a partial climatology was ruled out as the cause. A match rate is one number at
the end of a chain of seven stages, so this instruments the chain: how many troughs are
detected per timestep, how many survive association, how many survive the filters, and how
many are African. A deficit at the first step and a deficit at the last mean entirely
different things.

WHAT IS COMPARABLE AND WHAT IS NOT. Version 1's published record holds finished TRACKS, not
the candidates it detected, so its raw per-timestep detection count cannot be recovered.
What can be compared directly is OBSERVATIONS PER TIMESTEP: at each six-hourly step, how
many waves each record has in Africa. That is the same quantity on both sides and it is
what the tracks are made of.

So the funnel below is one-sided by necessity, and only its last row can be held against
version 1. Its value is showing which stage the port's own waves disappear at, which
narrows where to look.

    .venv/bin/python scripts/diagnose_detection.py --year 1990
    .venv/bin/python scripts/diagnose_detection.py --year 1990 --steps 200
"""

import argparse
import collections
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port import regions as G  # noqa: E402
from aew.v1port import validate as V  # noqa: E402
from aew.v1port.association import (associate_step, finalize_tracks,  # noqa: E402
                                    prune_stale_tracks)
from aew.v1port.detection import detect_troughs  # noqa: E402


def published_per_step(year, directory="data/aewc"):
    """Observations per timestep in version 1's record, keyed by time."""
    path = os.path.join(directory, f"ERA-Int_ew_700hPa_{year}_AFR.nc")
    counts = collections.Counter()
    total_tracks = 0
    for track in V.read_tracks(path):
        total_tracks += 1
        for t in track["time"]:
            counts[round(float(t), 4)] += 1
    return counts, total_tracks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--directory", default="data/eraint/v1port")
    ap.add_argument("--steps", type=int, default=None,
                    help="only the first N timesteps, for a quicker look")
    ap.add_argument("--climatology-years", nargs="+", type=int, default=None)
    args = ap.parse_args(argv)

    have = L.available_years(args.directory, "eraint")
    years = args.climatology_years or have
    print(f"climatology over {len(years)} years ({min(years)}-{max(years)})", flush=True)
    climatology = L.build_climatology(years, args.directory, "eraint")

    times, latgrid, longrid, u, v = L.load_year(args.year, args.directory, "eraint")
    curvature = P.curvature_from_winds(latgrid, longrid, u, v)
    anomaly = P.anomaly_from_climatology(curvature, times, climatology)
    del curvature

    coarse_threshold, fine_threshold = P.thresholds_for("ERA-Int", 700)
    advection = P._advection(latgrid, longrid, u, v, anomaly)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    decimate = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
                for n, f in (("u", u), ("anomaly", anomaly), ("advection", advection))}
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
    rows_f, cols_f = P._subset(latgrid, longrid, P.DOMAIN_LAT, P.DOMAIN_LON)
    rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
    latgrid_f = latgrid[np.ix_(rows_f, cols_f)]
    longrid_f = longrid[np.ix_(rows_f, cols_f)]
    latgrid_c = lat_c[np.ix_(rows_c, cols_c)]
    longrid_c = lon_c[np.ix_(rows_c, cols_c)]

    n_steps = args.steps or times.size
    print(f"tracking {n_steps} timesteps of {args.year}", flush=True)

    detected, tracks, states = [], [], []
    started = time.time()
    for step in range(n_steps):
        waves = detect_troughs(
            float(times[step]), latgrid_c, longrid_c,
            decimate["u"][step][np.ix_(rows_c, cols_c)],
            decimate["anomaly"][step][np.ix_(rows_c, cols_c)],
            decimate["advection"][step][np.ix_(rows_c, cols_c)],
            latgrid_f, longrid_f, anomaly[step][np.ix_(rows_f, cols_f)],
            coarse_threshold=coarse_threshold, fine_threshold=fine_threshold)
        detected.append(len(waves))
        u_median = P._median_over(clim.smooth9(u[step][np.ix_(rows_f, cols_f)]))
        v_median = P._median_over(clim.smooth9(v[step][np.ix_(rows_f, cols_f)]))
        tracks, states = associate_step(tracks, states, waves, step, u_median, v_median)
        tracks, states = prune_stale_tracks(tracks, states, step, float(times[step]),
                                            waves)
        if step and step % 200 == 0:
            print(f"   step {step}, {len(tracks)} live tracks, "
                  f"{time.time()-started:.0f}s", flush=True)

    after_prune = len(tracks)
    filtered = finalize_tracks(tracks, total_steps=n_steps)
    G.assign_regions(filtered)
    african = [t for t in filtered if t.get("region_name") == "AFR"]

    detected = np.asarray(detected)
    print(f"\nTHE PORT'S FUNNEL for {args.year}"
          f"{'' if args.steps is None else f' (first {n_steps} steps)'}")
    print(f"  troughs detected per timestep: mean {detected.mean():.2f}  "
          f"median {np.median(detected):.0f}  max {detected.max()}  "
          f"total {detected.sum()}")
    print(f"  timesteps with no trough at all: {int((detected == 0).sum())} of {n_steps}")
    print(f"  tracks alive after the last prune: {after_prune}")
    print(f"  tracks surviving the speed filter: {len(filtered)}")
    print(f"  of those, African: {len(african)}")
    lengths = np.array([len(t["time"]) for t in filtered]) if filtered else np.array([0])
    print(f"  track lengths: median {np.median(lengths):.0f}  mean {lengths.mean():.1f}  "
          f"max {lengths.max()}")

    if args.steps is None:
        published, published_tracks = published_per_step(args.year)
        port = collections.Counter()
        for t in african:
            for tt in t["time"]:
                port[round(float(tt), 4)] += 1
        shared = sorted(set(published) | set(port))
        pub_v = np.array([published.get(k, 0) for k in shared])
        port_v = np.array([port.get(k, 0) for k in shared])
        print(f"\nAFRICAN OBSERVATIONS PER TIMESTEP, the one quantity both records hold")
        print(f"  version 1: {pub_v.sum()} over {len(published)} timesteps, "
              f"mean {pub_v.sum()/max(len(shared),1):.2f}")
        print(f"  the port:  {port_v.sum()} over {len(port)} timesteps, "
              f"mean {port_v.sum()/max(len(shared),1):.2f}")
        print(f"  ratio port/version 1: {port_v.sum()/max(pub_v.sum(),1):.2f}")
        both = (pub_v > 0) & (port_v > 0)
        if both.any():
            print(f"  correlation where both have waves: "
                  f"{np.corrcoef(pub_v[both], port_v[both])[0,1]:.2f}")
        print(f"  tracks: version 1 {published_tracks}, port {len(african)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
