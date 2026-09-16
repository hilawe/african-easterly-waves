#!/usr/bin/env python3
"""Decompose the port-against-version-1 match rate into what the unmatched tracks are.

WHY THIS EXISTS. `run_validation.py` reports matched tracks over published tracks, and
that fraction is not what it looks like. The matcher is one-to-one, so it cannot pair
more tracks than the port produced, and the denominator counts version 1's duplication
defect, which holds one physical trough in several tracks. This reruns the comparison
and reports the accounting at the level of WAVES, then says where the genuinely missed
ones are lost.

This script is the source of every number in the project's write-up of the
unmatched tracks, so the write-up can be re-derived rather than trusted.

    .venv/bin/python scripts/diagnose_match_accounting.py --year 1990
    .venv/bin/python scripts/diagnose_match_accounting.py --year 1990 \
        --climatology-cache /tmp/climo.npz

BUDGET about four minutes for the climatology and three to track a year, unless a cache
is given. The cache stores the years it was built over and refuses a different set,
but it does NOT fingerprint the code, so delete it after any change to the curvature,
climatology, load or pipeline modules.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port import regions as G  # noqa: E402
from aew.v1port import validate as V  # noqa: E402
from aew.v1port.compare import match_tracks, separation_km  # noqa: E402
from aew.v1port.detection import (MAX_ZONAL_WIND, _prepare,  # noqa: E402
                                  detect_troughs, trough_axes)
from aew.v1port.geometry import great_circle_distance  # noqa: E402

DUPLICATE_KM = 100.0        # two tracks this close over 3+ shared steps are one wave
DUPLICATE_STEPS = 3
DETECTED_KM = 200.0         # "the port detected this wave here"


def wave_groups(tracks, cutoff_km=DUPLICATE_KM, min_steps=DUPLICATE_STEPS):
    """Group tracks that are the same physical wave held more than once.

    NOT SENSITIVE TO THE CUTOFF on the published record, because version 1's duplicates
    carry identical coordinates rather than nearby ones: between 25 and 300 km the group
    count moves from 272 to 268 on 1990. A cutoff that changed the answer would make
    every wave-level number below a choice rather than a measurement, so the sensitivity
    is worth rechecking on any new record.
    """
    n = len(tracks)
    partners = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            shared, distances = separation_km(tracks[i], tracks[j])
            if shared.size >= min_steps and float(np.median(distances)) <= cutoff_km:
                partners[i].append(j)
                partners[j].append(i)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, component = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            component.append(k)
            stack.extend(partners[k])
        groups.append(component)
    return groups


def _climatology(directory, prefix, cache):
    years = sorted(L.available_years(directory, prefix))
    if cache and os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:
            if z["years"].tolist() != years:
                raise SystemExit(f"{cache} was built over different years; delete it")
            return {"keys": [tuple(int(x) for x in r) for r in z["keys"]],
                    "mean": z["mean"], "counts": z["counts"], "short_steps": []}
    print(f"building climatology over {len(years)} years", flush=True)
    climatology = L.build_climatology(years, directory, prefix)
    if cache:
        np.savez_compressed(cache, years=np.asarray(years),
                            keys=np.asarray(climatology["keys"], dtype=int),
                            mean=climatology["mean"], counts=climatology["counts"])
    return climatology


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--directory", default="data/eraint/v1port")
    ap.add_argument("--prefix", default="eraint")
    ap.add_argument("--published", default="data/aewc")
    ap.add_argument("--tolerance-km", type=float, default=500.0)
    ap.add_argument("--climatology-cache", default=None)
    args = ap.parse_args(argv)

    climatology = _climatology(args.directory, args.prefix, args.climatology_cache)
    times, latgrid, longrid, u, v = L.load_year(args.year, args.directory, args.prefix)
    curvature = P.curvature_from_winds(latgrid, longrid, u, v)
    anomaly = P.anomaly_from_climatology(curvature, times, climatology)
    del curvature
    coarse_threshold, fine_threshold = P.thresholds_for("ERA-Int", 700)

    # The fields detection reads, prepared once and reused for the per-stage questions.
    advection = P._advection(latgrid, longrid, u, v, anomaly)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    coarse = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
              for n, f in (("u", u), ("anomaly", anomaly), ("advection", advection))}
    del advection
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
    rows_f, cols_f = P._subset(latgrid, longrid, P.DOMAIN_LAT, P.DOMAIN_LON)
    rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
    latgrid_f = latgrid[np.ix_(rows_f, cols_f)]
    longrid_f = longrid[np.ix_(rows_f, cols_f)]
    latgrid_c = lat_c[np.ix_(rows_c, cols_c)]
    longrid_c = lon_c[np.ix_(rows_c, cols_c)]
    lat_values, lon_values = latgrid_c[:, 0], longrid_c[0, :]

    detections = {}
    for step in range(times.size):
        waves = detect_troughs(
            float(times[step]), latgrid_c, longrid_c,
            coarse["u"][step][np.ix_(rows_c, cols_c)],
            coarse["anomaly"][step][np.ix_(rows_c, cols_c)],
            coarse["advection"][step][np.ix_(rows_c, cols_c)],
            latgrid_f, longrid_f, anomaly[step][np.ix_(rows_f, cols_f)],
            coarse_threshold=coarse_threshold, fine_threshold=fine_threshold)
        detections[step] = [(w["lat_mean"], w["lon_mean"]) for w in waves]
    print(f"detections: {sum(len(w) for w in detections.values())} over {times.size} "
          f"timesteps", flush=True)

    tracks = P.track_year(times, latgrid, longrid, u, v, anomaly,
                          coarse_threshold=coarse_threshold,
                          fine_threshold=fine_threshold)
    G.assign_regions(tracks)
    port = [t for t in tracks if t.get("region_name") == "AFR"]
    other_basins = [t for t in tracks if t.get("region_name") != "AFR"]
    published = V.read_tracks(os.path.join(
        args.published, f"ERA-Int_ew_700hPa_{args.year}_AFR.nc"))

    port_groups = wave_groups(port)
    published_groups = wave_groups(published)
    matching = match_tracks(port, published, tolerance_km=args.tolerance_km)
    group_of = {i: g for g, members in enumerate(published_groups) for i in members}
    matched_groups = {group_of[p["published"]] for p in matching["pairs"]}
    port_group_of = {i: g for g, members in enumerate(port_groups) for i in members}
    matched_port_groups = {port_group_of[p["port"]] for p in matching["pairs"]}

    pairs = len(matching["pairs"])
    ceiling = min(len(port), len(published)) / max(len(published), 1)
    print(f"\nTRACKS.  port {len(port)} African ({len(tracks)} all basins), "
          f"published {len(published)}")
    print(f"  matched pairs {pairs}")
    print(f"  over published tracks: {pairs/max(len(published),1):.1%}   "
          f"(the number run_validation reports)")
    print(f"  over port tracks:      {pairs/max(len(port),1):.1%}")
    print(f"  ONE-TO-ONE CEILING:    {ceiling:.1%}, since no track is paired twice")

    print(f"\nWAVES.  tracks grouped when within {DUPLICATE_KM:.0f} km over "
          f"{DUPLICATE_STEPS}+ shared steps")
    print(f"  version 1: {len(published)} tracks -> {len(published_groups)} waves "
          f"({len(published)/max(len(published_groups),1):.2f} tracks per wave)")
    print(f"  the port:  {len(port)} tracks -> {len(port_groups)} waves "
          f"({len(port)/max(len(port_groups),1):.2f} tracks per wave)")
    print(f"  version 1 waves the port found:  {len(matched_groups)} of "
          f"{len(published_groups)} ({len(matched_groups)/max(len(published_groups),1):.1%})")
    print(f"  port waves matching nothing:     "
          f"{len(port_groups)-len(matched_port_groups)}")

    unmatched = matching["published_unmatched"]
    duplicates = [j for j in unmatched if group_of[j] in matched_groups]
    print(f"\nTHE {len(unmatched)} UNMATCHED PUBLISHED TRACKS")
    print(f"  duplicate copies of a wave the port DID match: {len(duplicates)} "
          f"({len(duplicates)/max(len(unmatched),1):.1%})")
    print(f"  belonging to a wave the port missed:           "
          f"{len(unmatched)-len(duplicates)}")

    missed = [max((published[i] for i in members), key=lambda t: len(t["time"]))
              for g, members in enumerate(published_groups) if g not in matched_groups]
    found = [max((published[i] for i in members), key=lambda t: len(t["time"]))
             for g, members in enumerate(published_groups) if g in matched_groups]
    index = {round(float(t), 4): k for k, t in enumerate(times)}

    def coverage(wave_list, radius):
        out = []
        for track in wave_list:
            hits = 0
            for t, la, lo in zip(track["time"], track["meanlat"], track["meanlon"]):
                step = index.get(round(float(t), 4))
                found_here = detections.get(step) if step is not None else None
                if found_here:
                    d = great_circle_distance(
                        la, lo, np.array([w[0] for w in found_here]),
                        np.array([w[1] for w in found_here]), "km")
                    if float(np.min(d)) <= radius:
                        hits += 1
            out.append(hits / max(len(track["time"]), 1))
        return np.array(out) if out else np.zeros(1)

    print(f"\nIS THE WAVE DETECTED AT ALL (share of its observations with a detection)")
    print(f"{'within':>8} {'found':>10} {'missed':>10}")
    for radius in (200.0, 350.0, 500.0):
        print(f"{radius:>7.0f}k {coverage(found, radius).mean():>9.1%} "
              f"{coverage(missed, radius).mean():>9.1%}")
    never = int((coverage(missed, DETECTED_KM) == 0).sum())
    print(f"  missed waves never detected anywhere: {never} of {len(missed)}")

    # Where the undetected observations are lost, stage by stage.
    verdict = {"above threshold, no candidate": 0, "masked as westerly": 0,
               "anomaly below threshold": 0}
    axis_distance = []
    by_step = {}
    for track in missed:
        for t, la, lo in zip(track["time"], track["meanlat"], track["meanlon"]):
            step = index.get(round(float(t), 4))
            if step is None:
                continue
            here = detections.get(step) or []
            if here:
                d = great_circle_distance(la, lo, np.array([w[0] for w in here]),
                                          np.array([w[1] for w in here]), "km")
                if float(np.min(d)) <= DETECTED_KM:
                    continue
            by_step.setdefault(step, []).append((la, lo))

    for step, points in by_step.items():
        wind = clim.smooth9(coarse["u"][step][np.ix_(rows_c, cols_c)])
        field = _prepare(coarse["anomaly"][step][np.ix_(rows_c, cols_c)], lat_values)
        masked = clim.smooth9(coarse["advection"][step][np.ix_(rows_c, cols_c)])
        westerly = wind > MAX_ZONAL_WIND
        masked = np.where(westerly | (np.where(westerly, np.nan, field)
                                      < coarse_threshold), np.nan, masked)
        axes = trough_axes(latgrid_c, longrid_c, masked)
        if axes:
            axis_lat = np.concatenate([a[0] for a in axes])
            axis_lon = np.concatenate([a[1] for a in axes])
        else:
            axis_lat = axis_lon = np.array([])
        for la, lo in points:
            r = int(np.argmin(np.abs(lat_values - la)))
            c = int(np.argmin(np.abs(lon_values - lo)))
            if westerly[r, c]:
                verdict["masked as westerly"] += 1
            elif field[r, c] < coarse_threshold:
                verdict["anomaly below threshold"] += 1
            else:
                verdict["above threshold, no candidate"] += 1
                if axis_lat.size:
                    axis_distance.append(float(np.min(great_circle_distance(
                        la, lo, axis_lat, axis_lon, "km"))))
                else:
                    axis_distance.append(float("inf"))

    total = max(sum(verdict.values()), 1)
    print(f"\nAT THE {sum(verdict.values())} OBSERVATIONS WITH NO DETECTION WITHIN "
          f"{DETECTED_KM:.0f} KM, the port's own field there says:")
    for key, n in sorted(verdict.items(), key=lambda kv: -kv[1]):
        print(f"  {key:>32}: {n:5d} ({n/total:.1%})")

    if axis_distance:
        d = np.asarray(axis_distance)
        print(f"\nOF THE {len(d)} ABOVE-THRESHOLD CASES, distance to the nearest trough "
              f"axis BEFORE merging:")
        for lo, hi, label in ((0, 100, "under 100 km"), (100, 200, "100 to 200 km"),
                              (200, 350, "200 to 350 km"), (350, 500, "350 to 500 km"),
                              (500, np.inf, "over 500 km")):
            n = int(((d >= lo) & (d < hi)).sum())
            print(f"  {label:>16}: {n:5d} ({n/len(d):.0%})")
        print(f"  median {np.median(d[np.isfinite(d)]):.0f} km")
        print(f"  AN AXIS IS PRESENT within 200 km for {(d <= 200).mean():.0%}, and in "
              f"those cases the merge is what dropped it.")
        print(f"  None is within 500 km for {(d > 500).mean():.0%}, and in those the "
              f"contour never found the trough.")

    # Basin misassignment, which would be a different explanation entirely.
    elsewhere = 0
    for track in missed:
        for other in other_basins:
            shared, distances = separation_km(track, other)
            if (shared.size >= DUPLICATE_STEPS
                    and float(np.median(distances)) <= args.tolerance_km):
                elsewhere += 1
                break
    print(f"\nmissed waves matched by a port track filed under another basin: "
          f"{elsewhere} of {len(missed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
