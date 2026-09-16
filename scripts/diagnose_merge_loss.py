#!/usr/bin/env python3
"""Follow the waves version 1 keeps and the port loses through the three merge passes.

WHAT THIS ANSWERS. The accounting in the project's write-up of the unmatched tracks
found that at 83 percent of the observations where version 1 has a wave and the port
has none, the port's own curvature anomaly is above the detection threshold, and that
in 42 percent of those a trough axis passes within 200 km. So the field and the contour
both found the trough and something later dropped it. This says which pass.

HOW IT AVOIDS ANSWERING FROM A CHOSEN CASE. It traces EVERY qualifying observation
rather than one picked by hand, because a single case selected after the fact can be
made to show whatever the author already believes. One case is then printed in full as
an illustration of the majority mechanism.

THE TRACER IS A REPLICA of contours.merge_contours with recording added, and it is bound
to the real function by asserting their outputs agree on every call. A divergence stops
the run rather than producing a plausible wrong story.

    .venv/bin/python scripts/diagnose_merge_loss.py --year 1990
    .venv/bin/python scripts/diagnose_merge_loss.py --year 1990 --climatology-cache /tmp/c.npz

BUDGET about seven minutes without a climatology cache, three with one.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port import regions as G  # noqa: E402
from aew.v1port import validate as V  # noqa: E402
from aew.v1port.compare import match_tracks, separation_km  # noqa: E402
from aew.v1port.contours import (MERGE_DISTANCE_DEG, MIN_EXTENT_DEG,  # noqa: E402
                                 _binary_masks, _hull_contains, _select_region,
                                 merge_contours)
from aew.v1port.detection import (MAX_ZONAL_WIND, _prepare,  # noqa: E402
                                  detect_troughs, trough_axes)
from aew.v1port.geometry import great_circle_distance  # noqa: E402

DETECTED_KM = 200.0
AXIS_NEAR_KM = 100.0


def wave_groups(tracks, cutoff_km=100.0, min_steps=3):
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


def merge_traced(candidates, latgrid, longrid, curvature, threshold, watch):
    """merge_contours, plus the fate of candidate `watch`.

    Returns (output, fate, index_of_watch_in_output). `fate` is None while the watched
    candidate is still alive.
    """
    latgrid = np.asarray(latgrid, float)
    longrid = np.asarray(longrid, float)
    curvature = np.asarray(curvature, float)
    candidates = list(candidates)
    if not candidates:
        return [], "no candidates", None
    masks = _binary_masks(curvature, threshold)
    if not masks[0].any():
        return [], "nothing above threshold on this grid", None
    base_rows, base_cols = np.nonzero(masks[0])
    base_lat, base_lon = latgrid[base_rows, base_cols], longrid[base_rows, base_cols]

    remaining, merged, owner = list(range(len(candidates))), [], []
    watch_merged, fate = None, None
    while remaining:
        first, others = remaining[0], remaining[1:]
        here = candidates[first]
        d2 = ((here["lat_mean"] - base_lat) ** 2 + (here["lon_mean"] - base_lon) ** 2)
        seed_at = int(np.argmin(d2))
        seed = (base_rows[seed_at], base_cols[seed_at])
        region = _select_region(masks, seed, latgrid, longrid)
        lats, lons = latgrid[region], longrid[region]
        absorbed = []
        if others:
            inside = _hull_contains(
                lons, lats,
                np.array([candidates[i]["lon_mean"] for i in others]),
                np.array([candidates[i]["lat_mean"] for i in others]))
            absorbed = [others[i] for i in np.flatnonzero(inside)]
        merged.append({
            "time": candidates[absorbed[0]]["time"] if absorbed else here["time"],
            "lat_mean": float(np.median(lats)) if lats.size else here["lat_mean"],
            "lon_mean": float(np.median(lons)) if lons.size else here["lon_mean"],
            "region": region, "lat_wave": lats, "lon_wave": lons})
        owner.append(first)
        if watch == first:
            watch_merged = len(merged) - 1
        elif watch in absorbed and fate is None:
            fate = (f"PASS 1 absorbed: taken into the region of candidate {first} at "
                    f"({candidates[first]['lat_mean']:.1f}, "
                    f"{candidates[first]['lon_mean']:.1f}), which spans "
                    f"{np.ptp(lats):.1f} by {np.ptp(lons):.1f} degrees "
                    f"over {int(region.sum())} cells")
        drop = set(absorbed) | {first}
        remaining = [i for i in remaining if i not in drop]

    # NO EARLY RETURN ONCE THE WATCHED CANDIDATE IS DEAD. A first version returned here
    # with the pass-1 intermediate, which is not what merge_contours returns, and the
    # binding assertion below caught it on the first absorbed case. The remaining passes
    # still run so the output stays comparable; `watch_merged` is None, so their fate
    # checks cannot fire.
    if len(merged) <= 1:
        # FAITHFUL: a lone wave skips passes 2 and 3 entirely, so it is never tested
        # against the minimum extent. merge_contours returns `merged` here too.
        return merged, fate, (0 if watch_merged == 0 else None)

    kept, kept_idx, pending = [], [], list(range(len(merged)))
    while pending:
        first, others = pending[0], pending[1:]
        kept.append(merged[first])
        kept_idx.append(first)
        close = set()
        if others:
            degrees = great_circle_distance(
                merged[first]["lat_mean"], merged[first]["lon_mean"],
                np.array([merged[i]["lat_mean"] for i in others]),
                np.array([merged[i]["lon_mean"] for i in others]), "nm") / 60.0
            close = {others[i] for i in np.flatnonzero(degrees <= MERGE_DISTANCE_DEG)}
        if watch_merged in close and fate is None:
            fate = (f"PASS 2 five-degree separation: dropped as a duplicate of the wave "
                    f"at ({merged[first]['lat_mean']:.1f}, "
                    f"{merged[first]['lon_mean']:.1f})")
        pending = [i for i in pending if i != first and i not in close]

    out, out_idx = [], []
    for wave, ki in zip(kept, kept_idx):
        lats, lons = wave["lat_wave"], wave["lon_wave"]
        if lats.size == 0:
            if ki == watch_merged and fate is None:
                fate = ("PASS 3 empty region: the threshold ladder escalated past the "
                        "level where the seed cell itself survives")
            continue
        values = curvature[wave["region"]]
        peak = values == np.nanmax(values)
        refined = dict(wave)
        if peak.any():
            refined["lat_mean"] = (float(np.mean(lats[peak])) + wave["lat_mean"]) / 2.0
            refined["lon_mean"] = (float(np.mean(lons[peak])) + wave["lon_mean"]) / 2.0
        top, bottom = lats == np.max(lats), lats == np.min(lats)
        span = great_circle_distance(
            np.min(lats), (np.min(lons[bottom]) + np.max(lons[bottom])) / 2.0,
            np.max(lats), (np.min(lons[top]) + np.max(lons[top])) / 2.0, "nm") / 60.0
        if span >= MIN_EXTENT_DEG:
            out.append(refined)
            out_idx.append(ki)
        elif ki == watch_merged and fate is None:
            fate = (f"PASS 3 minimum extent: the region spans {float(span):.2f} degrees, "
                    f"under the {MIN_EXTENT_DEG} degree minimum")

    return out, fate, (out_idx.index(watch_merged) if watch_merged in out_idx else None)


def _bind(traced, candidates, latgrid, longrid, curvature, threshold):
    """The replica must agree with the real merge, or nothing it reports means anything."""
    real = merge_contours(candidates, latgrid, longrid, curvature, threshold)
    a = sorted((w["lat_mean"], w["lon_mean"]) for w in traced)
    b = sorted((w["lat_mean"], w["lon_mean"]) for w in real)
    if len(a) != len(b) or not np.allclose(a, b):
        raise AssertionError("the traced merge diverged from contours.merge_contours")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--directory", default="data/eraint/v1port")
    ap.add_argument("--prefix", default="eraint")
    ap.add_argument("--published", default="data/aewc")
    ap.add_argument("--climatology-cache", default=None)
    args = ap.parse_args(argv)

    years = sorted(L.available_years(args.directory, args.prefix))
    cache = args.climatology_cache
    if cache and os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:
            if z["years"].tolist() != years:
                raise SystemExit(f"{cache} was built over different years; delete it")
            climatology = {"keys": [tuple(int(x) for x in r) for r in z["keys"]],
                           "mean": z["mean"], "counts": z["counts"], "short_steps": []}
    else:
        print(f"building climatology over {len(years)} years", flush=True)
        climatology = L.build_climatology(years, args.directory, args.prefix)
        if cache:
            np.savez_compressed(cache, years=np.asarray(years),
                                keys=np.asarray(climatology["keys"], dtype=int),
                                mean=climatology["mean"], counts=climatology["counts"])

    times, latgrid, longrid, u, v = L.load_year(args.year, args.directory, args.prefix)
    curvature = P.curvature_from_winds(latgrid, longrid, u, v)
    anomaly = P.anomaly_from_climatology(curvature, times, climatology)
    del curvature
    advection = P._advection(latgrid, longrid, u, v, anomaly)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    coarse = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
              for n, f in (("u", u), ("anomaly", anomaly), ("advection", advection))}
    del advection
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
    rows_f, cols_f = P._subset(latgrid, longrid, P.DOMAIN_LAT, P.DOMAIN_LON)
    rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
    fine_lat = latgrid[np.ix_(rows_f, cols_f)]
    fine_lon = longrid[np.ix_(rows_f, cols_f)]
    coarse_lat = lat_c[np.ix_(rows_c, cols_c)]
    coarse_lon = lon_c[np.ix_(rows_c, cols_c)]
    coarse_threshold, fine_threshold = P.thresholds_for("ERA-Int", 700)
    lat_values, lon_values = coarse_lat[:, 0], coarse_lon[0, :]

    detections = {}
    for step in range(times.size):
        detections[step] = [
            (w["lat_mean"], w["lon_mean"]) for w in detect_troughs(
                float(times[step]), coarse_lat, coarse_lon,
                coarse["u"][step][np.ix_(rows_c, cols_c)],
                coarse["anomaly"][step][np.ix_(rows_c, cols_c)],
                coarse["advection"][step][np.ix_(rows_c, cols_c)],
                fine_lat, fine_lon, anomaly[step][np.ix_(rows_f, cols_f)],
                coarse_threshold=coarse_threshold, fine_threshold=fine_threshold)]

    tracks = P.track_year(times, latgrid, longrid, u, v, anomaly,
                          coarse_threshold=coarse_threshold,
                          fine_threshold=fine_threshold)
    G.assign_regions(tracks)
    port = [t for t in tracks if t.get("region_name") == "AFR"]
    published = V.read_tracks(os.path.join(
        args.published, f"ERA-Int_ew_700hPa_{args.year}_AFR.nc"))
    groups = wave_groups(published)
    matching = match_tracks(port, published, tolerance_km=500.0)
    group_of = {i: g for g, members in enumerate(groups) for i in members}
    matched = {group_of[p["published"]] for p in matching["pairs"]}
    missed = [max((published[i] for i in members), key=lambda t: len(t["time"]))
              for g, members in enumerate(groups) if g not in matched]
    index = {round(float(t), 4): k for k, t in enumerate(times)}
    print(f"{len(missed)} missed version 1 waves to follow", flush=True)

    verdicts, survived_km, illustration = Counter(), [], None
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
            wind = clim.smooth9(coarse["u"][step][np.ix_(rows_c, cols_c)])
            field = _prepare(coarse["anomaly"][step][np.ix_(rows_c, cols_c)], lat_values)
            westerly = wind > MAX_ZONAL_WIND
            masked_field = np.where(westerly, np.nan, field)
            weak = masked_field < coarse_threshold
            r = int(np.argmin(np.abs(lat_values - la)))
            c = int(np.argmin(np.abs(lon_values - lo)))
            if westerly[r, c] or field[r, c] < coarse_threshold:
                continue
            advection_masked = np.where(
                westerly | weak, np.nan,
                clim.smooth9(coarse["advection"][step][np.ix_(rows_c, cols_c)]))
            axes = trough_axes(coarse_lat, coarse_lon, advection_masked)
            if not axes:
                continue
            best, best_km = None, np.inf
            for k, (axis_lat, axis_lon) in enumerate(axes):
                near = float(np.min(great_circle_distance(la, lo, axis_lat, axis_lon,
                                                          "km")))
                if near < best_km:
                    best, best_km = k, near
            if best_km > AXIS_NEAR_KM:
                continue

            candidates = [{"time": float(t), "lat_mean": float(np.mean(al)),
                           "lon_mean": float(np.mean(ao))} for al, ao in axes]
            fine_field = _prepare(anomaly[step][np.ix_(rows_f, cols_f)], fine_lat[:, 0])
            fine_field = np.where(fine_field < fine_threshold, np.nan, fine_field)
            masked_field = np.where(weak, np.nan, masked_field)

            out, fate, at = merge_traced(candidates, coarse_lat, coarse_lon,
                                         masked_field, coarse_threshold, best)
            _bind(out, candidates, coarse_lat, coarse_lon, masked_field, coarse_threshold)
            stage = "coarse merge"
            if fate is None and at is not None:
                out2, fate, at2 = merge_traced(out, fine_lat, fine_lon, fine_field,
                                               fine_threshold, at)
                _bind(out2, out, fine_lat, fine_lon, fine_field, fine_threshold)
                stage = "fine merge"
                if fate is None:
                    if at2 is None:
                        fate = "left the merge without appearing in its output"
                    else:
                        moved = float(great_circle_distance(
                            la, lo, out2[at2]["lat_mean"], out2[at2]["lon_mean"], "km"))
                        survived_km.append(moved)
                        fate = "SURVIVED, and its center was relocated"
            elif fate is None:
                fate = "left the coarse merge without appearing in its output"
            verdicts[f"{stage}: {fate.split(':')[0]}"] += 1
            if illustration is None and fate.startswith("PASS 1") and 5 <= la <= 25:
                illustration = (float(t), float(la), float(lo), best_km,
                                float(great_circle_distance(
                                    la, lo, candidates[best]["lat_mean"],
                                    candidates[best]["lon_mean"], "km")), fate)

    total = max(sum(verdicts.values()), 1)
    print(f"\nWHICH PASS DROPS THE WAVE, over {total} observations where a trough axis")
    print(f"passes within {AXIS_NEAR_KM:.0f} km and no wave is detected within "
          f"{DETECTED_KM:.0f} km:\n")
    for key, n in verdicts.most_common():
        print(f"  {n:4d} ({n/total:5.1%})  {key}")
    if survived_km:
        s = np.asarray(survived_km)
        print(f"\n  of the {len(s)} that survived, the final center sits a median "
              f"{np.median(s):.0f} km from version 1's wave, and over 1000 km away in "
              f"{(s > 1000).mean():.0%} of them")
    if illustration:
        t, la, lo, axis_km, cand_km, fate = illustration
        print(f"\nONE CASE IN FULL, the majority mechanism, in the wave corridor:")
        print(f"  t={t}, version 1 has a wave at {la:.1f}N {abs(lo):.1f}"
              f"{'W' if lo < 0 else 'E'}")
        print(f"  the trough axis passes {axis_km:.0f} km from it, so the contour found it")
        print(f"  but the axis's MEAN, which is the position detection carries forward,")
        print(f"  sits {cand_km:.0f} km away")
        print(f"  {fate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
