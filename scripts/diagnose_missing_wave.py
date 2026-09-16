#!/usr/bin/env python3
"""Walk one wave version 1 finds and the port does not, stage by stage.

THE CASE THIS WAS BUILT FOR. On the 60-timestep oracle window, once version 1's own
duplicates are accounted for, the substantial thing it finds in the southern hemisphere
over eastern Africa and the port does not is ONE feature, near 11S and 27E, alive from
33027.00 to 33030.75. Version 1 emits it as two tracks whose positions are identical from
33028.50 onward, which is its duplication defect in its clearest form.

WHY ONE WORKED CASE. Every aggregate comparison in this project has been walked past by
the defect that mattered, and the curvature orientation error was found by taking a single
timestep and following it. This does the same for a single wave: at each timestep of
version 1's track, report what the port's pipeline holds at that position, in order, so
the stage that drops it can be named rather than guessed.

THE STAGES, in the order find_ews_f.m and the port both run them:

  1. the raw curvature anomaly on the fine grid
  2. the same field decimated to the coarse tracking grid
  3. smoothed, then sign-adjusted so cyclonic is positive in both hemispheres, which is
     what the threshold is compared against. THE SIGN FLIP IS THE ONE TO WATCH HERE,
     because this wave is south of the equator and every other unmatched case is not.
  4. the zonal wind, and whether the westerly mask removes the cell
  5. the coarse threshold gate
  6. whether a trough axis (the advection's zero contour) passes nearby
  7. whether a merged wave survives nearby
  8. whether any port TRACK exists nearby, which separates a detection failure from an
     association failure
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port.climo_cache import load_or_build  # noqa: E402
from aew.v1port.detection import (MAX_ZONAL_WIND, _prepare,  # noqa: E402
                                  detect_troughs, trough_axes)
from aew.v1port.geometry import great_circle_distance  # noqa: E402
from compare_tracker_oracle import load_case  # noqa: E402


def nearest_index(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle-dir", default=os.environ.get("AEW_ORACLE_DIR"))
    ap.add_argument("--track", type=int, default=41,
                    help="which version 1 track to follow, by index")
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--directory", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "eraint", "v1port_buffered"))
    args = ap.parse_args(argv)
    if not args.oracle_dir:
        ap.error("set AEW_ORACLE_DIR or pass --oracle-dir")

    v1, port, case, _ = load_case(args.oracle_dir)
    if not 0 <= args.track < len(v1):
        ap.error(f"track {args.track} is outside version 1's {len(v1)}")
    target = v1[args.track]
    print(f"case {case}, following version 1's track {args.track}: "
          f"{target['time'].size} steps, {target['time'].min():.2f} to "
          f"{target['time'].max():.2f}\n")

    climo = load_or_build(L.available_years(args.directory, "eraint"),
                          args.directory, "eraint", os.environ["AEW_CLIMO_CACHE"],
                          verbose=False)
    times, latgrid, longrid, u, v = L.load_year(args.year, args.directory, "eraint")
    curvature = P.curvature_from_winds(latgrid, longrid, u, v)
    anomaly = P.anomaly_from_climatology(curvature, times, climo)
    del curvature
    advection = P._advection(latgrid, longrid, u, v, anomaly)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    coarse = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
              for n, f in (("u", u), ("anomaly", anomaly), ("advection", advection))}
    del advection
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
    rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
    rows_f, cols_f = P._subset(latgrid, longrid, P.DOMAIN_LAT, P.DOMAIN_LON)
    LG, LO = lat_c[np.ix_(rows_c, cols_c)], lon_c[np.ix_(rows_c, cols_c)]
    FG, FO = latgrid[np.ix_(rows_f, cols_f)], longrid[np.ix_(rows_f, cols_f)]
    ct, ft = P.thresholds_for("ERA-Int", 700)
    print(f"coarse threshold {ct:.4g}, fine threshold {ft:.4g}, "
          f"westerly cut {MAX_ZONAL_WIND} m/s\n")

    print(f"{'time':>9} {'lat':>7} {'lon':>7} | {'anom raw':>10} {'coarse':>10} "
          f"{'PREPARED':>10} | {'>=thr?':>7} {'u':>6} {'west?':>6} | "
          f"{'axis km':>8} {'wave km':>8} {'track km':>9}")
    for k in range(target["time"].size):
        t, la, lo = (float(target["time"][k]), float(target["lat"][k]),
                     float(target["lon"][k]))
        step = nearest_index(times, t)
        r = nearest_index(LG[:, 0], la)
        c = nearest_index(LO[0, :], lo)
        rf = nearest_index(FG[:, 0], la)
        cf = nearest_index(FO[0, :], lo)

        raw = float(anomaly[step][np.ix_(rows_f, cols_f)][rf, cf])
        coarse_raw = float(coarse["anomaly"][step][np.ix_(rows_c, cols_c)][r, c])
        prepared_field = _prepare(coarse["anomaly"][step][np.ix_(rows_c, cols_c)],
                                  LG[:, 0])
        prepared = float(prepared_field[r, c])
        wind = clim.smooth9(coarse["u"][step][np.ix_(rows_c, cols_c)])
        uu = float(wind[r, c])
        westerly = uu > MAX_ZONAL_WIND

        adv = clim.smooth9(coarse["advection"][step][np.ix_(rows_c, cols_c)])
        masked = np.where(wind > MAX_ZONAL_WIND, np.nan, adv)
        field = np.where(wind > MAX_ZONAL_WIND, np.nan, prepared_field)
        weak = field < ct
        masked = np.where(weak, np.nan, masked)
        field = np.where(weak, np.nan, field)

        axis_km = np.inf
        for axis_lat, axis_lon in trough_axes(LG, LO, masked):
            d = great_circle_distance(la, lo, np.asarray(axis_lat),
                                      np.asarray(axis_lon))
            axis_km = min(axis_km, float(np.min(d)))
        # detect_troughs takes the RAW coarse and fine fields and does its own
        # smoothing, sign flip and masking, so it is handed the same arrays track_year
        # hands it rather than the prepared ones computed above for reporting.
        waves = detect_troughs(
            t, LG, LO,
            coarse["u"][step][np.ix_(rows_c, cols_c)],
            coarse["anomaly"][step][np.ix_(rows_c, cols_c)],
            coarse["advection"][step][np.ix_(rows_c, cols_c)],
            FG, FO, anomaly[step][np.ix_(rows_f, cols_f)], ct, ft)
        wave_km = np.inf
        for w in waves:
            wave_km = min(wave_km, float(great_circle_distance(
                la, lo, w["lat_mean"], w["lon_mean"])))
        track_km = np.inf
        for p in port:
            hit = np.isclose(p["time"], t)
            if hit.any():
                track_km = min(track_km, float(np.min(great_circle_distance(
                    la, lo, p["lat"][hit], p["lon"][hit]))))

        def fmt(x):
            return f"{x:8.0f}" if np.isfinite(x) else "     n/a"

        print(f"{t:9.2f} {la:+7.2f} {lo:+7.2f} | {raw:10.2e} {coarse_raw:10.2e} "
              f"{prepared:10.2e} | {str(prepared >= ct):>7} {uu:6.1f} "
              f"{str(bool(westerly)):>6} | {fmt(axis_km)} {fmt(wave_km)} "
              f"{fmt(track_km):>9}")

    print("\nHOW TO READ THE LAST THREE COLUMNS. 'axis km' is the distance to the nearest")
    print("trough axis the port draws, 'wave km' to the nearest wave surviving its merge,")
    print("and 'track km' to the nearest point of any port TRACK at that time. A large")
    print("axis distance is a detection failure; a small axis with a large track distance")
    print("is an association or prune failure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
