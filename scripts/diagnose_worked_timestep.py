#!/usr/bin/env python3
"""One worked timestep, end to end, against version 1's known positions.

Aggregates have twice supported a confident, detailed, wrong story in this project, both
times by dividing one count by another that was not the same quantity, so this looks at
ONE timestep where version 1 has several African waves and answers, per version 1
observation:

  1. What does the port's field say AT that position: the smoothed coarse zonal wind (is it
     masked as westerly), the prepared curvature anomaly (is it below the coarse
     threshold), and the advection.
  2. Is there a trough axis (a zero contour of the masked advection) near it, BEFORE any
     merging.
  3. Is there a merged wave near it after the coarse pass, and after the fine pass.

If the axes are near version 1's positions and the merged centers are not, the merge or
the center definition moves them. If no axis exists near a version 1 position, the field
or the masks differ there, and the field values in step 1 say which.

The replica of detect_troughs below is bound to the real one by a runtime assertion on
the FINAL WAVE CENTERS only: they must equal what detect_troughs returns on the same
inputs. The intermediates (masks, axes, the two merge passes) are the replica's own and
are not individually bound, so read them as this script's reconstruction rather than as
values the production path emitted.

    .venv/bin/python scripts/diagnose_worked_timestep.py --year 1990 --time 33067.5
    .venv/bin/python scripts/diagnose_worked_timestep.py --year 1990 --time 33067.5 \
        --climatology-cache /tmp/climo_eraint.npz --figure /tmp/worked_timestep.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port import validate as V  # noqa: E402
from aew.v1port.climo_cache import load_or_build  # noqa: E402
from aew.v1port.contours import merge_contours  # noqa: E402
from aew.v1port.detection import (MAX_ZONAL_WIND, _prepare,  # noqa: E402
                                  detect_troughs, trough_axes)
from aew.v1port.geometry import great_circle_distance  # noqa: E402




def v1_observations_at(year, time_days, directory="data/aewc"):
    """Version 1's African observations at one time, as (lat, lon, track_index) rows.

    These are PUBLISHED coordinates: version 1 smooths each track's coordinates with a
    five-point moving average before writing, so an individual position can sit a little
    off the raw detection, and the record's duplication defect means two rows can be the
    same physical trough held by two tracks. Both are noted where this is printed.
    """
    path = os.path.join(directory, f"ERA-Int_ew_700hPa_{year}_AFR.nc")
    rows = []
    for n, track in enumerate(V.read_tracks(path)):
        for t, la, lo in zip(track["time"], track["meanlat"], track["meanlon"]):
            if abs(float(t) - time_days) < 1e-6:
                rows.append((float(la), float(lo), n))
    return rows


def km_to_nearest(lat, lon, lats, lons):
    """Great-circle km from one point to the nearest of a set, and that point."""
    if len(lats) == 0:
        return float("inf"), None
    d = great_circle_distance(lat, lon, np.asarray(lats, dtype=float),
                              np.asarray(lons, dtype=float), "km")
    i = int(np.argmin(d))
    return float(d[i]), (float(lats[i]), float(lons[i]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--time", type=float, default=33067.5,
                    help="days since 1900 (default 1990-07-15T12, 12 v1 African waves)")
    ap.add_argument("--directory", default="data/eraint/v1port")
    ap.add_argument("--prefix", default="eraint")
    ap.add_argument("--climatology-cache", default=None)
    ap.add_argument("--figure", default=None, help="write a map PNG here")
    args = ap.parse_args(argv)

    have = L.available_years(args.directory, args.prefix)
    climatology = load_or_build(have, args.directory, args.prefix,
                                args.climatology_cache)

    times, latgrid, longrid, u, v = L.load_year(args.year, args.directory, args.prefix)
    step = int(np.argmin(np.abs(times - args.time)))
    if abs(float(times[step]) - args.time) > 1e-6:
        raise SystemExit(f"no timestep at {args.time}; nearest is {float(times[step])}")
    when = P.days_to_datetime64(float(times[step]))
    print(f"timestep {step} of {args.year}: {when} (t={float(times[step])})", flush=True)

    # The exact preparation track_year performs, on a slice around the step (each
    # timestep's fields are independent, and the slice keeps this run in seconds).
    lo, hi = max(0, step - 1), min(times.size, step + 2)
    at = step - lo
    curv = P.curvature_from_winds(latgrid, longrid, u[lo:hi], v[lo:hi])
    anom = P.anomaly_from_climatology(curv, times[lo:hi], climatology)
    advection = P._advection(latgrid, longrid, u[lo:hi], v[lo:hi], anom)

    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    coarse = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
              for n, f in (("u", u[lo:hi]), ("anomaly", anom), ("advection", advection))}
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
    rows_f, cols_f = P._subset(latgrid, longrid, P.DOMAIN_LAT, P.DOMAIN_LON)
    rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
    latgrid_f = latgrid[np.ix_(rows_f, cols_f)]
    longrid_f = longrid[np.ix_(rows_f, cols_f)]
    latgrid_c = lat_c[np.ix_(rows_c, cols_c)]
    longrid_c = lon_c[np.ix_(rows_c, cols_c)]
    u_c = coarse["u"][at][np.ix_(rows_c, cols_c)]
    anom_c = coarse["anomaly"][at][np.ix_(rows_c, cols_c)]
    adv_c = coarse["advection"][at][np.ix_(rows_c, cols_c)]
    anom_f = anom[at][np.ix_(rows_f, cols_f)]

    coarse_threshold, fine_threshold = P.thresholds_for("ERA-Int", 700)

    # ---- the replica of detect_troughs, intermediates kept --------------------------
    wind = clim.smooth9(np.asarray(u_c, dtype=float))
    curvature_c = _prepare(np.asarray(anom_c, dtype=float), latgrid_c[:, 0])
    advection_c = clim.smooth9(np.asarray(adv_c, dtype=float))
    curvature_f = _prepare(np.asarray(anom_f, dtype=float), latgrid_f[:, 0])

    westerly = wind > MAX_ZONAL_WIND
    advection_masked = np.where(westerly, np.nan, advection_c)
    curvature_masked = np.where(westerly, np.nan, curvature_c)
    weak = curvature_masked < coarse_threshold
    advection_masked = np.where(weak, np.nan, advection_masked)
    curvature_masked = np.where(weak, np.nan, curvature_masked)
    curvature_f_masked = np.where(curvature_f < fine_threshold, np.nan, curvature_f)

    axes = trough_axes(latgrid_c, longrid_c, advection_masked)
    candidates = [{"time": float(times[step]),
                   "lat_mean": float(np.mean(lats)),
                   "lon_mean": float(np.mean(lons))}
                  for lats, lons in axes]
    merged_coarse = merge_contours(candidates, latgrid_c, longrid_c,
                                   curvature_masked, coarse_threshold)
    merged_fine = merge_contours(merged_coarse, latgrid_f, longrid_f,
                                 curvature_f_masked, fine_threshold)

    # Bind the replica to the real path: same inputs, same final wave centers.
    real = detect_troughs(float(times[step]), latgrid_c, longrid_c, u_c, anom_c, adv_c,
                          latgrid_f, longrid_f, anom_f,
                          coarse_threshold=coarse_threshold,
                          fine_threshold=fine_threshold)
    replica_centers = sorted((w["lat_mean"], w["lon_mean"]) for w in merged_fine)
    real_centers = sorted((w["lat_mean"], w["lon_mean"]) for w in real)
    assert np.allclose(replica_centers, real_centers), \
        "the replica diverged from detect_troughs; every number below is suspect"

    # ---- version 1 at this time ------------------------------------------------------
    v1 = v1_observations_at(args.year, float(times[step]))
    print(f"\nversion 1 has {len(v1)} African observations at this time "
          f"(published SMOOTHED coordinates; duplicates possible across tracks)")
    print(f"the port: {len(axes)} trough axes -> {len(merged_coarse)} after the coarse "
          f"merge -> {len(merged_fine)} waves after the fine merge (whole domain)")

    axis_lats = np.concatenate([a[0] for a in axes]) if axes else np.array([])
    axis_lons = np.concatenate([a[1] for a in axes]) if axes else np.array([])
    cc_lat = [w["lat_mean"] for w in merged_coarse]
    cc_lon = [w["lon_mean"] for w in merged_coarse]
    ff_lat = [w["lat_mean"] for w in merged_fine]
    ff_lon = [w["lon_mean"] for w in merged_fine]

    def cell_at(lat, lon):
        r = int(np.argmin(np.abs(latgrid_c[:, 0] - lat)))
        c = int(np.argmin(np.abs(longrid_c[0, :] - lon)))
        return r, c

    print(f"\nPER VERSION 1 OBSERVATION (coarse threshold {coarse_threshold:.3g}):")
    print(f"{'lat':>6} {'lon':>7} | {'u_smooth':>8} {'masked?':>8} | {'anomaly':>9} "
          f"{'weak?':>6} | {'axis_km':>8} {'coarse_km':>9} {'fine_km':>8}")
    for la, lo_, n in v1:
        r, c = cell_at(la, lo_)
        uu = float(wind[r, c])
        aa = float(curvature_c[r, c])
        d_axis, _ = km_to_nearest(la, lo_, axis_lats, axis_lons)
        d_cc, _ = km_to_nearest(la, lo_, cc_lat, cc_lon)
        d_ff, _ = km_to_nearest(la, lo_, ff_lat, ff_lon)
        print(f"{la:6.1f} {lo_:7.1f} | {uu:8.2f} {str(bool(westerly[r, c])):>8} | "
              f"{aa:9.2e} {str(bool(aa < coarse_threshold)):>6} | "
              f"{d_axis:8.0f} {d_cc:9.0f} {d_ff:8.0f}")

    print(f"\nPORT WAVES after the fine merge, nearest version 1 observation:")
    v1lat = np.array([r[0] for r in v1])
    v1lon = np.array([r[1] for r in v1])
    for w in merged_fine:
        d, _ = km_to_nearest(w["lat_mean"], w["lon_mean"], v1lat, v1lon)
        print(f"  port wave at {w['lat_mean']:6.1f}, {w['lon_mean']:7.1f}   "
              f"nearest v1: {d:6.0f} km")

    # Check 3 from RECORD_SIZE_GAP.md: does the zero contour sit on the curvature ridge?
    print(f"\nAXIS vs CURVATURE PEAK, per coarse-merged wave (center refinement check):")
    for w in merged_coarse:
        values = np.where(w["region"], curvature_masked, np.nan)
        if not np.isfinite(values).any():
            continue
        peak = np.unravel_index(np.nanargmax(values), values.shape)
        d = float(great_circle_distance(w["lat_mean"], w["lon_mean"],
                                        float(latgrid_c[peak]), float(longrid_c[peak]),
                                        "km"))
        print(f"  center {w['lat_mean']:6.1f}, {w['lon_mean']:7.1f}   region peak at "
              f"{float(latgrid_c[peak]):6.1f}, {float(longrid_c[peak]):7.1f}   "
              f"{d:5.0f} km apart")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 7))
        shade = ax.pcolormesh(longrid_c, latgrid_c, curvature_c, cmap="viridis",
                              vmin=0, vmax=3 * coarse_threshold, shading="auto")
        fig.colorbar(shade, ax=ax, label="prepared coarse curvature anomaly (1/s)")
        ax.contourf(longrid_c, latgrid_c, westerly.astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["///"])
        for lats, lons in axes:
            ax.plot(lons, lats, color="white", lw=1.5)
        ax.plot(cc_lon, cc_lat, "o", mfc="none", mec="orange", ms=12,
                label="coarse-merged center")
        ax.plot(ff_lon, ff_lat, ".", color="red", ms=10, label="final wave center")
        ax.plot(v1lon, v1lat, "x", color="black", ms=12, mew=2.5,
                label="version 1 observation")
        ax.set_xlim(-42, 42)
        ax.set_ylim(-8, 32)
        ax.set_title(f"{when}  |  hatch = westerly mask, white = trough axes")
        ax.legend(loc="lower left")
        fig.savefig(args.figure, dpi=140, bbox_inches="tight")
        print(f"\nfigure written to {args.figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
