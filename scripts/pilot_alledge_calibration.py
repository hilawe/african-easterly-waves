#!/usr/bin/env python3
"""The calibration audit under the margin preparation's order: the two percentiles
recomputed over the same reference period with the smoother applied on the buffered
grids BEFORE the crop, beside the archived order, from the same inputs, population years
and domain, with the archived order required to reproduce the released thresholds
exactly before the other order is read.

WHY A SEPARATE INSTRUMENT. The production producer, `compute_thresholds.py`, and its
library, `aew.v1port.thresholds`, are bound by digest into the committed calibration
artifacts and a provenance gate refuses any change to them. This instrument therefore
reuses their pieces (the climatology builder, the curvature and anomaly chain, the
decimation, `transform_step`, the population files and the exact percentile) and writes
its own population loop with the order as a parameter. THE BIND: with the archived order,
crop to the domain then smooth, the pair it produces must equal the released artifact's
pair to the last bit, or it refuses, since that is the only evidence that its loop is the
producer's loop. With the declared order, smooth on the buffered grids then crop, every
cell of the population carries the interior-smoothed value, which is what the all-edge
margin replay's preparation gives the detector.

THE READING is two pairs of numbers and their relative differences. Whether the recomputed
pair changes any mask decision on the margin-prepared fields is the mask check's question
(`pilot_alledge_mask_check.py`), kept apart from this one.

    python3 scripts/pilot_alledge_calibration.py --directory <75 E inputs> --prefix era5 --climatology-years 1979 2010 \\
        --population-years 1979 2010 --lat-range -35 35 --lon-range -140 40 --released <pilot control 40E artifact> --scratch <dir> --out <fresh json>
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port import thresholds as T  # noqa: E402

ORDERS = ("crop_then_smooth", "smooth_then_crop")


def build_population(years, directory, prefix, climatology, scratch_dir, *, order, lat_range, lon_range, coarse_resolution=2.5, passes=1):
    """The producer's population loop with the order as a parameter, on its own pieces."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    os.makedirs(scratch_dir, exist_ok=True)
    files = T.PopulationFiles(scratch_dir)
    with open(files.fine_path, "wb") as fine_out, open(files.coarse_path, "wb") as coarse_out:
        for year in sorted(years):
            times, latgrid, longrid, curvature = L.curvature_for_year(year, directory, prefix)
            anomaly = clim.curvature_anomaly(curvature, P.days_to_datetime64(times), climatology)
            del curvature
            lats, lons = latgrid[:, 0], longrid[0, :]
            rows = np.where((lats >= lat_range[0]) & (lats <= lat_range[1]))[0]
            cols = np.where((lons >= lon_range[0]) & (lons <= lon_range[1]))[0]
            native = abs(float(lats[1] - lats[0]))
            _, stride = clim.decimation_shape(native, coarse_resolution)
            clats, clons = lats[::stride], lons[::stride]
            crows = np.where((clats >= lat_range[0]) & (clats <= lat_range[1]))[0]
            ccols = np.where((clons >= lon_range[0]) & (clons <= lon_range[1]))[0]
            fine_cells, coarse_cells = rows.size * cols.size, crows.size * ccols.size
            if files.fine_cells is None:
                files.fine_cells, files.coarse_cells = fine_cells, coarse_cells
                files.fine_shape, files.coarse_shape = (int(rows.size), int(cols.size)), (int(crows.size), int(ccols.size))
                files.fine_lats, files.fine_lons = lats[rows].copy(), lons[cols].copy()
                files.coarse_lats, files.coarse_lons = clats[crows].copy(), clons[ccols].copy()
            elif (files.fine_cells, files.coarse_cells) != (fine_cells, coarse_cells):
                raise ValueError(f"{year} yields a {fine_cells}/{coarse_cells}-cell domain where earlier years gave {files.fine_cells}/{files.coarse_cells}")
            for step in range(anomaly.shape[0]):
                decimated = clim.gaussian_decimate(anomaly[step], native, coarse_resolution)
                if order == "crop_then_smooth":
                    fine = T.transform_step(anomaly[step][np.ix_(rows, cols)], lats[rows], passes)
                    coarse = T.transform_step(decimated[np.ix_(crows, ccols)], clats[crows], passes)
                else:
                    fine = T.transform_step(anomaly[step], lats, passes)[np.ix_(rows, cols)]
                    coarse = T.transform_step(decimated, clats, passes)[np.ix_(crows, ccols)]
                fine.astype(np.float64).tofile(fine_out)
                coarse.astype(np.float64).tofile(coarse_out)
            files.steps += int(anomaly.shape[0])
            del anomaly
    for which, path, cells in (("fine", files.fine_path, files.fine_cells), ("coarse", files.coarse_path, files.coarse_cells)):
        if os.path.getsize(path) // 8 != cells * files.steps:
            raise RuntimeError(f"{which} population holds {os.path.getsize(path) // 8} values, not {cells * files.steps}")
    return files


def thresholds_for(files):
    out = {}
    for which, q in (("coarse", T.COARSE_Q), ("fine", T.FINE_Q)):
        value, count, upper = T.exact_threshold(files, which, q)
        out[which] = {"threshold": float(value), "finite_count": int(count), "domain_upper_bound": int(upper), "percentile": q}
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--directory", "--released", "--scratch", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--prefix", default="era5")
    ap.add_argument("--climatology-years", nargs=2, type=int, required=True)
    ap.add_argument("--population-years", nargs=2, type=int, required=True)
    ap.add_argument("--lat-range", nargs=2, type=float, required=True)
    ap.add_argument("--lon-range", nargs=2, type=float, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    with open(args.released, "rb") as fh:
        rel_blob = fh.read()
    released = json.loads(rel_blob)
    want = (float(released["coarse"]["threshold"]), float(released["fine"]["threshold"]))
    if (list(released["domain"]["lat_range"]) != list(args.lat_range) or list(released["domain"]["lon_range"]) != list(args.lon_range)
            or list(released["climatology_years"]) != list(args.climatology_years) or list(released["population_years"]) != list(args.population_years)):
        raise SystemExit("REFUSED: the released artifact's domain or years are not this run's")
    cy = list(range(args.climatology_years[0], args.climatology_years[1] + 1))
    py = list(range(args.population_years[0], args.population_years[1] + 1))
    inputs = {}
    for year in sorted(set(cy + py)):
        for var in ("u700", "v700"):
            path = os.path.join(args.directory, f"{args.prefix}_{var}_{year}_6h_region.nc")
            inputs[os.path.basename(path)] = _sha256(path)
    if any(inputs.get(k) != v for k, v in released.get("input_file_sha256", {}).items()):
        raise SystemExit("REFUSED: the inputs are not the files the released artifact names")
    t0 = time.perf_counter()
    climatology = L.build_climatology(cy, args.directory, args.prefix, progress=lambda y, n, t: print(f"   climatology {y} ({n}/{t})", flush=True))
    out = {"generated_by": "scripts/pilot_alledge_calibration.py", "script_sha256": X.digest(os.path.abspath(__file__)),
           "library_sha256": {"aew/v1port/thresholds.py": _sha256(T.__file__), "aew/v1port/climatology.py": _sha256(clim.__file__), "aew/v1port/load.py": _sha256(L.__file__), "aew/v1port/pipeline.py": _sha256(P.__file__)},
           "released_artifact": {"path": args.released, "sha256": hashlib.sha256(rel_blob).hexdigest(), "case_id": released.get("case_id"), "coarse": want[0], "fine": want[1]},
           "inputs": {"directory": args.directory, "prefix": args.prefix, "sha256": inputs}, "climatology_years": args.climatology_years, "population_years": args.population_years,
           "domain": {"lat_range": list(args.lat_range), "lon_range": list(args.lon_range)}, "percentiles": {"coarse": T.COARSE_Q, "fine": T.FINE_Q}, "estimator": "exact, linear interpolation",
           "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__}, "orders": {}}
    for order in ORDERS:
        scratch = os.path.join(args.scratch, order)
        os.makedirs(scratch, exist_ok=True)
        t1 = time.perf_counter()
        print(f"  building the {order} population", flush=True)
        files = build_population(py, args.directory, args.prefix, climatology, scratch, order=order, lat_range=tuple(args.lat_range), lon_range=tuple(args.lon_range))
        thr = thresholds_for(files)
        out["orders"][order] = {**thr, "steps": files.steps, "fine_shape": list(files.fine_shape), "coarse_shape": list(files.coarse_shape), "elapsed_seconds": round(time.perf_counter() - t1, 1)}
        for path in (files.fine_path, files.coarse_path):
            os.unlink(path)
        if order == "crop_then_smooth":
            got = (thr["coarse"]["threshold"], thr["fine"]["threshold"])
            out["bind"] = {"archived_order_reproduces_released_pair": got == want, "got": list(got), "released": list(want)}
            if got != want:
                X.publish_json(args.out, out, exclusive=True)
                raise SystemExit(f"REFUSED: the archived order gives {got}, not the released {want}; the loop is not the producer's")
        print(f"  {order}: coarse {thr['coarse']['threshold']!r} fine {thr['fine']['threshold']!r} ({out['orders'][order]['elapsed_seconds']} s)", flush=True)
    a, b = out["orders"]["crop_then_smooth"], out["orders"]["smooth_then_crop"]
    out["relative_difference"] = {w: (b[w]["threshold"] - a[w]["threshold"]) / a[w]["threshold"] for w in ("coarse", "fine")}
    out["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    X.publish_json(args.out, out, exclusive=True)
    print(f"done: archived {a['coarse']['threshold']!r}/{a['fine']['threshold']!r}, smooth-then-crop {b['coarse']['threshold']!r}/{b['fine']['threshold']!r}, relative {out['relative_difference']}, {out['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
