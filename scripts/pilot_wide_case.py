#!/usr/bin/env python3
"""A WIDE case for one season: the protocol's preprocessing (curvature from the winds,
the anomaly against the pinned climatology, its advection, the decimation) run on the
retained 75 E retrieval and cropped to a declared box 4 degrees wider than the control
run's tracking domain (the 40 E control's or the 60 E control's) on every side, so that an
all-edge margin preparation has interior-smoothed values available beyond each edge of
the domain.

BOUND TO THE RETAINED CONTROL CASE. The same arrays cropped to the control's own box must
equal the control's retained case bit for bit, every field at every step and both grids'
coordinates, or the build refuses. That binds this laptop's preprocessing of these inputs
to the compute-cluster run that produced the released control, and makes the wide case the
control's own fields with a border, nothing else. The inputs are hashed and must be the
files the control's record names, and the climatology cache must be the one it names.

WHAT THE WIDE CASE IS NOT. It is not a domain variant of the protocol (whose guard keeps a
variant 15 degrees inside the retrieval, which forbids any latitude beyond 35) and it is
never tracked on its own. It feeds `pilot_alledge_replay.py`, which prepares on it and
crops to the control grids before the masks, the contouring, the merges and the
association.

    python3 scripts/pilot_wide_case.py --year 1990 --inputs data/era5/pilot_east_75E --climo data/climo/east75E/climo_era5_1979_2010.npz \\
        --control-run data/protocol_runs/pilot_B/era5_1990 --control-case <B case> --out-dir <fresh dir>
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402
from export_protocol_case import case_id, publish_mat  # noqa: E402

MARGIN_DEGREES = 4.0            # two coarse cells beyond every edge of the control run's domain


def wide_box(domain, margin=MARGIN_DEGREES):
    """The control run's domain widened by the margin on every side, as (lat, lon) ranges,
    so the same builder serves the 40 E control and the 60 E control."""
    lat, lon = domain["lat"], domain["lon"]
    return (float(lat[0]) - margin, float(lat[1]) + margin), (float(lon[0]) - margin, float(lon[1]) + margin)


WIDE_LAT, WIDE_LON = wide_box({"lat": (-35.0, 35.0), "lon": (-140.0, 40.0)})      # the 40 E control's box, for reference
RAW_FIELDS = ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c", "u", "v", "currv_anom")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_pinned_climatology(cache):
    """The climatology dict from the cache whose DIGEST the control's record names.

    `climo_cache.load_or_build` establishes a cache's identity by a fingerprint over every
    reference year's input file, which this machine does not hold. Here the identity is
    established the other way, by `main` checking the cache file's sha256 against the
    control run's record before this is called, and the bind to the control case then
    checks the result. The dict is the one `load_or_build` returns, read the same way."""
    with np.load(cache, allow_pickle=False) as z:
        if "fingerprint" not in z:
            raise SystemExit(f"REFUSED: {cache} carries no fingerprint")
        keys = [tuple(int(x) for x in row) for row in z["keys"]]
        return {"keys": keys, "mean": z["mean"], "counts": z["counts"], "short_steps": []}


def preprocess(year, inputs, prefix, climo_cache):
    """The protocol's preprocessing on the full retrieval grid, as `export_protocol_case`
    runs it, returning the native and the decimated fields uncropped."""
    from aew.v1port import climatology as clim
    from aew.v1port import load as L
    from aew.v1port import pipeline as PL
    times, latgrid, longrid, u, v = L.load_year(year, inputs, prefix)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    curvature = PL.curvature_from_winds(latgrid, longrid, u, v)
    climo = read_pinned_climatology(climo_cache)
    anomaly = PL.anomaly_from_climatology(curvature, times, climo)
    del curvature
    advection = PL._advection(latgrid, longrid, u, v, anomaly)
    nominal = PL.COARSE_RESOLUTION_DEG
    coarse = {"u": clim.gaussian_decimate(u, native, nominal), "v": clim.gaussian_decimate(v, native, nominal),
              "anomaly": clim.gaussian_decimate(anomaly, native, nominal), "advection": clim.gaussian_decimate(advection, native, nominal)}
    del advection
    lat_c, lon_c = PL.coarse_grid(latgrid, longrid, native, nominal)
    return {"times": times, "latgrid": latgrid, "longrid": longrid, "u": u, "v": v, "anomaly": anomaly, "lat_c": lat_c, "lon_c": lon_c, "coarse": coarse, "native": native, "nominal": nominal}


def crop(pre, lat_range, lon_range):
    """The payload a case file holds, for one box, in the protocol's layout."""
    from aew.v1port import pipeline as PL
    rows_f, cols_f = PL._subset(pre["latgrid"], pre["longrid"], lat_range, lon_range)
    rows_c, cols_c = PL._subset(pre["lat_c"], pre["lon_c"], lat_range, lon_range)

    def cut_f(field):
        return field[:, rows_f, :][:, :, cols_f]

    def cut_c(field):
        return field[:, rows_c, :][:, :, cols_c]
    fine_lat, fine_lon = pre["latgrid"][np.ix_(rows_f, cols_f)], pre["longrid"][np.ix_(rows_f, cols_f)]
    coarse_lat, coarse_lon = pre["lat_c"][np.ix_(rows_c, cols_c)], pre["lon_c"][np.ix_(rows_c, cols_c)]
    return {"lat_c": coarse_lat[:, 0], "lon_c": coarse_lon[0, :], "latgrid": fine_lat, "longrid": fine_lon, "time": np.asarray(pre["times"], float),
            "u_c": cut_c(pre["coarse"]["u"]), "v_c": cut_c(pre["coarse"]["v"]), "currv_anom_c": cut_c(pre["coarse"]["anomaly"]), "advcurrv_anom_c": cut_c(pre["coarse"]["advection"]),
            "u": cut_f(pre["u"]), "v": cut_f(pre["v"]), "currv_anom": cut_f(pre["anomaly"])}


def bind_to_control(payload_control_box, case_b):
    """Every field and both grids of the control-box crop against the retained control case."""
    out = {}
    for key in ("lat_c", "lon_c"):
        out[key] = bool(np.array_equal(payload_control_box[key], case_b[key]))
    for key in ("latgrid", "longrid"):
        out[key] = bool(np.array_equal(payload_control_box[key], case_b[key]))
    out["time"] = bool(np.array_equal(payload_control_box["time"], case_b["time"]))
    for key in RAW_FIELDS:
        a, b = payload_control_box[key], case_b[key]
        out[key] = bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--inputs", "--climo", "--control-run", "--control-case", "--out-dir"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--prefix", default="era5")
    args = ap.parse_args(argv)
    try:
        os.mkdir(args.out_dir)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out_dir} exists and a wide case is never overwritten")
    run = P.load_run(args.control_run, year=args.year)
    ds = run["record"]["dataset_specific"]
    inputs = {}
    for name, want in ds["inputs_sha256"].items():
        path = os.path.join(args.inputs, name)
        got = _sha256(path)
        if got != want:
            raise SystemExit(f"REFUSED: {path} is not the input the control's record names")
        inputs[name] = got
    climo_sha = _sha256(args.climo)
    if climo_sha != ds["climo_cache_sha256"]:
        raise SystemExit("REFUSED: the climatology cache is not the one the control's record names")
    case_b = SD.load_case(args.control_case, run["record"])
    t0 = time.perf_counter()
    pre = preprocess(args.year, args.inputs, args.prefix, args.climo)
    control_box = crop(pre, tuple(run["record"]["protocol_settings"]["domain"]["lat"]), tuple(run["record"]["protocol_settings"]["domain"]["lon"]))
    bound = bind_to_control(control_box, case_b)
    if not all(bound.values()):
        raise SystemExit(f"REFUSED: the control-box crop does not reproduce the retained control case: {bound}")
    wide_lat, wide_lon = wide_box(run["record"]["protocol_settings"]["domain"])
    wide = crop(pre, wide_lat, wide_lon)
    wide["rean"], wide["level"] = ds["label"], float(run["record"]["protocol_settings"]["level_hpa"])
    wide["case_id"] = case_id(wide)
    elapsed = time.perf_counter() - t0
    record = {"generated_by": "scripts/pilot_wide_case.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year,
              "what": f"the protocol's preprocessing on the retained 75 E retrieval, cropped to a box {MARGIN_DEGREES:g} degrees wider than the control run's domain on every side, for the all-edge margin preparation; never tracked on its own",
              "box": {"lat": list(wide_lat), "lon": list(wide_lon)}, "margin_degrees": MARGIN_DEGREES, "control_domain": run["record"]["protocol_settings"]["domain"],
              "inputs_sha256": inputs, "climo_cache": args.climo, "climo_cache_sha256": climo_sha,
              "control_run": {"dir": args.control_run, "record_sha256": run["record_sha256"], "case_sha256": case_b["sha256"], "tracks_sha256": run["tracks_sha256"]},
              "bound_to_control_case": bound, "shapes": {"coarse": list(wide["u_c"].shape[1:]), "fine": list(wide["u"].shape[1:]), "steps": int(wide["time"].size)},
              "coarse_extent": {"lat": [float(wide["lat_c"].min()), float(wide["lat_c"].max())], "lon": [float(wide["lon_c"].min()), float(wide["lon_c"].max())]},
              "fine_extent": {"lat": [float(wide["latgrid"][:, 0].min()), float(wide["latgrid"][:, 0].max())], "lon": [float(wide["longrid"][0].min()), float(wide["longrid"][0].max())]},
              "case_id": wide["case_id"], "elapsed_seconds": round(elapsed, 1)}
    case_path = os.path.join(args.out_dir, "wide_case.mat")
    publish_mat(case_path, {**wide, "producer_json": json.dumps(record, sort_keys=True)})
    record["wide_case_sha256"] = _sha256(case_path)
    X.publish_json(os.path.join(args.out_dir, f"wide_case_{args.year}.json"), record, exclusive=True)
    print(f"wide case {args.year}: coarse {record['shapes']['coarse']} fine {record['shapes']['fine']}, bound to the control case {all(bound.values())}, {elapsed:.0f} s, sha256 {record['wide_case_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
