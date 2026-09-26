#!/usr/bin/env python3
"""The shared entry point of the reanalysis comparison protocol: one dataset, one
calendar year, one code path, every setting read from the retained manifest, ENFORCED
and not only recorded.

WHY A NEW ENTRY POINT. `export_tracker_case.py` hardcodes the ERA-Interim file prefix,
builds its climatology over every year on disk, labels the case ERA-Int, selects the
archived thresholds by that label, and carries a tracker-variant switch. A protocol run
must take the dataset, the years, the calibration and the tracker flags explicitly and
record them, so a run on either reanalysis cannot make one of those choices silently.

WHAT A REVIEW CHANGED. A first version copied the manifest into its record and enforced
almost none of it, accepted a calibration artifact on its prefix and years alone, checked
only the target year's files, let the tracker recompute its fields under its own defaults,
wrote a case without its settings at a path a later run overwrote, and its tests passed
with tracking removed. Now:

- The manifest's fixed settings are validated against the code's own constants and the
  supported values, and a manifest asking for anything else is refused before any file
  is read (section VALIDATE).
- Every year the climatology needs is preflighted against the manifest grid and
  digested, not only the target year.
- The wind files pass the retrieval validator (units, dimensions, epoch, payload) before
  any computation.
- The calibration artifact must carry the producer's own population, transformation,
  grid, domain, estimator and digest fields and they must fit the manifest.
- The retained case carries the protocol settings, the calibration identity, the
  thresholds, the tracker flags and the producer record inside it, lands in a run
  directory named by dataset and year, and no final path is ever overwritten.
- Tracking is REPLAYED FROM THE RETAINED CASE, the same loop phase B used against version
  1, with the thresholds and both tracker flags passed explicitly, so the tracks are bound
  to the bytes retained and nothing is prepared twice.
- The completion record is written last, after the case and the tracks are on disk.

TWO STAGES.
  inputs    preflight, retrieval-grade validation, curvature, decimation, crop, and the
            readiness readings. No climatology, no anomaly, no tracking. What the one-year
            ERA5 input-readiness pilot can run.
  tracking  the inputs stage, then the anomaly against the dataset's own climatology over
            the manifest years, the advection, the retained case, and the replayed tracks.

    .venv/bin/python3 scripts/export_protocol_case.py --manifest <json> --dataset eraint \\
        --year 1990 --stage tracking --calibration <thresholds json> --climo-cache <npz> --out-dir <dir>
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import detection as D  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port import thresholds as T  # noqa: E402
from aew.v1port.association import associate_step, finalize_tracks, prune_stale_tracks  # noqa: E402
from aew.v1port.climo_cache import climatology_fingerprint, load_or_build  # noqa: E402
from download_eraint_v1port import validate_file  # noqa: E402
from export_tracker_case import case_id  # noqa: E402

PROTOCOL_KEYS = ("years", "level_hpa", "hours", "grid", "decimation", "domain", "smoothing",
                 "sign_convention", "wind_mask", "tracker_flags", "initialization",
                 "calibration", "membership", "reporting")
SYNOPTIC_HOURS = ["00:00", "06:00", "12:00", "18:00"]
HEX64 = re.compile(r"[0-9a-f]{64}\Z")     # a full match, so a trailing newline does not pass
# THE FIXED BEHAVIOR, in the words the manifest must carry. These describe what the code
# does and are not configurable, so a manifest that says anything else is refused rather
# than recorded, which a review showed happening for the sign rule, the initialization,
# the smoothing placement and the order of decimation and cropping.
FIXED_TEXT = {
    ("decimation", "order"): "advection on the native grid, decimate, crop both grids, smooth and flip the southern-hemisphere sign inside detection",
    ("smoothing", "where"): "inside detection, on each cropped field",
    ("sign_convention",): "southern-hemisphere curvature reversed so cyclonic is positive in both hemispheres, in detection and in the calibration population",
    ("wind_mask",): "applied to the coarse fields only",
    ("initialization",): "each calendar year from January 1 00Z, started cold, both datasets",
    ("calibration", "estimator"): "exact, linear interpolation",
    ("reporting", "membership"): "first observation inside the season",
    ("membership", "rule"): "the archive's source-region rule: the first observation, each coordinate rounded to the nearest quarter degree with ties away from zero, tested against the region polygons in priority order, the first containing polygon naming the region, else OTH",
    ("reporting", "crossing_in"): "first observation before June 1 00Z and last observation at or after June 1 00Z",
    ("reporting", "crossing_out"): "first observation at or after June 1 00Z and before October 1 00Z, and last observation at or after October 1 00Z",
    ("reporting", "year_end"): "last observation on December 31, counted as potentially censored",
}
REGION_PRIORITY = ["NEP", "SEP", "CAM", "SAM", "NAL", "SAL", "AFR"]
TRACKING_SOURCES = ("src/aew/v1port/", "scripts/export_protocol_case.py")


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_manifest(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    manifest = json.loads(blob.decode())
    missing = [k for k in PROTOCOL_KEYS + ("datasets",) if k not in manifest]
    if missing:
        raise SystemExit(f"REFUSED: the manifest lacks {missing}")
    problems = validate_manifest(manifest)
    if problems:
        raise SystemExit("REFUSED: the manifest asks for what this protocol does not support: " + "; ".join(problems))
    return manifest, hashlib.sha256(blob).hexdigest()


def validate_manifest(m):
    """Every fixed setting against the code's constants and the supported values. The
    protocol makes none of these configurable, so a manifest that differs is refused
    rather than recorded. Returns the list of problems."""
    p = []
    g, dec, dom, cal, flags = m["grid"], m["decimation"], m["domain"], m["calibration"], m["tracker_flags"]
    if list(m["hours"]) != SYNOPTIC_HOURS:
        p.append(f"hours {m['hours']} are not the four synoptic hours the loader and preflight assume")
    if int(m["level_hpa"]) != 700:
        p.append("level is not 700 hPa")
    n, w, s, e = g["area_nwse"]
    if (n, s, w, e) != (g["lat_first"], g["lat_last"], g["lon_first"], g["lon_last"]):
        p.append("grid.area_nwse disagrees with the grid's first and last coordinates")
    if g["row_order"] != "descending" or g["lat_first"] <= g["lat_last"]:
        p.append("the grid must be latitude-descending, which is how the coordinates are built")
    lat = np.arange(g["lat_first"], g["lat_last"] - g["spacing_deg"] / 2, -g["spacing_deg"])
    lon = np.arange(g["lon_first"], g["lon_last"] + g["spacing_deg"] / 2, g["spacing_deg"])
    if [lat.size, lon.size] != list(g["shape"]):
        p.append(f"grid shape {g['shape']} is not what the coordinates give, {lat.size}x{lon.size}")
    stride = int(np.floor(float(dec["nominal_deg"]) / float(g["spacing_deg"])))
    if stride != int(dec["stride_on_one_degree_input"]) or not np.isclose(stride * g["spacing_deg"], dec["achieved_spacing_deg"]):
        p.append(f"decimation stride {dec['stride_on_one_degree_input']} and achieved spacing "
                 f"{dec['achieved_spacing_deg']} do not follow from nominal {dec['nominal_deg']} on {g['spacing_deg']} degrees")
    if float(dec["nominal_deg"]) != P.COARSE_RESOLUTION_DEG:
        p.append(f"nominal decimation {dec['nominal_deg']} is not the code's {P.COARSE_RESOLUTION_DEG}")
    if tuple(dom["lat"]) != tuple(P.DOMAIN_LAT) or tuple(dom["lon"]) != tuple(P.DOMAIN_LON):
        p.append(f"domain {dom} is not the code's {P.DOMAIN_LAT} by {P.DOMAIN_LON}")
    if int(m["smoothing"]["passes"]) != 1:
        p.append("the code smooths once inside detection, and the manifest asks for another count")
    for keys, text in FIXED_TEXT.items():
        node = m
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if node != text:
            p.append(f"{'.'.join(keys)} is not the fixed behavior this code implements")
    mem, rep = m["membership"], m["reporting"]
    if list(mem.get("priority", [])) != REGION_PRIORITY or mem.get("product_region") != "AFR":
        p.append("membership priority or product region is not the archive's rule")
    if list(rep.get("season_months", [])) != [6, 7, 8, 9]:
        p.append("the reporting season is not June to September")
    if set(flags) != {"exclusive", "absorb"} or not all(isinstance(v, bool) for v in flags.values()):
        p.append("tracker_flags must be exactly exclusive and absorb, both booleans")
    if cal["rule"] != "A" or cal["percentiles"] != {"coarse": 55.0, "fine": 66.0} \
            or cal.get("wind_mask_in_population") is not False or cal.get("smoothing_adjustment") != "none":
        p.append("the calibration block is not Rule A as the protocol fixes it")
    y0, y1 = (int(x) for x in m["years"])
    if [int(x) for x in cal["climatology_years"]] != [y0, y1] or [int(x) for x in cal["population_years"]] != [y0, y1]:
        p.append("the calibration years are not the protocol years")
    if y1 < y0:
        p.append("the protocol years are reversed")
    return p


def dataset_entry(manifest, name):
    if name not in manifest["datasets"]:
        raise SystemExit(f"REFUSED: the manifest names no dataset {name!r}, only {sorted(manifest['datasets'])}")
    return manifest["datasets"][name]


def expected_grid(manifest):
    g = manifest["grid"]
    lat = np.arange(g["lat_first"], g["lat_last"] - g["spacing_deg"] / 2, -g["spacing_deg"])
    lon = np.arange(g["lon_first"], g["lon_last"] + g["spacing_deg"] / 2, g["spacing_deg"])
    return lat, lon


def area_and_grid_text(manifest):
    g = manifest["grid"]
    return ("/".join(f"{x:g}" for x in g["area_nwse"]), f"{g['spacing_deg']:g}/{g['spacing_deg']:g}")


def input_paths(dataset, year):
    return {var: os.path.join(dataset["directory"], f"{dataset['prefix']}_{var}_{year}_6h_region.nc")
            for var in ("u700", "v700")}


def validate_inputs(manifest, dataset, years):
    """Every year's two files through the retrieval validator (grid, calendar, epoch,
    variable, dimensions, units, payload) and the threshold preflight (identical
    coordinates across years and files). Returns the preflight metadata and the digests."""
    area, grid = area_and_grid_text(manifest)
    digests = {}
    for year in years:
        for var, path in input_paths(dataset, year).items():
            if not os.path.exists(path):
                raise SystemExit(f"REFUSED: {path} is absent")
            reason = validate_file(path, year, var, area, grid)
            if reason is not None:
                raise SystemExit(f"REFUSED: {os.path.basename(path)} fails the retrieval validator: {reason}")
            digests[os.path.basename(path)] = _sha256(path)
    want_lat, want_lon = expected_grid(manifest)
    preflight = T.preflight_years(years, dataset["directory"], dataset["prefix"],
                                  expected_lat=want_lat, expected_lon=want_lon,
                                  expected_level_hpa=float(manifest["level_hpa"]))
    return preflight, digests


def check_calibration(artifact_path, manifest, dataset):
    """A calibration artifact is accepted only if the producer's own fields say it was
    computed for this dataset, over the manifest's years, on the manifest grid and domain,
    under Rule A's transformation, population and estimator, with finite thresholds and
    the input digests recorded."""
    with open(artifact_path, "rb") as fh:
        blob = fh.read()
    art = json.loads(blob.decode())
    cal, g, dom = manifest["calibration"], manifest["grid"], manifest["domain"]
    y0, y1 = (int(x) for x in manifest["years"])
    problems = []

    def need(cond, text):
        if not cond:
            problems.append(text)
    need(art.get("schema") == "thresholds-v2", f"schema {art.get('schema')!r} is not thresholds-v2")
    need(art.get("prefix") == dataset["prefix"], f"prefix {art.get('prefix')!r} is not {dataset['prefix']!r}")
    need([int(x) for x in art.get("climatology_years", [])] == [y0, y1], f"climatology_years {art.get('climatology_years')} are not {[y0, y1]}")
    need([int(x) for x in art.get("population_years", [])] == [y0, y1], f"population_years {art.get('population_years')} are not {[y0, y1]}")
    need(art.get("percentiles") == cal["percentiles"], f"percentiles {art.get('percentiles')} are not {cal['percentiles']}")
    need(art.get("estimator") == "exact" and art.get("percentile_method") == "linear",
         "the estimator is not exact with linear interpolation")
    tr = art.get("transformation") or {}
    need(tr.get("id") == "T0" and int(tr.get("smoothing_passes", -1)) == 1 and tr.get("coarse_scale") is None,
         f"transformation {tr} is not T0 with one smoothing pass and no coarse scale")
    pop = art.get("population") or {}
    need(pop.get("wind_mask") == "none" and int(pop.get("smoothing_passes", -1)) == 1 and pop.get("hemispheres") == "both",
         f"population {pop} is not the rule's (no wind mask, one pass, both hemispheres)")
    need(float(art.get("coarse_resolution", -1)) == float(manifest["decimation"]["nominal_deg"]),
         f"coarse_resolution {art.get('coarse_resolution')} is not the nominal decimation")
    d = art.get("domain") or {}
    need(list(d.get("lat_range", [])) == list(dom["lat"]) and list(d.get("lon_range", [])) == list(dom["lon"]),
         f"domain {d} is not the manifest's {dom}")
    pf = art.get("preflight") or {}
    need(pf.get("n_lat") == g["shape"][0] and pf.get("n_lon") == g["shape"][1]
         and pf.get("lat_first") == g["lat_first"] and pf.get("lat_last") == g["lat_last"]
         and pf.get("lon_first") == g["lon_first"] and pf.get("lon_last") == g["lon_last"]
         and float(pf.get("level_hpa", -1)) == float(manifest["level_hpa"]),
         "the artifact's preflight grid is not the manifest grid")
    need(int(art.get("subsample_stride", 0)) == 1, "the population was subsampled")
    ct = (art.get("coarse") or {}).get("threshold")
    ft = (art.get("fine") or {}).get("threshold")
    need(ct is not None and ft is not None and np.isfinite(float(ct)) and np.isfinite(float(ft)) and float(ct) > 0 and float(ft) > 0,
         "the thresholds are not finite positive numbers")
    inputs = art.get("input_file_sha256") or {}
    expected_files = {f"{dataset['prefix']}_{var}_{y}_6h_region.nc" for y in range(y0, y1 + 1) for var in ("u700", "v700")}
    need(all(isinstance(inputs.get(f), str) and HEX64.fullmatch(inputs.get(f, "")) for f in expected_files),
         "the artifact does not record a full digest for every population input file")
    if problems:
        raise SystemExit("REFUSED: the calibration artifact does not fit the manifest: " + "; ".join(problems))
    return float(ct), float(ft), hashlib.sha256(blob).hexdigest(), inputs


def source_digests():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources = {}
    for root, _dirs, files in os.walk(os.path.join(repo, "src", "aew", "v1port")):
        for name in sorted(files):
            if name.endswith(".py"):
                full = os.path.join(root, name)
                sources[os.path.relpath(full, repo)] = _sha256(full)
    sources["scripts/export_protocol_case.py"] = _sha256(os.path.abspath(__file__))
    return sources


def producer_record(manifest, manifest_sha256, dataset_name, dataset, year, stage, inputs_sha256, extra):
    """Protocol settings apart from dataset-specific values, so a comparison can require
    the first block equal and expect the second to differ."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*argv):
        try:
            return subprocess.run(["git", "-C", repo, *argv], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"producer": "scripts/export_protocol_case.py",
            "protocol_settings": {**{k: manifest[k] for k in PROTOCOL_KEYS}, "manifest_sha256": manifest_sha256},
            "dataset_specific": {"dataset": dataset_name, "label": dataset["label"], "prefix": dataset["prefix"],
                                 "year": year, "inputs_sha256": inputs_sha256, **extra},
            "stage": stage, "git_head": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "src", "scripts") or ""),
            "source_sha256": source_digests(),
            "produced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def inputs_stage(manifest, dataset, year, preflight):
    """Curvature, decimation and crop for one already-validated year. Returns the arrays
    the tracking stage continues from and the readiness readings."""
    times, latgrid, longrid, u, v = L.load_year(year, dataset["directory"], dataset["prefix"])
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    if not np.isclose(native, float(manifest["grid"]["spacing_deg"])):
        raise SystemExit(f"REFUSED: the native spacing is {native}, not the manifest's {manifest['grid']['spacing_deg']}")
    curvature = P.curvature_from_winds(latgrid, longrid, u, v)
    nominal = float(manifest["decimation"]["nominal_deg"])
    coarse = {name: clim.gaussian_decimate(f, native, nominal) for name, f in (("u", u), ("v", v), ("curvature", curvature))}
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, nominal)
    rows_f, cols_f = P._subset(latgrid, longrid, tuple(manifest["domain"]["lat"]), tuple(manifest["domain"]["lon"]))
    rows_c, cols_c = P._subset(lat_c, lon_c, tuple(manifest["domain"]["lat"]), tuple(manifest["domain"]["lon"]))
    achieved = abs(float(lat_c[1, 0] - lat_c[0, 0]))
    if not np.isclose(achieved, float(manifest["decimation"]["achieved_spacing_deg"])):
        raise SystemExit(f"REFUSED: the achieved coarse spacing is {achieved}, not the manifest's "
                         f"{manifest['decimation']['achieved_spacing_deg']}")
    if len(rows_c) < 3 or len(cols_c) < 3:
        raise SystemExit(f"REFUSED: the domain crop leaves a coarse grid of {len(rows_c)}x{len(cols_c)}, too small to track")
    u_c = coarse["u"][:, rows_c, :][:, :, cols_c]
    readiness = {
        "preflight": preflight, "timesteps": int(times.size), "native_spacing_deg": native,
        "achieved_coarse_spacing_deg": achieved,
        "fine_shape": [int(len(rows_f)), int(len(cols_f))], "coarse_shape": [int(len(rows_c)), int(len(cols_c))],
        "finite_fraction": {name: float(np.mean(np.isfinite(f[:, rows_f, :][:, :, cols_f]))) for name, f in (("u", u), ("v", v), ("curvature", curvature))},
        "coarse_finite_fraction": {name: float(np.mean(np.isfinite(f[:, rows_c, :][:, :, cols_c]))) for name, f in coarse.items()},
        "wind_mask_fraction_coarse": float(np.mean(u_c > D.MAX_ZONAL_WIND)),
        "curvature_range": [float(np.nanmin(curvature)), float(np.nanmax(curvature))]}
    arrays = {"times": times, "latgrid": latgrid, "longrid": longrid, "u": u, "v": v, "curvature": curvature,
              "native": native, "nominal": nominal, "lat_c": lat_c, "lon_c": lon_c,
              "rows_f": rows_f, "cols_f": cols_f, "rows_c": rows_c, "cols_c": cols_c, "coarse": coarse}
    return readiness, arrays


def track_case(payload, coarse_threshold, fine_threshold, exclusive, absorb):
    """The port's tracks FROM A RETAINED CASE, the loop phase B replayed against version 1,
    with the thresholds and both tracker flags explicit. Nothing is prepared again."""
    times = np.asarray(payload["time"], float).ravel()
    lat_c, lon_c = np.asarray(payload["lat_c"], float).ravel(), np.asarray(payload["lon_c"], float).ravel()
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    tracks, states = [], []
    for step in range(times.size):
        t = float(times[step])
        waves = D.detect_troughs(t, latgrid_c, longrid_c, payload["u_c"][step], payload["currv_anom_c"][step],
                                 payload["advcurrv_anom_c"][step], payload["latgrid"], payload["longrid"],
                                 payload["currv_anom"][step], coarse_threshold=coarse_threshold,
                                 fine_threshold=fine_threshold, absorb=absorb)
        um = P._median_over(clim.smooth9(payload["u"][step]))
        vm = P._median_over(clim.smooth9(payload["v"][step]))
        tracks, states = associate_step(tracks, states, waves, step, um, vm, exclusive=exclusive)
        tracks, states = prune_stale_tracks(tracks, states, step, t, waves)
    return finalize_tracks(tracks, total_steps=times.size)


def tracking_stage(manifest, manifest_sha256, dataset_name, dataset, year, arrays, climo_cache,
                   calibration, run_dir, inputs_sha256, readiness):
    from scipy.io import savemat
    y0, y1 = (int(x) for x in manifest["years"])
    years = list(range(y0, y1 + 1))
    if not y0 <= year <= y1:
        raise SystemExit(f"REFUSED: the year {year} is outside the protocol years {y0} to {y1}")
    # EVERY CLIMATOLOGY YEAR IS VALIDATED AND DIGESTED, not only the target year, and each
    # phase is timed so the record says where a tracked year's time goes
    phases, t0 = {}, time.perf_counter()
    preflight_all, climatology_inputs = validate_inputs(manifest, dataset, years)
    phases["validate_climatology_inputs_seconds"] = round(time.perf_counter() - t0, 1)
    ct, ft, calibration_sha256, calibration_inputs = check_calibration(calibration, manifest, dataset)
    # THE CALIBRATION'S INPUTS ARE THE CLIMATOLOGY'S INPUTS, digest for digest, strictly:
    # a missing or null digest is a mismatch, not a pass
    mismatched = sorted(k for k in climatology_inputs if calibration_inputs.get(k) != climatology_inputs[k])
    if mismatched:
        raise SystemExit(f"REFUSED: the calibration was computed on different or unrecorded bytes for {mismatched[:4]}")
    cache = os.fspath(climo_cache)
    cache = cache if cache.endswith(".npz") else cache + ".npz"
    t0 = time.perf_counter()
    climo = load_or_build(years, dataset["directory"], dataset["prefix"], cache, verbose=False)
    phases["climatology_load_seconds"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    a = arrays
    anomaly = P.anomaly_from_climatology(a["curvature"], a["times"], climo)
    advection = P._advection(a["latgrid"], a["longrid"], a["u"], a["v"], anomaly)
    coarse_anom = clim.gaussian_decimate(anomaly, a["native"], a["nominal"])
    coarse_adv = clim.gaussian_decimate(advection, a["native"], a["nominal"])
    rows_f, cols_f, rows_c, cols_c = a["rows_f"], a["cols_f"], a["rows_c"], a["cols_c"]

    def cut_f(field):
        return field[:, rows_f, :][:, :, cols_f]

    def cut_c(field):
        return field[:, rows_c, :][:, :, cols_c]
    fine_lat, fine_lon = a["latgrid"][np.ix_(rows_f, cols_f)], a["longrid"][np.ix_(rows_f, cols_f)]
    coarse_lat, coarse_lon = a["lat_c"][np.ix_(rows_c, cols_c)], a["lon_c"][np.ix_(rows_c, cols_c)]
    flags = manifest["tracker_flags"]
    payload = {"lat_c": coarse_lat[:, 0], "lon_c": coarse_lon[0, :], "latgrid": fine_lat, "longrid": fine_lon,
               "time": np.asarray(a["times"], float),
               "u_c": cut_c(a["coarse"]["u"]), "v_c": cut_c(a["coarse"]["v"]),
               "currv_anom_c": cut_c(coarse_anom), "advcurrv_anom_c": cut_c(coarse_adv),
               "u": cut_f(a["u"]), "v": cut_f(a["v"]), "currv_anom": cut_f(anomaly),
               "rean": dataset["label"], "level": float(manifest["level_hpa"])}
    payload["case_id"] = case_id(payload)
    extra = {"climatology_fingerprint": climatology_fingerprint(years, dataset["directory"], dataset["prefix"]),
             "climo_cache_sha256": _sha256(cache), "climatology_inputs_sha256": climatology_inputs,
             "calibration_path": os.path.abspath(calibration), "calibration_sha256": calibration_sha256,
             "coarse_threshold": ct, "fine_threshold": ft, "tracker_flags": flags,
             "case_id": payload["case_id"], "readiness": readiness, "preflight_all_years": preflight_all}
    phases["anomaly_advection_and_case_seconds"] = round(time.perf_counter() - t0, 1)
    extra["phase_seconds"] = phases
    record = producer_record(manifest, manifest_sha256, dataset_name, dataset, year, "tracking", inputs_sha256, extra)
    # THE CASE CARRIES ITS SETTINGS, and every final file is published EXCLUSIVELY: written
    # to a temporary path and linked into place, which the filesystem refuses if the name
    # exists, so a check-then-write race cannot overwrite a retained file
    case_path, port_path = os.path.join(run_dir, "tracker_case.mat"), os.path.join(run_dir, "tracker_port.mat")
    publish_mat(case_path, {**payload, "producer_json": json.dumps(record, sort_keys=True)})
    t0 = time.perf_counter()
    tracks = track_case(payload, ct, ft, exclusive=bool(flags["exclusive"]), absorb=bool(flags["absorb"]))
    phases["tracking_seconds"] = round(time.perf_counter() - t0, 1)
    record["dataset_specific"]["case_sha256"] = _sha256(case_path)
    port = {"n": float(len(tracks)), "case_id": payload["case_id"], "exclusive": float(bool(flags["exclusive"])),
            "absorb": float(bool(flags["absorb"])), "producer_json": json.dumps(record, sort_keys=True)}
    for i, t in enumerate(tracks):
        port[f"lat{i}"] = np.asarray(t["meanlat"], float)
        port[f"lon{i}"] = np.asarray(t["meanlon"], float)
        port[f"time{i}"] = np.asarray(t["time"], float)
    publish_mat(port_path, port)
    record["dataset_specific"]["tracks_sha256"] = _sha256(port_path)
    return record, len(tracks)


def publish_mat(path, payload):
    """Write a .mat file to a private temporary name and link it into place; the link is
    refused atomically when the final name exists."""
    import tempfile
    from scipy.io import savemat
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".publish-", suffix=".mat")
    os.close(fd)
    try:
        savemat(tmp, payload, do_compression=True)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {path} exists and a retained file is never overwritten")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def publish_record(path, record):
    """The completion record, published exclusively and last."""
    import exact_tracks as X
    try:
        X.publish_json(path, record, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {path} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--stage", choices=("inputs", "tracking"), required=True)
    ap.add_argument("--calibration", default=None, help="thresholds artifact, required for tracking")
    ap.add_argument("--climo-cache", default=None, help="climatology cache path, required for tracking")
    ap.add_argument("--out-dir", required=True, help="the run lands in <out-dir>/<dataset>_<year>/")
    args = ap.parse_args(argv)
    manifest, manifest_sha256 = load_manifest(args.manifest)
    dataset = dataset_entry(manifest, args.dataset)
    run_dir = os.path.join(args.out_dir, f"{args.dataset}_{args.year}")
    # THE RUN DIRECTORY IS RESERVED BEFORE ANY WORK, and a second run of the same dataset
    # and year cannot enter it; the inputs stage shares the directory the tracking stage
    # will use, so the reservation is per stage file rather than per directory
    os.makedirs(run_dir, exist_ok=True)
    stage_marker = os.path.join(run_dir, f"{args.stage}_{args.dataset}_{args.year}.json")
    if os.path.exists(stage_marker):
        raise SystemExit(f"REFUSED: {stage_marker} exists, this stage already ran here")
    started = time.perf_counter()
    preflight, inputs_sha256 = validate_inputs(manifest, dataset, [args.year])
    readiness, arrays = inputs_stage(manifest, dataset, args.year, preflight)
    if args.stage == "inputs":
        record = producer_record(manifest, manifest_sha256, args.dataset, dataset, args.year, "inputs",
                                 inputs_sha256, {"readiness": readiness})
        record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        out = os.path.join(run_dir, f"inputs_{args.dataset}_{args.year}.json")
        publish_record(out, record)
        print(f"inputs stage {args.dataset} {args.year}: {readiness['timesteps']} steps, fine "
              f"{readiness['fine_shape']}, coarse {readiness['coarse_shape']} at {readiness['achieved_coarse_spacing_deg']} deg, "
              f"wind mask {readiness['wind_mask_fraction_coarse']:.3f}; wrote {out}")
        return 0
    if not args.calibration or not args.climo_cache:
        raise SystemExit("REFUSED: the tracking stage needs --calibration and --climo-cache")
    record, n = tracking_stage(manifest, manifest_sha256, args.dataset, dataset, args.year, arrays,
                               args.climo_cache, args.calibration, run_dir, inputs_sha256, readiness)
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    out = os.path.join(run_dir, f"tracking_{args.dataset}_{args.year}.json")     # the completion record, published last
    publish_record(out, record)
    print(f"tracking stage {args.dataset} {args.year}: {n} tracks under thresholds "
          f"{record['dataset_specific']['coarse_threshold']:.3e} / {record['dataset_specific']['fine_threshold']:.3e}; "
          f"case and tracks retained in {run_dir}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
