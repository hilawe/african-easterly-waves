#!/usr/bin/env python3
"""Compute version 1's two detection thresholds, under the frozen contracts.

THE REWRITE, and what it replaces. The previous program coupled the climatology period
and the percentile-population period under one --years argument, so the documented
experiment, the archive's own 1981-2010 climatology against its 1979-2010 tracking loop,
could not be expressed at all. It sampled with an unpinned generator at one seed, reported
a within-sample half-sample range that two rounds of review watched being misread as
precision, fingerprinted two of the six modules the numbers pass through, and truncated
every hash. Each of those is now a rule in the project's written threshold contract,
and this program is the contracts executed in order, with the mechanics in
`aew.v1port.thresholds` where tests and a mutation catalog bind them.

THE PERIODS ARE TWO REQUIRED ARGUMENTS. So are the domain bounds, because the buffered
trees carry a computational halo the tracker never runs over, and a run that sampled it by
default would not be any named case. There are deliberately no defaults for any of these.

THE ESTIMATOR IS EXACT BY DEFAULT: the transformed populations are written to scratch
files once, and the memory-bounded percentile walks them in monthly blocks, bit-equal to
NumPy over the whole population, with the finite count recorded and refused if it exceeds
the domain arithmetic. --estimator ladder is the fallback for grids where exact is not
feasible, running five PCG64-pinned seeds per level up a predeclared per-step ladder, the median
of five as the value, a strict sub-1-percent between-seed gate on both thresholds, and
EXHAUSTION IS AN INVALID RUN (exit 2), never a result.

THE ARTIFACT IS THE FULL SCHEMA: an untruncated SHA-256 per source module the numbers
pass through, full input-file hashes, the library versions, both periods separately, the
transformation, the estimator's complete record, and no sampling diagnostics of any kind
in exact mode, their absence being the record that nothing was drawn.

    .venv/bin/python scripts/compute_thresholds.py \\
        --directory data/eraint/v1port_buffered --prefix eraint \\
        --climatology-years 1981 2010 --population-years 1979 2010 \\
        --lat-range -35 35 --lon-range -140 40 \\
        --transformation T0 --estimator exact --case-id P1-T0 \\
        --expect 7.16e-7 2.80e-6 --out docs/aewc_v2/artifacts/thresholds_P1-T0.json
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import load as L  # noqa: E402
from aew.v1port import thresholds as T  # noqa: E402

# EVERY MODULE THE NUMBERS PASS THROUGH, full hashes, no truncation. Omitting a numerical
# dependency makes the fingerprint certify a provenance it did not check, which happened
# once with vorticity.py, and truncation was called out as short of publication grade.
SOURCE_FILES = (
    "src/aew/v1port/load.py",
    "src/aew/v1port/climatology.py",
    "src/aew/v1port/pipeline.py",
    "src/aew/v1port/vorticity.py",
    "src/aew/v1port/geometry.py",
    "src/aew/v1port/percentile.py",
    "src/aew/v1port/thresholds.py",
    "scripts/compute_thresholds.py",
)

CRITERION = 0.10                                 # the project's reproduction criterion


def judge_against_expected(coarse_value, fine_value, want_coarse, want_fine):
    """The artifact's `expected` block: SIGNED relative differences and the verdict.

    value / expected - 1, the same definition diagnose_threshold_population.py writes
    under the same key, so a negative number means the run landed BELOW version 1. The
    first version stored the absolute difference and printed it with a forced plus
    sign, so a run 37 percent low read as "+37%". The criterion is applied to the
    absolute value of EACH side separately, so one side within and the other far low
    still fails. A pure function so every sign pattern can be tested by hand, which the
    real fixtures cannot do: their coarse threshold is negative and is refused as an
    expected value before this is reached.
    """
    rc = coarse_value / want_coarse - 1.0
    rf = fine_value / want_fine - 1.0
    worst = max(abs(rc), abs(rf))
    return {"coarse": want_coarse, "fine": want_fine,
            "coarse_relative_difference": rc, "fine_relative_difference": rf,
            "sign_convention": "value / expected - 1; negative means the run "
                               "is below the expected value",
            "criterion": CRITERION,
            "reproduces": bool(worst <= CRITERION)}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    root = _repo_root()
    return {rel: _sha256_file(os.path.join(root, rel)) for rel in SOURCE_FILES}


def environment():
    versions = {"platform": platform.platform(),
                "python": platform.python_version(), "numpy": np.__version__}
    for name in ("scipy", "netCDF4"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:                                   # noqa: BLE001
            versions[name] = "unavailable"
    return versions


def input_hashes(directory, prefix, years):
    out = {}
    for year in sorted(set(years)):
        for var in ("u700", "v700"):
            name = f"{prefix}_{var}_{year}_6h_region.nc"
            path = os.path.join(directory, name)
            out[name] = _sha256_file(path) if os.path.exists(path) else "absent"
    return out


def year_span(pair, label):
    lo, hi = int(pair[0]), int(pair[1])
    if hi < lo:
        raise SystemExit(f"{label} {lo}..{hi} runs backwards")
    return list(range(lo, hi + 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--directory", default="data/eraint/v1port_buffered")
    ap.add_argument("--prefix", default="eraint")
    ap.add_argument("--climatology-years", nargs=2, type=int, required=True,
                    metavar=("START", "END"),
                    help="the years the anomaly climatology is built over. The archive's "
                         "own driver uses 1981 2010 here while looping 1979 2010, which "
                         "is why this and --population-years are separate")
    ap.add_argument("--population-years", nargs=2, type=int, required=True,
                    metavar=("START", "END"),
                    help="the years the percentile population is drawn over")
    ap.add_argument("--lat-range", nargs=2, type=float, required=True,
                    metavar=("LOW", "HIGH"),
                    help="required, with no default, because the buffered trees carry a computational "
                         "halo, and sampling it silently would be no named case")
    ap.add_argument("--lon-range", nargs=2, type=float, required=True,
                    metavar=("LOW", "HIGH"))
    ap.add_argument("--transformation", choices=sorted(T.TRANSFORMATIONS), default="T0",
                    help="T0 code-as-archived; T1 two-pass smoothing; T2/T3 coarse "
                         "scaled up/down; T4/T5 the interactions")
    ap.add_argument("--estimator", choices=("exact", "ladder"), default="exact")
    ap.add_argument("--coarse-resolution", type=float, default=2.5)
    ap.add_argument("--subsample", type=int, default=1,
                    help="stride applied to the WINDS before curvature, putting a half "
                         "degree retrieval on whole degrees when 2")
    ap.add_argument("--expect", nargs=2, type=float, default=None,
                    metavar=("COARSE", "FINE"),
                    help="version 1's own pair, to judge reproduction against the "
                         "10 percent criterion")
    ap.add_argument("--case-id", default=None,
                    help="a matrix cell id (P1-T0..P2-T5) is an ENFORCED CLAIM, checked "
                         "against the frozen registry before anything runs, with the "
                         "canonical artifact path required; diagnostic labels must "
                         "carry a diag-, smoke- or test- prefix; anything else in the "
                         "matrix-looking namespace is refused")
    ap.add_argument("--scratch", default=None,
                    help="directory for the population files (about 5.3 GiB for thirty "
                         "one-degree years); a temporary directory by default")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--out", default=None, help="write the artifact as JSON")
    ap.add_argument("--allow-partial-years", action="store_true",
                    help="accept years that do not cover their whole calendar; refused "
                         "for matrix cells, whose periods mean complete years, and the "
                         "actual step counts are recorded either way")
    args = ap.parse_args(argv)

    climatology_years = year_span(args.climatology_years, "--climatology-years")
    population_years = year_span(args.population_years, "--population-years")
    passes, coarse_scale = T.TRANSFORMATIONS[args.transformation]

    # CHEAP GUARDS BEFORE EXPENSIVE WORK. Reversed bounds would build an empty domain and
    # fail obscurely an hour in, and a non-positive expected threshold would repeat, in
    # the denominator of the comparison, the signed-denominator defect already corrected
    # in the fallback.
    if args.lat_range[0] >= args.lat_range[1]:
        raise SystemExit(f"--lat-range {list(args.lat_range)} is empty or reversed")
    if args.lon_range[0] >= args.lon_range[1]:
        raise SystemExit(f"--lon-range {list(args.lon_range)} is empty or reversed")
    if args.expect is not None and (args.expect[0] <= 0 or args.expect[1] <= 0):
        raise SystemExit(
            f"--expect {list(args.expect)} holds a non-positive threshold, which cannot "
            f"anchor a relative difference")

    # A MATRIX IDENTIFIER IS A CLAIM, CHECKED BEFORE ANYTHING RUNS. A review demonstrated
    # a ladder T5 diagnostic labelled P1-T0 writing a successful artifact, so a case id
    # matching the matrix pattern now binds every setting to the frozen cell, and a
    # mislabelled run is refused with the differences listed rather than recorded.
    kind = T.classify_case_id(args.case_id)
    if kind == "reserved":
        print(f"REFUSED: case id {args.case_id!r} sits in the matrix-looking namespace "
              f"without being a cell. Valid cells are P1-T0..P1-T5 and P2-T0..P2-T5; "
              f"diagnostic labels must start with one of "
              f"{', '.join(T.FREE_CASE_PREFIXES)}. A typo in an intended matrix id must "
              f"not run unprotected as a free label.", flush=True)
        return 2
    case = T.matrix_case(args.case_id)
    if case is not None:
        supplied = {
            "climatology_years": (climatology_years[0], climatology_years[-1]),
            "population_years": (population_years[0], population_years[-1]),
            "transformation": args.transformation, "estimator": args.estimator,
            "lat_range": tuple(args.lat_range), "lon_range": tuple(args.lon_range),
            "coarse_resolution": args.coarse_resolution, "subsample": args.subsample,
            "prefix": args.prefix,
            "expect": tuple(args.expect) if args.expect else None,
            "out": args.out,
        }
        violations = T.validate_case_settings(case, supplied)
        if args.allow_partial_years:
            violations.append(
                "allow-partial-years: a matrix cell's period means complete years")
        # THE ARTIFACT PATH IS PART OF THE CASE. The contract puts gating artifacts
        # under docs/aewc_v2/artifacts, and an official-looking P1-T0 written to /tmp or
        # the gitignored data tree is a provenance hole with a correct label on it.
        canonical = os.path.join(_repo_root(), "docs", "aewc_v2", "artifacts",
                                 f"thresholds_{args.case_id}.json")
        if args.out and os.path.abspath(args.out) != canonical:
            violations.append(
                f"out: a matrix cell's artifact belongs at {canonical}, "
                f"not {os.path.abspath(args.out)}")
        if violations:
            print(f"REFUSED: this run is not the {args.case_id} it claims to be:",
                  flush=True)
            for v in violations:
                print(f"  - {v}", flush=True)
            return 2

    # PROVENANCE IS CAPTURED BEFORE THE COMPUTATION and compared again after it, because
    # hashes taken only at the end can describe a tree the numbers did not come from,
    # which matters most during active development and retrieval.
    provenance_before = (source_hashes(),
                         input_hashes(args.directory, args.prefix,
                                      climatology_years + population_years))

    print("  preflighting the input years", flush=True)
    expected_lat = expected_lon = None
    if case is not None:
        expected_lat, expected_lon = T.expected_buffered_grid()
    # 700 hPa IS THE PROGRAM'S PHYSICS, not an option: every threshold this computes is
    # a 700 hPa threshold, so the level evidence is required unconditionally.
    preflight = T.preflight_years(
        set(climatology_years + population_years), args.directory, args.prefix,
        require_full_calendar=not args.allow_partial_years,
        expected_lat=expected_lat, expected_lon=expected_lon,
        expected_level_hpa=700.0)

    print(f"thresholds for {args.prefix}: climatology {climatology_years[0]}-"
          f"{climatology_years[-1]}, population {population_years[0]}-"
          f"{population_years[-1]}, {args.transformation} "
          f"({passes} smoothing pass{'es' if passes > 1 else ''}, coarse scale "
          f"{coarse_scale or 'none'}), estimator {args.estimator}", flush=True)
    print(f"  domain lat {list(args.lat_range)}, lon {list(args.lon_range)}", flush=True)

    started = time.time()
    print("  building the climatology", flush=True)
    climatology = L.build_climatology(
        climatology_years, args.directory, args.prefix,
        progress=lambda y, n, t: print(f"     {y} ({n}/{t})", flush=True),
        subsample=args.subsample)

    scratch = args.scratch or tempfile.mkdtemp(prefix="aew_thresholds_")
    os.makedirs(scratch, exist_ok=True)
    try:
        print(f"  building the transformed populations under {scratch}", flush=True)
        files = T.build_population_files(
            population_years, args.directory, args.prefix, climatology, scratch,
            passes=passes, lat_range=tuple(args.lat_range),
            lon_range=tuple(args.lon_range),
            coarse_resolution=args.coarse_resolution, subsample=args.subsample)
        print(f"     {files.steps} timesteps, fine {files.fine_cells} cells/step, "
              f"coarse {files.coarse_cells} cells/step", flush=True)

        # A MATRIX CELL'S DOMAIN MUST PRODUCE THE FROZEN GRIDS. Cell counts alone
        # cannot catch a grid shifted by half a cell with the same dimensions, so the
        # shapes are part of the claim.
        if case is not None:
            got = (tuple(files.fine_shape), tuple(files.coarse_shape))
            want = (case["fine_shape"], case["coarse_shape"])
            if got != want:
                print(f"REFUSED: {args.case_id} requires fine/coarse grids "
                      f"{want[0]}/{want[1]} and this tree produced {got[0]}/{got[1]}",
                      flush=True)
                return 2
            # and the cropped coordinate VECTORS, derived from the frozen buffered grid
            # by the builder's own crop arithmetic, not just their shapes
            blat, blon = T.expected_buffered_grid()
            lat_lo, lat_hi = case["lat_range"]
            lon_lo, lon_hi = case["lon_range"]
            want_fine_lats = blat[(blat >= lat_lo) & (blat <= lat_hi)]
            want_fine_lons = blon[(blon >= lon_lo) & (blon <= lon_hi)]
            want_coarse_lats = blat[::2][(blat[::2] >= lat_lo) & (blat[::2] <= lat_hi)]
            want_coarse_lons = blon[::2][(blon[::2] >= lon_lo) & (blon[::2] <= lon_hi)]
            if not (np.array_equal(files.fine_lats, want_fine_lats)
                    and np.array_equal(files.fine_lons, want_fine_lons)
                    and np.array_equal(files.coarse_lats, want_coarse_lats)
                    and np.array_equal(files.coarse_lons, want_coarse_lons)):
                print(f"REFUSED: {args.case_id} produced cropped coordinates that "
                      f"differ from the frozen case's, at matching shapes; the grids "
                      f"are not the case's grids", flush=True)
                return 2

        artifact = {
            "schema": "thresholds-v2",
            "generated_by": "scripts/compute_thresholds.py",
            "case_id": args.case_id,
            "directory": os.path.abspath(args.directory),
            "prefix": args.prefix,
            "preflight": preflight,
            "grids": {"fine_shape": list(files.fine_shape),
                      "coarse_shape": list(files.coarse_shape),
                      "timesteps": files.steps,
                      "fine_lat_sha256": T._array_sha256(files.fine_lats),
                      "fine_lon_sha256": T._array_sha256(files.fine_lons),
                      "coarse_lat_sha256": T._array_sha256(files.coarse_lats),
                      "coarse_lon_sha256": T._array_sha256(files.coarse_lons)},
            "source_sha256": provenance_before[0],
            "environment": environment(),
            "input_file_sha256": provenance_before[1],
            "climatology_years": [climatology_years[0], climatology_years[-1]],
            "population_years": [population_years[0], population_years[-1]],
            "domain": {"lat_range": list(args.lat_range),
                       "lon_range": list(args.lon_range)},
            "coarse_resolution": args.coarse_resolution,
            "subsample_stride": args.subsample,
            "transformation": {"id": args.transformation, "smoothing_passes": passes,
                               "coarse_scale": coarse_scale},
            # the population definition, stated in the artifact rather than inferred
            # from the transformation id, so a future masked or seasonal variant is
            # distinguishable from these cells by reading the artifact alone
            "population": {
                "field": "curvature vorticity anomaly against the six-hourly "
                         "calendar climatology",
                "grids": "the cropped input grid (fine) and the decimated then cropped "
                         "grid (coarse)",
                "order": "decimate the buffered anomaly, crop both grids to the domain, "
                         "smooth on the cropped grids, flip the southern-hemisphere sign",
                "smoothing_passes": passes,
                "wind_mask": "none",
                "hemispheres": "both",
                "timesteps": "every six-hourly step of every population year",
                "values": "every finite cell"},
            "percentiles": {"coarse": T.COARSE_Q, "fine": T.FINE_Q},
            "percentile_method": "linear",
            "estimator": args.estimator,
        }

        status = 0
        coarse_value = fine_value = None
        if args.estimator == "exact":
            print("  exact percentiles over the population files", flush=True)
            results = {}
            for which, q in (("coarse", T.COARSE_Q), ("fine", T.FINE_Q)):
                value, count, upper = T.exact_threshold(files, which, q)
                results[which] = {"unscaled": value, "finite_count": count,
                                  "domain_upper_bound": upper}
            coarse_value = T.apply_coarse_scale(results["coarse"]["unscaled"],
                                                coarse_scale)
            fine_value = results["fine"]["unscaled"]
            artifact["coarse"] = {"threshold": coarse_value,
                                  "threshold_unscaled": results["coarse"]["unscaled"],
                                  "finite_count": results["coarse"]["finite_count"],
                                  "domain_upper_bound":
                                      results["coarse"]["domain_upper_bound"]}
            artifact["fine"] = {"threshold": fine_value,
                                "finite_count": results["fine"]["finite_count"],
                                "domain_upper_bound":
                                    results["fine"]["domain_upper_bound"]}
        else:
            print("  the fallback ladder, five pinned seeds per level", flush=True)
            outcome, records = T.run_ladder(files)
            artifact["between_seed"] = {
                "seeds": list(T.SEEDS), "generator": "PCG64",
                "range_formula": "(max - min) / abs(median)",
                "gate": "both < 0.01 strictly",
                "per_step_ladder": records,
            }
            if outcome is None:
                artifact["outcome"] = "exhausted"
                artifact["coarse"] = artifact["fine"] = None
                print("  THE LADDER IS EXHAUSTED at every level through "
                      f"{T.LADDER[-1]}. Per the contract this is an INVALID RUN, not a "
                      "result: use exact mode, or revise the contract in a dated block.",
                      flush=True)
                status = 2
            else:
                artifact["outcome"] = {"per_step": outcome["per_step"]}
                coarse_value = T.apply_coarse_scale(outcome["coarse"]["median"],
                                                    coarse_scale)
                fine_value = outcome["fine"]["median"]
                artifact["coarse"] = {"threshold": coarse_value,
                                      "threshold_unscaled": outcome["coarse"]["median"]}
                artifact["fine"] = {"threshold": fine_value}

        if status == 0:
            print(f"\n  coarse ({T.COARSE_Q:.0f}th percentile, decimated grid): "
                  f"{coarse_value:.6e}", flush=True)
            print(f"  fine   ({T.FINE_Q:.0f}th percentile, input grid):    "
                  f"{fine_value:.6e}", flush=True)
            if args.expect is not None:
                want_coarse, want_fine = args.expect
                judged = judge_against_expected(coarse_value, fine_value,
                                                want_coarse, want_fine)
                artifact["expected"] = judged
                rc, rf = (judged["coarse_relative_difference"],
                          judged["fine_relative_difference"])
                print(f"\n  AGAINST THE EXPECTED PAIR: coarse {rc:+.1%}, fine {rf:+.1%} "
                      f"(criterion {CRITERION:.0%} either way; negative is below "
                      f"version 1)", flush=True)
                if not judged["reproduces"]:
                    print("  THE RECIPE DOES NOT REPRODUCE the expected pair.",
                          flush=True)
                    status = 1

        # THE TREE MUST NOT HAVE MOVED UNDER THE RUN. Recompute both hash sets and
        # refuse to write an artifact whose provenance describes a later state than the
        # one the numbers came from.
        provenance_after = (source_hashes(),
                            input_hashes(args.directory, args.prefix,
                                         climatology_years + population_years))
        if provenance_after != provenance_before:
            moved = sorted(
                {k for k in provenance_before[0]
                 if provenance_after[0].get(k) != provenance_before[0][k]}
                | {k for k in provenance_before[1]
                   if provenance_after[1].get(k) != provenance_before[1][k]})
            print(f"REFUSED: files changed during the run ({', '.join(moved)}); the "
                  f"numbers came from the earlier state and no artifact is written",
                  flush=True)
            return 2

        if args.out:
            # ATOMIC: serialize to a sibling temp file and replace, so a failure during
            # writing cannot truncate an existing artifact into an empty official file
            out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
            os.makedirs(out_dir, exist_ok=True)
            tmp_path = os.path.join(out_dir, os.path.basename(args.out) + ".tmp")
            try:
                with open(tmp_path, "w") as fh:
                    json.dump(artifact, fh, indent=2, sort_keys=True)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, args.out)
            except BaseException:
                # a failed serialization removes its debris; the existing artifact is
                # untouched either way, which is the property the test binds
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            print(f"\n  written to {args.out}", flush=True)
        print(f"  total {time.time() - started:.0f}s", flush=True)
        return status
    finally:
        if not args.keep_scratch and args.scratch is None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
