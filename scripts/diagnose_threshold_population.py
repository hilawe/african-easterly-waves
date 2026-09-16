#!/usr/bin/env python3
"""Which POPULATION the archived percentiles were drawn from, as a recorded experiment.

WHY THIS EXISTS AS A SCRIPT. The first version of this comparison was a throwaway file in a
scratch directory, and its numbers went into a findings document as prose with no command,
no artifact and no input fingerprint. That is the same provenance failure the project had
just criticised in a figure it inherited from a review, and a reviewer pointed out that
being right about the numbers does not repair it. This produces a machine-readable artifact
with input hashes so the result can be regenerated and checked.

THE QUESTION. `src_readme.docx` inside the archive says the thresholds came from "the full
period of each reanalysis's curvature vorticity DATA for the tracking domain". The data
paper says "curvature vorticity ANOMALIES". The tracker gates the ANOMALY. Those are
different distributions, and thresholds drawn from raw curvature but applied to anomalies
would sit too high against what they gate, which is the direction of the disagreement
between the port's recomputed pair and version 1's published one.

HOW IT IS TESTED. `sample_anomaly_values` subtracts a climatology, so handing it a ZERO
climatology makes its "anomaly" the raw curvature. Everything else, the smoothing, the
decimation, the hemispheric sign flip, the sampling and the seed, is then held identical
between the two populations, and only the subtraction differs. That is what makes the
comparison attributable.

WHAT A RESULT HERE CAN AND CANNOT SETTLE. It bears on the tracker-equivalent recipe only.
Ruling out raw curvature carried through THIS chain does not rule out every reading of
"curvature vorticity data", and a masked or positive-only population would need its own
run. State any conclusion at that width.

WHERE THE ARTIFACT GOES. Under `docs/`, NOT under `data/`, because `/data/` is gitignored
and an artifact that cannot be committed is not provenance. Every existing threshold
artifact in this project sits in the ignored tree, which is worth fixing separately.

    .venv/bin/python scripts/diagnose_threshold_population.py --years 1981 1982 1983 \
        --out docs/aewc_v2/artifacts/threshold_population_diagnostic.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import load as L  # noqa: E402

# version 1's published ERA-Interim 700 hPa pair, from find_ews_f.m's hardcoded table
V1_COARSE, V1_FINE = 7.16e-7, 2.80e-6
COARSE_PERCENTILE, FINE_PERCENTILE = 55.0, 66.0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# EVERY MODULE THE NUMBERS PASS THROUGH, not a convenient subset. An earlier version of
# this list omitted `vorticity.py`, which supplies `component_vorticity` and so produces the
# curvature the whole diagnostic is about, reached through `pipeline.curvature_from_winds`.
# A fingerprint that misses a numerical dependency is worse than none, because it certifies
# a provenance it did not check.
_SOURCE_FILES = (
    "src/aew/v1port/load.py",
    "src/aew/v1port/climatology.py",
    "src/aew/v1port/pipeline.py",
    "src/aew/v1port/vorticity.py",
    "src/aew/v1port/geometry.py",
    "scripts/diagnose_threshold_population.py",
)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source_hashes():
    """Full SHA-256 per file, rather than one combined prefix.

    PER FILE, because a single combined digest says only that SOMETHING changed. When it
    differs from a committed artifact's, the useful question is which module moved, and a
    combined hash cannot answer it. FULL rather than truncated, because this is meant to
    stand behind a published number.
    """
    root = _repo_root()
    out = {}
    for rel in _SOURCE_FILES:
        with open(os.path.join(root, rel), "rb") as fh:
            out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _environment():
    """The library versions the arithmetic actually ran on."""
    import platform

    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("scipy", "netCDF4"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:                                   # noqa: BLE001
            versions[name] = "unavailable"
    return versions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--directory", default="data/eraint/v1port_buffered")
    ap.add_argument("--prefix", default="eraint")
    ap.add_argument("--years", type=int, nargs="+", default=[1981, 1982, 1983])
    ap.add_argument("--lat-range", type=float, nargs=2, default=[-35.0, 35.0])
    ap.add_argument("--lon-range", type=float, nargs=2, default=[-140.0, 40.0])
    ap.add_argument("--coarse-resolution", type=float, default=2.5)
    ap.add_argument("--per-step", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the result as JSON")
    args = ap.parse_args(argv)

    years = list(args.years)
    print(f"threshold population diagnostic, {args.prefix}, {len(years)} years "
          f"({min(years)}-{max(years)})", flush=True)
    print(f"  region lat {args.lat_range}, lon {args.lon_range}, seed {args.seed}",
          flush=True)

    climatology = L.build_climatology(
        years, args.directory, args.prefix,
        progress=lambda y, n, t: print(f"     climatology {y} ({n}/{t})", flush=True))
    # THE ONLY DIFFERENCE BETWEEN THE TWO RUNS. A zero mean makes the subtraction a no-op,
    # so the "anomaly" the sampler returns is the raw curvature and every other step of the
    # chain is bit-identical between them.
    zero = {k: (np.zeros_like(v) if k == "mean" else v) for k, v in climatology.items()}

    results = {}
    for label, clim in (("anomaly", climatology), ("raw_curvature", zero)):
        coarse_parts, fine_parts = [], []
        rng = np.random.default_rng(args.seed)
        for year in years:
            times, latgrid, longrid, curvature = L.curvature_for_year(
                year, args.directory, args.prefix)
            native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
            c, f = L.sample_anomaly_values(
                curvature, times, clim, latgrid[:, 0], native_resolution=native,
                coarse_resolution=args.coarse_resolution, per_step=args.per_step, rng=rng,
                lon_values=longrid[0, :], lat_range=args.lat_range,
                lon_range=args.lon_range)
            coarse_parts.append(c)
            fine_parts.append(f)
            del curvature
        coarse = float(np.percentile(np.concatenate(coarse_parts), COARSE_PERCENTILE))
        fine = float(np.percentile(np.concatenate(fine_parts), FINE_PERCENTILE))
        results[label] = {
            "coarse": coarse, "fine": fine,
            "coarse_relative_difference": coarse / V1_COARSE - 1.0,
            "fine_relative_difference": fine / V1_FINE - 1.0,
            "coarse_samples": int(sum(p.size for p in coarse_parts)),
            "fine_samples": int(sum(p.size for p in fine_parts))}
        print(f"\n  population {label}:", flush=True)
        print(f"    coarse P{COARSE_PERCENTILE:.0f} {coarse:.8e} vs {V1_COARSE:.3e}  "
              f"({coarse / V1_COARSE - 1.0:+.1%})", flush=True)
        print(f"    fine   P{FINE_PERCENTILE:.0f} {fine:.8e} vs {V1_FINE:.3e}  "
              f"({fine / V1_FINE - 1.0:+.1%})", flush=True)

    raw = results["raw_curvature"]
    print()
    if raw["coarse"] <= 0 < V1_COARSE:
        print("THE TRACKER-EQUIVALENT RAW-CURVATURE RECIPE IS INCOMPATIBLE with version 1's "
              "published pair. Its coarse percentile is NEGATIVE where version 1's is "
              "positive, so no scaling reconciles them. This rules out raw curvature carried "
              "through this chain. It does NOT rule out a masked or positive-only reading of "
              "the readme's \"curvature vorticity data\", which needs its own run.")
    else:
        print("The raw-curvature population does not separate from version 1's pair by sign "
              "on this sample, so the comparison is a matter of degree and the relative "
              "differences above are the whole result.")

    files = {}
    for year in years:
        for var in ("u700", "v700"):
            path = os.path.join(args.directory, f"{args.prefix}_{var}_{year}_6h_region.nc")
            if os.path.exists(path):
                files[os.path.basename(path)] = _sha256(path)
    artifact = {
        "generated_by": "scripts/diagnose_threshold_population.py",
        "source_sha256": _source_hashes(),
        "environment": _environment(),
        "input_file_sha256": files,
        "years": years, "prefix": args.prefix,
        "sample_region": {"lat_range": args.lat_range, "lon_range": args.lon_range},
        "seed": args.seed, "per_step": args.per_step,
        "coarse_resolution": args.coarse_resolution,
        "percentiles": {"coarse": COARSE_PERCENTILE, "fine": FINE_PERCENTILE},
        "expected": {"coarse": V1_COARSE, "fine": V1_FINE},
        "populations": results,
        "what_this_can_settle": (
            "The tracker-equivalent recipe only. Raw curvature here means raw curvature "
            "carried through the port's own smoothing, decimation, sign transformation and "
            "sampling. A masked or positive-only population is untested."),
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(artifact, fh, indent=2, sort_keys=True)
        print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
