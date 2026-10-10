#!/usr/bin/env python3
"""The prepared detection inputs of two runs, cell for cell: the eastern pilot brief's
section 7 item 1, second stage, between the input control (B) and the retained baseline (A).

Each run is a directory holding one tracking record and the record names its retained
case file by digest. Each case file is read into memory ONCE, those bytes are hashed and
must equal the record's digest, and every variable is parsed from those same bytes, so the
file cannot change between the check and the read. Every numeric array either file holds is
compared, and the two files must hold the same numeric variables, so nothing is left
unexamined. A variable passes only when the shapes match, the missing cells (NaN) are the
same cells, and every other cell is exactly equal. A difference is reported with its count,
its largest absolute size, its first location, the median magnitude of A's values, and the
fewest and most differing cells in any one column of the last axis (longitude for the
fields), never rounded away. The provenance strings the case also carries (reanalysis name,
case identifier, producer record) differ between runs by design and are listed as not
compared.

    python3 scripts/compare_case_fields.py --a <run dir A> --a-case <tracker_case.mat of A> \\
        --b <run dir B> --b-case <tracker_case.mat of B> --year 1990 --out <json>
"""
import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

FIELDS = ("time", "lat_c", "lon_c", "latgrid", "longrid", "u_c", "v_c", "currv_anom_c", "advcurrv_anom_c",
          "u", "v", "currv_anom", "level")      # the variables a protocol case is known to hold; every one is required
NOT_COMPARED = ("rean", "case_id", "producer_json")


def bound_case(run_dir, case_path, year):
    """The run's record and its case file's bytes, read once and refused unless their
    digest is the record's. Every later read parses these bytes, never the path again."""
    import hashlib
    run = P.load_run(run_dir, year=year)
    with open(case_path, "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != run["record"]["dataset_specific"].get("case_sha256"):
        raise SystemExit(f"REFUSED: {case_path} is not the case the record in {run_dir} names")
    return run, digest, blob


def numeric_inventory(blob):
    """The names of every non-character variable the case holds."""
    from scipy.io import whosmat
    return {name for name, _, cls in whosmat(io.BytesIO(blob)) if cls != "char"}


def compare_variable(a, b):
    """Exact comparison of one array pair, with the missing cells compared as cells."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "passed": False, "problem": "shapes differ"}
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    mask_diff = int(np.count_nonzero(nan_a != nan_b))
    both = ~nan_a & ~nan_b
    unequal = both & (a != b)
    n = int(np.count_nonzero(unequal))
    finite_a = a[~nan_a]
    out = {"shape": list(a.shape), "cells": int(a.size), "missing_cells_a": int(nan_a.sum()), "missing_cells_b": int(nan_b.sum()),
           "missing_mask_cells_differing": mask_diff, "value_cells_differing": n,
           "max_abs_difference": float(np.max(np.abs(a[unequal] - b[unequal]))) if n else 0.0,
           "median_abs_value_a": float(np.median(np.abs(finite_a))) if finite_a.size else None,
           "passed": mask_diff == 0 and n == 0}
    if n:
        out["first_differing_index"] = [int(i) for i in np.argwhere(unequal)[0]]
        if a.ndim >= 2:
            per_column = np.count_nonzero(unequal.reshape(-1, a.shape[-1]), axis=0)
            out["differing_cells_per_last_axis_column"] = {"columns": int(a.shape[-1]), "min": int(per_column.min()), "max": int(per_column.max()),
                                                           "columns_with_none": int(np.count_nonzero(per_column == 0))}
    return out


def compare(a_dir, a_case, b_dir, b_case, year):
    from scipy.io import loadmat
    a_run, a_digest, a_blob = bound_case(a_dir, a_case, year)
    b_run, b_digest, b_blob = bound_case(b_dir, b_case, year)
    names_a, names_b = numeric_inventory(a_blob), numeric_inventory(b_blob)
    missing = [f for f in FIELDS if f not in names_a or f not in names_b]
    if missing:
        raise SystemExit(f"REFUSED: the case files do not both hold {missing}, so the comparison would be partial")
    if names_a != names_b:
        raise SystemExit(f"REFUSED: the case files hold different numeric variables ({sorted(names_a ^ names_b)}), so the comparison would be partial")
    results = {}
    for f in list(FIELDS) + sorted(names_a - set(FIELDS)):                  # every numeric variable, the known ones first
        va = loadmat(io.BytesIO(a_blob), variable_names=[f])[f]
        vb = loadmat(io.BytesIO(b_blob), variable_names=[f])[f]
        results[f] = compare_variable(va, vb)
        del va, vb
    extra = []                                                              # nothing numeric is left out: the inventories are equal and all of it is compared
    return {"generated_by": "scripts/compare_case_fields.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": year,
            "a": {"dir": a_dir, "record_sha256": a_run["record_sha256"], "tracks_sha256": a_run["tracks_sha256"], "case": a_case, "case_sha256": a_digest},
            "b": {"dir": b_dir, "record_sha256": b_run["record_sha256"], "tracks_sha256": b_run["tracks_sha256"], "case": b_case, "case_sha256": b_digest},
            "rule": "shapes equal, the same missing cells, every other cell exactly equal",
            "fields": results, "not_compared": list(NOT_COMPARED), "other_variables_not_compared": extra,
            "bytes_read_once": "each case file was read once, hashed, and every variable parsed from those bytes",
            "passed": all(r["passed"] for r in results.values())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--a", "--a-case", "--b", "--b-case", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    out = compare(args.a, args.a_case, args.b, args.b_case, args.year)
    try:
        X.publish_json(args.out, out, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    failed = [f for f, r in out["fields"].items() if not r["passed"]]
    print(f"{args.year}: {'PASSED' if out['passed'] else 'DIFFERENT in ' + ', '.join(failed)} over {len(out['fields'])} fields; wrote {args.out}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
