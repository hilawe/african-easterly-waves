#!/usr/bin/env python3
"""Write the twelve-cell threshold matrix report from the committed artifacts.

ONE NUMBER, ONE SOURCE. Every figure in the report is read from a committed
thresholds_P*-T*.json artifact under docs/aewc_v2/artifacts and formatted here. Nothing
is recomputed and nothing is typed in. The report exists so the twelve cells are
REPORTED TOGETHER, which the contract requires, because reporting only the near ones
is selection by another route.

WHAT IT REFUSES, because a partial or mixed report reads as a complete one:
  - a missing cell, unless --allow-missing is given, in which case the row says MISSING
    and the report's title says the matrix is incomplete;
  - an artifact whose case_id disagrees with its filename;
  - artifacts produced by different source versions (their source_sha256 sets differ),
    since a table mixing code versions is not one experiment;
  - an artifact without the `expected` block, since the report is about the criterion;
  - an artifact whose recorded source hashes differ from the files now in the tree, so
    the table cannot describe code that no longer exists (the suite's provenance gate
    checks the same thing, and the generator must not rely on it having run);
  - an artifact whose recorded settings differ from what its case id claims in the
    frozen matrix registry, checked with the same validate_case_settings the CLI uses;
  - an artifact whose transformation object (passes and coarse scale, not only the id)
    differs from the frozen table, whose coarse threshold is not the recorded unscaled
    value scaled in the declared direction, or whose expected block (both relative
    differences, the criterion and the verdict) is not what the CLI's own judgement
    recomputes from the recorded thresholds. A review forged P1-T0's thresholds to zero
    and its metadata to 99 passes while keeping the hashes and id, and the first version
    tabled it. What this still cannot catch is a forgery that rewrites every field
    consistently; only a rerun from the inputs establishes the numbers themselves.

    .venv/bin/python scripts/report_threshold_matrix.py --out <report path>

The project writes the report beside the contract it serves, under the same docs
directory as the artifacts.
"""
import argparse
import importlib.util
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from aew.v1port import thresholds as T  # noqa: E402


def _cli_module():
    """scripts/compute_thresholds.py, for its SOURCE_FILES and source_hashes(), so the
    generator checks the same fingerprint the producer wrote."""
    spec = importlib.util.spec_from_file_location(
        "compute_thresholds_for_report",
        os.path.join(_ROOT, "scripts", "compute_thresholds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def artifact_settings(a):
    """The settings an artifact records, in the shape validate_case_settings compares."""
    return {"climatology_years": tuple(a.get("climatology_years", ())),
            "population_years": tuple(a.get("population_years", ())),
            "transformation": (a.get("transformation") or {}).get("id"),
            "estimator": a.get("estimator"),
            "lat_range": tuple((a.get("domain") or {}).get("lat_range", ())),
            "lon_range": tuple((a.get("domain") or {}).get("lon_range", ())),
            "coarse_resolution": a.get("coarse_resolution"),
            "subsample": a.get("subsample_stride"),
            "prefix": a.get("prefix"),
            "expect": ((a.get("expected") or {}).get("coarse"),
                       (a.get("expected") or {}).get("fine")),
            "out": "recorded"}

PERIODS = ("P1", "P2")
TRANSFORMATIONS = ("T0", "T1", "T2", "T3", "T4", "T5")
PERIOD_TEXT = {"P1": "clim 1981-2010, pop 1979-2010", "P2": "clim and pop 1980-2010"}
TRANSFORMATION_TEXT = {
    "T0": "code as archived", "T1": "two smoothing passes", "T2": "coarse scaled up",
    "T3": "coarse scaled down", "T4": "two passes, coarse up",
    "T5": "two passes, coarse down"}


def internal_consistency_problems(cid, a, cli):
    """Every way an artifact's displayed fields disagree with its recorded thresholds.

    The transformation object must be the frozen table's entry for the id, the coarse
    threshold must be the recorded unscaled value scaled in the declared direction, and
    the expected block must be what judge_against_expected() recomputes from the
    thresholds. Exact equality throughout, since every value is derived from numbers the
    same artifact records.
    """
    out = []
    tid = cid.split("-")[1]
    passes, scale = T.TRANSFORMATIONS[tid]
    want_tr = {"id": tid, "smoothing_passes": passes, "coarse_scale": scale}
    if a.get("transformation") != want_tr:
        out.append(f"transformation records {a.get('transformation')!r}, "
                   f"the frozen table says {want_tr!r}")
    coarse = a.get("coarse") or {}
    fine = a.get("fine") or {}
    if "threshold_unscaled" not in coarse or "threshold" not in coarse \
            or "threshold" not in fine:
        out.append("coarse threshold, coarse threshold_unscaled and fine threshold "
                   "must all be recorded")
        return out
    if coarse["threshold"] != T.apply_coarse_scale(coarse["threshold_unscaled"], scale):
        out.append("coarse threshold is not the recorded unscaled value scaled "
                   f"{scale!r}")
    exp = a.get("expected") or {}
    if "coarse" in exp and "fine" in exp:
        want = cli.judge_against_expected(coarse["threshold"], fine["threshold"],
                                          exp["coarse"], exp["fine"])
        for key in ("coarse_relative_difference", "fine_relative_difference",
                    "criterion", "reproduces"):
            if exp.get(key) != want[key]:
                out.append(f"expected.{key} is {exp.get(key)!r}, recomputed from the "
                           f"thresholds it is {want[key]!r}")
    return out


def cell_ids():
    return [f"{p}-{t}" for t in TRANSFORMATIONS for p in PERIODS]


def load_matrix(artifact_dir, allow_missing=False):
    """The twelve artifacts keyed by id, or a refusal string in `problems`."""
    cells, problems = {}, []
    cli = _cli_module()
    current_hashes = cli.source_hashes()
    for cid in cell_ids():
        path = os.path.join(artifact_dir, f"thresholds_{cid}.json")
        if not os.path.exists(path):
            if allow_missing:
                cells[cid] = None
                continue
            problems.append(f"{cid}: no artifact at {path}")
            continue
        with open(path) as fh:
            a = json.load(fh)
        if a.get("case_id") != cid:
            problems.append(f"{cid}: artifact says case_id {a.get('case_id')!r}")
        if not isinstance(a.get("expected"), dict):
            problems.append(f"{cid}: no expected block, the report is about the criterion")
        recorded = a.get("source_sha256") or {}
        for rel, digest in current_hashes.items():
            if recorded.get(rel) != digest:
                problems.append(f"{cid}: {rel} differs from the file now in the tree")
        for rel in set(recorded) - set(current_hashes):
            problems.append(f"{cid}: fingerprints {rel}, which the producer does not")
        for violation in T.validate_case_settings(T.matrix_case(cid), artifact_settings(a)):
            problems.append(f"{cid}: {violation}")
        problems.extend(f"{cid}: {v}" for v in internal_consistency_problems(cid, a, cli))
        cells[cid] = a
    present = [a for a in cells.values() if a is not None]
    if present:
        versions = {json.dumps(a.get("source_sha256"), sort_keys=True) for a in present}
        if len(versions) != 1:
            problems.append("the artifacts were produced by "
                            f"{len(versions)} different source versions, and one table "
                            "may not mix them")
    return cells, problems


def render(cells):
    lines = []
    missing = [cid for cid, a in cells.items() if a is None]
    title = "# The threshold run matrix, all twelve cells"
    if missing:
        title += f" (INCOMPLETE, {len(missing)} missing: {', '.join(missing)})"
    lines.append(title)
    lines.append("")
    lines.append("Every number below is read from the committed artifact named in its "
                 "row. The relative difference is value / expected - 1 against version "
                 "1's ERA-Interim pair (7.16e-07 coarse, 2.80e-06 fine), so a negative "
                 "number is below version 1. The criterion is 10 percent on the "
                 "absolute value of each side separately. Generated by "
                 "scripts/report_threshold_matrix.py. Do not edit by hand.")
    lines.append("")
    lines.append("| cell | periods | transformation | coarse (55th) | coarse vs v1 | "
                 "fine (66th) | fine vs v1 | criterion | steps | artifact |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cid, a in cells.items():
        period, tr = cid.split("-")
        if a is None:
            lines.append(f"| {cid} | {PERIOD_TEXT[period]} | {TRANSFORMATION_TEXT[tr]} "
                         f"| MISSING | | | | | | |")
            continue
        e = a["expected"]
        verdict = "passes" if e["reproduces"] else "FAILS"
        lines.append(
            f"| {cid} | {PERIOD_TEXT[period]} | {TRANSFORMATION_TEXT[tr]} "
            f"| {a['coarse']['threshold']:.6e} | {e['coarse_relative_difference']:+.1%} "
            f"| {a['fine']['threshold']:.6e} | {e['fine_relative_difference']:+.1%} "
            f"| {verdict} | {a['grids']['timesteps']:,} | thresholds_{cid}.json |")
    present = [a for a in cells.values() if a is not None]
    if present:
        lines.append("")
        lines.append(f"All {len(present)} artifacts carry the same source fingerprint "
                     f"({len(present[0]['source_sha256'])} modules), estimator "
                     f"{sorted({a['estimator'] for a in present})}.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--artifact-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "aewc_v2", "artifacts"))
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cells, problems = load_matrix(args.artifact_dir, args.allow_missing)
    if problems:
        for p in problems:
            print(f"REFUSED: {p}", flush=True)
        return 2
    text = render(cells)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"written to {args.out}", flush=True)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
