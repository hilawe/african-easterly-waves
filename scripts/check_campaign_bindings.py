#!/usr/bin/env python3
"""Confirm the year and dataset bindings of a protocol campaign, read-only.

WHY. The campaign's two annual series correlate 0.081. That is not evidence of a defect,
but it makes the associations consequential: a record filed under one year that holds
another year's tracks, or a comparison whose sides are not the runs its row names,
would produce exactly such a number. This check follows the retained chain for every
run and every year and refuses on the first broken link. It reuses the checks the
instruments already perform (the retrieval validator and the comparison gate's producer
check) rather than restating them.

THE CHAIN, per run `<dataset>_<year>`:
  1. the tracking record is filed under its own dataset and year, and passes the
     comparison gate's producer check against the retained manifest;
  2. the collected tracks file has the digest the record names, carries the record's
     case id, and embeds a producer record for the same dataset and year with the same
     raw-input digests;
  3. every observation of every track falls inside that calendar year, and the file
     holds the number of tracks it declares;
  4. each raw input the record names exists in the dataset's directory, has the digest
     the record names, carries the year in its name, and passes the retrieval validator
     for that year on the manifest's grid (the time vector read from the file itself);
  5. the calibration artifact the record names has the digest the record names and the
     thresholds the record carries.
Per year: the paired artifact names the year, its two sides carry the two records' case
ids and the manifest's digest, its input digests are the two collected tracks files'
digests, and the summary row names the artifact's digest and copies its season counts.

UNCHECKED NEVER MEANS PASSED. Every link is recorded as checked with its verdict, the
artifact carries the count of links checked, and the command exits nonzero unless every
link of every run and year passed. Nothing is written except the one artifact.

    .venv/bin/python3 scripts/check_campaign_bindings.py --manifest <json> \\
        --evidence docs/aewc_v2/evidence/validation/protocol_campaign \\
        --artifacts docs/aewc_v2/artifacts/protocol_campaign --summary <json> --out <json>
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

DATASETS = ("eraint", "era5")
SIDES = {"eraint": "v1", "era5": "port"}


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _day(year, month, day):
    return float((dt.date(year, month, day) - dt.date(1900, 1, 1)).days)


def _link(links, name, ok, detail=""):
    links.append({"link": name, "passed": bool(ok), "detail": detail})
    return bool(ok)


def check_run(evidence, dataset, year, manifest, manifest_sha256, raw_validator, calibration_path):
    """Links 1 to 5 for one run. `raw_validator(path, year, var)` returns None when the
    raw file passes the retrieval validator, else a reason; it is injected so the check
    can be tested without a retrieved year on disk, and a run whose raw inputs were not
    validated is reported as such and never as passed."""
    from scipy.io import loadmat
    import season_metrics as S
    links = []
    run = os.path.join(evidence, f"{dataset}_{year}")
    record_path = os.path.join(run, f"tracking_{dataset}_{year}.json")
    if not os.path.exists(record_path):
        _link(links, "record present", False, record_path)
        return links
    record = json.load(open(record_path))
    d = record["dataset_specific"]
    _link(links, "record filed under its dataset and year", d.get("dataset") == dataset and d.get("year") == year,
          f"record says {d.get('dataset')} {d.get('year')}")
    problems = S.producer_problems(record, manifest, manifest_sha256, year, SIDES[dataset])
    _link(links, "record passes the comparison gate's producer check", not problems, "; ".join(problems))
    tracks_path = os.path.join(run, "tracker_port.mat")
    if not os.path.exists(tracks_path):
        _link(links, "tracks file present", False, tracks_path)
        return links
    blob = open(tracks_path, "rb").read()
    _link(links, "tracks digest equals the record's", hashlib.sha256(blob).hexdigest() == d.get("tracks_sha256"))
    try:
        raw = loadmat(io.BytesIO(blob))
    except (OSError, ValueError) as exc:
        _link(links, "tracks file readable", False, str(exc))
        return links
    _link(links, "tracks file carries the record's case id", str(raw["case_id"][0]) == d.get("case_id"))
    try:
        embedded = json.loads(str(raw["producer_json"][0]))["dataset_specific"]
        _link(links, "embedded producer record names the same dataset, year and raw inputs",
              embedded.get("dataset") == dataset and embedded.get("year") == year
              and embedded.get("inputs_sha256") == d.get("inputs_sha256"),
              f"embedded says {embedded.get('dataset')} {embedded.get('year')}")
    except (KeyError, ValueError, IndexError) as exc:
        _link(links, "embedded producer record names the same dataset, year and raw inputs", False, f"unreadable: {exc}")
    n = int(np.asarray(raw["n"]).ravel()[0])
    indices = sorted(int(k[4:]) for k in raw if k.startswith("time") and k[4:].isdigit())
    contiguous = indices == list(range(n))
    _link(links, "tracks file holds the number of tracks it declares, contiguously indexed", contiguous,
          f"declares {n}, holds indices {indices[:5]}{'...' if len(indices) > 5 else ''} ({len(indices)})")
    lo, hi = _day(year, 1, 1), _day(year + 1, 1, 1)
    times = np.concatenate([np.asarray(raw[f"time{i}"], float).ravel() for i in indices]) if indices else np.array([])
    inside = bool(times.size) and bool(np.all((times >= lo) & (times < hi)))
    _link(links, "every observation falls inside the calendar year", inside,
          f"{times.size} observations, first {times.min() if times.size else None}, last {times.max() if times.size else None}, year spans [{lo}, {hi})")
    import export_protocol_case as E
    expected = E.input_paths(manifest["datasets"][dataset], year)      # {var: path}, the entry point's own rule
    inputs = d.get("inputs_sha256") or {}
    _link(links, "record names exactly the year's two canonical input files",
          sorted(inputs) == sorted(os.path.basename(p) for p in expected.values()),
          f"record names {sorted(inputs)}, the entry point expects {sorted(os.path.basename(p) for p in expected.values())}")
    for var, path in sorted(expected.items()):
        name = os.path.basename(path)
        if not os.path.exists(path):
            _link(links, f"raw input {name} present", False, path)
            continue
        _link(links, f"raw input {name} digest equals the record's", _sha256(path) == inputs.get(name))
        reason = raw_validator(path, year, var)
        _link(links, f"raw input {name} passes the retrieval validator for {year} as {var}", reason is None, reason or "")
        links[-1]["raw_validator_called"] = True
    cal = calibration_path(dataset)
    if cal is None or not os.path.exists(cal):
        _link(links, "calibration artifact present", False, str(cal))
    else:
        art = json.load(open(cal))
        _link(links, "calibration digest equals the record's", _sha256(cal) == d.get("calibration_sha256"))
        coarse, fine = (art.get("coarse") or {}).get("threshold"), (art.get("fine") or {}).get("threshold")
        _link(links, "record thresholds equal the calibration artifact's",
              coarse == d.get("coarse_threshold") and fine == d.get("fine_threshold"),
              f"artifact {coarse}, {fine}, record {d.get('coarse_threshold')}, {d.get('fine_threshold')}")
    return links


def check_year(evidence, artifacts, summary_rows, year, manifest_sha256):
    links = []
    path = os.path.join(artifacts, f"paired_{year}_eraint_era5.json")
    if not os.path.exists(path):
        _link(links, "paired artifact present", False, path)
        return links
    blob = open(path, "rb").read()
    art = json.loads(blob.decode())
    _link(links, "artifact names the year", art.get("year") == year, f"artifact says {art.get('year')}")
    for dataset, side in SIDES.items():
        rec_path = os.path.join(evidence, f"{dataset}_{year}", f"tracking_{dataset}_{year}.json")
        case = json.load(open(rec_path))["dataset_specific"]["case_id"] if os.path.exists(rec_path) else None
        s = art.get("sides", {}).get(side, {})
        _link(links, f"side {side} carries the {dataset} record's case id", case is not None and s.get("case_id") == case,
              f"artifact {s.get('case_id')}, record {case}")
        _link(links, f"side {side} names the dataset and the manifest digest",
              s.get("dataset") == dataset and s.get("manifest_sha256") == manifest_sha256)
        tracks = os.path.join(evidence, f"{dataset}_{year}", "tracker_port.mat")
        named = (art.get("inputs_sha256") or {}).get(side, {})
        _link(links, f"side {side} input digest is the collected tracks file's",
              os.path.exists(tracks) and named.get("sha256") == _sha256(tracks))
    row = summary_rows.get(str(year))
    if row is None:
        _link(links, "summary row present", False)
        return links
    _link(links, "summary row names the artifact's digest", row.get("artifact_sha256") == hashlib.sha256(blob).hexdigest())
    season = art["comparison"]["season"]
    _link(links, "summary row copies the artifact's season counts",
          row.get("africa_origin", {}).get("v1") == season.get("v1") and row.get("africa_origin", {}).get("port") == season.get("port"))
    return links


def run_check(years, manifest, manifest_sha256, evidence, artifacts, summary_rows, raw_validator, calibration_path):
    """Every run and year of `years`, with the accounting a verdict rests on: the number
    of raw-validator calls performed against the number a bound campaign needs (two per
    run), so an unperformed validation can never read as passed, and a refusal of an
    empty range, since a check over nothing is not a check. Returns the artifact body."""
    years = list(years)
    if not years:
        raise SystemExit("REFUSED: an empty year range checks nothing and cannot bind anything")
    runs, checks = {}, {}
    for year in years:
        for dataset in DATASETS:
            runs[f"{dataset}_{year}"] = check_run(evidence, dataset, year, manifest, manifest_sha256, raw_validator, calibration_path)
        checks[str(year)] = check_year(evidence, artifacts, summary_rows, year, manifest_sha256)
    all_links = [l for ls in list(runs.values()) + list(checks.values()) for l in ls]
    failed = [(k, l) for k, ls in list(runs.items()) + list(checks.items()) for l in ls if not l["passed"]]
    validator_calls = sum(1 for l in all_links if l.get("raw_validator_called"))
    validations_needed = 2 * len(runs)
    raw_validated = validator_calls == validations_needed
    bound = not failed and raw_validated and bool(all_links)
    return {"years": [int(y) for y in years], "runs_checked": len(runs), "years_checked": len(checks),
            "links_checked": len(all_links), "links_failed": len(failed),
            "raw_validator_calls": validator_calls, "raw_validations_needed": validations_needed,
            "raw_inputs_validated": raw_validated, "verdict": "bound" if bound else "NOT BOUND",
            "runs": runs, "years_detail": checks, "failed": [{"where": k, **l} for k, l in failed]}


def real_raw_validator(manifest):
    """The retrieval validator bound to the manifest's area and grid, called with the
    file, the year the record names and the variable read from the file name."""
    import download_eraint_v1port as D
    import export_protocol_case as E
    area, grid = E.area_and_grid_text(manifest)

    def raw_validator(path, year, var):
        return D.validate_file(path, year, var, area, grid)
    return raw_validator


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", default="1979-2010")
    ap.add_argument("--calibration-dir", default="docs/aewc_v2/artifacts")
    args = ap.parse_args(argv)
    import export_protocol_case as E
    manifest, manifest_sha256 = E.load_manifest(args.manifest)

    raw_validator = real_raw_validator(manifest)

    def calibration_path(dataset):
        return os.path.join(args.calibration_dir, f"thresholds_protocol_{dataset}_1979_2010.json")
    y0, y1 = (int(x) for x in args.years.split("-"))
    if y1 < y0:
        raise SystemExit(f"REFUSED: the year range {args.years} is reversed")
    rows = json.load(open(args.summary))["years"]
    body = run_check(range(y0, y1 + 1), manifest, manifest_sha256, args.evidence, args.artifacts, rows, raw_validator, calibration_path)
    out = {"generated_by": "scripts/check_campaign_bindings.py", "script_sha256": X.digest(__file__),
           "manifest_sha256": manifest_sha256, "summary_sha256": _sha256(args.summary), **body}
    try:
        X.publish_json(args.out, out, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    print(f"{out['verdict']}: {out['runs_checked']} runs and {out['years_checked']} years, {out['links_checked']} links checked, "
          f"{out['links_failed']} failed, {out['raw_validator_calls']} of {out['raw_validations_needed']} raw validations performed, wrote {args.out}")
    for f in out["failed"][:20]:
        print(f"  {f['where']}: {f['link']}: {f['detail']}")
    return 0 if out["verdict"] == "bound" else 1


if __name__ == "__main__":
    sys.exit(main())
