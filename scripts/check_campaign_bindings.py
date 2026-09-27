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


def check_run(evidence, dataset, year, manifest, manifest_sha256, raw_validator, calibration_path, campaign=None):
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
    import campaign_record_ok as R
    identity = R.problems(record_path)               # the driver's own predicate, so the two cannot disagree
    _link(links, "record filed under its dataset and year (the driver's own predicate)", not identity, "; ".join(identity))
    if identity and any("unreadable" in p for p in identity):
        return links
    record = json.load(open(record_path))
    d = record.get("dataset_specific") or {}
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
    complete = contiguous and all(f"lat{i}" in raw and f"lon{i}" in raw
                                  and np.asarray(raw[f"lat{i}"]).size == np.asarray(raw[f"lon{i}"]).size == np.asarray(raw[f"time{i}"]).size
                                  for i in indices)
    _link(links, "tracks file holds the number of tracks it declares, contiguously indexed, each with time, lat and lon of one length",
          complete, f"declares {n}, holds time indices {indices[:5]}{'...' if len(indices) > 5 else ''} ({len(indices)})"
          + ("" if contiguous else ", not 0..n-1") + ("" if complete or not contiguous else ", a lat or lon array is missing or of another length"))
    _link(links, "tracks file holds at least one track", n >= 1 and bool(indices), f"declares {n}")
    lo, hi = _day(year, 1, 1), _day(year + 1, 1, 1)
    times = np.concatenate([np.asarray(raw[f"time{i}"], float).ravel() for i in indices]) if indices else np.array([])
    inside = bool(np.all((times >= lo) & (times < hi))) if times.size else True     # vacuous for no tracks, refused by the link above
    _link(links, "every observation falls inside the calendar year", inside,
          f"{times.size} observations, first {times.min() if times.size else None}, last {times.max() if times.size else None}, year spans [{lo}, {hi})")
    import export_protocol_case as E
    expected = E.input_paths(manifest["datasets"][dataset], year)      # {var: path}, the entry point's own rule
    inputs = d.get("inputs_sha256") or {}
    _link(links, "record names exactly the year's two canonical input files",
          sorted(inputs) == sorted(os.path.basename(p) for p in expected.values()),
          f"record names {sorted(inputs)}, the entry point expects {sorted(os.path.basename(p) for p in expected.values())}")
    _link(links, "the expected input paths are distinct", len(set(expected.values())) == len(expected), str(sorted(expected.values())))
    for var, path in sorted(expected.items()):
        name = os.path.basename(path)
        if not os.path.exists(path):
            _link(links, f"raw input {name} present", False, path)
            links[-1]["raw_validation_expected"] = {"path": path, "year": year, "var": var}
            continue
        _link(links, f"raw input {name} digest equals the record's", _sha256(path) == inputs.get(name))
        reason = raw_validator(path, year, var)
        _link(links, f"raw input {name} passes the retrieval validator for {year} as {var}", reason is None, reason or "")
        links[-1]["raw_validation_expected"] = {"path": path, "year": year, "var": var}
        links[-1]["raw_validator_called"] = {"path": path, "year": year, "var": var}
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
    if campaign is not None:
        case = os.path.join(campaign, f"{dataset}_{year}", "tracker_case.mat")
        if not os.path.exists(case):
            _link(links, "retained case present with the record's digest", False, f"{case} is absent")
        else:
            _link(links, "retained case present with the record's digest", _sha256(case) == d.get("case_sha256"), case)
    else:
        _link(links, "retained case present with the record's digest", False, "no campaign root was given, so the case was not read")
    return links


def check_year(evidence, artifacts, summary_rows, year, manifest_path, regions_dir, record_dir, published_dir):
    """The per-year links. The binding of the artifact to the collected runs, the manifest,
    the tracks, the auxiliary inputs and the instrument is THE COLLECTOR'S OWN CHECK,
    called here as one link so the two cannot drift; the row link rebuilds the row from
    the artifact with the collector's own row builder and requires equality field by
    field, so every copied value is bound and not only two counts."""
    import collect_protocol_campaign as C
    links = []
    path = os.path.join(artifacts, f"paired_{year}_eraint_era5.json")
    if not os.path.exists(path):
        _link(links, "paired artifact present", False, path)
        return links
    try:
        art = json.load(open(path))
    except (OSError, ValueError) as exc:
        _link(links, "paired artifact readable", False, str(exc))
        return links
    if not isinstance(art, dict) or not isinstance(art.get("comparison"), dict) or not isinstance(art.get("sides"), dict):
        _link(links, "paired artifact is an artifact object with comparison and sides", False, f"top level is {type(art).__name__}")
        return links
    _link(links, "artifact names the year", art.get("year") == year, f"artifact says {art.get('year')}")
    try:
        problems = C.existing_artifact_problems(path, evidence, manifest_path, year, regions_dir, record_dir, published_dir)
    except Exception as exc:                          # a reporting boundary: a failed link, never a lost report
        problems = [f"the artifact's structure defeated the check: {type(exc).__name__}: {exc}"]
    _link(links, "artifact is bound to the collected runs, the manifest, the tracks, the auxiliary inputs and the instrument (the collector's check)",
          not problems, "; ".join(problems))
    row = summary_rows.get(str(year)) if isinstance(summary_rows, dict) else None
    if not isinstance(row, dict):
        _link(links, "summary row present and an object", False,
              f"rows are {type(summary_rows).__name__}, row is {type(row).__name__}")
        return links
    try:
        rebuilt = C.row_from_artifact(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        _link(links, "summary row equals the row rebuilt from the artifact, every copied field", False, f"the row could not be rebuilt: {exc}")
        return links
    row, rebuilt = json.loads(json.dumps(row)), json.loads(json.dumps(rebuilt))       # as the summary stores them
    differing = sorted(k for k in set(row) | set(rebuilt) if (k in row) != (k in rebuilt) or row.get(k) != rebuilt.get(k))
    _link(links, "summary row equals the row rebuilt from the artifact, every copied field", not differing,
          "differing fields: " + ", ".join(differing) if differing else "")
    return links


def run_check(years, manifest, manifest_sha256, evidence, artifacts, summary_rows, raw_validator, calibration_path,
              manifest_path=None, regions_dir=None, record_dir=None, published_dir=None, campaign=None,
              summary_aggregates=None, bind_aggregates=False):
    """Every run and year of `years`, with the accounting a verdict rests on: the set of
    (path, year, variable) validations performed must equal the set the entry point's own
    input rule expects for every run, so an unperformed validation can never read as
    passed and a count cannot stand in for the inventory, and a refusal of an empty
    range, since a check over nothing is not a check. Returns the artifact body."""
    years = list(years)
    if not years:
        raise SystemExit("REFUSED: an empty year range checks nothing and cannot bind anything")
    runs, checks = {}, {}
    for year in years:
        for dataset in DATASETS:
            runs[f"{dataset}_{year}"] = check_run(evidence, dataset, year, manifest, manifest_sha256, raw_validator, calibration_path, campaign)
        checks[str(year)] = check_year(evidence, artifacts, summary_rows, year, manifest_path, regions_dir, record_dir, published_dir)
    import export_protocol_case as E
    if bind_aggregates or summary_aggregates is not None:
        # the command always binds the summary's aggregates; a summary without an aggregates
        # object is a failed link, never a skipped one, and a row the aggregation cannot read
        # is a failed link rather than an exception that loses the whole report
        import collect_protocol_campaign as C
        checks["summary"] = []
        if not isinstance(summary_aggregates, dict):
            _link(checks["summary"], "summary carries an aggregates object", False, f"aggregates is {type(summary_aggregates).__name__}")
        else:
            try:
                recomputed = json.loads(json.dumps(C.aggregates(summary_rows)))
            except Exception as exc:                 # a reporting boundary: any failure here is a failed link, never a lost report
                recomputed = None
                _link(checks["summary"], "summary aggregates equal those recomputed from its rows", False,
                      f"the rows could not be aggregated: {type(exc).__name__}: {exc}")
            if recomputed is not None:
                _link(checks["summary"], "summary aggregates equal those recomputed from its rows", recomputed == json.loads(json.dumps(summary_aggregates)))
    all_links = [l for ls in list(runs.values()) + list(checks.values()) for l in ls]
    failed = [(k, l) for k, ls in list(runs.items()) + list(checks.items()) for l in ls if not l["passed"]]
    performed = sorted(json.dumps(l["raw_validator_called"], sort_keys=True) for l in all_links if l.get("raw_validator_called"))
    # what a bound campaign needs is derived from every requested run through the entry point's own rule,
    # never from the links that happened to be emitted, so a run that was never reached still counts as owed
    expected = sorted(json.dumps({"path": p, "year": y, "var": v}, sort_keys=True)
                      for y in years for ds in DATASETS for v, p in E.input_paths(manifest["datasets"][ds], y).items())
    validator_calls, validations_needed = len(performed), len(expected)
    raw_validated = bool(expected) and performed == expected
    bound = not failed and raw_validated and bool(all_links)
    return {"years": [int(y) for y in years], "runs_checked": len(runs), "years_checked": len(years),
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
    ap.add_argument("--campaign", default="data/protocol_runs/campaign", help="the campaign root holding the retained cases")
    ap.add_argument("--regions-dir", default="data/aewc_v2_pilot/v1_src")
    ap.add_argument("--record-dir", default="data/aewc")
    ap.add_argument("--published-dir", default="data/aewc")
    args = ap.parse_args(argv)
    import export_protocol_case as E
    manifest, manifest_sha256 = E.load_manifest(args.manifest)

    raw_validator = real_raw_validator(manifest)

    def calibration_path(dataset):
        return os.path.join(args.calibration_dir, f"thresholds_protocol_{dataset}_1979_2010.json")
    y0, y1 = (int(x) for x in args.years.split("-"))
    if y1 < y0:
        raise SystemExit(f"REFUSED: the year range {args.years} is reversed")
    # the summary is read once, guarded: a file that is not a summary object yields rows and
    # aggregates the check names as failed links, and a report is still published
    summary_blob = None
    try:
        with open(args.summary, "rb") as fh:
            summary_blob = fh.read()
        summary = json.loads(summary_blob.decode())
    except (OSError, ValueError) as exc:
        summary = {"unreadable": str(exc)}
    if not isinstance(summary, dict):
        summary = {"not_an_object": type(summary).__name__}
    rows, aggregates = summary.get("years"), summary.get("aggregates")
    body = run_check(range(y0, y1 + 1), manifest, manifest_sha256, args.evidence, args.artifacts, rows, raw_validator, calibration_path,
                     args.manifest, args.regions_dir, args.record_dir, args.published_dir, args.campaign,
                     aggregates, bind_aggregates=True)
    out = {"generated_by": "scripts/check_campaign_bindings.py", "script_sha256": X.digest(__file__),
           "manifest_sha256": manifest_sha256,
           "summary_sha256": hashlib.sha256(summary_blob).hexdigest() if summary_blob is not None else None, **body}
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
