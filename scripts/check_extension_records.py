#!/usr/bin/env python3
"""Check a campaign tracked under an extension manifest against what it must be bound to.

An extension manifest (see export_protocol_case.py) tracks years after the protocol years
under the protocol's own climatology and thresholds, held fixed. This script decides
whether every record of such a campaign is a protocol run of its year under that
manifest, and it reuses the checks that already exist rather than restating them:

- campaign_record_ok.problems: the record names the dataset and year of its path, the
  tracks beside it have the digest it names, and it was made under the manifest's digest;
- season_metrics.producer_problems: every protocol setting equals the manifest's, key by
  key, the stage is tracking, the dataset identity resolves through the manifest, and the
  source inventory is the complete tracking inventory;
- export_protocol_case.load_manifest and check_calibration: the extension is its base
  plus later tracking years only, and the calibration artifact fits the manifest.

THE CLIMATOLOGY IS COMPARED AS A MAPPING. Each record's climatology inputs, file name to
SHA-256, must equal the mapping every record of the RETAINED ORIGINAL CAMPAIGN carries,
entry for entry, so a changed digest under an unchanged file name is a mismatch, and so
is an added or a missing file. The retained records must agree with one another first.
The calibration's own input digests must equal the same mapping.

The rest of each record: the extension block names the manifest's tracking, climatology
and calibration years and its base; the thresholds and calibration digest are the
artifact's; the tracker's source digests equal the retained campaign's; the year's two
input files, read from the given directory, have the digests the record names; every
observation lies in the record's calendar year; and the record names a commit and a
clean tree. An absent input file, an unknown tree state, or a missing year is a problem,
never a pass. Every tracking year must have exactly one record: every tracking record
anywhere under the campaign root is inventoried, and one at any path but its year's
canonical path, a duplicate run or a kept attempt for instance, fails the campaign, as
does any run directory that is not a tracking year's. The inventory does not follow
symbolic links, so any symbolic link under the root fails the campaign too, since a record
reached through one would be outside the inventory.

    python3 scripts/check_extension_records.py --runs <campaign root> \\
        --manifest docs/aewc_v2/protocol/manifest_extension_era5_2011_2025_2026-10-03.json \\
        --retained docs/aewc_v2/evidence/validation/protocol_campaign \\
        --inputs-dir <directory holding the tracked years' input files> [--out <report json>]
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign_record_ok as R  # noqa: E402
import exact_tracks as X  # noqa: E402
import export_protocol_case as E  # noqa: E402
import season_metrics as S  # noqa: E402

DEFAULT_CALIBRATION = "docs/aewc_v2/artifacts/thresholds_protocol_{prefix}_1979_2010.json"
TRACKER_SOURCE_PREFIX = "src/aew/v1port/"


def _sha256(path):
    return X.digest(path)


def retained_reference(retained, dataset, years):
    """The climatology mapping and tracker source digests every retained record of the
    original campaign carries, which must agree across those records. Returns
    (mapping, tracker_sources, problems)."""
    mappings, sources, problems = {}, {}, []
    for y in years:
        path = os.path.join(retained, f"{dataset}_{y}", f"tracking_{dataset}_{y}.json")
        try:
            d = json.load(open(path))
        except (OSError, ValueError) as exc:
            problems.append(f"retained record {dataset} {y} is unreadable: {exc}")
            continue
        mappings[y] = (d.get("dataset_specific") or {}).get("climatology_inputs_sha256")
        sources[y] = {k: v for k, v in (d.get("source_sha256") or {}).items() if k.startswith(TRACKER_SOURCE_PREFIX)}
    if problems:
        return None, None, problems
    first = years[0]
    if not isinstance(mappings[first], dict) or not mappings[first]:
        return None, None, [f"retained record {dataset} {first} carries no climatology mapping"]
    disagree = [y for y in years if mappings[y] != mappings[first]]
    if disagree:
        problems.append(f"retained records disagree on the climatology mapping for {disagree[:4]}")
    disagree = [y for y in years if sources[y] != sources[first]]
    if disagree:
        problems.append(f"retained records disagree on the tracker source for {disagree[:4]}")
    if not sources[first]:
        problems.append("retained records carry no tracker source digests")
    return mappings[first], sources[first], problems


def mapping_problems(named, reference, label):
    """Entry-by-entry comparison of a file-name-to-digest mapping with the reference."""
    if not isinstance(named, dict):
        return [f"{label} is not a mapping"]
    out = []
    for name in sorted(set(reference) - set(named)):
        out.append(f"{label} lacks {name}")
    for name in sorted(set(named) - set(reference)):
        out.append(f"{label} adds {name}")
    for name in sorted(set(named) & set(reference)):
        if named[name] != reference[name]:
            out.append(f"{label} names a different digest for {name}")
    return out


def record_problems(runs, year, manifest, manifest_sha256, dataset_name, reference, tracker_sources,
                    calibration, inputs_dir):
    """Every reason one record is not a protocol run of `year` under the extension manifest."""
    dataset = manifest["datasets"][dataset_name]
    run = os.path.join(runs, f"{dataset_name}_{year}")
    rec_path = os.path.join(run, f"tracking_{dataset_name}_{year}.json")
    if not os.path.exists(rec_path):
        return [f"no record at {rec_path}"]
    p = list(R.problems(rec_path, manifest_sha256=manifest_sha256))
    try:
        record = json.load(open(rec_path))
    except (OSError, ValueError) as exc:
        return p + [f"unreadable record: {exc}"]
    p += S.producer_problems(record, manifest, manifest_sha256, year, "record")
    d = record.get("dataset_specific") or {}
    ext = d.get("manifest_extension") or {}
    if ext.get("tracking_years") != E.tracking_years(manifest) or ext.get("extends") != manifest["extends"] \
            or ext.get("climatology_and_calibration_years") != [int(x) for x in manifest["years"]]:
        p.append("the extension block is not the manifest's tracking, climatology and calibration years and base")
    ct, ft, cal_sha, cal_inputs = calibration
    if d.get("coarse_threshold") != ct or d.get("fine_threshold") != ft:
        p.append("the thresholds are not the calibration artifact's")
    if d.get("calibration_sha256") != cal_sha:
        p.append("the calibration digest is not the artifact's")
    p += mapping_problems(d.get("climatology_inputs_sha256"), reference, "the climatology inputs")
    sources = {k: v for k, v in (record.get("source_sha256") or {}).items() if k.startswith(TRACKER_SOURCE_PREFIX)}
    if sources != tracker_sources:
        p.append("the tracker source is not the retained campaign's")
    for var in ("u700", "v700"):
        name = f"{dataset['prefix']}_{var}_{year}_6h_region.nc"
        path = os.path.join(inputs_dir, name)
        named = (d.get("inputs_sha256") or {}).get(name)
        if not os.path.exists(path):
            p.append(f"input {name} is absent from {inputs_dir}, so its digest is unchecked")
        elif named != _sha256(path):
            p.append(f"input {name} does not have the digest the record names")
    tracks_path = os.path.join(run, "tracker_port.mat")
    if os.path.exists(tracks_path):
        with open(tracks_path, "rb") as fh:
            tracks, _case = S.read_mat_tracks(fh.read())
        lo, hi = S._day(year, 1, 1), S._day(year + 1, 1, 1)
        if any(np.any(t["time"] < lo) or np.any(t["time"] >= hi) for t in tracks):
            p.append(f"observations lie outside {year}")
    else:
        p.append("no tracks file beside the record")
    head = record.get("git_head")
    if not (isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head)):
        p.append(f"the record names no commit ({head!r})")
    if record.get("git_dirty") is not False:
        p.append(f"the record's tree state is {record.get('git_dirty')!r}, not a checked clean tree")
    return p


def check(runs, manifest_path, retained, inputs_dir, dataset_name="era5", calibration_path=None):
    """The whole campaign. Returns a report whose `passed` is true only when every tracking
    year has one record and no record has a problem."""
    manifest, manifest_sha256 = E.load_manifest(manifest_path)
    if "extends" not in manifest:
        raise SystemExit("REFUSED: the manifest is not an extension")
    if dataset_name not in manifest["datasets"]:
        raise SystemExit(f"REFUSED: {dataset_name} is not a manifest dataset")
    dataset = manifest["datasets"][dataset_name]
    calibration_path = calibration_path or DEFAULT_CALIBRATION.format(prefix=dataset["prefix"])
    ct, ft, cal_sha, cal_inputs = E.check_calibration(calibration_path, manifest, dataset)
    y0, y1 = (int(x) for x in manifest["years"])
    reference, tracker_sources, problems = retained_reference(retained, dataset_name, list(range(y0, y1 + 1)))
    if problems:
        raise SystemExit("REFUSED: " + "; ".join(problems))
    cal_problems = mapping_problems(cal_inputs, reference, "the calibration's inputs")
    t0, t1 = E.tracking_years(manifest)
    years = {}
    for year in range(t0, t1 + 1):
        years[str(year)] = record_problems(runs, year, manifest, manifest_sha256, dataset_name, reference,
                                           tracker_sources, (ct, ft, cal_sha, cal_inputs), inputs_dir)
    canonical = {f"{dataset_name}_{y}" for y in range(t0, t1 + 1)}
    stray = sorted(name for name in os.listdir(runs)
                   if os.path.isdir(os.path.join(runs, name)) and name.startswith(f"{dataset_name}_") and name not in canonical)
    expected = {os.path.join(runs, f"{dataset_name}_{y}", f"tracking_{dataset_name}_{y}.json") for y in range(t0, t1 + 1)}
    found = {os.path.join(d, f) for d, _dirs, files in os.walk(runs) for f in files
             if f.startswith("tracking_") and f.endswith(".json")}
    extra = sorted(os.path.relpath(path, runs) for path in found - expected)
    links = sorted(os.path.relpath(os.path.join(d, n), runs) for d, dirs, files in os.walk(runs)
                   for n in dirs + files if os.path.islink(os.path.join(d, n)))
    return {"generated_by": "scripts/check_extension_records.py", "script_sha256": _sha256(os.path.abspath(__file__)),
            "manifest": manifest_path, "manifest_sha256": manifest_sha256, "dataset": dataset_name,
            "tracking_years": [t0, t1], "calibration_sha256": cal_sha,
            "retained_climatology_mapping_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
            "calibration_problems": cal_problems, "run_directories_not_a_tracking_year": stray,
            "tracking_records_outside_the_canonical_paths": extra, "symbolic_links_under_the_root": links,
            "years": years,
            "passed": not cal_problems and not stray and not extra and not links and all(not v for v in years.values())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, help="the campaign root holding <dataset>_<year> run directories")
    ap.add_argument("--manifest", required=True, help="the extension manifest the campaign was run under")
    ap.add_argument("--retained", required=True, help="the retained original campaign's evidence directory")
    ap.add_argument("--inputs-dir", required=True, help="the directory holding the tracked years' input files")
    ap.add_argument("--dataset", default="era5")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", default=None, help="publish the report here, never over an existing file")
    args = ap.parse_args(argv)
    report = check(args.runs, args.manifest, args.retained, args.inputs_dir, args.dataset, args.calibration)
    for year, problems in report["years"].items():
        print(f"{year}: {'OK' if not problems else '; '.join(problems)}")
    for line in (report["calibration_problems"]
                 + [f"a run directory that is not a tracking year's: {s}" for s in report["run_directories_not_a_tracking_year"]]
                 + [f"a tracking record outside the canonical paths: {s}" for s in report["tracking_records_outside_the_canonical_paths"]]
                 + [f"a symbolic link under the campaign root: {s}" for s in report["symbolic_links_under_the_root"]]):
        print(line)
    print("PASSED" if report["passed"] else "NOT PASSED")
    if args.out:
        try:
            X.publish_json(args.out, report, exclusive=True)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {args.out} exists and reports are never overwritten")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
