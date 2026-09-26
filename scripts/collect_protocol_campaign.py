#!/usr/bin/env python3
"""Collect the protocol campaign: the small files of every run into evidence, one paired
comparison per year through the reviewed instrument, and one summary read from those
artifacts.

WHAT IT DOES, and nothing more. For each run directory `<dataset>_<year>` under the
campaign root it copies the tracks, the tracking record and the run log into the
evidence directory and writes a pointer to the 387 MB case left outside git, refusing to
overwrite anything already there. For each year with both datasets' records it runs
`season_metrics.py` in reanalysis mode against the retained manifest, publishing one
artifact per year exclusively. It then reads every per-year artifact and writes a
summary of the season-level and monthly lines by year, with each year's artifact digest,
so the campaign page is read from one file whose every number traces to a per-year
artifact that traces to two records.

The summary computes nothing new: every value is copied from a per-year artifact. A
year whose runs are missing or whose comparison the gate refused is listed as such and
not silently dropped.

    .venv/bin/python3 scripts/collect_protocol_campaign.py --campaign data/protocol_runs/campaign \\
        --evidence docs/aewc_v2/evidence/validation/protocol_campaign --manifest <json> \\
        --artifacts docs/aewc_v2/artifacts --summary <json>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

SMALL_FILES = ("tracker_port.mat", "run.log")
DATASETS = ("eraint", "era5")


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def collect_run(campaign, evidence, dataset, year):
    """Copy one run's small files into evidence, never over an existing file. Returns the
    evidence directory, or None when the run has no completion record."""
    src = os.path.join(campaign, f"{dataset}_{year}")
    record = os.path.join(src, f"tracking_{dataset}_{year}.json")
    if not os.path.exists(record):
        return None
    dst = os.path.join(evidence, f"{dataset}_{year}")
    os.makedirs(dst, exist_ok=True)
    for name in SMALL_FILES + (f"tracking_{dataset}_{year}.json",):
        target = os.path.join(dst, name)
        if os.path.exists(target):
            if _sha256(target) != _sha256(os.path.join(src, name)):
                raise SystemExit(f"REFUSED: {target} exists with different bytes and is never overwritten")
            continue
        shutil.copyfile(os.path.join(src, name), target)
    pointer = os.path.join(dst, "CASE_LOCATION.txt")
    if not os.path.exists(pointer):
        with open(pointer, "w") as fh:
            fh.write(f"The retained case for this run, about 387 MB, lives outside git at "
                     f"{os.path.abspath(os.path.join(src, 'tracker_case.mat'))}. Its digest is "
                     f"case_sha256 in tracking_{dataset}_{year}.json.\n")
    return dst


def auxiliary_digests(regions_dir, record_dir, published_dir, year):
    """The digests of the comparison's other inputs exactly as the instrument records
    them: the region polygons, the published record files over the scale years, and the
    published year file for `year` when one exists (None when it does not)."""
    import season_metrics as S
    try:
        _, regions = S.load_regions(regions_dir)
    except (OSError, ValueError, KeyError) as exc:
        regions = {"unreadable": str(exc)}
    y0, y1 = (int(x) for x in S.RECORD_YEARS.split("-"))      # the instrument's own default, not a copy
    record = {}
    for y in range(y0, y1 + 1):
        p = os.path.join(record_dir, f"ERA-Int_ew_700hPa_{y}_AFR.nc")
        if os.path.exists(p):
            record[str(y)] = _sha256(p)
    published = os.path.join(published_dir, f"ERA-Int_ew_700hPa_{year}_AFR.nc")
    return regions, record, (_sha256(published) if os.path.exists(published) else None)


def existing_artifact_problems(path, evidence, manifest, year, regions_dir, record_dir, published_dir):
    """Why an existing per-year artifact is NOT the comparison of the current runs under
    the current inputs, or an empty list when it is. A review named the gap: a file at
    the expected path was reused on existence alone, so a stale artifact from an earlier
    campaign, or one whose sides were other runs, would have been summarized as this
    year's. A second review named the rest of the key: the region polygons, the
    published record files and the published year file are inputs the instrument reads
    and digests, so a changed polygon or archive file would also leave an obsolete
    comparison reusable. Every digest the artifact records is compared with the
    corresponding current input."""
    problems = []
    try:
        art = json.load(open(path))
    except (OSError, ValueError) as exc:
        return [f"unreadable: {exc}"]
    if art.get("year") != year:
        problems.append(f"artifact year {art.get('year')} is not {year}")
    regions, record, published = auxiliary_digests(regions_dir, record_dir, published_dir, year)
    if art.get("region_polygons_sha256") != regions:
        problems.append("region polygon digests are not the current regions directory's")
    if art.get("published_record_sha256") != record:
        problems.append("published record digests are not the current record directory's")
    named_published = ((art.get("inputs_sha256") or {}).get("published_year_file") or {}).get("sha256")
    if named_published != published:
        problems.append("the published year file's presence or digest differs from the current published directory's")
    manifest_sha = _sha256(manifest)
    for dataset, side in (("eraint", "v1"), ("era5", "port")):
        record = os.path.join(evidence, f"{dataset}_{year}", f"tracking_{dataset}_{year}.json")
        tracks = os.path.join(evidence, f"{dataset}_{year}", "tracker_port.mat")
        case = json.load(open(record))["dataset_specific"].get("case_id") if os.path.exists(record) else None
        s = (art.get("sides") or {}).get(side) or {}
        if case is None or s.get("case_id") != case:
            problems.append(f"side {side} case id {s.get('case_id')} is not the collected {dataset} record's {case}")
        if s.get("manifest_sha256") != manifest_sha:
            problems.append(f"side {side} manifest digest is not the current manifest's")
        named = ((art.get("inputs_sha256") or {}).get(side) or {}).get("sha256")
        if not os.path.exists(tracks) or named != _sha256(tracks):
            problems.append(f"side {side} input digest is not the collected tracks file's")
    return problems


def compare_year(evidence, artifacts, manifest, year, regions_dir, record_dir, published_dir, runner):
    """One paired comparison, published exclusively; `runner` is season_metrics.main. An
    artifact already at the path is reused only when it is bound to the current runs
    and manifest, and is otherwise reported as a refusal, never overwritten."""
    a = os.path.join(evidence, f"eraint_{year}", "tracker_port.mat")
    b = os.path.join(evidence, f"era5_{year}", "tracker_port.mat")
    out = os.path.join(artifacts, f"paired_{year}_eraint_era5.json")
    if os.path.exists(out):
        problems = existing_artifact_problems(out, evidence, manifest, year, regions_dir, record_dir, published_dir)
        if problems:
            return None, "refused: an artifact exists at the path and is not this year's comparison of the collected runs (" + "; ".join(problems) + "), and artifacts are never overwritten"
        return out, "exists"
    argv = ["--mode", "reanalysis", "--manifest", manifest, "--v1", a, "--port", b, "--year", str(year),
            "--regions-dir", regions_dir, "--record-dir", record_dir, "--out", out]
    published = os.path.join(published_dir, f"ERA-Int_ew_700hPa_{year}_AFR.nc")
    if os.path.exists(published):
        argv += ["--published-year-file", published]
    try:
        runner(argv)
    except SystemExit as exc:
        return None, f"refused: {exc}"
    return out, "published"


def summarize(per_year):
    """The season-level and monthly lines of every per-year artifact, copied, keyed by year,
    with the artifact digest, plus the years that have no artifact and why."""
    rows, missing = {}, {}
    for year, (path, status) in sorted(per_year.items()):
        if path is None:
            missing[str(year)] = status
            continue
        with open(path, "rb") as fh:
            blob = fh.read()
        art = json.loads(blob.decode())
        s = art["comparison"]["season"]
        rows[str(year)] = {
            "artifact": os.path.basename(path), "artifact_sha256": hashlib.sha256(blob).hexdigest(),
            "sides": {k: art["sides"][k].get("case_id") for k in ("v1", "port")},
            "tracks_in_year": {k: art["columns"][k]["tracks_all"] for k in ("v1", "port")},
            "season_whole_domain": {k: art["columns"][k]["tracks_in_season_whole_domain"] for k in ("v1", "port")},
            "africa_origin": {"v1": s["v1"], "port": s["port"], "port_minus_v1": s["port_minus_v1"],
                              "over_published_sd": s.get("difference_over_published_sd")},
            "distinct_waves": {k: s["distinct_waves"][k] for k in ("v1", "port", "port_minus_v1")},
            "duplication_fraction": {"v1": s["duplication"]["v1_fraction"], "port": s["duplication"]["port_fraction"]},
            "lifetime_percentiles": {p: (v["v1"], v["port"]) for p, v in s["distributions"]["lifetime"]["percentiles"].items()} if "distributions" in s else None,
            "months": {str(m): {"v1": art["comparison"][f"month {m}"]["v1"], "port": art["comparison"][f"month {m}"]["port"]} for m in (6, 7, 8, 9)},
            "bands": {k[len("band "):]: {"v1": v["v1"], "port": v["port"]} for k, v in art["comparison"].items() if k.startswith("band ")},
            "boundaries": {k: {b: art["columns"][k]["boundaries"][b]["tracks"] for b in art["columns"][k]["boundaries"]} for k in ("v1", "port")},
            "published_context_tracks": (art.get("published_context_column") or {}).get("tracks_in_season")}
    return {"years": rows, "years_without_a_comparison": missing, "aggregates": aggregates(rows)}


def aggregates(rows):
    """Multi-year statistics of the per-year lines, so the campaign page reads them from
    this artifact: per side the mean and the interannual standard deviation (ddof 1) of
    the Africa-origin season count, the distinct-wave count and the tracks in the year,
    the mean and standard deviation of the ERA5 minus ERA-Interim difference, how many
    years each side is lower, the Pearson correlation of the two annual series, the mean
    starts by month and by ten-degree genesis-longitude band, and, over the years the
    archive covers, each side's correlation with the archive's count.
    Nothing here is a test of significance, and no contrast is called a change."""
    import numpy as np
    if not rows:
        return None
    years = sorted(rows)
    v1 = np.array([rows[y]["africa_origin"]["v1"] for y in years], float)
    port = np.array([rows[y]["africa_origin"]["port"] for y in years], float)
    w1 = np.array([rows[y]["distinct_waves"]["v1"] for y in years], float)
    w2 = np.array([rows[y]["distinct_waves"]["port"] for y in years], float)
    t1 = np.array([rows[y]["tracks_in_year"]["v1"] for y in years], float)
    t2 = np.array([rows[y]["tracks_in_year"]["port"] for y in years], float)
    arch = np.array([rows[y]["published_context_tracks"] if rows[y]["published_context_tracks"] is not None else np.nan for y in years], float)
    have = ~np.isnan(arch)

    def sd(x):
        return float(np.std(x, ddof=1)) if x.size > 1 else None

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and np.std(a) > 0 and np.std(b) > 0 else None
    out = {"years": [int(y) for y in years], "n_years": len(years),
           "africa_origin": {"mean": {"v1": float(v1.mean()), "port": float(port.mean())},
                             "interannual_sd": {"v1": sd(v1), "port": sd(port)},
                             "difference_port_minus_v1": {"mean": float((port - v1).mean()), "sd": sd(port - v1),
                                                          "years_port_lower": int((port < v1).sum()), "years_port_higher": int((port > v1).sum()),
                                                          "years_equal": int((port == v1).sum())},
                             "pearson_correlation_v1_port": corr(v1, port)},
           "distinct_waves": {"mean": {"v1": float(w1.mean()), "port": float(w2.mean())},
                              "difference_port_minus_v1": {"mean": float((w2 - w1).mean()), "sd": sd(w2 - w1)},
                              "pearson_correlation_v1_port": corr(w1, w2)},
           "tracks_in_year": {"mean": {"v1": float(t1.mean()), "port": float(t2.mean())}},
           "season_whole_domain": {"mean": {k: float(np.mean([rows[y]["season_whole_domain"][k] for y in years])) for k in ("v1", "port")}},
           "duplication_fraction": {"mean": {k: float(np.mean([rows[y]["duplication_fraction"][k] for y in years])) for k in ("v1", "port")}},
           "boundaries": {"mean": {k: {b: float(np.mean([rows[y]["boundaries"][k][b] for y in years]))
                                       for b in ("crossing_in", "crossing_out", "year_end_potentially_censored")} for k in ("v1", "port")}},
           "months_mean_starts": {m: {"v1": float(np.mean([rows[y]["months"][m]["v1"] for y in years])),
                                      "port": float(np.mean([rows[y]["months"][m]["port"] for y in years]))} for m in ("6", "7", "8", "9")},
           "bands_mean_starts": {b: {"v1": float(np.mean([rows[y]["bands"][b]["v1"] for y in years])),
                                     "port": float(np.mean([rows[y]["bands"][b]["port"] for y in years]))}
                                 for b in sorted(set.intersection(*(set(rows[y]["bands"]) for y in years)), key=lambda b: int(b.split("..")[0]))},
           "archive": {"years_with_archive": int(have.sum()),
                       "mean": float(np.nanmean(arch)) if have.any() else None,
                       "interannual_sd": sd(arch[have]) if have.any() else None,
                       "pearson_correlation_v1_archive": corr(v1[have], arch[have]) if have.any() else None,
                       "pearson_correlation_port_archive": corr(port[have], arch[have]) if have.any() else None}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--years", default="1979-2010")
    ap.add_argument("--regions-dir", default="data/aewc_v2_pilot/v1_src")
    ap.add_argument("--record-dir", default="data/aewc")
    ap.add_argument("--published-dir", default="data/aewc")
    args = ap.parse_args(argv)
    import season_metrics
    y0, y1 = (int(x) for x in args.years.split("-"))
    per_year, collected = {}, {}
    for year in range(y0, y1 + 1):
        dirs = {ds: collect_run(args.campaign, args.evidence, ds, year) for ds in DATASETS}
        collected[str(year)] = {ds: bool(d) for ds, d in dirs.items()}
        if all(dirs.values()):
            per_year[year] = compare_year(args.evidence, args.artifacts, args.manifest, year, args.regions_dir,
                                          args.record_dir, args.published_dir, season_metrics.main)
        else:
            per_year[year] = (None, "a run is missing: " + ", ".join(ds for ds, d in dirs.items() if not d))
    summary = summarize(per_year)
    summary.update({"generated_by": "scripts/collect_protocol_campaign.py", "script_sha256": X.digest(__file__),
                    "git_head_at_launch": X.repository_head(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "manifest_sha256": _sha256(args.manifest), "runs_collected": collected})
    try:
        X.publish_json(args.summary, summary, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.summary} exists and artifacts are never overwritten")
    print(f"collected {sum(1 for v in collected.values() if all(v.values()))} complete years, "
          f"{len(summary['years'])} compared, {len(summary['years_without_a_comparison'])} without; wrote {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
