#!/usr/bin/env python3
"""Assemble the versioned ERA5 1979 to 2025 track product from two retained campaigns.

The original years, 1979 to 2010, are the retained protocol campaign's ERA5 runs, copied
byte for byte. The later years, 2011 to 2025, are the campaign tracked under the ERA5
extension manifest, accepted only when check_extension_records.py passes it. Nothing is
tracked here, and no file of either campaign is written to.

THE SEASON SUMMARY REUSES THE INSTRUMENT'S DEFINITIONS. Each year's row holds the ERA5
column of the retained campaign summary's row: tracks in the year, June to September
tracks over the whole domain, Africa-origin June to September tracks by the archive's
source-region rule, their distinct waves, duplication fraction, lifetime percentiles,
starts by month and by ten-degree genesis band, and the boundary counts. The row is
computed with season_metrics.py's own functions, and before anything is written the rows
for 1979 to 2010 are recomputed from the retained tracks and must equal the retained
summary's ERA5 column field for field, or the build is refused. One deliberate difference
is stated in the summary: a row here reports lifetime percentiles for any year with an
Africa-origin track, where the comparison summary reports them only when its sparse rule
allows, which it did for every year 1979 to 2010. No multi-year statistic is computed, and
no value of the later years is compared with the earlier ones.

THE PRODUCT DIRECTORY IS THE RELEASE. It is assembled under a temporary name beside the
target, its completion marker written last, and renamed into place, so a target that
exists is never written into and a product without its marker is incomplete. Copies of
its manifest and season summary are then exported to the artifacts directory, a step
that can be repeated (--artifacts-only) after a failure: a copy present must be
byte-identical, and a missing one is made. No destination may lie inside either campaign,
resolved through symbolic links, and that is checked before anything is created. The
artifact export writes only through a directory descriptor whose own ancestry is checked,
so a path replaced after the check cannot redirect it. The product build stages through
paths, and a directory replaced by another process while it runs is outside what it
guards. The extension checker refuses any symbolic link in the extension campaign, and the
retained campaign is under git, where a stray file shows.

    python3 scripts/build_era5_record_product.py --out <product directory> --artifacts docs/aewc_v2/artifacts \\
        --extension-runs data/protocol_runs/extension_era5_2011_2025 --inputs-dir data/era5/v1port_ext_2011_2025
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign_record_ok as R  # noqa: E402
import check_extension_records as C  # noqa: E402
import exact_tracks as X  # noqa: E402
import season_metrics as S  # noqa: E402

PRODUCT_ID = "aewc-v2-era5-1979-2025"
VERSION = "1.0.0-rc3"
DATASET = "era5"
BASE_MANIFEST = "docs/aewc_v2/protocol/manifest_2026-09-25.json"
EXTENSION_MANIFEST = "docs/aewc_v2/protocol/manifest_extension_era5_2011_2025_2026-10-03.json"
CALIBRATION = "docs/aewc_v2/artifacts/thresholds_protocol_era5_1979_2010.json"
RETAINED = "docs/aewc_v2/evidence/validation/protocol_campaign"
RETAINED_SUMMARY = "docs/aewc_v2/artifacts/protocol_campaign_summary_v5_2026-10-02.json"
ROW_FIELDS = ("tracks_in_year", "season_whole_domain", "africa_origin", "distinct_waves", "duplication_fraction",
              "lifetime_percentiles", "months", "bands", "boundaries")


def _sha(path):
    return X.digest(path)


def season_row(tracks_all, year, regions):
    """One year's ERA5 row, from the instrument's own functions, in the retained summary's
    field layout for one side."""
    whole = [t for t in tracks_all if S.in_season(t, year)]
    africa = [t for t in whole if S.in_record_domain(t, regions)]
    feats = [S.features(t, i) for i, t in enumerate(africa)]
    groups = S.group_values(africa, feats)
    lifetimes = np.asarray([f["lifetime"] for f in feats], float)
    return {"tracks_in_year": len(tracks_all), "season_whole_domain": len(whole),
            "africa_origin": groups["season"]["count"], "distinct_waves": groups["season"]["distinct_waves"],
            "duplication_fraction": groups["season"]["duplication_fraction"],
            "lifetime_percentiles": ({str(p): float(v) for p, v in zip(S.PERCENTILES, np.percentile(lifetimes, S.PERCENTILES))}
                                     if lifetimes.size else None),
            "months": {str(m): groups[f"month {m}"]["count"] for m in S.SEASON_MONTHS},
            "bands": {k[len("band "):]: v["count"] for k, v in groups.items() if k.startswith("band ")},
            "boundaries": {b: v["tracks"] for b, v in S.boundary_lines(tracks_all, year).items()}}


def retained_row(row):
    """The ERA5 column of a retained campaign summary row (side B, keyed port)."""
    return {"tracks_in_year": row["tracks_in_year"]["port"], "season_whole_domain": row["season_whole_domain"]["port"],
            "africa_origin": row["africa_origin"]["port"], "distinct_waves": row["distinct_waves"]["port"],
            "duplication_fraction": row["duplication_fraction"]["port"],
            "lifetime_percentiles": ({p: v[1] for p, v in row["lifetime_percentiles"].items()}
                                     if row.get("lifetime_percentiles") is not None else None),
            "months": {m: v["port"] for m, v in row["months"].items()},
            "bands": {b: v["port"] for b, v in row["bands"].items()},
            "boundaries": dict(row["boundaries"]["port"])}


def row_differences(computed, retained):
    return [f for f in ROW_FIELDS if computed.get(f) != retained.get(f)]


def _tracks(path):
    with open(path, "rb") as fh:
        tracks, _case = S.read_mat_tracks(fh.read())
    return tracks


def original_years(retained, summary_path, base_manifest, regions):
    """The retained campaign's ERA5 years, each checked and summarized. Returns
    {year: (run directory, row)} or raises with every reason."""
    summary = json.load(open(summary_path))
    base_sha = _sha(base_manifest)
    if summary.get("manifest_sha256") != base_sha:
        raise SystemExit("REFUSED: the retained summary was not made under the base manifest")
    manifest = json.load(open(base_manifest))
    y0, y1 = (int(x) for x in manifest["years"])
    out, problems = {}, []
    for year in range(y0, y1 + 1):
        run = os.path.join(retained, f"{DATASET}_{year}")
        rec = os.path.join(run, f"tracking_{DATASET}_{year}.json")
        p = R.problems(rec, manifest_sha256=base_sha)
        if not os.path.exists(os.path.join(run, "tracker_port.mat")):
            p.append("no tracks file")
        if p:
            problems.append(f"{year}: " + "; ".join(p))
            continue
        row = season_row(_tracks(os.path.join(run, "tracker_port.mat")), year, regions)
        kept = (summary.get("years") or {}).get(str(year))
        if kept is None:
            problems.append(f"{year}: the retained summary has no row")
            continue
        diff = row_differences(row, retained_row(kept))
        if diff:
            problems.append(f"{year}: the recomputed row differs from the retained summary in {diff}")
            continue
        out[year] = (run, row)
    if problems:
        raise SystemExit("REFUSED: the original years do not reproduce the retained campaign: " + " | ".join(problems[:6]))
    return out


MARKER = "RELEASE_COMPLETE.json"
ARTIFACT_FILES = (("manifest", "MANIFEST.json"), ("season_summary", "season_summary_africa_jjas.json"))


def _inside(path, root):
    """Whether `path`, resolved through symbolic links even where it does not exist yet, is
    `root` or lies under it."""
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def refuse_destinations_inside(destinations, campaigns):
    for dest in destinations:
        for root in campaigns:
            if _inside(dest, root):
                raise SystemExit(f"REFUSED: {dest} lies inside the campaign {root}, which is never written to")


def artifact_names(artifacts, version):
    return {k: os.path.join(artifacts, f"era5_record_1979_2025_{version}_{k}.json") for k, _ in ARTIFACT_FILES}


def product_identity(blobs):
    """The product name and version that the manifest's and season summary's bytes declare,
    which must agree. Exported copies are named by this, never by the version of the builder
    doing the export, so a product built earlier keeps its own version. It reads the same
    bytes that are published, so nothing changed between the two can split them."""
    declared = {}
    for _, inside in ARTIFACT_FILES:
        doc = json.loads(blobs[inside].decode())
        declared[inside] = (doc.get("product"), doc.get("version"))
    if len(set(declared.values())) != 1:
        raise SystemExit(f"REFUSED: the product's files disagree on its identity: {declared}")
    product, version = next(iter(declared.values()))
    if product != PRODUCT_ID or not isinstance(version, str) or not re.fullmatch(r"[0-9A-Za-z.+-]+", version):
        raise SystemExit(f"REFUSED: the product declares {product!r} version {version!r}, not a version of {PRODUCT_ID}")
    return product, version


def build(out, extension_runs, inputs_dir, regions_dir, retained=RETAINED, summary_path=RETAINED_SUMMARY,
          base_manifest=BASE_MANIFEST, extension_manifest=EXTENSION_MANIFEST, calibration=CALIBRATION):
    """The product directory, assembled under a temporary name and renamed into place with
    its completion marker written last. It is the release; the artifact copies are made
    from it by export_artifacts."""
    refuse_destinations_inside([out], [retained, extension_runs])
    if os.path.exists(out):
        raise SystemExit(f"REFUSED: {out} exists and a product is never written into")
    regions, region_digests = S.load_regions(regions_dir)
    report = C.check(extension_runs, extension_manifest, retained, inputs_dir, DATASET, calibration)
    if not report["passed"]:
        raise SystemExit("REFUSED: the extension campaign does not pass check_extension_records.py")
    original = original_years(retained, summary_path, base_manifest, regions)
    t0, t1 = report["tracking_years"]
    later = {y: (os.path.join(extension_runs, f"{DATASET}_{y}"),
                 season_row(_tracks(os.path.join(extension_runs, f"{DATASET}_{y}", "tracker_port.mat")), y, regions))
             for y in range(t0, t1 + 1)}
    if set(original) & set(later) or sorted(set(original) | set(later)) != list(range(min(original), max(later) + 1)):
        raise SystemExit("REFUSED: the two campaigns do not tile the years once each")
    parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(parent, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=".product-", dir=parent)
    files, rows = {}, {}
    try:
        for sub in ("tracks", "records", "protocol"):
            os.makedirs(os.path.join(stage, sub))

        def place(src, rel, origin):
            dst = os.path.join(stage, rel)
            shutil.copyfile(src, dst)
            digest = _sha(dst)
            if digest != _sha(src):
                raise SystemExit(f"REFUSED: the copy of {src} does not have its digest")
            files[rel] = {"sha256": digest, "bytes": os.path.getsize(dst), "origin": origin}
        for label, years in (("original campaign, retained evidence", original), ("extension campaign", later)):
            for year, (run, row) in sorted(years.items()):
                place(os.path.join(run, "tracker_port.mat"), f"tracks/{DATASET}_{year}_tracks.mat", label)
                place(os.path.join(run, f"tracking_{DATASET}_{year}.json"), f"records/tracking_{DATASET}_{year}.json", label)
                rows[str(year)] = {**row, "campaign": label}
        for src in (base_manifest, extension_manifest, calibration):
            place(src, f"protocol/{os.path.basename(src)}", "protocol declaration")
        script = {"build_era5_record_product.py": _sha(os.path.abspath(__file__)),
                  "season_metrics.py": _sha(S.__file__), "check_extension_records.py": _sha(C.__file__)}
        season = {"product": PRODUCT_ID, "version": VERSION, "generated_by": "scripts/build_era5_record_product.py",
                  "script_sha256": script, "git_head_at_build": X.repository_head(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                  "definitions": {"season": f"first observation in months {list(S.SEASON_MONTHS)} of the year",
                                  "africa_origin": f"the archive's source-region rule, region {S.RECORD_REGION}, as in season_metrics.py",
                                  "distinct_waves": f"connected components sharing at least {S.WAVE_SHARED_STEPS} timesteps at mean "
                                                    f"separation at most {S.WAVE_SEPARATION_DEG} deg",
                                  "lifetime": "observations recorded, not days",
                                  "lifetime_percentiles": "reported for every year with an Africa-origin track; the retained "
                                                          "comparison summary reports them only where its sparse rule allows, "
                                                          "which held for every year 1979 to 2010",
                                  "bands": "ten-degree bands of genesis longitude, [lo, hi)",
                                  "rows": "per year only, and no statistic across years is computed here"},
                  "region_polygons_sha256": region_digests,
                  "original_years_check": f"the rows for {min(original)} to {max(original)} were recomputed from the retained "
                                          f"tracks and equal the ERA5 column of {os.path.basename(summary_path)} in every field",
                  "years": rows}
        manifest = {"product": PRODUCT_ID, "version": VERSION, "generated_by": "scripts/build_era5_record_product.py",
                    "script_sha256": script, "git_head_at_build": season["git_head_at_build"],
                    "coverage": {"dataset": "ERA5 700 hPa", "years": [min(original), max(later)],
                                 "original_years": [min(original), max(original)], "extension_years": [t0, t1]},
                    "calibration": {"climatology_and_threshold_years": [min(original), max(original)],
                                    "note": "the later years are tracked under the earlier years' climatology and thresholds, held fixed"},
                    "base_manifest_sha256": _sha(base_manifest), "extension_manifest_sha256": report["manifest_sha256"],
                    "calibration_sha256": report["calibration_sha256"],
                    "extension_check": {k: report[k] for k in ("script_sha256", "passed", "retained_climatology_mapping_sha256")},
                    "files": files}
        X.publish_json(os.path.join(stage, "season_summary_africa_jjas.json"), season, exclusive=True)
        X.publish_json(os.path.join(stage, "MANIFEST.json"), manifest, exclusive=True)
        X.publish_json(os.path.join(stage, MARKER), {name: _sha(os.path.join(stage, name)) for _, name in ARTIFACT_FILES},
                       exclusive=True)
        os.rename(stage, out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return out


def verify_product(out):
    """Why `out` is not a complete product, or an empty list: the completion marker names
    the manifest and season summary by digest, every file the manifest lists has its
    digest, and nothing else is there."""
    marker = os.path.join(out, MARKER)
    if not os.path.exists(marker):
        return [f"{out} has no completion marker"]
    try:
        named = json.load(open(marker))
    except ValueError as exc:
        return [f"the completion marker is unreadable: {exc}"]
    required = {name for _, name in ARTIFACT_FILES}
    if not isinstance(named, dict) or set(named) != required:
        return [f"the completion marker names {sorted(named) if isinstance(named, dict) else named!r}, not exactly {sorted(required)}"]
    problems = [f"{name} does not have the digest the completion marker names"
                for name, digest in named.items()
                if not os.path.exists(os.path.join(out, name)) or _sha(os.path.join(out, name)) != digest]
    if problems:
        return problems
    listed = json.load(open(os.path.join(out, "MANIFEST.json")))["files"]
    for rel, entry in listed.items():
        path = os.path.join(out, rel)
        if not os.path.exists(path) or _sha(path) != entry["sha256"]:
            problems.append(f"{rel} does not have the digest the manifest lists")
    present = {os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs}
    extra = sorted(present - set(listed) - {name for _, name in ARTIFACT_FILES} - {MARKER})
    problems += [f"{rel} is in the product but not in its manifest" for rel in extra]
    return problems


def _ancestry(dir_fd):
    """The (device, inode) of the directory open at `dir_fd` and of each of its ancestors up
    to the root, read through descriptors, so no path is resolved again."""
    ids, cur = [], os.dup(dir_fd)
    try:
        while True:
            st = os.fstat(cur)
            ids.append((st.st_dev, st.st_ino))
            parent = os.open("..", os.O_RDONLY | os.O_DIRECTORY, dir_fd=cur)
            os.close(cur)
            cur = parent
            pst = os.fstat(cur)
            if (pst.st_dev, pst.st_ino) == ids[-1]:
                return ids
    finally:
        os.close(cur)


def publish_bytes(dir_fd, name, blob):
    """Write bytes under a private temporary name in the directory open at `dir_fd` and
    link them into place there, which the filesystem refuses when `name` exists."""
    tmp = f".publish-{os.getpid()}-{os.urandom(8).hex()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        os.link(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
    finally:
        os.unlink(tmp, dir_fd=dir_fd)


def export_artifacts(out, artifacts, campaigns=(RETAINED,)):
    """Copy the complete product's manifest and season summary to the artifacts directory.
    Safe to repeat: a copy already present must be byte-identical, and one missing is made.
    The copies are named by the version the product itself declares, which its manifest and
    season summary must agree on, so a product built by an earlier builder is exported under
    its own version.
    THE DIRECTORY IS OPENED ONCE and every read and write goes through that descriptor, and
    the opened directory's own ancestry, read through descriptors, must contain neither
    campaign, so replacing the path after the check cannot redirect the writes. Returns the
    artifact paths."""
    refuse_destinations_inside([artifacts], campaigns)
    problems = verify_product(out)
    if problems:
        raise SystemExit("REFUSED: the product is not complete: " + "; ".join(problems[:4]))
    # EACH EXPORTED FILE IS READ ONCE. Those bytes must be the ones the completion marker
    # names, the version is read from them, and they are what is published.
    with open(os.path.join(out, MARKER)) as fh:
        marker = json.load(fh)
    blobs = {}
    for _, inside in ARTIFACT_FILES:
        with open(os.path.join(out, inside), "rb") as fh:
            blobs[inside] = fh.read()
        if hashlib.sha256(blobs[inside]).hexdigest() != marker.get(inside):
            raise SystemExit(f"REFUSED: {inside} changed after the product was verified")
    _product, version = product_identity(blobs)
    os.makedirs(artifacts, exist_ok=True)
    names = artifact_names(artifacts, version)
    dir_fd = os.open(artifacts, os.O_RDONLY | os.O_DIRECTORY)
    try:
        roots = {(st.st_dev, st.st_ino) for st in (os.stat(c) for c in campaigns if os.path.exists(c))}
        if roots & set(_ancestry(dir_fd)):
            raise SystemExit(f"REFUSED: the directory opened as {artifacts} lies inside a campaign, which is never written to")
        for key, inside in ARTIFACT_FILES:
            blob = blobs[inside]
            name = os.path.basename(names[key])
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
            except FileNotFoundError:
                publish_bytes(dir_fd, name, blob)
                continue
            with os.fdopen(fd, "rb") as fh:
                if fh.read() != blob:
                    raise SystemExit(f"REFUSED: {names[key]} exists and is not the product's {inside}, and artifacts are never overwritten")
    finally:
        os.close(dir_fd)
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the product directory, which must not exist unless --artifacts-only")
    ap.add_argument("--artifacts", required=True, help="where copies of the product manifest and season summary are published")
    ap.add_argument("--extension-runs", required=True)
    ap.add_argument("--inputs-dir", required=True, help="the directory holding the extension years' input files")
    ap.add_argument("--regions-dir", default="data/aewc_v2_pilot/v1_src")
    ap.add_argument("--artifacts-only", action="store_true", help="export the artifacts of an existing, complete product")
    args = ap.parse_args(argv)
    campaigns = [RETAINED, args.extension_runs]
    refuse_destinations_inside([args.out, args.artifacts], campaigns)
    out = args.out if args.artifacts_only else build(args.out, args.extension_runs, args.inputs_dir, args.regions_dir)
    names = export_artifacts(out, args.artifacts, campaigns)
    season = json.load(open(os.path.join(out, "season_summary_africa_jjas.json")))
    for y, row in season["years"].items():
        print(f"{y}: {row['africa_origin']:4d} Africa-origin JJAS tracks, {row['season_whole_domain']:4d} whole-domain JJAS, {row['campaign']}")
    print(f"product {out} is complete; artifacts {names['manifest']}, {names['season_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
