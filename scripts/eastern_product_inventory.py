#!/usr/bin/env python3
"""The inventory of the eastern descriptive product (`EASTERN_DESCRIPTIVE_PRODUCT_2026-10-19.md`):
the current version of every record the product's layers rest on, grouped by layer, with
each file's size and digest and one combined digest per group, read from the files
themselves. A group whose patterns match no file is refused, and earlier versions of a
corrected record are left out because the current version is named explicitly.

    python3 scripts/eastern_product_inventory.py --root . --out-json PATH --out-md PATH
"""

import argparse
import glob
import hashlib
import json
import os

EXTENT = "docs/aewc_v2/evidence/validation/eastern_pilot_2026-10-05/reports/extent"
GROUPS = (
    ("L1", "month records of the 700 hPa band means, 24 seasons", "EASTERN_STANDING_CLIMATOLOGY_2026-10-18.md", (f"{EXTENT}/climatology/eval60_regime_????_??.json.gz",)),
    ("L1", "standing climatology", "EASTERN_STANDING_CLIMATOLOGY_2026-10-18.md", (f"{EXTENT}/eval60_clim_*",)),
    ("L1", "standing-element flag", "EASTERN_STANDING_FLAG_2026-10-18.md", (f"{EXTENT}/eval60_flag_*",)),
    ("L2", "characterization months and composite", "EASTERN_REGIME_CHARACTERIZATION_2026-10-18.md",
     (f"{EXTENT}/eval60_regime_????_??.json", f"{EXTENT}/eval60_regime_????_??_*.png", f"{EXTENT}/eval60_regime_composite.*", f"{EXTENT}/eval60_regime_month_draw.json", f"{EXTENT}/eval60_regime_summary.md")),
    ("L2", "anchored transient ledger", "EASTERN_T1_LEDGER_2026-10-18.md", (f"{EXTENT}/eval60_t1_*",)),
    ("L2", "candidate positions, ten months", "EASTERN_CANDIDATE_POSITIONS_2026-10-18.md", (f"{EXTENT}/eval60_cand_*",)),
    ("L2", "anomaly minima", "EASTERN_BACKGROUND_REMOVAL_2026-10-19.md",
     (f"{EXTENT}/eval60_anom_constructions_r1.*", f"{EXTENT}/eval60_anom_archive.json", f"{EXTENT}/eval60_anom_tables.md", f"{EXTENT}/eval60_anom_recurrence.png")),
    ("L2", "minima histories", "EASTERN_MINIMA_HISTORIES_2026-10-19.md", (f"{EXTENT}/eval60_hist_*_r1.*",)),
    ("L2", "westward-history subset, version 3", "EASTERN_WESTWARD_HISTORIES_SUBSET_2026-10-22.md", ("docs/aewc_v2/eastern_westward_histories_v3_r2/*",)),
    ("L2", "westward histories, lag regression", "EASTERN_LAG_REGRESSION_2026-10-21.md", (f"{EXTENT}/eval60_lagreg_*_r2.*",)),
    ("L2", "westward histories, injection recovery", "EASTERN_INJECTION_RECOVERY_2026-10-21.md", (f"{EXTENT}/eval60_inject_*_r1.*",)),
    ("L2", "westward histories, average shape", "EASTERN_HISTORY_SHAPE_2026-10-21.md", (f"{EXTENT}/eval60_shape_*_r2.*",)),
    ("L2", "westward histories, coherence", "EASTERN_HISTORY_COHERENCE_2026-10-22.md", (f"{EXTENT}/eval60_coherence_*_r1.*",)),
    ("L2", "westward histories, lobe position", "EASTERN_LOBE_POSITION_2026-10-22.md",
     (f"{EXTENT}/eval60_lobe_constructions_r1.*", f"{EXTENT}/eval60_lobe_archive_r1.json", f"{EXTENT}/eval60_lobe_tables_r1.md")),
    ("L2", "westward histories, standing lobes read together", "EASTERN_JOINT_STANDING_2026-10-22.md", (f"{EXTENT}/eval60_joint_*",)),
    ("L2", "westward histories, before the first day", "EASTERN_PRESTART_2026-10-22.md", (f"{EXTENT}/eval60_prestart_*",)),
    ("L2", "westward histories, daily profile over a fixed cohort", "EASTERN_MATCHED_PROFILE_2026-10-23.md", (f"{EXTENT}/eval60_matched_profile_r1.*",)),
    ("L2", "westward histories, fixed-cohort profile against matched noise", "EASTERN_MATCHED_NOISE_2026-10-23.md", (f"{EXTENT}/eval60_matched_noise_r4.*",)),
    ("L2", "westward histories, a known rise against matched noise", "EASTERN_KNOWN_RISE_2026-10-24.md",
     (f"{EXTENT}/eval60_known_rise_clean_g2_m*_r1.*", f"{EXTENT}/eval60_known_rise_noise*_r2.*", f"{EXTENT}/eval60_known_rise_map_r2.*",
      f"{EXTENT}/eval60_known_rise_diagnostics_r1.*")),
    ("L3", "descriptive layer, version 1", "EASTERN_DESCRIPTIVE_LAYER_2026-10-17.md", ("docs/aewc_v2/eastern_descriptive_layer_v1/*",)),
    ("L4", "band means extended to 10 W", "EASTERN_CROSSING_TEST_2026-10-19.md", (f"{EXTENT}/xband/eval60_xband_????_??.json.gz",)),
    ("L4", "crossing test", "EASTERN_CROSSING_TEST_2026-10-19.md", (f"{EXTENT}/eval60_cross_constructions.*", f"{EXTENT}/eval60_cross_*_r1.*")),
    ("L4", "QTrack look-back", "EASTERN_QTRACK_LOOKBACK_2026-10-19.md",
     (f"{EXTENT}/eval60_lookback_constructions_r2.*", f"{EXTENT}/eval60_lookback_archive_r1.json", f"{EXTENT}/eval60_lookback_tables_r1.md",
      f"{EXTENT}/eval60_lookback_composite_r1.png", f"{EXTENT}/eval60_lookback_maxima_posthoc_r1.md", f"{EXTENT}/eval60_lookback_exact_check.md")),
    ("L4", "two-sided look-back", "EASTERN_TWO_SIDED_LOOKBACK_2026-10-19.md", (f"{EXTENT}/eval60_twosided_*_r1.*",)),
)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_files(root, patterns):
    """The sorted, de-duplicated files a group's patterns match under root, as paths
    relative to root, refused when a pattern matches nothing."""
    out = set()
    for pat in patterns:
        hits = [p for p in glob.glob(os.path.join(root, pat)) if os.path.isfile(p)]
        if not hits:
            raise SystemExit(f"REFUSED: no file matches {pat}")
        out.update(os.path.relpath(p, root) for p in hits)
    return sorted(out)


def inventory(root, groups=GROUPS):
    rows = []
    for layer, label, record, patterns in groups:
        files = [{"path": f, "bytes": os.path.getsize(os.path.join(root, f)), "sha256": sha256_of(os.path.join(root, f))} for f in group_files(root, patterns)]
        combined = hashlib.sha256("".join(f"{f['path']}\t{f['sha256']}\n" for f in files).encode()).hexdigest()
        rows.append({"layer": layer, "group": label, "record": f"docs/aewc_v2/{record}", "files": len(files), "bytes": sum(f["bytes"] for f in files), "sha256": combined, "contents": files})
    return rows


def inventory_text(rows):
    L = ["# eastern descriptive product, inventory", "", "built by scripts/eastern_product_inventory.py from the files themselves, current versions only, each group's digest over its sorted path and file digests", "",
         "| layer | group | record | files | bytes | combined sha256, first 16 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['layer']} | {r['group']} | {os.path.basename(r['record'])} | {r['files']} | {r['bytes']:,} | {r['sha256'][:16]} |")
    L += ["", f"total {sum(r['files'] for r in rows)} files, {sum(r['bytes'] for r in rows):,} bytes"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args(argv)
    for p in (a.out_json, a.out_md):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    rows = inventory(a.root)
    with open(a.out_json, "w", encoding="utf-8") as fh:
        json.dump({"groups": rows}, fh, indent=1)
    text = inventory_text(rows)
    with open(a.out_md, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
