#!/usr/bin/env python3
"""Write the Atlantic, repeat-free subset of QTrack v2 (ERA5_WITH_EPAC), year by year.

The rules live in `aew.qtrack` and are decisions recorded there: Atlantic means any step
in basin codes 2, 6 or 7 with a first valid code that is not Pacific (an inferred key,
see that module), and repeats are systems sharing at least `--min-shared` identical
positions, the longest of each cluster kept along with the longest bearer of each storm
name the survivor lacks, listed for adjudication.
Every input and the module are hashed into the summary, which also carries the count of
repeats that WOULD be removed at thresholds 2, 4 and 8, so the chosen threshold is a
legible choice rather than a hidden one. Output files keep the published schema and the
original system numbers.

    .venv/bin/python scripts/filter_qtrack_atlantic.py \\
        --directory data/aewc_v2_pilot/ERA5_WITH_EPAC \\
        --out-directory data/aewc_v2_pilot/ERA5_ATLANTIC \\
        --summary <summary.json>
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew import qtrack as Q  # noqa: E402

SENSITIVITY_THRESHOLDS = (2, 4, 8)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--directory", default="data/aewc_v2_pilot/ERA5_WITH_EPAC")
    ap.add_argument("--out-directory", default="data/aewc_v2_pilot/ERA5_ATLANTIC")
    ap.add_argument("--min-shared", type=int, default=Q.DEFAULT_MIN_SHARED)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.directory, "*.nc")))
    if not files:
        print(f"REFUSED: no .nc files under {args.directory}", flush=True)
        return 2
    os.makedirs(args.out_directory, exist_ok=True)
    years, totals = {}, {}
    for path in files:
        m = re.search(r"(\d{4})\.nc$", os.path.basename(path))
        if not m:
            print(f"REFUSED: cannot read a year from {os.path.basename(path)}", flush=True)
            return 2
        year = m.group(1)
        data = Q.read_year(path)
        r = Q.filter_year(data, args.min_shared)
        # sensitivity under BOTH rules, so the artifact says what each number counts:
        # the systems the rule removes (retention included) and the repeat clusters
        # themselves, which retention does not alter
        sens = {}
        for k in SENSITIVITY_THRESHOLDS:
            rk = Q.filter_year(data, k)
            sens[str(k)] = {"systems_removed": rk["n_duplicates_removed"],
                            "repeat_clusters": sum(1 for g in rk["clusters"] if len(g) > 1)}
        dst = os.path.join(args.out_directory, f"ERA5_AEW_tracks_atlantic_{year}.nc")
        tmp = dst + ".partial"
        Q.write_subset(path, tmp, r["keep"], args.min_shared)
        os.replace(tmp, dst)
        years[year] = {k: r[k] for k in ("n_systems", "n_atlantic",
                                          "n_duplicates_removed", "n_kept",
                                          "n_developers_kept",
                                          "n_developers_removed_as_duplicate",
                                          "n_retained_for_name")}
        years[year]["duplicates_removed_at_threshold"] = sens
        # SYSTEM NUMBERS, the published one-based coordinate, never array indices: a
        # review found the adjudication list pointing a person at the wrong system
        sysno = lambda i: int(data["system"][i])  # noqa: E731
        years[year]["repeat_clusters"] = [[sysno(i) for i in g]
                                          for g in r["clusters"] if len(g) > 1]
        # the clusters a person still has to adjudicate: a developer kept alongside
        # its positional twin because dropping it would drop a storm name
        years[year]["retained_for_name"] = [
            {"cluster": [sysno(i) for i in g],
             "names": {str(sysno(i)): str(data["name"][i]) for i in g}}
            for g in r["clusters"] if len(g) > 1
            and any(r["retained_for_name"][i] for i in g)]
        years[year]["input_sha256"] = _sha256(path)
        years[year]["output_sha256"] = _sha256(dst)
        for k, v in years[year].items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
        print(f"  {year}: {r['n_systems']:4d} systems, {r['n_atlantic']:4d} Atlantic, "
              f"{r['n_duplicates_removed']:3d} repeats removed, {r['n_kept']:4d} kept "
              f"({r['n_developers_kept']} developers, {r['n_retained_for_name']} "
              f"retained for a name)", flush=True)
    summary = {
        "generated_by": "scripts/filter_qtrack_atlantic.py",
        "rules": {"atlantic": "any valid step with basin_des in " + str(sorted(Q.ATLANTIC_CODES))
                              + " and first valid code not in " + str(sorted(Q.PACIFIC_CODES)),
                  "identifiers": "every system in this summary is the published one-based "
                                 "`system` coordinate, not an array index",
                  "basin_key": {str(k): v for k, v in Q.BASIN_KEY.items()},
                  "basin_key_status": "inferred from the tracks "
                                      "(docs/aewc_v2/artifacts/qtrack_basin_codes.json), "
                                      "awaiting the authors' confirmation",
                  "repeats": f"systems sharing >= {args.min_shared} identical "
                             "(lon, lat) positions, clustered transitively among Atlantic "
                             "systems, longest kept, ties to the lower index, plus the "
                             "longest bearer of each storm name the survivor does not "
                             "carry (listed per year under retained_for_name for "
                             "adjudication)",
                  "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
                  "sensitivity_fields": "per year, duplicates_removed_at_threshold[k] holds "
                                        "systems_removed under the full rule and "
                                        "repeat_clusters, the cluster count, which "
                                        "retention does not change"},
        "totals": totals,
        "years": years,
        "source_sha256": {"src/aew/qtrack.py": _sha256(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "src", "aew", "qtrack.py")),
            "scripts/filter_qtrack_atlantic.py": _sha256(os.path.abspath(__file__))},
        "what_this_does_not_settle": "whether the basin key is the authors' own, and "
                                     "whether a shared-position threshold of one day is "
                                     "the right definition of a repeat. Both are recorded "
                                     "decisions."}
    print(f"totals: {totals}", flush=True)
    if args.summary:
        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
        print(f"written to {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
