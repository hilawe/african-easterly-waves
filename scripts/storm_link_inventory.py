#!/usr/bin/env python3
"""The annual storm-link inventory: per year, the validated genesis events by event basin,
the unresolved storm keys, and the retained tracks reported BESIDE them, with no rate.

WHY NO RATE. Storm keys and tracks are different units. The 512 keys are the tagged
storms of the population, resolved by the join to 499 validated links and 13 unresolved
keys; the 3,932 retained tracks are the waves the filter kept. Dividing one by the other
would describe linked storms per retained track, not the fraction of waves that
developed, because an untagged track is not a verified nondeveloping wave, and because a
storm on a shared tail needs an explicit storm-to-track attribution rule before it can
be counted against tracks. The inventory therefore reports the two units side by side
and leaves any development probability to a later, separately declared definition.

INPUTS, both committed artifacts: the joined link artifact (one resolution per key with
its event basin) and the Atlantic filter summary (retained tracks and storm keys per
year). The script REFUSES a pair whose years differ, a year whose key count differs
between the two, and per-year sums that do not reproduce the joined artifact's own
totals, so the table can never disagree with its sources.

    .venv/bin/python scripts/storm_link_inventory.py \\
        --joined docs/aewc_v2/artifacts/qtrack_storm_link_joined_2026-09-17.json \\
        --filter-summary docs/aewc_v2/artifacts/qtrack_atlantic_filter_summary.json \\
        --out <artifact.json> --report <report.md>
"""
import argparse
import collections
import hashlib
import json
import os
import sys

BASIN_COLUMNS = ("NA", "EP")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def inventory(joined, filter_summary):
    """Per-year rows and totals; raises on any disagreement between the two sources."""
    per_year = collections.defaultdict(lambda: {"storm_keys": 0, "unresolved_keys": 0,
                                                **{f"events_{b}": 0 for b in BASIN_COLUMNS}})
    other_basins = collections.Counter()
    for key, res in joined["resolutions"].items():
        year = key.split("|")[0]
        row = per_year[year]
        row["storm_keys"] += 1
        if res["state"] == "validated":
            basin = res["event_basin"]
            if basin in BASIN_COLUMNS:
                row[f"events_{basin}"] += 1
            else:
                other_basins[basin] += 1
        elif res["state"] == "unresolved":
            row["unresolved_keys"] += 1
        else:
            raise ValueError(f"{key} has an unknown state {res['state']!r}")
    if other_basins:
        raise ValueError(f"validated keys with an event basin outside {BASIN_COLUMNS}: "
                         f"{dict(other_basins)}")
    filt_years = filter_summary["years"]
    if set(per_year) != set(filt_years):
        raise ValueError(f"years differ: joined {sorted(set(per_year) - set(filt_years))} "
                         f"only, filter {sorted(set(filt_years) - set(per_year))} only")
    rows = {}
    for year in sorted(per_year):
        f = filt_years[year]
        if per_year[year]["storm_keys"] != f["n_distinct_storms_kept"]:
            raise ValueError(f"{year}: {per_year[year]['storm_keys']} keys in the join but "
                             f"{f['n_distinct_storms_kept']} storm keys in the filter summary")
        rows[year] = {"retained_tracks": f["n_kept"], **per_year[year]}
    totals = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    checks = {"storm_keys": joined["keys"],
              "storm_keys_in_filter_summary": filter_summary["totals"]["n_distinct_storms_kept"],
              "validated": joined["validated"],
              "unresolved_keys": len(joined["unresolved"]),
              "retained_tracks": filter_summary["totals"]["n_kept"]}
    validated_sum = sum(totals[f"events_{b}"] for b in BASIN_COLUMNS)
    if totals["storm_keys"] != checks["storm_keys"]:
        raise ValueError("per-year storm keys do not sum to the joined artifact's key count")
    if totals["storm_keys"] != checks["storm_keys_in_filter_summary"]:
        raise ValueError("per-year storm keys do not sum to the filter summary's storm-key "
                         "total")
    if validated_sum != checks["validated"]:
        raise ValueError("per-year events do not sum to the joined artifact's validated count")
    if totals["unresolved_keys"] != checks["unresolved_keys"]:
        raise ValueError("per-year unresolved keys do not sum to the joined artifact's list")
    if totals["retained_tracks"] != checks["retained_tracks"]:
        raise ValueError("per-year retained tracks do not sum to the filter summary's total")
    by_basin = joined.get("validated_by_event_basin", {})
    for b in BASIN_COLUMNS:
        if totals[f"events_{b}"] != by_basin.get(b, 0):
            raise ValueError(f"events in {b} do not equal the joined artifact's own count")
    reasons = collections.Counter(u.get("reason", "reason not recorded")
                                  for u in joined["unresolved"])
    return {"years": rows, "totals": totals, "reconciled_against": checks,
            "unresolved_reasons": dict(reasons)}


def render_report(inv, joined_name, filter_name):
    lines = ["# Annual storm-link inventory", "",
             "Per year, the storm keys of the QTrack Atlantic population resolved by the "
             "two-basin join to a validated genesis event in the North Atlantic (NA) or "
             "eastern Pacific (EP) best track, the keys left unresolved, and the retained "
             "tracks of the same year reported beside them. The two right-hand groups are "
             "different units, storms and waves, and no rate is formed from them here. An "
             "untagged track is not a verified nondeveloping wave, and a storm on a shared "
             "tail needs a declared storm-to-track attribution rule before it can be "
             "counted against tracks. The event basin is the basin recorded by the "
             "International Best Track Archive for Climate Stewardship (IBTrACS) at the "
             "genesis event carried on the validated link, never the wave track's own "
             "code.", "",
             f"Sources are the committed artifacts `{joined_name}` and `{filter_name}`. The "
             "generator refuses any year or total on which they disagree.", "",
             "| Year | Storm keys | NA events | EP events | Unresolved keys | Retained tracks |",
             "|---|---:|---:|---:|---:|---:|"]
    for year, r in inv["years"].items():
        lines.append(f"| {year} | {r['storm_keys']} | {r['events_NA']} | {r['events_EP']} | "
                     f"{r['unresolved_keys']} | {r['retained_tracks']} |")
    t = inv["totals"]
    lines.append(f"| All years | {t['storm_keys']} | {t['events_NA']} | {t['events_EP']} | "
                 f"{t['unresolved_keys']} | {t['retained_tracks']} |")
    # Every number and every reason below is read from the inventory, never typed: a
    # review found a first version stating fixed counts and a fixed reason that its own
    # synthetic test contradicted.
    lines += ["", f"The {t['unresolved_keys']} unresolved keys belong to the "
                  f"{t['storm_keys']}-key population. They are not counted as validated "
                  "genesis events. Their unresolved status does not establish whether the "
                  "associated tracks developed. Their recorded reasons are listed below.", ""]
    for why, n in sorted(inv["unresolved_reasons"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {n} {why}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--joined", required=True)
    ap.add_argument("--filter-summary", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    with open(args.joined) as fh:
        joined = json.load(fh)
    with open(args.filter_summary) as fh:
        filt = json.load(fh)
    try:
        inv = inventory(joined, filt)
    except ValueError as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    print("totals:", inv["totals"], flush=True)
    out = dict(inv)
    out.update({"generated_by": "scripts/storm_link_inventory.py",
                "inputs": {"joined": {"path": args.joined, "sha256": _sha256(args.joined)},
                           "filter_summary": {"path": args.filter_summary,
                                              "sha256": _sha256(args.filter_summary)}},
                "source_sha256": {"scripts/storm_link_inventory.py":
                                  _sha256(os.path.abspath(__file__))},
                "no_rate": "storm keys and retained tracks are different units; no "
                           "development rate is formed here, and one needs a declared "
                           "eligible wave unit and a storm-to-track attribution rule for "
                           "shared tails first"})
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}", flush=True)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(render_report(inv, os.path.basename(args.joined),
                                   os.path.basename(args.filter_summary)))
        print(f"report written to {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
