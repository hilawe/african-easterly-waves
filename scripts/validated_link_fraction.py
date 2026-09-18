#!/usr/bin/env python3
"""The validated-link fraction over shared-tail groups: a descriptive statistic under
the declaration recorded in the project handoff, implemented after it and bound by it.

THE UNIT is a shared-tail group of retained tracks, defined within each year by the
existing grouping rule (four or more identical positions, transitive), with isolated
tracks treated as singleton groups. THIS IS A DATA GROUPING, NOT AN IDENTIFIED PHYSICAL
WAVE: shared tails do not prove that their different track histories represent one
wave, observation ownership decides which copy of a position to count and not which
wave produced a storm, and genesis does not always occur on the shared tail.

THE STATISTIC is the proportion of these groups containing at least one distinct storm
identifier with a validated best-track link. It supports a descriptive fraction of
groups with validated links. It leaves development probability and physical wave
identity unsettled, and nothing here is named a development probability.

THE RULES that complete it, each bound by a test:
  - every retained group is in the denominator (groups = retained tracks minus, for
    each shared-tail group, one less than its size);
  - groups with validated links, groups with only unresolved tags, and groups with no
    storm tags are reported separately, and the three sum to the denominator;
  - unresolved-key counts are preserved even within groups that also hold validated
    links;
  - a group may hold North Atlantic links, eastern Pacific links, or both, and is
    reported by that composition rather than forced onto one basin;
  - distinct linked storm identifiers are reported beside the group counts.

INPUTS, all committed artifacts: the filter summary (retained tracks and shared-tail
groups per year), the first link artifact (the tracks each storm key sits on), and the
joined artifact (each key's resolution and event basin). The script REFUSES, before any
row: a link file whose digest is not one the joined artifact records as its input (a
review substituted one track number in a copy of the link file and moved a storm out
of its group unseen); a filter summary and link artifact that do not cover the same
years, do not carry equal per-year population hashes, or do not record the same filter
module hash; a key set that differs between the link and joined artifacts; a validated
key whose event basin is outside the supported basins (checked per key, before any
group composition is reduced); a key whose tracks span more than one group (none does;
a rule would be needed); and any total that does not reconcile against its source.
The filter summary's PAIR count (107 over 44 years) is reconciled under its own name
and kept apart from the GROUP count (97), which is not a source total.

    .venv/bin/python scripts/validated_link_fraction.py \\
        --filter-summary docs/aewc_v2/artifacts/qtrack_atlantic_filter_summary.json \\
        --links docs/aewc_v2/artifacts/qtrack_storm_link_2026-09-17.json \\
        --joined docs/aewc_v2/artifacts/qtrack_storm_link_joined_2026-09-17.json \\
        --out <artifact.json> --report <report.md>
"""
import argparse
import collections
import hashlib
import json
import os
import sys

BASINS = ("NA", "EP")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def check_provenance(filter_summary, links, joined, links_sha256):
    """Raise unless the link artifact is the one the join resolved and it describes the
    filter summary's population."""
    inputs = joined.get("inputs") or {}
    recorded = {v.get("sha256") for v in inputs.values() if isinstance(v, dict)}
    if not recorded:
        raise ValueError("the joined artifact records no input digests")
    if links_sha256 is None or links_sha256 not in recorded:
        raise ValueError("the link file's digest is not an input the joined artifact "
                         "records, so its track membership is not the membership the "
                         "join resolved")
    if set(filter_summary["years"]) != set(links["years"]):
        raise ValueError("the filter summary and the link artifact cover different years")
    for year in sorted(filter_summary["years"]):
        hf = filter_summary["years"][year].get("input_sha256")
        hl = links["years"][year].get("input_sha256")
        if hf is None or hl is None or hf != hl:
            raise ValueError(f"population hash for {year} differs or is missing between "
                             f"the filter summary and the link artifact")
    qf = (filter_summary.get("source_sha256") or {}).get("src/aew/qtrack.py")
    ql = (links.get("source_sha256") or {}).get("src/aew/qtrack.py")
    if qf is None or ql is None or qf != ql:
        raise ValueError("the filter summary and the link artifact do not record the same "
                         "filter module hash")


def group_of(systems, index):
    """The one group a key's tracks sit in: a shared-tail group's index, or the
    singleton of its system number. Raises when the tracks span groups."""
    groups = {index.get(int(s), ("single", int(s))) for s in systems}
    if len(groups) != 1:
        raise ValueError(f"tracks {list(systems)} span more than one group: "
                         f"{sorted(map(str, groups))}")
    return groups.pop()


def fraction(filter_summary, links, joined, links_sha256=None):
    """Per-year rows and totals of the validated-link fraction; raises on any
    disagreement between the three sources or on unbound provenance."""
    check_provenance(filter_summary, links, joined, links_sha256)
    link_rows = {}
    for year, v in links["years"].items():
        for r in v["rows"]:
            link_rows[f"{year}|{r['name']}|{r['qtrack_genesis']}"] = r
    if set(link_rows) != set(joined["resolutions"]):
        raise ValueError("the link and joined artifacts hold different key sets")
    years = {}
    for year in sorted(filter_summary["years"]):
        f = filter_summary["years"][year]
        shared = [[int(s) for s in g] for g in f["shared_tail_groups"]]
        index = {s: ("shared", gi) for gi, g in enumerate(shared) for s in g}
        n_groups = f["n_kept"] - sum(len(g) - 1 for g in shared)
        per_group = collections.defaultdict(lambda: {"validated": 0, "unresolved": 0,
                                                     "basins": set(), "sids": set()})
        for key, res in joined["resolutions"].items():
            if not key.startswith(year + "|"):
                continue
            g = group_of(link_rows[key]["systems"], index)
            if res["state"] == "validated":
                if res["event_basin"] not in BASINS:
                    raise ValueError(f"{key}: event basin {res['event_basin']!r} is "
                                     f"outside {BASINS}")
                per_group[g]["validated"] += 1
                per_group[g]["basins"].add(res["event_basin"])
                per_group[g]["sids"].add(res["sid"])
            elif res["state"] == "unresolved":
                per_group[g]["unresolved"] += 1
            else:
                raise ValueError(f"{key} has an unknown state {res['state']!r}")
        linked = [g for g in per_group.values() if g["validated"]]
        only_unresolved = [g for g in per_group.values() if not g["validated"]]
        composition = collections.Counter()
        for g in linked:
            b = g["basins"]
            if b == {"NA"}:
                composition["NA"] += 1
            elif b == {"EP"}:
                composition["EP"] += 1
            elif b == {"NA", "EP"}:
                composition["both"] += 1
            else:
                raise ValueError(f"{year}: a linked group holds the basin set "
                                 f"{sorted(b)}, not one of the three supported sets")
        sids = set().union(*(g["sids"] for g in linked)) if linked else set()
        n_linked, n_only = len(linked), len(only_unresolved)
        years[year] = {
            "retained_tracks": f["n_kept"],
            "shared_tail_groups": len(shared),
            "shared_tail_pairs": f["n_shared_tail_pairs"],
            "groups": n_groups,
            "groups_with_validated_links": n_linked,
            "groups_with_only_unresolved_tags": n_only,
            "groups_with_no_storm_tags": n_groups - n_linked - n_only,
            "linked_groups_by_basin": {b: composition.get(b, 0) for b in (*BASINS, "both")},
            "validated_keys": sum(g["validated"] for g in linked),
            "unresolved_keys": sum(g["unresolved"] for g in per_group.values()),
            "unresolved_keys_in_linked_groups": sum(g["unresolved"] for g in linked),
            "distinct_linked_storms": len(sids),
            "validated_link_fraction": n_linked / n_groups if n_groups else None}
        if years[year]["groups_with_no_storm_tags"] < 0:
            raise ValueError(f"{year}: more tagged groups than groups")
    count_keys = [k for k in next(iter(years.values())) if k not in
                  ("validated_link_fraction", "linked_groups_by_basin")]
    totals = {k: sum(r[k] for r in years.values()) for k in count_keys}
    totals["linked_groups_by_basin"] = {b: sum(r["linked_groups_by_basin"][b]
                                               for r in years.values())
                                        for b in (*BASINS, "both")}
    totals["validated_link_fraction"] = (totals["groups_with_validated_links"]
                                         / totals["groups"])
    # Reconciled totals only; the group count (97 over 44 years) is derived from the
    # group lists and has no source total, so it is NOT here, and the filter summary's
    # PAIR count (107) is reconciled under its own name and never read as groups.
    checks = {"retained_tracks": filter_summary["totals"]["n_kept"],
              "shared_tail_pairs": filter_summary["totals"]["n_shared_tail_pairs"],
              "validated_keys": joined["validated"],
              "unresolved_keys": len(joined["unresolved"]),
              "distinct_linked_storms": joined["distinct_storms_validated"]}
    for k, expected in checks.items():
        if totals[k] != expected:
            raise ValueError(f"per-year {k} sum to {totals[k]}, the source says {expected}")
    if totals["groups_with_validated_links"] + totals["groups_with_only_unresolved_tags"] \
            + totals["groups_with_no_storm_tags"] != totals["groups"]:
        raise ValueError("the three group categories do not sum to the groups")
    return {"years": years, "totals": totals, "reconciled_against": checks}


def render_report(res, names):
    t = res["totals"]
    lines = ["# Validated-link fraction over shared-tail groups", "",
             "A descriptive statistic under the declaration recorded in the project "
             "handoff. The unit is a shared-tail group of retained tracks, defined within "
             "each year by the existing grouping rule, with isolated tracks as singleton "
             "groups. It is a data grouping and not an identified physical wave. The "
             "fraction is the proportion of groups holding at least one distinct storm "
             "identifier with a validated best-track link. Development and physical wave "
             "identity are left unsettled by it.", "",
             "Every retained group is in the denominator. Groups with validated links, "
             "groups whose only tags are unresolved keys, and groups with no storm tags "
             "are reported separately and sum to the groups. Unresolved keys are counted "
             "even inside linked groups. A linked group is reported by the basins of its "
             "links, North Atlantic (NA), eastern Pacific (EP), or both, and no single "
             "basin is forced onto a group holding different storms. Distinct linked "
             "storm identifiers are reported beside the group counts.", "",
             f"Sources are the committed artifacts `{names[0]}`, `{names[1]}` and "
             f"`{names[2]}`. The generator refuses any total on which they disagree.", "",
             "| Year | Tracks | Groups | Linked groups | NA only | EP only | Both | "
             "Only unresolved | No tags | Unresolved keys | Distinct storms | Fraction |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(label, r):
        b = r["linked_groups_by_basin"]
        return (f"| {label} | {r['retained_tracks']} | {r['groups']} | "
                f"{r['groups_with_validated_links']} | {b['NA']} | {b['EP']} | {b['both']} | "
                f"{r['groups_with_only_unresolved_tags']} | {r['groups_with_no_storm_tags']} | "
                f"{r['unresolved_keys']} | {r['distinct_linked_storms']} | "
                f"{r['validated_link_fraction']:.3f} |")
    for year, r in res["years"].items():
        lines.append(row(year, r))
    lines.append(row("All years", t))
    lines += ["", f"Over all years, {t['groups_with_validated_links']} of {t['groups']} "
                  f"groups hold a validated link, a fraction of "
                  f"{t['validated_link_fraction']:.3f}. The {t['unresolved_keys']} "
                  f"unresolved keys are counted in full, "
                  f"{t['unresolved_keys_in_linked_groups']} of them inside groups that "
                  "also hold a validated link. The fraction describes groups of stored "
                  "tracks with validated links and nothing more.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--filter-summary", required=True)
    ap.add_argument("--links", required=True)
    ap.add_argument("--joined", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    loaded = []
    for path in (args.filter_summary, args.links, args.joined):
        with open(path) as fh:
            loaded.append(json.load(fh))
    try:
        res = fraction(*loaded, links_sha256=_sha256(args.links))
    except ValueError as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    print("totals:", {k: v for k, v in res["totals"].items()}, flush=True)
    out = dict(res)
    out.update({"generated_by": "scripts/validated_link_fraction.py",
                "inputs": {"filter_summary": {"path": args.filter_summary,
                                              "sha256": _sha256(args.filter_summary)},
                           "links": {"path": args.links, "sha256": _sha256(args.links)},
                           "joined": {"path": args.joined, "sha256": _sha256(args.joined)}},
                "source_sha256": {"scripts/validated_link_fraction.py":
                                  _sha256(os.path.abspath(__file__))},
                "unit": "shared-tail group of retained tracks, a data grouping and not an "
                        "identified physical wave",
                "what_this_does_not_establish": "development probability or physical wave "
                                                "identity"})
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}", flush=True)
    if args.report:
        names = tuple(os.path.basename(p) for p in (args.filter_summary, args.links,
                                                    args.joined))
        with open(args.report, "w") as fh:
            fh.write(render_report(res, names))
        print(f"report written to {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
