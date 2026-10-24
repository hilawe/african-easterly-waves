#!/usr/bin/env python3
"""The builder of the eastern westward-history subset, version 3
(`EASTERN_WESTWARD_HISTORIES_SUBSET_2026-10-22.md`, sections 1 to 9, 10 and 12). It recomputes the westward histories of the
declared combination (RB9 with the LN linker) from the 96 regime month records, refuses unless
the month records are the ones the stored minima-history record was made from and every
history's summary equals that record in order, compares the rebuilt tables with a retained
earlier version when one is given, and writes a table of histories, a table of their daily
members and a manifest whose population descriptors are read from records made from the same
inputs. Version 2 adds each history's position against the season's available days and a table
of which histories and days enter each descriptor, read with the records' own functions and
refused unless its counts sum to the records' and the readings recomputed from the same rows
equal them. Version 3 adds the days each descriptor read for each history, checked against the
readers' day-by-day eligibility, and the daily profile over the cohort of histories it reads on
all five days, read from its own record and refused unless that cohort is the participation
table's. Every history carries the travel status 'unresolved'. A
history is a sequence of linked minima, not an identified feature. The builder changes no saved
file.

    python3 scripts/build_eastern_westward_histories.py --out-dir DIR --travel-source PATH [--reference RETAINED_DIR] [--build-note TEXT]

The travel-status source is the document that states why every history's travel status is
unresolved, given by path relative to the root and recorded with its digest.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import eastern_joint_standing as G  # noqa: E402
import eastern_matched_profile as MP  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_wobble_climatology as W  # noqa: E402

A = W.A
P, C, Sh, L = G.P, G.C, G.H, G.L
EXT = "docs/aewc_v2/evidence/validation/eastern_pilot_2026-10-05/reports/extent"
RECORDS = f"{EXT}/climatology"
STORED = f"{EXT}/eval60_hist_archive_r1.json"
SOURCES = {"shape": f"{EXT}/eval60_shape_archive_r2.json", "coherence": f"{EXT}/eval60_coherence_archive_r1.json",
           "lobe": f"{EXT}/eval60_lobe_archive_r1.json", "joint": f"{EXT}/eval60_joint_archive.json",
           "prestart": f"{EXT}/eval60_prestart_constructions.json", "matched_profile": f"{EXT}/eval60_matched_profile_r1.json"}
XBAND = f"{EXT}/xband"
INJECT = f"{EXT}/eval60_inject_archive_r1.json"
VERSION = 3
VERSION_BUILT = {1: "2026-10-22", 2: "2026-10-23", 3: "2026-10-23"}
DESCRIPTORS = ("average_structure", "coherence_departure", "lobe_contrast", "first_day", "daily_profile", "daily_profile_matched")
PRIMARY_BANDS = ("15 to 19.5 N", "20 to 24.5 N")
PRIMARY_HISTORIES = 86
TRAVEL_STATUS = "unresolved"
HIST_COLUMNS = ["history_id", "season", "band", "primary", "first_date", "last_date", "days", "first_lon", "last_lon", "net_deg",
                "speed_deg_per_day", "westward_step_fraction", "deepest_anomaly_ms", "cut_east", "cut_west", "ends_at_or_west_of_40",
                "rule_class", "travel_status", "starts_on_first_available_day", "ends_on_last_available_day"]
PARTICIPATION_COLUMNS = ["descriptor", "history_id", "participates", "days_used", "days_excluded", "exclusion_reason", "used_offsets"]
MEMBER_COLUMNS = ["history_id", "member", "date", "day_of_season", "lon", "anomaly_ms"]
COLUMN_DEFINITIONS = {
    "history_id": "season, band code and the history's order among that season and band's westward histories",
    "season": "the June to September season",
    "band": "the latitude band of the 700 hPa band mean",
    "primary": "true in 15 to 24.5 N, the population every pooled descriptor was measured on",
    "first_date": "the date of the first member, where the linker first joined a westward minimum, not a physical onset",
    "last_date": "the date of the last member, after which no minimum was linked",
    "days": "the number of members, one per consecutive day",
    "first_lon": "the first member's longitude, degrees east",
    "last_lon": "the last member's longitude, degrees east",
    "net_deg": "last_lon minus first_lon, degrees",
    "speed_deg_per_day": "net_deg over days minus one, the stored record's speed",
    "westward_step_fraction": "the share of day-to-day steps that move west, at least 0.75 by the class rule",
    "deepest_anomaly_ms": "the most negative member anomaly against the centered nine-day mean, m/s",
    "cut_east": "the first member lies within 3 degrees of 75 E, the box's eastern edge, so the history may have begun east of the box; proximity, not demonstrated truncation",
    "cut_west": "the last member lies within 3 degrees of 20 E, the box's western edge, so the history may continue west of the box; proximity, not demonstrated truncation",
    "ends_at_or_west_of_40": "the last member lies at or west of 40 E",
    "rule_class": "westward, the minima-history record's class rule (at least 4 days, net at most -8 degrees, at least 0.75 of steps west)",
    "travel_status": "unresolved for every history, see the manifest's travel_status entry",
    "member": "the member's order in its history, from 0",
    "date": "the member's date",
    "day_of_season": "days since 1 June of the season",
    "lon": "the member's longitude, degrees east",
    "anomaly_ms": "the member's anomaly against the centered nine-day mean of the daily band mean, m/s, exact at stored precision",
    "starts_on_first_available_day": "the first member falls on the season's first day with a nine-day background, so the history may have begun before the data allow it to be seen",
    "ends_on_last_available_day": "the last member falls on the season's last day with a nine-day background, so the history may have continued after the data allow it to be seen",
    "descriptor": "the population descriptor of the manifest the row concerns",
    "participates": "whether any of the history's days enter the descriptor",
    "days_used": "the history-days the descriptor's reader read for this history",
    "days_excluded": "the history-days the reader would read that it skipped",
    "exclusion_reason": "why each skipped day, or the whole history, was left out",
    "used_offsets": ("the days after the history's first day that the descriptor read, separated by spaces; for coherence_departure the "
                     "anomaly at the first longitude on those days, which may fall after the last member, and for the others the member on "
                     "that day"),
}


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def band_code(band):
    """'15 to 19.5 N' gives '15N'."""
    return band.split(" ")[0] + "N"


def load_records(records_dir):
    files = sorted(glob.glob(os.path.join(records_dir, "eval60_regime_????_??.json.gz")))
    recs = [W.SC.load_record(p) for p in files]
    W.SC.check_archive_identity(files, recs)
    by = {}
    for r in recs:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    return by, files


def month_digests(files):
    return {os.path.basename(p): sha256_of(p) for p in files}


def check_inputs(digests, stored):
    """The month records read now are the ones the stored minima-history record was made from,
    file by file."""
    expected = stored["generated_by"]["month_records"]
    if digests != expected:
        diff = sorted(k for k in set(digests) | set(expected) if digests.get(k) != expected.get(k))
        raise SystemExit(f"REFUSED: the month records differ from those the stored history record was made from ({diff[:5]})")


def available_days(by):
    """The first and last day of each season and band on which the declared representation is
    defined, as the linker saw them through the minima-history script's own function."""
    rep, _ = W.HISTORIES
    month_of = A.month_index(); out = {}
    for b in list(by[W.YEARS[0]][W.MONTHS[0]]["bands"]):
        for y in W.YEARS:
            mins = HI.rep_minima(rep, A.season_hundredths(by[y], b)[1], month_of, W.LONS)
            ok = [d for d, m in enumerate(mins) if m is not None]
            out[(y, b)] = (ok[0], ok[-1])
    return out


def recompute(by):
    """(season, band, dates, history) for every westward history of the declared combination,
    in the stored record's order: band, then season, then the linker's order."""
    rep, linker = W.HISTORIES
    month_of = A.month_index(); out = []
    bands = list(by[W.YEARS[0]][W.MONTHS[0]]["bands"])
    for b in sorted(bands):
        for y in W.YEARS:
            dates, H = A.season_hundredths(by[y], b)
            for h in HI.link(HI.rep_minima(rep, H, month_of, W.LONS), linker):
                if HI.classify(h) == "westward":
                    out.append((y, b, dates, h))
    return out


def check_identity(items, stored):
    """Every recomputed history's summary, by the minima-history script's own summary function,
    equals the stored record's in order, and every history's members fall on consecutive days.
    A summary binds the season, band, first date, member count and end longitudes, not the
    interior longitudes or any depth, which the input digests and the retained reference bind."""
    rep, linker = W.HISTORIES
    expected = stored["tables"][HI.combo(rep, linker)]["westward_histories"]
    mine = json.loads(json.dumps([HI.summarize_westward(h, dates, y, b) for y, b, dates, h in items]))
    if len(mine) != len(expected):
        raise SystemExit(f"REFUSED: {len(mine)} recomputed westward histories, not the stored {len(expected)}")
    for k, (m, e) in enumerate(zip(mine, expected)):
        if m != e:
            raise SystemExit(f"REFUSED: recomputed history {k} differs from the stored record ({m} against {e})")
    for y, b, dates, h in items:
        if h["days"] != list(range(h["days"][0], h["days"][0] + len(h["days"]))):
            raise SystemExit(f"REFUSED: a history in {y} {b} has members on non-consecutive days")


def history_rows(items, avail):
    rows, members, seen = [], [], {}
    for y, b, dates, h in items:
        k = seen[(y, b)] = seen.get((y, b), 0) + 1
        hid = f"{y}-{band_code(b)}-{k:02d}"
        x = h["lons"]; n = len(x)
        steps = [x[i + 1] - x[i] for i in range(n - 1)]
        s = HI.summarize_westward(h, dates, y, b)
        rows.append({"history_id": hid, "season": y, "band": b, "primary": b in PRIMARY_BANDS, "first_date": dates[h["days"][0]],
                     "last_date": dates[h["days"][-1]], "days": n, "first_lon": x[0], "last_lon": x[-1], "net_deg": s["net_deg"],
                     "speed_deg_per_day": s["speed_deg_per_day"], "westward_step_fraction": sum(1 for d in steps if d < 0) / len(steps),
                     "deepest_anomaly_ms": min(h["depths"]), "cut_east": s["cut_east"], "cut_west": s["cut_west"],
                     "ends_at_or_west_of_40": s["ends_at_or_west_of_40"], "rule_class": HI.classify(h), "travel_status": TRAVEL_STATUS,
                     "starts_on_first_available_day": h["days"][0] == avail[(y, b)][0], "ends_on_last_available_day": h["days"][-1] == avail[(y, b)][1]})
        june1 = datetime.date(y, 6, 1)
        for i, (d, lon, dep) in enumerate(zip(h["days"], h["lons"], h["depths"])):
            date = dates[d]
            members.append({"history_id": hid, "member": i, "date": date,
                            "day_of_season": (datetime.date.fromisoformat(date) - june1).days, "lon": lon, "anomaly_ms": dep})
    return rows, members


def _source(name, root, digests, stored_sha):
    """A descriptor record, refused unless it was made from the same month records and the same
    stored history record as the subset."""
    path = SOURCES[name]
    with open(os.path.join(root, path), encoding="utf-8") as fh:
        rec = json.load(fh)
    g = rec["generated_by"]
    if g.get("records") != digests:
        raise SystemExit(f"REFUSED: the {name} record was made from other month records")
    if (g.get("histories") or {}).get("sha256") != stored_sha:
        raise SystemExit(f"REFUSED: the {name} record was made from another history record")
    return rec, {"path": path, "sha256": sha256_of(os.path.join(root, path))}


def same_inputs(gen, joint_rec, name):
    """The five numerical inputs a record or a load names are the joint record's, file by file,
    by the matched-profile script's own check, or the build is refused."""
    missing = [k for k in MP.INPUTS if k not in (gen or {})]
    if missing:
        raise SystemExit(f"REFUSED: the {name} names no {missing[0]} input")
    MP.check_same_inputs(gen, joint_rec)


def _population(rec_value, name):
    if rec_value != PRIMARY_HISTORIES:
        raise SystemExit(f"REFUSED: the {name} record measured {rec_value} histories, not the {PRIMARY_HISTORIES} of 15 to 24.5 N")


def descriptors(root, digests, stored_sha):
    """The pooled descriptors of the 86 histories in 15 to 24.5 N, each read from the record that
    measured it with that record's path and digest, refused when a record was made from other
    inputs or its population is not the 86 histories."""
    pop = "the 86 westward histories in 15 to 24.5 N, pooled; a property of the population, not of any one history"
    out = []
    rec, src = _source("shape", root, digests, stored_sha); p = rec["results"]["composite"]["primary"]; _population(p["histories"], "shape")
    out.append({"name": "average_structure", "value": {"center_ms": p["center"], "east_flank": p["east"], "west_flank": p["west"], "member_days": p["members"]},
                "definition": "the own-month anomaly averaged around the member-days, its center and its largest flank values east and west as fractions of the center",
                "population": pop, "source": src})
    rec, src = _source("coherence", root, digests, stored_sha); c = rec["results"]["measure"]["primary"]; _population(c["histories"], "coherence")
    out.append({"name": "coherence_departure", "value": c["departure"],
                "definition": "the own-month anomaly at the first longitude averaged over 2 to 4 days after the first day, summed, over its summed first-day value",
                "population": pop, "source": src})
    rec, src = _source("lobe", root, digests, stored_sha); l = rec["results"]["measure"]["primary"]; _population(l["histories"], "lobe")
    out.append({"name": "lobe_contrast", "value": {"carried": l["carried"], "pinned": l["pinned"], "contrast": l["contrast"], "history_days": l["samples"]},
                "definition": "on days 2 and 3, the other-season anomaly 6 degrees east of the current minimum (carried) and of the first longitude (pinned), summed, over the summed magnitude at the minimum",
                "population": pop, "source": src})
    rec, src = _source("joint", root, digests, stored_sha); j = rec["results"]["measure"]; _population(j["histories"], "joint")
    out.append({"name": "first_day", "value": {"ratio": j["first_day"], "starts": j["profile_days"][0]},
                "definition": "on the first day, the other-season anomaly 6 degrees east of the first longitude, summed, over the summed magnitude at the first longitude",
                "population": pop, "source": src})
    out.append({"name": "daily_profile", "value": {"ratio_by_day": j["profile"], "history_days_by_day": j["profile_days"]},
                "definition": "exploratory and outside every declared reading: on each of days 0 to 4, the other-season anomaly 6 degrees east of that day's member over the summed magnitude at those members, over a sample that changes by day",
                "population": pop, "source": src})
    joint_src, joint_rec = src, rec
    rec, src = _source("matched_profile", root, digests, stored_sha); m = rec["results"]
    if rec["generated_by"].get("joint_archive", {}).get("sha256") != joint_src["sha256"]:
        raise SystemExit("REFUSED: the matched-profile record was read against another joint record")
    same_inputs(rec["generated_by"], joint_rec, "matched-profile record")
    _population(m["changing"]["histories"], "matched-profile")
    if (m["changing"]["by_day"], m["changing"]["days"]) != (j["profile"], j["profile_days"]):
        raise SystemExit("REFUSED: the matched-profile record's curve over every history differs from the joint record")
    cohort = m["cohort"]
    if m["matched"]["histories"] != len(cohort) or m["matched"]["days"] != [len(cohort)] * len(G.PROFILE_DAYS):
        raise SystemExit("REFUSED: the matched-profile record's cohort is not read on every day")
    out.append({"name": "daily_profile_matched",
                "value": {"ratio_by_day": m["matched"]["by_day"], "histories": len(cohort), "numerator_by_day_ms": m["matched"]["numerator"],
                          "denominator_by_day_ms": m["matched"]["denominator"]},
                "definition": "exploratory and outside every declared reading: the daily profile's ratio on each of days 0 to 4 over a cohort held fixed "
                              "across the days, with each day's summed anomaly east of the members and summed magnitude at them; no noise reference was computed",
                "population": (f"the {len(cohort)} westward histories in 15 to 24.5 N whose daily_profile participation row has days_used "
                               f"{len(G.PROFILE_DAYS)}, pooled; selected by length and start position, so a property of that cohort, not of the "
                               f"{PRIMARY_HISTORIES}, and not of any one history"),
                "cohort": cohort, "source": src})
    rec, src = _source("prestart", root, digests, stored_sha)
    out.append({"name": "prestart", "value": None, "status": "not measured",
                "definition": "the other-season anomaly at the first longitude over the 2 days before the first day; the declared stopping rule failed in "
                              f"{len(rec['verdict']['failures'])} of its 16 checks, so the archive was not measured",
                "population": pop, "source": src})
    return out


def history_participation(hid, h, b1_rows, b0_rows, band):
    """The five descriptor rows of one history, read by the records' own functions on the
    history's rows: the shape record's composite, the coherence record's end series, the
    lobe-position record's accumulator, and the joint record's daily-profile rule."""
    x = h["lons"]; n = len(x); off = lambda lon: lon + P.OFFSET > A.LONS[-1]
    offsets = lambda ks: " ".join(str(k) for k in ks)
    comp = Sh.composite({(0, band): b1_rows}, {(0, band): [h]}, (band,))
    center = int(comp["count"][list(Sh.DX).index(0.0)])
    rows = [{"descriptor": "average_structure", "history_id": hid, "participates": center > 0, "days_used": center, "days_excluded": 0, "exclusion_reason": "",
             "used_offsets": offsets(range(n))}]
    e = C.ends(b1_rows, h)
    rows.append({"descriptor": "coherence_departure", "history_id": hid, "participates": e is not None, "days_used": 1 + len(C.LAG_DAYS) if e else 0,
                 "days_excluded": 0 if e else 1 + len(C.LAG_DAYS),
                 "exclusion_reason": "" if e else f"fewer than {C.CURVE_DAYS[-1]} days of the season after the first day or before the last day",
                 "used_offsets": offsets((0,) + tuple(C.LAG_DAYS)) if e else ""})
    acc = P.new_acc(); P.accumulate(b0_rows, h, acc)
    lobe_days, before = [], 0
    for k in P.CP_DAYS:                                        # the lobe-position reader on the history cut after day k
        a_k = P.new_acc(); P.accumulate(b0_rows, {"days": h["days"][:k + 1], "lons": x[:k + 1]}, a_k)
        lobe_days.append(a_k["samples"] - before); before = a_k["samples"]
    if before != acc["samples"]:
        raise SystemExit(f"REFUSED: {hid}'s lobe-position days differ from the reader's count")
    why = [f"day {k} beyond the last member" for k in P.CP_DAYS if k >= n]
    why += [f"day {k} with a position {P.OFFSET:g} degrees east lying east of {A.LONS[-1]:g} E" for k in P.CP_DAYS if k < n and (off(x[0]) or off(x[k]))]
    rows.append({"descriptor": "lobe_contrast", "history_id": hid, "participates": acc["samples"] > 0, "days_used": acc["samples"],
                 "days_excluded": len(P.CP_DAYS) - acc["samples"], "exclusion_reason": ", ".join(why),
                 "used_offsets": offsets(k for k in P.CP_DAYS if k < n and not off(x[0]) and not off(x[k]))})
    used = [k for k in G.PROFILE_DAYS if k < n and not off(x[k])]
    mask_days = [int(k in used) for k in G.PROFILE_DAYS]; mask_off = [int(k < n and off(x[k])) for k in G.PROFILE_DAYS]
    try:                                                       # the joint record's own reader on this one history
        pr = G.profile({(0, band): b0_rows}, {(0, band): [h]}, (band,))
        if 0 not in used or (pr["days"], pr["off_grid"]) != (mask_days, mask_off):
            raise SystemExit(f"REFUSED: {hid}'s daily-profile days differ from the joint reader's")
    except SystemExit as ex:
        if "no first day enters" not in str(ex) or 0 in used:
            raise
    rows.append({"descriptor": "first_day", "history_id": hid, "participates": 0 in used, "days_used": int(0 in used), "days_excluded": int(0 not in used),
                 "exclusion_reason": "" if 0 in used else f"the position {P.OFFSET:g} degrees east of the first longitude lies east of {A.LONS[-1]:g} E",
                 "used_offsets": offsets([0] if 0 in used else [])})
    why = [f"day {k} beyond the last member" if k >= n else f"day {k} with its position {P.OFFSET:g} degrees east lying east of {A.LONS[-1]:g} E"
           for k in G.PROFILE_DAYS if k not in used]
    rows.append({"descriptor": "daily_profile", "history_id": hid, "participates": bool(used), "days_used": len(used),
                 "days_excluded": len(G.PROFILE_DAYS) - len(used), "exclusion_reason": ", ".join(why), "used_offsets": offsets(used)})
    return rows, {"history_id": hid, "center": center, "coherence": e is not None, "lobe_samples": acc["samples"], "lobe_off_grid": acc["off_grid"],
                  "lobe_days": lobe_days, "profile_days": mask_days, "profile_off_grid": mask_off}


def expected_fields(mask):
    """The participates, days_used and days_excluded each descriptor row must carry, from a
    history's eligibility as its readers gave it."""
    nl, nc, npd = len(P.CP_DAYS), 1 + len(C.LAG_DAYS), len(G.PROFILE_DAYS)
    return {"average_structure": (mask["center"] > 0, mask["center"], 0),
            "coherence_departure": (mask["coherence"], nc if mask["coherence"] else 0, 0 if mask["coherence"] else nc),
            "lobe_contrast": (mask["lobe_samples"] > 0, mask["lobe_samples"], nl - mask["lobe_samples"]),
            "first_day": (bool(mask["profile_days"][0]), mask["profile_days"][0], 1 - mask["profile_days"][0]),
            "daily_profile": (sum(mask["profile_days"]) > 0, sum(mask["profile_days"]), npd - sum(mask["profile_days"]))}


def expected_offsets(mask):
    """The days after the first day each descriptor must list, from the readers' day-by-day
    eligibility: every member counted at the composite's center, the coherence reader's
    reference and lag days, and the lobe-position and joint readers' days."""
    pick = lambda days, flags: [k for k, f in zip(days, flags) if f]
    return {"average_structure": list(range(mask["center"])),
            "coherence_departure": [0, *C.LAG_DAYS] if mask["coherence"] else [],
            "lobe_contrast": pick(P.CP_DAYS, mask["lobe_days"]),
            "first_day": pick(G.PROFILE_DAYS[:1], mask["profile_days"][:1]),
            "daily_profile": pick(G.PROFILE_DAYS, mask["profile_days"])}


def check_rows(rows, masks):
    """Every exported row agrees with its history's eligibility, lists the days the readers
    read, a reason is given exactly when days were skipped, and each history has one row per
    descriptor, or the build is refused."""
    by = {}
    for r in rows:
        key = (r["descriptor"], r["history_id"])
        if key in by:
            raise SystemExit(f"REFUSED: two participation rows for {key}")
        by[key] = r
    for mask in masks:
        days = expected_offsets(mask)
        for d, (part, used, skipped) in expected_fields(mask).items():
            r = by.pop((d, mask["history_id"]), None)
            if r is None:
                raise SystemExit(f"REFUSED: no {d} row for {mask['history_id']}")
            if (r["participates"], r["days_used"], r["days_excluded"]) != (part, used, skipped) or bool(r["exclusion_reason"]) != (skipped > 0):
                raise SystemExit(f"REFUSED: the {d} row of {mask['history_id']} disagrees with its eligibility")
            listed = [int(t) for t in r["used_offsets"].split()]
            if listed != days[d] or len(listed) != r["days_used"]:
                raise SystemExit(f"REFUSED: the {d} row of {mask['history_id']} lists days {listed}, not the readers' {days[d]}")
    if by:
        raise SystemExit(f"REFUSED: participation rows for histories outside the population ({sorted(by)[:3]})")


def check_pooled(rows, extras, recomputed, records):
    """The per-history counts sum to the pooled counts the readers give over the same histories,
    and those pooled readings equal the stored records, or the build is refused."""
    def total(name, field):
        return sum(r[field] for r in rows if r["descriptor"] == name)
    shape, coh, lobe, prof = recomputed["shape"], recomputed["coherence"], recomputed["lobe"], recomputed["profile"]
    checks = {
        "average_structure member-days": (total("average_structure", "days_used"), shape["members"]),
        "coherence_departure histories": (sum(1 for r in rows if r["descriptor"] == "coherence_departure" and r["participates"]), coh["histories"]),
        "lobe_contrast history-days": (total("lobe_contrast", "days_used"), lobe["samples"]),
        "lobe_contrast days off the grid": (sum(e["lobe_off_grid"] for e in extras), lobe["off_grid"]),
        "daily_profile history-days by day": ([sum(e["profile_days"][k] for e in extras) for k in range(len(G.PROFILE_DAYS))], prof["days"]),
        "daily_profile days off the grid by day": ([sum(e["profile_off_grid"][k] for e in extras) for k in range(len(G.PROFILE_DAYS))], prof["off_grid"]),
        "daily_profile exported history-days": (total("daily_profile", "days_used"), sum(prof["days"])),
        "coherence_departure exported history-days": (total("coherence_departure", "days_used"), coh["histories"] * (1 + len(C.LAG_DAYS))),
        "first_day starts": (total("first_day", "days_used"), prof["days"][0]),
        "shape record": ({k: shape[k] for k in ("center", "east", "west", "members", "histories")},
                         {k: records["shape"]["results"]["composite"]["primary"][k] for k in ("center", "east", "west", "members", "histories")}),
        "coherence record": (coh["departure"], records["coherence"]["results"]["measure"]["primary"]["departure"]),
        "lobe record": ({k: lobe[k] for k in ("carried", "pinned", "contrast", "samples", "off_grid")},
                        {k: records["lobe"]["results"]["measure"]["primary"][k] for k in ("carried", "pinned", "contrast", "samples", "off_grid")}),
        "joint record": ({"profile": prof["by_day"], "profile_days": prof["days"], "profile_off_grid": prof["off_grid"]},
                         {k: records["joint"]["results"]["measure"][k] for k in ("profile", "profile_days", "profile_off_grid")}),
    }
    for name, (mine, theirs) in checks.items():
        if mine != theirs:
            raise SystemExit(f"REFUSED: {name}, {mine} against {theirs}")
    return {name: "equal" for name in checks}


def participation(items, hist, by_x, records):
    """Every primary history's descriptor rows in the subset's order, the two loads of the
    histories refused unless they agree, and the pooled checks."""
    month_of = A.month_index()
    b1 = {k: Sh.b1_rows(L.season_samples(by_x[k[0]], k[1]), month_of)[1] for k in hist}
    b0 = {}
    for b in L.BAND_LAT:
        rows = P.b0_rows({y: P.daily_hundredths(L.season_samples(by_x[y], b)) for y in W.YEARS}, month_of)
        for y in W.YEARS:
            b0[(y, b)] = rows[y]
    ids, seen = {}, {}
    for y, b, dates, h in items:
        k = seen[(y, b)] = seen.get((y, b), 0) + 1
        ids.setdefault((y, b), []).append((f"{y}-{band_code(b)}-{k:02d}", h))
    for key in set(ids) | {k for k, v in hist.items() if v}:
        if [(h["days"], h["lons"]) for _, h in ids.get(key, [])] != [(h["days"], h["lons"]) for h in hist.get(key, [])]:
            raise SystemExit(f"REFUSED: the two loads of the histories differ in {key}")
    out, extras = [], []
    for y, b, dates, h in items:
        if b not in PRIMARY_BANDS:
            continue
        hid = next(i for i, hh in ids[(y, b)] if hh is h)
        r, e = history_participation(hid, h, b1[(y, b)], b0[(y, b)], b)
        out.extend(r); extras.append(e)
    check_rows(out, extras)
    recomputed = {"shape": Sh.flank_readers(Sh.composite(b1, hist, PRIMARY_BANDS)), "coherence": C.measure(b1, hist, PRIMARY_BANDS),
                  "lobe": P.measure({b: {y: b0[(y, b)] for y in W.YEARS} for b in L.BAND_LAT}, hist, PRIMARY_BANDS),
                  "profile": G.profile(b0, hist, PRIMARY_BANDS)}
    return out, check_pooled(out, extras, recomputed, records)


def check_cohort(desc, part):
    """The matched-profile cohort is the set of histories whose daily_profile row lists every
    day, or the build is refused."""
    cohort = next(x for x in desc if x["name"] == "daily_profile_matched")["cohort"]
    full = [r["history_id"] for r in part if r["descriptor"] == "daily_profile" and r["days_used"] == len(G.PROFILE_DAYS)]
    if sorted(cohort) != sorted(full):
        raise SystemExit(f"REFUSED: the matched-profile cohort ({len(cohort)}) is not the {len(full)} histories read on every day of the daily profile")
    return "equal"


def failure_modes(stored, travel_source):
    """The known ways the subset misrepresents the flow, with the numbers read from the stored record."""
    rep, linker = W.HISTORIES
    bands = stored["tables"][HI.combo(rep, linker)]["bands"]
    at_net = {b: {"-6": v["westward_at_net"]["-6"], "-8": v["westward"], "-10": v["westward_at_net"]["-10"]} for b, v in bands.items()}
    return [
        {"mode": "a weak slow passage through a wobbling standing feature breaks into histories before and after it, so one passage may appear as two histories",
         "source": "EASTERN_MINIMA_HISTORIES_2026-10-19.md"},
        {"mode": "the centered nine-day background is undefined within four days of each season's ends, so no member falls on the first or last four days of a season",
         "source": "EASTERN_MINIMA_HISTORIES_2026-10-19.md"},
        {"mode": "the net-displacement threshold of -8 degrees shapes the population and its speeds; westward counts by band at -6, -8 and -10 degrees",
         "value": at_net, "source": STORED},
        {"mode": "the first and last members are where the linker joined and stopped, not physical onset or decay; histories within 3 degrees of the box's edges are flagged cut_east or cut_west as possibly truncated",
         "source": "EASTERN_JOINT_STANDING_2026-10-22.md"},
        {"mode": "outside 15 to 24.5 N the westward counts do not exceed the mean of the spread-matched noise ensemble under every combination, and that mean is not a false-history rate",
         "source": "EASTERN_MINIMA_HISTORIES_2026-10-19.md"},
        {"mode": "whether the histories travel as troughs with their southerly lobe or move past southerly anomaly standing near where they begin is unresolved, and the pooled descriptors do not separate them from minima that noise produces beyond five placements of one noise model",
         "source": travel_source},
    ]


def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a product file is never overwritten")


def csv_text(columns, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns); w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def member_digest(members):
    """A digest of every member's history, date, longitude and anomaly in order, the canonical
    member sequence a later build is compared with."""
    canon = json.dumps([[m["history_id"], m["date"], m["lon"], m["anomaly_ms"]] for m in members], separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def implementation():
    """The digests of the modules whose functions computed the members."""
    return {os.path.relpath(os.path.abspath(m.__file__), ROOT): sha256_of(os.path.abspath(m.__file__)) for m in (HI, A, W, W.SC)}


def check_reference(ref_dir, tables):
    """The retained earlier version, refused unless its tables match its own manifest, and every
    table it holds, restricted to its columns, equals the rebuilt table byte for byte. A later
    version may add columns and tables, and may change none of the reference's."""
    with open(os.path.join(ref_dir, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    compared = {}
    for k, spec in man["tables"].items():
        p = os.path.join(ref_dir, f"{k}.csv")
        if sha256_of(p) != spec["sha256"]:
            raise SystemExit(f"REFUSED: the reference {k}.csv does not match its manifest")
        if k not in tables or not set(spec["columns"]) <= set(tables[k][0]):
            raise SystemExit(f"REFUSED: the rebuilt {k}.csv lacks the reference's columns")
        mine = csv_text(spec["columns"], [{c: r[c] for c in spec["columns"]} for r in tables[k][1]])
        with open(p, encoding="utf-8", newline="") as fh:
            if fh.read() != mine:
                raise SystemExit(f"REFUSED: the rebuilt {k}.csv differs from the retained reference")
        compared[k] = spec["columns"]
    return {"path": ref_dir, "manifest_sha256": sha256_of(os.path.join(ref_dir, "manifest.json")), "performed": True, "result": "equal",
            "compared_columns": compared}


def write_text(path, text):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--travel-source", required=True, help="the document stating why the travel status is unresolved, relative to the root")
    ap.add_argument("--reference", default=None, help="a retained earlier version whose tables the rebuild must equal")
    ap.add_argument("--build-note", default="")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out_dir)
    paths = {k: os.path.join(out, f) for k, f in (("histories", "histories.csv"), ("members", "members.csv"), ("participation", "participation.csv"),
                                                   ("manifest", "manifest.json"))}
    refuse_existing(paths.values())
    if not os.path.isfile(os.path.join(a.root, a.travel_source)):
        raise SystemExit(f"REFUSED: the travel-status source {a.travel_source} does not exist")
    with open(os.path.join(a.root, STORED), encoding="utf-8") as fh:
        stored = json.load(fh)
    stored_sha = sha256_of(os.path.join(a.root, STORED))
    by, files = load_records(os.path.join(a.root, RECORDS))
    digests = month_digests(files)
    check_inputs(digests, stored)
    items = recompute(by)
    check_identity(items, stored)
    rows, members = history_rows(items, available_days(by))
    desc = descriptors(a.root, digests, stored_sha)
    ns = argparse.Namespace(shape=os.path.join(a.root, SOURCES["shape"]), inject=os.path.join(a.root, INJECT), records=os.path.join(a.root, RECORDS),
                            histories=os.path.join(a.root, STORED), xband=os.path.join(a.root, XBAND))
    hist, _, by_x, _, loaded = C.load_inputs(ns)
    records = {k: _source(k, a.root, digests, stored_sha)[0] for k in ("shape", "coherence", "lobe", "joint")}
    same_inputs(loaded, records["joint"], "builder's load")
    part, part_checks = participation(items, hist, by_x, records)
    part_checks["daily_profile_matched cohort"] = check_cohort(desc, part)
    tables = {"histories":(HIST_COLUMNS, rows), "members": (MEMBER_COLUMNS, members), "participation": (PARTICIPATION_COLUMNS, part)}
    texts = {k: csv_text(cols, rs) for k, (cols, rs) in tables.items()}
    ref = (check_reference(os.path.abspath(a.reference), tables) if a.reference
           else {"path": None, "performed": False, "result": "not performed, no retained version was given"})
    if ref["path"]:
        ref["path"] = os.path.relpath(ref["path"], a.root)
    os.makedirs(out, exist_ok=True)
    for k in tables:
        write_text(paths[k], texts[k])
    rel = lambda p: os.path.relpath(p, a.root)
    by_band = {b: sum(1 for r in rows if r["band"] == b) for b in sorted({r["band"] for r in rows})}
    rep, linker = W.HISTORIES
    manifest = {
        "product": f"eastern westward-history subset, version {VERSION}", "part_of": "EASTERN_DESCRIPTIVE_PRODUCT_2026-10-19.md, layer L2",
        "declared": "EASTERN_WESTWARD_HISTORIES_SUBSET_2026-10-22.md", "built": VERSION_BUILT[VERSION], "first_version_built": VERSION_BUILT[1],
        "builder": {"path": "scripts/build_eastern_westward_histories.py", "sha256": sha256_of(os.path.abspath(__file__))},
        "label": "linked minima and their positions, depths and dates, with rule-defined classes; no identity, no tracked disturbance, no physical classification",
        "definition": {"representation": rep, "linker": linker, "max_link_deg": HI.S_MAX, "floor_ms": HI.REP_SPEC[rep][1], "prominence_ms": A.PROMINENCE,
                       "class_rule": {"min_days": HI.W_MIN_LEN, "max_net_deg": HI.W_NET, "min_westward_step_fraction": HI.W_FRAC},
                       "field": "ERA5 700 hPa meridional wind, band means of four latitude bands on 20 to 75 E at 0.5 degrees, daily means of six-hourly values",
                       "seasons": f"June to September, {W.YEARS[0]} to {W.YEARS[-1]} without 2005, which is held out and unopened"},
        "build_note": a.build_note,
        "inputs": {"month_records": {"dir": RECORDS, "count": len(files), "sha256": digests},
                   "stored_histories": {"path": STORED, "sha256": stored_sha}},
        "identity_check": ("the month records equal those the stored history record was made from, file by file; every recomputed history's "
                           "summary (season, band, first date, member count, end longitudes) equals the stored record's in order; every "
                           "history's members fall on consecutive days; every descriptor record was made from the same month records and "
                           "history record; the inputs the builder loads and those the matched-profile record names equal the joint "
                           "record's, file by file. The summaries do not bind interior longitudes or depths, which the member digest and the "
                           "reference check carry"),
        "member_digest": member_digest(members), "implementation": implementation(), "reference_check": ref,
        "counts": {"histories": len(rows), "by_band": by_band, "primary": sum(1 for r in rows if r["primary"]), "members": len(members),
                   "starts_on_first_available_day": sum(1 for r in rows if r["starts_on_first_available_day"]),
                   "ends_on_last_available_day": sum(1 for r in rows if r["ends_on_last_available_day"]),
                   "participation_rows": len(part)},
        "tables": {k: {"path": rel(paths[k]), "sha256": sha256_of(paths[k]), "rows": len(rs), "columns": cols} for k, (cols, rs) in tables.items()},
        "participation_check": part_checks,
        "column_definitions": COLUMN_DEFINITIONS,
        "travel_status": {"vocabulary": [TRAVEL_STATUS], "value": TRAVEL_STATUS,
                          "meaning": "the records do not separate troughs traveling with their southerly lobe from minima moving past southerly anomaly standing near where they begin",
                          "source": {"path": a.travel_source, "sha256": sha256_of(os.path.join(a.root, a.travel_source))}},
        "descriptors": desc, "failure_modes": failure_modes(stored, a.travel_source),
        "use": "the population the westward-history records measured, listed member by member so that any later per-history study starts from the same histories",
    }
    with open(paths["manifest"], "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"{len(rows)} histories ({by_band}), {manifest['counts']['primary']} primary, {len(members)} members, {len(desc)} descriptors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
