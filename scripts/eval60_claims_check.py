#!/usr/bin/env python3
"""The 60 E evaluation's record-claims check. Every numerical and coverage-class claim the
evaluation document makes about its retained records is stated once here as the document
states it, read from the retained record and compared.

This is a bounded list, not a prose validator. Each claim names the document section,
what the document says, the record it rests on and the value read from it. A claim whose
record or input cannot be read is UNCHECKED and never passed, and the exit status says
so. It is 0 when every claim is checked and matches, 1 when any mismatches, and 2 when
any is unchecked and none mismatches. Passing says the document's numbers are the
records'. It says nothing about the readings of section 6 or the assessment of section 7.
Instrument settings the document states (the 240 K cold threshold, the 400 K refusal,
the caption's 56 characters and 110 dots per inch, the test's 100 rejected rows) are
not record measurements and are outside the list, as are the results of the scratch
regenerations of section 11, which are not retained, and section 13's statements about
what the diagnostic document records for the released runs, which are about a document
and not a record of this unit.

    python3 scripts/eval60_claims_check.py --evidence docs/aewc_v2/evidence/validation/eastern_pilot_2026-10-05/reports/extent --out <record.json>
"""
import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_tracks as X  # noqa: E402

EPOCH_1900_DAYS_1970 = 25567.0


class Unchecked(Exception):
    """The record or input a claim needs cannot be read, so the claim is not checked."""


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return json.loads(fh.read())
    with open(path) as fh:
        return json.load(fh)


class Evidence:
    """The retained records, read once each and remembered with their digests."""

    def __init__(self, root, wind_files=None):
        self.root, self.cache, self.digests = root, {}, {}
        self.wind_files = wind_files or {}

    def rec(self, name):
        """A record and its digest. A gzipped replay is digested on its JSON bytes, which
        is how every record of this project binds a replay, and a plain record on its file."""
        if name not in self.cache:
            path = os.path.join(self.root, name)
            if not os.path.exists(path):
                raise Unchecked(f"{name} is not under the evidence directory")
            if path.endswith(".gz"):
                with gzip.open(path, "rb") as fh:
                    blob = fh.read()
                self.cache[name] = json.loads(blob)
                self.digests[name] = hashlib.sha256(blob).hexdigest()
            else:
                self.cache[name] = load(path)
                self.digests[name] = X.digest(path)
        return self.cache[name]

    def wind_bounds(self, year):
        path = self.wind_files.get(year)
        if not path or not os.path.exists(path):
            raise Unchecked(f"the {year} wind file is not available here")
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            secs = np.asarray(ds.variables["valid_time"][:], float)
        t = secs / 86400.0 + EPOCH_1900_DAYS_1970
        return float(t.min()), float(t.max())


def pct(a, b):
    return round(100.0 * a / b, 1)


def round_like(value, document_value):
    """The record's value rounded to the decimals the document prints."""
    text = repr(float(document_value))
    decimals = len(text.split(".")[1]) if "." in text and not text.endswith(".0") else 0
    return round(float(value), decimals)


def bins_sum(bins, lo, hi):
    """The divergence-longitude bins whose lower edge lies in [lo, hi)."""
    total = 0
    for key, n in bins.items():
        edge = float(key.strip("[)").split(",")[0])
        if lo <= edge < hi:
            total += n
    return total


CONTEXT = {1990: {1314: "eval60_context_1990_track1314.json", 804: "eval60_context_1990_track804.json", 885: "eval60_context_1990_track885.json", 785: "eval60_context_1990_track785.json"},
           2006: {1286: "eval60_context_2006_track1286.json", 1137: "eval60_context_2006_track1137.json", 907: "eval60_context_2006_track907.json", 914: "eval60_context_2006_track914.json"}}
EVENT = "eval60_context_event_2006_r2.json"
CASES = [(1990, 1314, "20 Sep 12Z, 17 N 45 E", 46, 37.3, "3 Oct 18Z, 16 N 38 E", (108, 163)),
         (1990, 804, "11 Jun 00Z, 19 N 48.25 E", 26, 48.25, "17 Jun 18Z, 13 N 51 E", (0, 111)),
         (1990, 885, "26 Jun 06Z, 19.5 N 45 E", 20, 45.0, "1 Jul 18Z, 24.25 N 59.5 E", (39, 101)),
         (1990, 785, "7 Jun 06Z, 14.5 N 37.5 E", 11, 28.6, "10 Jun 00Z, 13 N 29.5 E", (0, 79)),
         (2006, 1286, "3 Sep 06Z, 17 N 54.25 E", 32, 51.75, "11 Sep 12Z, 12 N 52 E", (123, 123)),
         (2006, 1137, "2 Aug 12Z, 18.75 N 50.5 E", 25, 32.0, "10 Aug 18Z, 12.5 N 32 E", (123, 123)),
         (2006, 907, "16 Jun 12Z, 13.5 N 55 E", 21, 54.5, "21 Jun 18Z, 13 N 54.5 E", (0, 99)),
         (2006, 914, "17 Jun 12Z, 14 N 32 E", 12, 19.0, "21 Jun 00Z, 7.5 N 19 E", (0, 85))]


def stamp(date_iso, lat, lon):
    """The section 5 table's cell form, '3 Oct 18Z, 16 N 38 E', from a record's date and position."""
    import datetime as dt
    d = dt.datetime.fromisoformat(date_iso)
    return f"{d.day} {d.strftime('%b')} {d.strftime('%H')}Z, {lat:g} N {lon:g} E"


def claims(ev):
    """The list, each entry (id, section, statement, document value, record value)."""
    out = []
    rep = {y: ev.rec(f"eval60_replay_margin60_own_{y}.json.gz") for y in (1990, 2006)}
    cmp_ = {y: ev.rec(f"eval60_compare_audits60_{y}.json") for y in (1990, 2006)}
    coh = {y: ev.rec(f"eval60_cohort_audits60_{y}.json") for y in (1990, 2006)}
    scr = {y: ev.rec(f"eval60_screen_{y}.json") for y in (1990, 2006)}
    scr_t = {y: ev.rec(f"eastern_start_screen_{y}_r2.json") for y in (1990, 2006)}
    smp = {y: ev.rec(f"eval60_sample_{y}.json") for y in (1990, 2006)}

    # section 3, the replays
    for y, n, el, shared, archived in ((1990, 1857, 444.5, 509, 1990), (2006, 1917, 447.9, 562, 2007)):
        r = rep[y]; t = r["thresholds"]
        out.append((f"s3.finished.{y}", "3", f"the {y} replay finishes this many tracks", n, len(r["finished"])))
        out.append((f"s3.elapsed.{y}", "3", f"the {y} replay took this many seconds", el, round(float(r["elapsed_seconds"]), 1)))
        out.append((f"s3.domain_calibration.{y}", "3", "the replay applied the audit as a domain calibration, not a transfer", (True, False), (t["domain_calibration"], t["transfer"])))
        out.append((f"s3.pair.{y}", "3", "the applied pair is the 60 E domain's own", (4.6021628338653876e-07, 2.5821907501727775e-06), (t["coarse"], t["fine"])))
        out.append((f"s3.control_pair.{y}", "3", "the control's own pair recorded beside it", [4.842862320963436e-07, 2.5815610232489423e-06], t["control_run_pair"]))
        out.append((f"s3.audit.{y}", "3", "the audit digest begins c58ee92c19db", "c58ee92c19db", t["calibration_audit"]["sha256"][:12]))
        out.append((f"s3.audited_pair.{y}", "3", "the audit's released pair is the D run's 60 E pair", [4.7050118223104945e-07, 2.5920937221956206e-06], t["calibration_audit"]["audited_released_pair"]))
        g = r["archive_comparison"]
        out.append((f"s3.archive_shared.{y}", "3", f"shares this many histories with the C archive's {archived}", (shared, archived), (g["shared"], g["tracks_b"])))
        v = r["preparation_verification"]
        out.append((f"s3.interior.{y}", "3", "0 interior cells differ at any step", 0, v["interior_differing_max"]))
        out.append((f"s3.ring_every_step.{y}", "3", "ring differences at every step for every field", True, all(n_ == v["steps"] for n_ in v["ring_differing_steps"].values())))
        ring = v["ring_differing_total"]
        coarse = {k: n_ for k, n_ in ring.items() if k.startswith("coarse")}; fine = {k: n_ for k, n_ in ring.items() if k.startswith("fine")}
        out.append((f"s3.ring_all_but_few.{y}", "3", "on all but a few ring cells over the season (each field's count within 0.1 % of its grid's largest)", True,
                    all(n_ >= 0.999 * max(coarse.values()) for n_ in coarse.values()) and all(n_ >= 0.999 * max(fine.values()) for n_ in fine.values())))

    # section 3, the audit-to-audit contrast
    doc = {1990: dict(ref=1918, cand=1857, ident=768, pct=40.0, mutual=975, amb=(116, 87), unm=(59, 27), div=869, within=9, conv=106, bins=(94, 67, 23, 32, 84, 383, 145, 41), minmax=(-139.0, 59.5), changed=60),
           2006: dict(ref=1970, cand=1917, ident=751, pct=38.1, mutual=1043, amb=(114, 91), unm=(62, 32), div=918, within=14, conv=125, bins=(82, 93, 48, 33, 73, 395, 163, 31), minmax=(-139.5, 59.5), changed=62)}
    for y, changed40 in ((1990, 57), (2006, 60)):
        k40 = ev.rec(f"extent_compare_calibration40_{y}.json")["counts"]
        out.append((f"s3.recalibration40.{y}", "3", "the 40 E recalibration changed this share of the season's histories, percent, rounded", changed40, int(round(100.0 * (1 - k40["identical_pairs"] / k40["native_finished"])))))
    for y in (1990, 2006):
        c, d = cmp_[y], doc[y]; k, b = c["counts"], c["band"]
        out.append((f"s3.contrast.{y}", "3", "contrast and band", ("calibration_audits", 58.0), (c["contrast"], float(c["band_west_edge_lon"]))))
        out.append((f"s3.ref_cand.{y}", "3", "reference and candidate finished counts", (d["ref"], d["cand"]), (k["native_finished"], k["margin_finished"])))
        out.append((f"s3.identical.{y}", "3", "identical histories and their share of the reference", (d["ident"], d["pct"]), (k["identical_pairs"], pct(k["identical_pairs"], k["native_finished"]))))
        out.append((f"s3.mutual.{y}", "3", "unique mutual best matches", d["mutual"], k["mutual_changed_pairs"]))
        out.append((f"s3.ambiguous.{y}", "3", "ambiguous reference and candidate histories", d["amb"], (k["ambiguous_native"], k["ambiguous_margin"])))
        out.append((f"s3.unmatched.{y}", "3", "unmatched reference and candidate histories", d["unm"], (k["unmatched_native"], k["unmatched_margin"])))
        out.append((f"s3.diverging.{y}", "3", "diverging pairs, those within 2 degrees of 60 E, and converging pairs", (d["div"], d["within"], d["conv"]),
                    (b["diverging_pairs_within_band"] + b["diverging_pairs_west_of_band"], b["diverging_pairs_within_band"], b["converging_pairs"])))
        bins = c["divergence_longitude_bins"]
        out.append((f"s3.bins.{y}", "3", "divergence longitudes at or east of 50 E, 40 to 50, 30 to 40, 20 to 30, 0 to 20, 0 to 100 W, 100 to 130 W, west of 130 W", d["bins"],
                    (bins_sum(bins, 50, 999), bins_sum(bins, 40, 50), bins_sum(bins, 30, 40), bins_sum(bins, 20, 30), bins_sum(bins, 0, 20), bins_sum(bins, -100, 0), bins_sum(bins, -130, -100), bins_sum(bins, -999, -130))))
        lons = c["divergence_longitudes_sorted"]
        out.append((f"s3.minmax.{y}", "3", "divergence longitudes run from this westernmost to this easternmost", d["minmax"], (float(lons[0]), float(lons[-1]))))
        out.append((f"s7.changed.{y}", "7", "the domain's own pair changes this share of the season's histories (percent, rounded)", d["changed"], int(round(100.0 * (1 - k["identical_pairs"] / k["native_finished"])))))

    # section 4, the cohort
    doc = {1990: dict(n=(63, 59), atl=(28.6, 28.8), left=1, entered=1, mutual=27, diff=(0, 1, 1), exact=44, nocp=7),
           2006: dict(n=(68, 63), atl=(26.5, 30.2), left=0, entered=1, mutual=33, diff=(0, 2, 1), exact=41, nocp=7)}
    for y in (1990, 2006):
        h, d = coh[y], doc[y]
        rn, rm = h["cohort_tracks"]["native"], h["cohort_tracks"]["margin"]
        cor = h["correspondence"]
        out.append((f"s4.cohort.{y}", "4", "cohort members, reference then candidate", d["n"], (len(rn), len(rm))))
        atl_n = sum(1 for t in rn if str(t.get("outcome")).startswith("atlantic")); atl_m = sum(1 for t in rm if str(t.get("outcome")).startswith("atlantic"))
        out.append((f"s4.atlantic.{y}", "4", "Atlantic-side fraction of the cohort, percent, reference then candidate", d["atl"], (pct(atl_n, len(rn)), pct(atl_m, len(rm)))))
        out.append((f"s4.left_entered.{y}", "4", "members leaving and entering through a counterpart", (d["left"], d["entered"]), (cor["left_the_cohort_through_a_counterpart"]["count"], cor["entered_the_cohort_through_a_counterpart"]["count"])))
        out.append((f"s4.mutual.{y}", "4", "mutual pairs in both cohorts", d["mutual"], cor["members_in_both_cohorts_through_a_counterpart"]["mutual_pairs"]))
        s = cor["mutual_pairs_in_both_cohorts"]["summaries_differing"]
        out.append((f"s4.differing.{y}", "4", "of those pairs, Atlantic-side outcome, first-latitude band and genesis band differing", d["diff"], (s["outcome"], s["first_latitude_band"], s["genesis_longitude_band"])))
        out.append((f"s4.exact.{y}", "4", "percent of the reference cohort keeping an exact history", d["exact"], int(round(100.0 * cor["members_in_both_cohorts_through_a_counterpart"]["exact_histories"] / len(rn)))))
        cls = cor["native_cohort_by_class"]
        out.append((f"s4.no_counterpart.{y}", "4", "reference members without a clean counterpart (ambiguous plus unmatched)", d["nocp"], cls.get("ambiguous", 0) + cls.get("unmatched", 0)))

    # section 4, the screens
    for y, sel, west, entry, sel_t in ((1990, 12, 31.85, 6, 12), (2006, 15, 23.67, 7, 16)):
        s = scr[y]
        out.append((f"s4.screen.{y}", "4", "summer starts east of 40 E under the 60 E pair, those reaching 20 E, the westernmost stored longitude", (sel, 0, west), (s["selected"], s["reaching"], round(float(s["westernmost_lon_among_selected"]), 2))))
        out.append((f"s4.entry.{y}", "4", "of them, with a first entry into Africa", entry, sum(1 for r in s["rows"] if r.get("first_entry_into_africa"))))
        out.append((f"s4.screen_transferred.{y}", "4", "the same screen under the transferred pair", (sel_t, 0), (scr_t[y]["selected"], scr_t[y]["reaching"])))

    # section 5, the sample and the cases
    for y, east, comp, pop, med in ((1990, [1314, 804, 885], 785, 12, 11), (2006, [1286, 1137, 907], 914, 15, 12)):
        m = smp[y]
        out.append((f"s5.sample.{y}", "5", "eastern cases by stored observations, the comparison case, the eastern population, the comparison median", (east, comp, pop, med),
                    (list(m["eastern"]), m["comparison"], len(m["eastern_population"]), m["comparison_observations_median"])))
    e = smp[2006]["event"]
    out.append(("s5.event", "5", "no finished history within 5 degrees and 24 hours of the point, and the nearest observation's history, degrees and hours", (None, 1616, 0.5, 1536.0),
                (e["index"], e["nearest"]["index"], round(float(e["nearest"]["distance_deg"]), 2), round(float(e["nearest"]["hours"]), 1))))
    out.append(("s5.event_point", "5", "the event point", (10.5, 29.0, 38960.0), (e["point"]["lat"], e["point"]["lon"], e["point"]["days_since_1900"])))
    for y, idx, first, nobs, west, last, cov in CASES:
        m = smp[y]
        case = next((c for c in m["eastern_cases"] if c["index"] == idx), None) or (m["comparison_case"] if m["comparison_case"]["index"] == idx else None)
        if case is None:
            raise Unchecked(f"history {idx} is not in the {y} sample record")
        out.append((f"s5.first.{y}.{idx}", "5", "first detection", first, stamp(case["first"]["date"], case["first"]["lat"], case["first"]["lon"])))
        out.append((f"s5.obs.{y}.{idx}", "5", "stored observations and westernmost stored longitude (to the document's decimals)", (nobs, west), (case["observations"], round_like(case["westernmost_lon"], west))))
        ctx = ev.rec(CONTEXT[y][idx])
        tr = ctx["track"]
        import datetime as dt
        last_date = (dt.datetime(1900, 1, 1) + dt.timedelta(days=float(tr["time"][-1]))).isoformat()
        out.append((f"s5.last.{y}.{idx}", "5", "last observation", last, stamp(last_date, tr["lat"][-1], tr["lon"][-1])))
        sm = ctx["cloud"]["summary"]
        out.append((f"s5.coverage.{y}.{idx}", "5", "cloud coverage, retained of timesteps", cov, (sm["retained"], sm["timesteps"])))
        out.append((f"s5.replay_bound.{y}.{idx}", "5", "the context record binds the replay by digest", True, bool(ctx["replay"]) and ctx["replay"]["sha256"] == smp[y]["replay"]["sha256"]))

    # sections 10 and 11, the context records
    ctxs = {name: ev.rec(name) for y in CONTEXT for name in CONTEXT[y].values()}
    ctxs[EVENT] = ev.rec(EVENT)
    reads = [(f["path"], f["sha256"]) for r in ctxs.values() for f in r["cloud"]["files"]]
    out.append(("s11.reads", "11", "retained imagery file reads, distinct files, shared between the event window and history 1286", (474, 403, 71),
                (len(reads), len(set(reads)), len({(f["path"], f["sha256"]) for f in ctxs[EVENT]["cloud"]["files"]} & {(f["path"], f["sha256"]) for f in ctxs[CONTEXT[2006][1286]]["cloud"]["files"]}))))
    out.append(("s11.classes", "11", "coverage classes in the retained records are retained and missing only", {"retained", "missing"}, set(c for r in ctxs.values() for c in r["cloud"]["coverage"])))
    out.append(("s11.digests", "11", "the eight track records carry one instrument digest and the event record another", ("7b59e653aa4f", "77755029ffc2"),
                (sorted({r["script_sha256"][:12] for n, r in ctxs.items() if n != EVENT}), ctxs[EVENT]["script_sha256"][:12])))
    out[-1] = (out[-1][0], out[-1][1], out[-1][2], (["7b59e653aa4f"], "77755029ffc2"), out[-1][4])
    out.append(("s9.replay_binding", "9", "the eight track records bind a replay and the event record binds none", (8, False), (sum(1 for n, r in ctxs.items() if n != EVENT and r["replay"]), bool(ctxs[EVENT]["replay"]))))
    for y, names in ((1990, list(CONTEXT[1990].values())), (2006, list(CONTEXT[2006].values()) + [EVENT])):
        try:
            b0, b1 = ev.wind_bounds(y)
            inside = all(b0 <= float(ctxs[n]["window"]["time"][0]) and float(ctxs[n]["window"]["time"][1]) <= b1 for n in names)
            out.append((f"s11.windows_inside.{y}", "11", "every window lies inside its wind file's time range", True, inside))
        except Unchecked as exc:
            out.append((f"s11.windows_inside.{y}", "11", "every window lies inside its wind file's time range", True, exc))
    cf = np.array([[np.nan if x is None else float(x) for x in row] for row in ctxs[EVENT]["cloud"]["cold_fraction"]], float)
    finite = cf[np.isfinite(cf)]
    out.append(("s10.event_cold", "10", "the event record's cold fraction, minimum, maximum and percent of cells strictly between 0 and 1", (0.0, 0.94, 81),
                (round(float(finite.min()), 2), round(float(finite.max()), 2), int(round(100.0 * float(np.mean((finite > 0) & (finite < 1))))))))
    out.extend(claims_cohort_detail(coh))
    out.extend(claims_screen_detail(scr))
    out.extend(claims_coverage_and_bindings(ev, ctxs, smp, rep))
    out.extend(claims_trace(ev, rep[2006]))
    return out


GENESIS_KEYS = ("-20..-10", "-10..+0", "+0..+10", "+10..+20", "+20..+30", "+30..+40", "+40..+50")
LAT_BANDS = ("0 to 5", "5 to 10", "10 to 15", "15 to 20", "20 to 25")


def claims_cohort_detail(coh):
    """Section 4, the cohort table's rows, the distributions and the counterpart rows."""
    out = []
    table = {(1990, "native"): (1918, 103, 40, 63, 18, 1, 44, 46, 18, 13, 28.3), (1990, "margin"): (1857, 99, 40, 59, 17, 1, 41, 44, 16, 12, 27.3),
             (2006, "native"): (1970, 102, 34, 68, 18, 2, 48, 44, 19, 14, 31.8), (2006, "margin"): (1917, 98, 35, 63, 19, 2, 42, 44, 19, 15, 34.1)}
    genesis = {(1990, "native"): [7, 7, 8, 17, 10, 12, 2], (1990, "margin"): [7, 6, 9, 17, 10, 9, 1], (2006, "native"): [5, 13, 13, 9, 16, 9, 3], (2006, "margin"): [5, 14, 11, 7, 16, 8, 2]}
    latitude = {(1990, "native"): [(12, 0), (10, 1), (13, 7), (12, 5), (16, 5)], (1990, "margin"): [(10, 0), (9, 1), (12, 6), (11, 5), (17, 5)],
                (2006, "native"): [(4, 0), (12, 0), (18, 6), (20, 8), (14, 4)], (2006, "margin"): [(4, 0), (10, 0), (17, 5), (18, 9), (14, 5)]}
    stats = {(1990, "native"): (14.7, 13, 4.0, 16.5), (1990, "margin"): (14.9, 14, 4.25, 14.0), (2006, "native"): (13.5, 12, 3.625, 13.0), (2006, "margin"): (13.6, 12, 3.5, 11.5)}
    classes = {(1990, "native"): (28, 26, 2, 2, 5), (1990, "margin"): (28, 26, 2, 0, 3), (2006, "native"): (28, 27, 6, 6, 1), (2006, "margin"): (28, 27, 7, 1, 0)}
    for y in (1990, 2006):
        h = coh[y]
        for side in ("native", "margin"):
            r = h["runs"][side]; tot = r["outcomes_by_first_latitude_band"]["total"]; g = r["grouping_sensitivity"]; d = r["description"]
            out.append((f"s4.table.{y}.{side}", "4", "finished, archive rule, outside 0 to 25 N, cohort, Atlantic-side, Gulf-only, none, grouping units, grouped, grouped Atlantic-side, grouped percent", table[(y, side)],
                        (r["finished_tracks"], r["archive_rule_n"], r["outside_cohort_latitude"], r["cohort_n"], tot["atlantic_side"], tot["gulf_only"], tot["none"],
                         g["units"], g["grouped_units"], g["total_row"]["atlantic_side"], pct(g["total_row"]["atlantic_side"], g["units"]))))
            out.append((f"s4.genesis.{y}.{side}", "4", "genesis longitude by ten-degree band from 20 to 10 W upward to 40 to 50 E", genesis[(y, side)], [d["genesis_longitude_bands_10deg"].get(k, 0) for k in GENESIS_KEYS]))
            ob = r["outcomes_by_first_latitude_band"]
            out.append((f"s4.latitude.{y}.{side}", "4", "first-latitude bands from 0 to 5 N upward, members with the Atlantic-side count", latitude[(y, side)], [(ob[b]["cohort"], ob[b]["atlantic_side"]) for b in LAT_BANDS]))
            out.append((f"s4.stats.{y}.{side}", "4", "observations per track mean and median, duration median in days, start-longitude median", stats[(y, side)],
                        (round(d["observations_per_track"]["mean"], 1), d["observations_per_track"]["percentiles"]["50"], d["duration_days"]["50"], d["start_longitude_percentiles"]["50"])))
            cls = h["correspondence"][f"{side}_cohort_by_class"]
            out.append((f"s4.classes.{y}.{side}", "4", "members by class, identical, mutual diverging, mutual converging, ambiguous, unmatched", classes[(y, side)],
                        tuple(cls.get(k, 0) for k in ("identical", "mutual_diverging", "mutual_converging", "ambiguous", "unmatched"))))
        cor = h["correspondence"]
        left, entered = cor["left_the_cohort_through_a_counterpart"], cor["entered_the_cohort_through_a_counterpart"]
        if y == 1990:
            out.append(("s4.counterparts.1990", "4", "one leaves and one enters through converging pairs whose counterparts start at 5.25 S 20 E and 5 S 31.75 E", (1, "mutual_converging", [-5.25, 20.0], 1, "mutual_converging", [-5.0, 31.75]),
                        (left["count"], left["rows"][0]["kind"], left["rows"][0]["margin_first"], entered["count"], entered["rows"][0]["kind"], entered["rows"][0]["native_first"])))
        else:
            out.append(("s4.counterparts.2006", "4", "none leaves and one enters, its counterpart starting at 25.25 N 18.5 E", (0, 1, [25.25, 18.5]), (left["count"], entered["count"], entered["rows"][0]["native_first"])))
        rn, rm = h["runs"]["native"], h["runs"]["margin"]
        tn, tm = rn["outcomes_by_first_latitude_band"]["total"], rm["outcomes_by_first_latitude_band"]["total"]
        out.append((f"s4.moves.{y}", "4", "the cohort moves by this many members and the Atlantic-side fraction by this many points", (-4, 0.2) if y == 1990 else (-5, 3.7),
                    (rm["cohort_n"] - rn["cohort_n"], round(pct(tm["atlantic_side"], rm["cohort_n"]) - pct(tn["atlantic_side"], rn["cohort_n"]), 1))))
    return out


def claims_screen_detail(scr):
    """Section 4, the screen's months, longitudes, entries and the one entry at first detection."""
    import datetime as dt
    rows = {y: scr[y]["rows"] for y in (1990, 2006)}
    allrows = rows[1990] + rows[2006]
    months = sorted({dt.datetime.fromisoformat(r["first"]["date"]).month for r in allrows})
    lons = [r["first"]["lon"] for r in allrows]
    entries = [r for r in allrows if r.get("first_entry_into_africa")]
    at_first = [r for r in entries if r["first_entry_into_africa"]["index"] == 0]
    return [("s4.screen_span", "4", "first detections span June to September and 41 to 59.5 E", ([6, 7, 8, 9], 41.0, 59.5), (months, min(lons), max(lons))),
            ("s4.screen_entries", "4", "first entries into Africa lie at 35.3 to 49.7 E, one of them at its first detection, 2006 history 1185 at 42 E", (35.3, 49.7, 1, 1185, 42.0),
             (round(min(r["first_entry_into_africa"]["lon"] for r in entries), 1), round(max(r["first_entry_into_africa"]["lon"] for r in entries), 1), len(at_first), at_first[0]["index"] if at_first else None, at_first[0]["first_entry_into_africa"]["lon"] if at_first else None)),
            ("s4.screen_never", "4", "the other 6 and 8 never enter the polygon", (6, 8), tuple(sum(1 for r in rows[y] if not r.get("first_entry_into_africa")) for y in (1990, 2006)))]


def claims_coverage_and_bindings(ev, ctxs, smp, rep):
    """Sections 5, 7, 9 and 11, the imagery inventory, the comparison populations, the
    coverage classes of the nine windows, the records' bindings, the wind files' bounds
    and the imagery reads' timestamps and digests where the data are present."""
    import datetime as dt
    out = []
    summaries = {n: r["cloud"]["summary"] for n, r in ctxs.items()}
    full = sum(1 for s in summaries.values() if s["retained"] == s["timesteps"]); none = sum(1 for s in summaries.values() if s["retained"] == 0)
    out.append(("s7.coverage_windows", "7", "cloud context in 5 of the 9 windows, 3 in full and 2 in part, missing in 4", (5, 3, 2, 4), (9 - none, full, 9 - none - full, none)))
    out.append(("s5.event_coverage", "5", "the event window's coverage", (81, 81), (summaries[EVENT]["retained"], summaries[EVENT]["timesteps"])))
    out.append(("s5.populations", "5", "the comparison populations and their medians", (12, 11, 22, 12), (len(smp[1990]["comparison_population"]), smp[1990]["comparison_observations_median"], len(smp[2006]["comparison_population"]), smp[2006]["comparison_observations_median"])))
    out.append(("s5.nearest_days", "5", "the nearest stored observation to the event point is 64 days away", 64, smp[2006]["event"]["nearest"]["hours"] / 24.0))
    hexes = lambda s: isinstance(s, str) and len(s) == 64
    out.append(("s9.wind_bindings", "9", "every context record binds its wind file by digest", 9, sum(1 for r in ctxs.values() if hexes(r["wind"]["file_sha256"]))))
    out.append(("s9.imagery_bindings", "9", "every imagery file read is bound by digest", True, all(hexes(f["sha256"]) for r in ctxs.values() for f in r["cloud"]["files"])))
    out.append(("s9.replay_digest", "9", "the eight track records bind the retained replays themselves, by the digest of the replay's bytes", 8,
                sum(1 for n, r in ctxs.items() if n != EVENT and r["replay"] and r["replay"]["sha256"] == ev.digests[f"eval60_replay_margin60_own_{r['replay']['year']}.json.gz"])))
    for y in (1990, 2006):
        try:
            b0, b1 = ev.wind_bounds(y)
            d0 = (dt.datetime(y, 1, 1) - dt.datetime(1900, 1, 1)).days; d1 = (dt.datetime(y, 12, 31, 18) - dt.datetime(1900, 1, 1)).total_seconds() / 86400.0
            out.append((f"s11.wind_bounds.{y}", "11", "the wind file runs from 1 January 00Z to 31 December 18Z", (float(d0), d1), (b0, b1)))
        except Unchecked as exc:
            out.append((f"s11.wind_bounds.{y}", "11", "the wind file runs from 1 January 00Z to 31 December 18Z", True, exc))
    imagery = ev.wind_files.get("imagery")
    if imagery and os.path.isdir(imagery):
        names = os.listdir(imagery)
        counts = {}
        for y in (1990, 2006):
            for m in (6, 7, 8, 9, 10):
                counts[(y, m)] = sum(1 for n in names if n.startswith(f"gridsat_{y}{m:02d}"))
        out.append(("s5.inventory", "5", "every three-hourly file of July, August and September for both seasons, 248, 248 and 240, and none for June or October",
                    [(248, 248, 240, 0, 0), (248, 248, 240, 0, 0)], [tuple(counts[(y, m)] for m in (7, 8, 9, 6, 10)) for y in (1990, 2006)]))
        try:
            import xarray as xr
            bad_digest, bad_stamp, n_reads = 0, 0, 0
            seen = {}
            for r in ctxs.values():
                requested = [t for t, c in zip(r["cloud"]["times"], r["cloud"]["coverage"]) if c == "retained"]
                for when, f in zip(requested, r["cloud"]["files"]):
                    n_reads += 1
                    want = dt.datetime.fromisoformat(when) if isinstance(when, str) else dt.datetime(1900, 1, 1) + dt.timedelta(days=float(when))
                    if f["path"] not in seen:
                        with xr.open_dataset(f["path"]) as ds:
                            stamped = np.asarray(ds["time"].values).ravel()[0].astype("datetime64[s]").astype(dt.datetime)
                        seen[f["path"]] = (X.digest(f["path"]), stamped)
                    digest, stamped = seen[f["path"]]
                    bad_digest += digest != f["sha256"]
                    bad_stamp += stamped != want
            out.append(("s11.imagery_reads", "11", "every retained imagery file read decodes to the timestep it was read for, with its digest rebound", (474, 0, 0), (n_reads, bad_stamp, bad_digest)))
        except (OSError, ImportError) as exc:
            out.append(("s11.imagery_reads", "11", "every retained imagery file read decodes to the timestep it was read for, with its digest rebound", (474, 0, 0), Unchecked(f"the imagery could not be read, {exc}")))
    else:
        out.append(("s5.inventory", "5", "every three-hourly file of July, August and September for both seasons, 248, 248 and 240, and none for June or October", True, Unchecked("the imagery directory is not available here")))
        out.append(("s11.imagery_reads", "11", "every retained imagery file read decodes to the timestep it was read for, with its digest rebound", True, Unchecked("the imagery directory is not available here")))
    return out


def claims_trace(ev, replay_2006):
    """Section 13, the documented event's trace under the frozen candidate."""
    out = []
    try:
        r = ev.rec("eval60_event_trace_2006.json.gz")
    except Unchecked as exc:
        return [("s13.trace", "13", "the trace record is retained", True, exc)]
    t, rep = r["trace"], r["reproduces"]
    mine = [(f["time"], f["lat"], f["lon"]) for f in r["finished"]]
    theirs = [(f["time"], f["lat"], f["lon"]) for f in replay_2006["finished"]]
    out.append(("s13.bound", "13", "the traced run's finished histories are the retained replay's, compared here array by array, and the run named that replay", (True, 1917, True),
                (mine == theirs, len(mine), bool(rep) and rep["sha256"] == ev.digests["eval60_replay_margin60_own_2006.json.gz"])))
    out.append(("s13.window", "13", "the family, its window, its box and the logged steps", ("A_preHelene_literature", ["2006-09-01T00", "2006-09-06T00"], [0.0, 30.0], [15.0, 60.0], 21),
                (t["family"], t["window"], t["box"]["lat"], t["box"]["lon"], len(t["logged_steps"]))))
    out.append(("s13.point", "13", "the trace point, the released product's seed point for the event", (10.5, 29.0, 38960.0), (t["point"]["lat"], t["point"]["lon"], t["point"]["days_since_1900"])))
    by_step = {s["step"]: s for s in t["steps"]}
    seed = next(s for s in t["steps"] if s["date"] == "2006-09-02T00:00Z")
    n = seed["nearest_to_point"]; c = seed["candidates"][n["candidate"]]
    out.append(("s13.seed", "13", "at 2 September 00Z the nearest candidate is at the point, with this many points and these wind medians, and a track is seeded from it", (0.0, 10.5, 29.0, 41, -2.53, -1.57, True),
                (n["distance_deg"], n["lat"], n["lon"], c["n_points"], round(c["u_median"], 2), round(c["v_median"], 2), any(s["candidate"] == n["candidate"] for s in seed["seeds"]))))
    birth = next(s["birth"] for s in seed["seeds"] if s["candidate"] == n["candidate"])
    before = [by_step[k]["nearest_to_point"]["distance_deg"] for k in sorted(by_step) if k < seed["step"]]
    out.append(("s13.before", "13", "at the four steps before, the nearest candidates lie 8 to 14 degrees away", (4, True), (len(before), all(8.0 <= d <= 14.0 for d in before))))
    h = t["histories"][birth]
    obs = [(round(a, 2), round(o, 2)) for a, o in zip(h["lat_claimed"], h["lon_claimed"])]
    out.append(("s13.history", "13", "the track's eight observations", [(10.5, 29.0), (12.0, 31.5), (13.0, 30.0), (12.25, 30.5), (15.5, 29.5), (17.5, 26.5), (14.0, 28.0), (14.5, 27.0)], obs))
    last = by_step[h["steps"][-1]]; c8 = last["candidates"][next(cl["candidate"] for cl in last["claims"] if cl["birth"] == birth)]
    out.append(("s13.eighth", "13", "the eighth observation's region, points and wind medians", (90, -5.51, -3.96), (c8["n_points"], round(c8["u_median"], 2), round(c8["v_median"], 2))))
    s988 = next(s for s in t["steps"] if s["date"] == "2006-09-05T00:00Z"); m = next(x for x in s988["matching"] if x["birth"] == birth)
    cand = s988["candidates"][m["candidates_inside_polygon"][0]] if m["candidates_inside_polygon"] else None
    out.append(("s13.loss_00z", "13", "at 5 September 00Z one candidate inside the six-hour polygon, at this position, speed and distance, not chosen", (1, 15.5, 39.0, 59.8, 768, None),
                (len(m["candidates_inside_polygon"]), cand["lat_mean"] if cand else None, cand["lon_mean"] if cand else None,
                 round(list(m["implied_speed_ms"].values())[0], 1) if m["implied_speed_ms"] else None, int(round(list(m["distance_to_prediction_nm"].values())[0])) if m["distance_to_prediction_nm"] else None, m["chosen"])))
    s989 = next(s for s in t["steps"] if s["date"] == "2006-09-05T06:00Z"); m2 = next(x for x in s989["matching"] if x["birth"] == birth)
    out.append(("s13.loss_06z", "13", "at 06Z nothing inside the twelve-hour polygon", ("12hr", 0), (m2["pass"], len(m2["candidates_inside_polygon"]))))
    la, lo = h["lat_claimed"][-1], h["lon_claimed"][-1]
    gap = [all(float(np.hypot(c["lat_mean"] - la, c["lon_mean"] - lo)) > 10.0 for c in by_step[k]["candidates"]) for k in sorted(by_step) if h["steps"][-1] < k <= t["pruned_at"][birth]["step"]]
    out.append(("s13.gap", "13", "no candidate within 10 degrees of the track's last position at the three steps after it", (3, True), (len(gap), all(gap))))
    p = t["pruned_at"][birth]
    out.append(("s13.prune", "13", "the prune removes it at 5 September 12Z with eight observations and the last 0.75 days old", ("2006-09-05T12:00Z", 8, 0.75), (by_step[p["step"]]["date"], p["n_obs"], round(p["time"] - h["times"][-1], 2))))
    s991 = next(s for s in t["steps"] if s["date"] == "2006-09-05T18:00Z"); new = next((s for s in s991["seeds"] if abs(s["lat"] - 15.0) < 1e-9 and abs(s["lon"] - 22.0) < 1e-9), None)
    row = next((x for x in t["summary"] if new and x["birth"] == new["birth"]), None)
    out.append(("s13.reseed", "13", "at 18Z a track is seeded at 15.0 N 22.0 E from a 52-point candidate and finishes with 34 observations as finished history 1298", (True, 52, "finished", 34, 1298),
                (new is not None, s991["candidates"][new["candidate"]]["n_points"] if new else None, row["fate"] if row else None, row["n_obs"] if row else None, row["finished_index"] if row else None)))
    f = replay_2006["finished"][1298]
    import datetime as dt
    out.append(("s13.finished_1298", "13", "finished history 1298 runs from 15.0 N 22.0 E to 18.5 N 43.5 W on 16 September", (15.0, 22.0, 18.5, -43.5, "2006-09-16"),
                (f["lat"][0], f["lon"][0], round(f["lat"][-1], 2), round(f["lon"][-1], 2), (dt.datetime(1900, 1, 1) + dt.timedelta(days=float(f["time"][-1]))).strftime("%Y-%m-%d"))))
    pre = sorted((by_step[k]["nearest_to_point"]["lat"], by_step[k]["nearest_to_point"]["lon"]) for k in sorted(by_step) if k < seed["step"])
    out.append(("s13.pre_seed_positions", "13", "the four nearest candidates before the seed, over the highlands at 12.5 to 13.75 N and 36.5 to 37 E or to the north at 19.5 to 24 N",
                sorted([(12.5, 37.0), (13.75, 36.5), (19.5, 22.75), (24.0, 28.0)]), pre))
    dates = [(dt.datetime(1900, 1, 1) + dt.timedelta(days=float(x))).strftime("%d %HZ") for x in h["times"]]
    out.append(("s13.history_times", "13", "the eight observations' days and hours", ["02 00Z", "02 12Z", "03 00Z", "03 12Z", "04 00Z", "04 06Z", "04 12Z", "04 18Z"], dates))
    out.append(("s13.pass_00z", "13", "the 5 September 00Z read is the six-hour pass", "6hr", m["pass"]))
    out.append(("s13.displacements", "13", "the rejected candidate lies 12 degrees east of the last position and the reseed 5 degrees west of it, 13 degrees from the 40 E edge and 33 from the 60 E edge",
                (12.0, 5.0, 13.0, 33.0), (round(cand["lon_mean"] - lo, 2) if cand else None, round(lo - (new["lon"] if new else float("nan")), 2), round(40.0 - lo, 2), round(60.0 - lo, 2))))
    for label, idx in (("C", 1345), ("B", 1171)):
        d = ev.wind_files.get(f"run_{label}")
        try:
            if not d or not os.path.isdir(d):
                raise Unchecked(f"the released {label} run's tracks are not available here")
            import pilot_track_crosswalk as P
            tr = P.load_run(d, year=2006)["tracks"][idx]
            same = bool(np.array_equal(np.asarray(tr["time"], float), np.asarray(f["time"], float)) and np.array_equal(np.asarray(tr["lat"], float), np.asarray(f["lat"], float)) and np.array_equal(np.asarray(tr["lon"], float), np.asarray(f["lon"], float)))
            out.append((f"s13.identical_{label}{idx}", "13", f"finished history 1298 is identical to {label}{idx} in every time and position", True, same))
        except Unchecked as exc:
            out.append((f"s13.identical_{label}{idx}", "13", f"finished history 1298 is identical to {label}{idx} in every time and position", True, exc))
    return out


def evaluate(items):
    """Compare each claim's document value with its record value. A record value that is
    an Unchecked exception is UNCHECKED. Returns the rows and the exit status."""
    rows, status = [], 0
    for cid, section, text, expected, actual in items:
        if isinstance(actual, Unchecked):
            rows.append({"id": cid, "section": section, "claim": text, "document": _plain(expected), "record": None, "status": "UNCHECKED", "reason": str(actual)})
            status = max(status, 2)
            continue
        ok = _equal(expected, actual)
        rows.append({"id": cid, "section": section, "claim": text, "document": _plain(expected), "record": _plain(actual), "status": "MATCH" if ok else "MISMATCH"})
        if not ok:
            status = 1
    if status == 2 and any(r["status"] == "MISMATCH" for r in rows):
        status = 1
    return rows, status


def _plain(v):
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _equal(a, b):
    """Exact equality, with a document integer never matching a record value that is not
    itself that integer, so a fractional count cannot pass as zero."""
    if isinstance(a, set) or isinstance(b, set):
        return set(a) == set(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int) and not isinstance(b, bool):
        return isinstance(b, (int, float)) and float(b) == float(a) and float(b).is_integer()
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wind-1990", default="data/era5/pilot_east_75E/era5_v700_1990_6h_region.nc")
    ap.add_argument("--wind-2006", default="data/era5/pilot_east_75E/era5_v700_2006_6h_region.nc")
    ap.add_argument("--run-b", default="data/protocol_runs/pilot_B/era5_2006", help="the released 40 E control run directory for 2006, for section 13's identity claims")
    ap.add_argument("--run-c", default="data/protocol_runs/pilot_C/era5_2006", help="the released 60 E run directory for 2006, for section 13's identity claims")
    ap.add_argument("--imagery", default="data/gridsat_jas", help="the retained imagery directory, for the inventory and the imagery reads' timestamps and digests")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    ev = Evidence(args.evidence, {1990: args.wind_1990, 2006: args.wind_2006, "run_B": args.run_b, "run_C": args.run_c, "imagery": args.imagery})
    try:
        items = claims(ev)
    except Unchecked as exc:                           # a record the list needs is missing, so the whole list is unchecked, reported and never passed
        items = [("evidence", "all", "the retained records the list reads are present", True, exc)]
    rows, status = evaluate(items)
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("MATCH", "MISMATCH", "UNCHECKED")}
    out = {"generated_by": "scripts/eval60_claims_check.py", "script_sha256": X.digest(os.path.abspath(__file__)), "evidence": args.evidence,
           "records": {n: d for n, d in sorted(ev.digests.items())}, "counts": counts, "status": {0: "all checked and matching", 1: "mismatch", 2: "unchecked"}[status],
           "scope": "the numerical and coverage-class claims of the evaluation document's sections 3 to 5, 7, 9, 10, 11 and 13 against the retained records. Not the readings of section 6, not the assessment.",
           "claims": rows}
    X.publish_json(args.out, out, exclusive=True)
    for r in rows:
        if r["status"] != "MATCH":
            print(f"{r['status']:9s} {r['id']}: document {r['document']} record {r.get('record')} {r.get('reason', '')}")
    print(f"{counts['MATCH']} matching, {counts['MISMATCH']} mismatching, {counts['UNCHECKED']} unchecked of {len(rows)} claims. {out['status']}.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
