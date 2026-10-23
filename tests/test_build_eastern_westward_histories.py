"""Tests for the eastern westward-history subset builder (`EASTERN_WESTWARD_HISTORIES_SUBSET_2026-10-22.md`)."""

import csv
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import build_eastern_westward_histories as B  # noqa: E402

HI, W, A = B.HI, B.W, B.A
AVAIL = {(1990, b): (4, 117) for b in ("10 to 14.5 N", "15 to 19.5 N", "20 to 24.5 N", "25 to 29.5 N")}
DATES = [f"1990-06-{d:02d}" for d in range(1, 31)] + [f"1990-07-{d:02d}" for d in range(1, 32)]


def item(days, lons, depths=None, year=1990, band="15 to 19.5 N"):
    h = {"days": list(days), "lons": list(lons), "depths": list(depths or [-3.0] * len(lons))}
    return (year, band, DATES, h)


def stored_for(items):
    return {"tables": {HI.combo(*W.HISTORIES): {"westward_histories": json.loads(json.dumps([HI.summarize_westward(h, d, y, b) for y, b, d, h in items]))}}}


def test_band_codes():
    assert [B.band_code(b) for b in ("10 to 14.5 N", "15 to 19.5 N", "20 to 24.5 N", "25 to 29.5 N")] == ["10N", "15N", "20N", "25N"]


def test_the_identity_check_compares_every_summary_in_order():
    items = [item([3, 4, 5, 6], [60.0, 58.0, 55.0, 51.5]), item([10, 11, 12, 13], [70.0, 66.0, 64.0, 61.0], band="20 to 24.5 N")]
    B.check_identity(items, stored_for(items))
    with pytest.raises(SystemExit, match="recomputed history 1 differs"):
        B.check_identity(items, stored_for([items[0], item([10, 11, 12, 13], [70.0, 66.0, 64.0, 61.5], band="20 to 24.5 N")]))
    with pytest.raises(SystemExit, match="1 recomputed"):
        B.check_identity(items[:1], stored_for(items))
    with pytest.raises(SystemExit, match="recomputed history 0 differs"):
        B.check_identity(items[::-1], stored_for(items))                      # the order is part of the identity
    gap = [item([3, 4, 6, 7], [60.0, 58.0, 55.0, 51.5])]
    with pytest.raises(SystemExit, match="non-consecutive days"):
        B.check_identity(gap, stored_for(gap))


def test_history_and_member_rows():
    items = [item([3, 4, 5, 6], [60.0, 58.0, 58.5, 52.0], [-2.5, -6.25, -4.0, -3.0]), item([20, 21, 22, 23], [70.0, 66.0, 64.0, 61.0]),
             item([8, 9, 10, 11], [65.0, 62.0, 60.0, 56.0], band="25 to 29.5 N")]
    rows, members = B.history_rows(items, AVAIL)
    assert [r["history_id"] for r in rows] == ["1990-15N-01", "1990-15N-02", "1990-25N-01"]
    r = rows[0]
    assert (r["first_date"], r["last_date"], r["days"], r["net_deg"], r["deepest_anomaly_ms"]) == ("1990-06-04", "1990-06-07", 4, -8.0, -6.25)
    assert r["westward_step_fraction"] == 2 / 3 and r["speed_deg_per_day"] == -8.0 / 3 and r["primary"] is True and rows[2]["primary"] is False
    assert {x["travel_status"] for x in rows} == {"unresolved"} and r["rule_class"] == HI.classify(items[0][3])
    assert (r["starts_on_first_available_day"], r["ends_on_last_available_day"]) == (False, False)
    edge, _ = B.history_rows([item([4, 5, 6, 7], [60.0, 58.0, 55.0, 51.5]), item([57, 58, 59, 60], [60.0, 58.0, 55.0, 51.5])],
                             {(1990, "15 to 19.5 N"): (4, 60)})
    assert [(e["starts_on_first_available_day"], e["ends_on_last_available_day"]) for e in edge] == [(True, False), (False, True)]
    assert len(members) == 12 and members[1] == {"history_id": "1990-15N-01", "member": 1, "date": "1990-06-05", "day_of_season": 4, "lon": 58.0, "anomaly_ms": -6.25}
    still, _ = B.history_rows([item([3, 4, 5, 6], [60.0, 58.0, 58.0, 52.0])], AVAIL)
    assert still[0]["westward_step_fraction"] == 2 / 3                       # a step that stays put is not westward
    second = [m for m in members if m["history_id"] == "1990-15N-02"]
    assert second[-1]["date"] == "1990-06-24" and second[-1]["day_of_season"] == 23


def test_recompute_keeps_westward_histories_by_band_then_season(monkeypatch):
    monkeypatch.setattr(W, "YEARS", (1990, 1991))
    by = {y: {W.MONTHS[0]: {"bands": {"20 to 24.5 N": 22.25, "15 to 19.5 N": 17.25}}} for y in (1990, 1991)}
    monkeypatch.setattr(A, "season_hundredths", lambda months, b: ([f"{b}"], (months, b)))
    monkeypatch.setattr(HI, "rep_minima", lambda rep, H, month_of, lons: H)
    def fake_link(H, linker):
        assert linker == "LN"
        y = next(k for k, v in by.items() if v is H[0])
        return [{"tag": (y, H[1], "w")}, {"tag": (y, H[1], "s")}]
    monkeypatch.setattr(HI, "link", fake_link)
    monkeypatch.setattr(HI, "classify", lambda h: "westward" if h["tag"][2] == "w" else "standing")
    out = B.recompute(by)
    assert [(y, b) for y, b, _, _ in out] == [(1990, "15 to 19.5 N"), (1991, "15 to 19.5 N"), (1990, "20 to 24.5 N"), (1991, "20 to 24.5 N")]
    assert all(h["tag"][2] == "w" for _, _, _, h in out)


TRAVEL = os.path.join("notes", "travel_source.md")
DIGESTS = {"eval60_regime_1990_06.json.gz": "a" * 64}
STORED_SHA = "b" * 64


COHORT = ["1990-15N-01", "1990-20N-02"]
LOADED = {"shape": {"path": "s", "sha256": "1" * 64}, "inject": {"path": "i", "sha256": "2" * 64}, "xband": {"eval60_xband_1990_06.json.gz": "3" * 64}}


def matched_record(n=86, cohort=COHORT, profile=(0.13, 0.24), days=None):
    k = len(cohort)
    return {"results": {"changing": {"by_day": list(profile), "days": [77, 81], "histories": n},
                        "matched": {"by_day": [0.17] * 5, "days": [k] * 5 if days is None else days, "histories": k,
                                    "numerator": [4.0] * 5, "denominator": [20.0] * 5}, "cohort": list(cohort)}}


def write_sources(root, n=86, failures=5, digests=DIGESTS, stored_sha=STORED_SHA, matched=None, joint_sha=None, matched_inputs=None):
    recs = {"shape": {"results": {"composite": {"primary": {"center": -5.0, "east": 0.4, "west": 0.1, "members": 513, "histories": n}}}},
            "coherence": {"results": {"measure": {"primary": {"departure": -0.38, "histories": n}}}},
            "lobe": {"results": {"measure": {"primary": {"carried": 0.47, "pinned": 0.16, "contrast": 0.31, "samples": 154, "histories": n}}}},
            "joint": {"results": {"measure": {"first_day": 0.13, "profile": [0.13, 0.24], "profile_days": [77, 81], "histories": n}}},
            "matched_profile": matched or matched_record(n),
            "prestart": {"verdict": {"adequate": False, "failures": ["x"] * failures}}}
    for k, v in recs.items():
        v["generated_by"] = {"records": dict(digests), "histories": {"path": B.STORED, "sha256": stored_sha}, **json.loads(json.dumps(LOADED))}
        if k == "matched_profile":
            v["generated_by"].update(matched_inputs or {})                                             # read against the joint record just written
            v["generated_by"]["joint_archive"] = {"sha256": joint_sha or hashlib.sha256(open(os.path.join(root, B.SOURCES["joint"]), "rb").read()).hexdigest()}
        p = os.path.join(root, B.SOURCES[k]); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(v, fh)


def test_descriptors_are_read_with_their_sources_and_population(tmp_path):
    write_sources(str(tmp_path))
    d = {x["name"]: x for x in B.descriptors(str(tmp_path), DIGESTS, STORED_SHA)}
    assert list(d) == ["average_structure", "coherence_departure", "lobe_contrast", "first_day", "daily_profile", "daily_profile_matched", "prestart"]
    assert d["coherence_departure"]["value"] == -0.38 and d["lobe_contrast"]["value"]["contrast"] == 0.31 and d["first_day"]["value"] == {"ratio": 0.13, "starts": 77}
    assert d["average_structure"]["value"]["member_days"] == 513 and d["daily_profile"]["value"]["history_days_by_day"] == [77, 81]
    assert d["lobe_contrast"]["value"]["history_days"] == 154
    assert d["prestart"]["value"] is None and d["prestart"]["status"] == "not measured" and "failed in 5 of its 16" in d["prestart"]["definition"]
    p = tmp_path / B.SOURCES["lobe"]
    assert d["lobe_contrast"]["source"] == {"path": B.SOURCES["lobe"], "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    assert all("not of any one history" in x["population"] for x in d.values())
    m = d["daily_profile_matched"]
    assert m["value"] == {"ratio_by_day": [0.17] * 5, "histories": 2, "numerator_by_day_ms": [4.0] * 5, "denominator_by_day_ms": [20.0] * 5}
    assert m["cohort"] == COHORT and m["population"].startswith("the 2 westward histories") and "days_used 5" in m["population"]
    assert m["source"]["path"] == B.SOURCES["matched_profile"]
    for kw, match in (({"joint_sha": "e" * 64}, "read against another joint record"),
                      ({"matched": matched_record(profile=(0.13, 0.25))}, "curve over every history differs"),
                      ({"matched": matched_record(days=[2, 2, 2, 2, 1])}, "cohort is not read on every day"),
                      ({"matched": matched_record(n=85)}, "matched-profile record measured 85 histories"),
                      ({"matched_inputs": {"xband": {"eval60_xband_1990_06.json.gz": "0" * 64}}}, "the xband input differs"),
                      ({"matched_inputs": {"shape": {"path": "s", "sha256": "0" * 64}}}, "the shape input differs"),
                      ({"matched_inputs": {"inject": {"path": "i", "sha256": "0" * 64}}}, "the inject input differs"),
                      ({"matched_inputs": {"histories": {"path": B.STORED, "sha256": "0" * 64}}}, "made from another history record")):
        write_sources(str(tmp_path), **kw)
        with pytest.raises(SystemExit, match=match):
            B.descriptors(str(tmp_path), DIGESTS, STORED_SHA)
    write_sources(str(tmp_path))
    write_sources(str(tmp_path), failures=4)
    assert "failed in 4 of its 16" in {x["name"]: x for x in B.descriptors(str(tmp_path), DIGESTS, STORED_SHA)}["prestart"]["definition"]
    write_sources(str(tmp_path), n=104)
    with pytest.raises(SystemExit, match="shape record measured 104 histories"):
        B.descriptors(str(tmp_path), DIGESTS, STORED_SHA)
    write_sources(str(tmp_path), digests={"eval60_regime_1990_06.json.gz": "c" * 64})
    with pytest.raises(SystemExit, match="shape record was made from other month records"):
        B.descriptors(str(tmp_path), DIGESTS, STORED_SHA)
    write_sources(str(tmp_path), stored_sha="d" * 64)
    with pytest.raises(SystemExit, match="shape record was made from another history record"):
        B.descriptors(str(tmp_path), DIGESTS, STORED_SHA)


def test_the_month_records_must_be_those_of_the_stored_history_record(tmp_path):
    stored = {"generated_by": {"month_records": {"a.json.gz": "1" * 64, "b.json.gz": "2" * 64}}}
    B.check_inputs({"a.json.gz": "1" * 64, "b.json.gz": "2" * 64}, stored)
    with pytest.raises(SystemExit, match=r"differ from those the stored history record was made from \(\['b.json.gz'\]\)"):
        B.check_inputs({"a.json.gz": "1" * 64, "b.json.gz": "3" * 64}, stored)
    with pytest.raises(SystemExit, match="differ"):
        B.check_inputs({"a.json.gz": "1" * 64}, stored)
    f = tmp_path / "x.json.gz"; f.write_bytes(b"abc")
    assert B.month_digests([str(f)]) == {"x.json.gz": hashlib.sha256(b"abc").hexdigest()}


def test_the_member_digest_and_the_retained_reference_bind_interior_positions_and_depths(tmp_path):
    items = [item([3, 4, 5, 6, 7], [54.0, 51.5, 51.5, 46.5, 42.5], [-2.76, -4.16, -9.04, -6.18, -4.58])]
    rows, members = B.history_rows(items, AVAIL)
    old_cols = B.HIST_COLUMNS[:-2]                                             # a version 1 reference, without the later columns
    tables = {"histories": (B.HIST_COLUMNS, rows), "members": (B.MEMBER_COLUMNS, members)}
    ref = tmp_path / "ref"; ref.mkdir()
    B.write_text(str(ref / "histories.csv"), B.csv_text(old_cols, [{c: r[c] for c in old_cols} for r in rows]))
    B.write_text(str(ref / "members.csv"), B.csv_text(B.MEMBER_COLUMNS, members))
    def write_manifest():
        (ref / "manifest.json").write_text(json.dumps({"tables": {k: {"sha256": hashlib.sha256((ref / f"{k}.csv").read_bytes()).hexdigest(), "columns": c}
                                                                for k, c in (("histories", old_cols), ("members", B.MEMBER_COLUMNS))}}))
    write_manifest()
    got = B.check_reference(str(ref), tables)
    assert got["result"] == "equal" and got["compared_columns"]["histories"] == old_cols    # added columns are allowed
    with pytest.raises(SystemExit, match="lacks the reference's columns"):
        B.check_reference(str(ref), {"histories": (old_cols[:-1], rows), "members": tables["members"]})
    for change in ("lon", "depth"):
        it = [item([3, 4, 5, 6, 7], [54.0, 52.0 if change == "lon" else 51.5, 51.5, 46.5, 42.5], [-2.76, -5.16 if change == "depth" else -4.16, -9.04, -6.18, -4.58])]
        assert B.check_identity(it, stored_for(items)) is None                 # the summaries cannot see an interior change
        r2, m2 = B.history_rows(it, AVAIL)
        assert B.member_digest(m2) != B.member_digest(members)
        with pytest.raises(SystemExit, match="differs from the retained reference"):
            B.check_reference(str(ref), {"histories": (B.HIST_COLUMNS, r2), "members": (B.MEMBER_COLUMNS, m2)})
    (ref / "members.csv").write_text(B.csv_text(B.MEMBER_COLUMNS, members).replace("-4.16", "-4.17"))
    with pytest.raises(SystemExit, match="reference members.csv does not match its manifest"):
        B.check_reference(str(ref), tables)


def test_available_days_are_where_the_representation_is_defined(monkeypatch):
    monkeypatch.setattr(W, "YEARS", (1990,))
    by = {1990: {W.MONTHS[0]: {"bands": {"15 to 19.5 N": 17.25}}}}
    monkeypatch.setattr(A, "season_hundredths", lambda months, b: (None, "H"))
    monkeypatch.setattr(HI, "rep_minima", lambda rep, H, month_of, lons: [None] * 4 + [[]] * 3 + [[{"lon": 50.0}]] + [None] * 2)
    assert B.available_days(by) == {(1990, "15 to 19.5 N"): (4, 7)}


def rows_with(values):
    rows = [np.zeros(len(A.LONS)) for _ in range(A.SEASON_DAYS)]
    for d, lon, v in values:
        rows[d][B.P.col(lon)] = v
    return rows


def test_one_historys_participation_by_the_readers_own_functions():
    h = {"days": [20, 21, 22, 23, 24], "lons": [60.0, 58.0, 56.0, 54.0, 52.0], "depths": [-3.0] * 5}
    b1 = rows_with([(d, x, -2.0) for d, x in zip(h["days"], h["lons"])])
    b0 = rows_with([(d, x, -2.0) for d, x in zip(h["days"], h["lons"])])
    rows, extra = B.history_participation("1990-15N-01", h, b1, b0, "15 to 19.5 N")
    by = {r["descriptor"]: r for r in rows}
    assert list(by) == ["average_structure", "coherence_departure", "lobe_contrast", "first_day", "daily_profile"]
    assert by["average_structure"]["days_used"] == 5 and by["coherence_departure"]["days_used"] == 4 and by["lobe_contrast"]["days_used"] == 2
    assert by["first_day"]["participates"] and by["daily_profile"]["days_used"] == 5 and extra["profile_off_grid"] == [0] * 5
    assert [by[d]["used_offsets"] for d in by] == ["0 1 2 3 4", "0 2 3 4", "2 3", "0", "0 1 2 3 4"] and extra["lobe_days"] == [1, 1]
    east = {"days": [20, 21, 22], "lons": [70.0, 67.0, 65.0], "depths": [-3.0] * 3}         # 76 E is off the grid, and day 3 is missing
    rows, extra = B.history_participation("1990-15N-02", east, b1, b0, "15 to 19.5 N")
    by = {r["descriptor"]: r for r in rows}
    assert by["first_day"]["participates"] is False and "east of 75 E" in by["first_day"]["exclusion_reason"]
    assert by["lobe_contrast"]["days_used"] == 0 and by["lobe_contrast"]["days_excluded"] == 2 and extra["lobe_off_grid"] == 1
    assert by["lobe_contrast"]["exclusion_reason"] == "day 3 beyond the last member, day 2 with a position 6 degrees east lying east of 75 E"
    assert by["daily_profile"]["days_used"] == 2 and extra["profile_days"] == [0, 1, 1, 0, 0] and extra["profile_off_grid"] == [1, 0, 0, 0, 0]
    assert (by["first_day"]["used_offsets"], by["lobe_contrast"]["used_offsets"], by["daily_profile"]["used_offsets"]) == ("", "", "1 2")
    B.check_rows(rows, [extra])                                                # a skipped first day lists no day
    back ={"days": [20, 21, 22, 23, 24], "lons": [68.0, 69.5, 70.0, 66.0, 63.0], "depths": [-3.0] * 5}  # east on days 1 and 2, then west
    bb = rows_with([(d, x, -2.0) for d, x in zip(back["days"], back["lons"])])                # the history's own member anomalies
    rows, extra = B.history_participation("1990-15N-04", back, bb, bb, "15 to 19.5 N")
    by = {r["descriptor"]: r for r in rows}
    assert by["lobe_contrast"]["used_offsets"] == "3" and extra["lobe_days"] == [0, 1] and by["daily_profile"]["used_offsets"] == "0 3 4"
    B.check_rows(rows, [extra])
    early = {"days": [2, 3, 4, 5], "lons": [60.0, 58.0, 56.0, 54.0], "depths": [-3.0] * 4}  # fewer than 6 days of season before its last day
    eb = rows_with([(d, x, -2.0) for d, x in zip(early["days"], early["lons"])])              # the early history's own member anomalies
    rows, extra = B.history_participation("1990-15N-03", early, eb, eb, "15 to 19.5 N")
    by = {r["descriptor"]: r for r in rows}
    assert by["coherence_departure"]["participates"] is False and by["coherence_departure"]["days_excluded"] == 4
    assert by["coherence_departure"]["used_offsets"] == "" and by["average_structure"]["used_offsets"] == "0 1 2 3"
    B.check_rows(rows, [extra])                                                # a skipped coherence reading lists no day


def test_participation_covers_the_primary_histories_and_needs_both_loads_to_agree(monkeypatch):
    monkeypatch.setattr(W, "YEARS", (1990,))
    monkeypatch.setattr(B.L, "BAND_LAT", {"15 to 19.5 N": 17.25, "25 to 29.5 N": 27.25})
    monkeypatch.setattr(B.L, "season_samples", lambda months, b: None)
    monkeypatch.setattr(B.Sh, "b1_rows", lambda S, month_of: (None, "b1"))
    monkeypatch.setattr(B.P, "daily_hundredths", lambda S: None)
    monkeypatch.setattr(B.P, "b0_rows", lambda Hb, month_of: {y: "b0" for y in Hb})
    for mod, name in ((B.Sh, "composite"), (B.Sh, "flank_readers"), (B.C, "measure"), (B.P, "measure"), (B.G, "profile")):
        monkeypatch.setattr(mod, name, lambda *a, **k: {})
    seen = []
    monkeypatch.setattr(B, "history_participation", lambda hid, h, b1, b0, band: (seen.append(hid) or [{"history_id": hid}], {}))
    monkeypatch.setattr(B, "check_pooled", lambda rows, extras, rc, rec: {"all": "equal"})
    real_check_rows = B.check_rows
    monkeypatch.setattr(B, "check_rows", lambda rows, masks: None)
    items = [item([3, 4, 5, 6], [60.0, 58.0, 55.0, 51.5]), item([8, 9, 10, 11], [65.0, 62.0, 60.0, 56.0], band="25 to 29.5 N")]
    hist = {(y, b): [h] for y, b, _, h in items}
    rows, checks = B.participation(items, hist, {1990: {}}, {})
    assert seen == ["1990-15N-01"] and rows == [{"history_id": "1990-15N-01"}] and checks == {"all": "equal"}     # outer bands carry no descriptor
    other = {k: [dict(v[0], lons=[61.0] + v[0]["lons"][1:])] for k, v in hist.items()}
    with pytest.raises(SystemExit, match="two loads of the histories differ"):
        B.participation(items, other, {1990: {}}, {})
    monkeypatch.setattr(B, "check_rows", real_check_rows)                     # rows that do not match their eligibility are refused
    one_row = lambda hid: [{"descriptor": "first_day", "history_id": hid, "participates": False, "days_used": 0, "days_excluded": 1, "exclusion_reason": "x"}]
    monkeypatch.setattr(B, "history_participation", lambda hid, h, b1, b0, band: (one_row(hid), {"history_id": hid, "center": 4, "coherence": False,
                                                                                    "lobe_samples": 0, "lobe_off_grid": 0, "lobe_days": [0, 0],
                                                                                    "profile_days": [0] * 5, "profile_off_grid": [0] * 5}))
    with pytest.raises(SystemExit, match="no average_structure row"):
        B.participation(items, hist, {1990: {}}, {})


def test_every_exported_row_must_agree_with_its_historys_eligibility():
    four = {"days": [20, 21, 22, 23], "lons": [60.0, 58.0, 56.0, 54.0], "depths": [-3.0] * 4}
    b = rows_with([(d, x, -2.0) for d, x in zip(four["days"], four["lons"])])
    rows, mask = B.history_participation("1990-15N-01", four, b, b, "15 to 19.5 N")
    prof = next(r for r in rows if r["descriptor"] == "daily_profile")
    assert (prof["days_used"], prof["days_excluded"], prof["exclusion_reason"]) == (4, 1, "day 4 beyond the last member")    # the four-member case
    B.check_rows(rows, [mask])
    for field, value in (("days_used", 5), ("days_excluded", 0), ("participates", False), ("exclusion_reason", ""), ("days_used", 0)):
        bad = json.loads(json.dumps(rows)); next(r for r in bad if r["descriptor"] == "daily_profile")[field] = value
        with pytest.raises(SystemExit, match="daily_profile row of 1990-15N-01 disagrees"):
            B.check_rows(bad, [mask])
    for d, listed in (("lobe_contrast", "2"), ("lobe_contrast", "3 2"), ("daily_profile", "0 1 2 4"), ("coherence_departure", "0 1 2 3"),
                      ("average_structure", "1 2 3 4"), ("first_day", "1")):
        bad = json.loads(json.dumps(rows)); next(r for r in bad if r["descriptor"] == d)["used_offsets"] = listed
        with pytest.raises(SystemExit, match=f"the {d} row of 1990-15N-01 lists days"):
            B.check_rows(bad, [mask])
    moved = dict(mask, lobe_days=[0, 1], lobe_samples=1)                     # one lobe day read, but which one
    bad = json.loads(json.dumps(rows)); r = next(r for r in bad if r["descriptor"] == "lobe_contrast")
    r.update(days_used=1, days_excluded=1, exclusion_reason="x", used_offsets="2")
    with pytest.raises(SystemExit, match=r"lobe_contrast row of 1990-15N-01 lists days \[2\], not the readers' \[3\]"):
        B.check_rows(bad, [moved])
    r["used_offsets"] = "3"
    B.check_rows(bad, [moved])
    with pytest.raises(SystemExit, match="two participation rows"):
        B.check_rows(rows + rows[:1], [mask])
    with pytest.raises(SystemExit, match="no lobe_contrast row"):
        B.check_rows([r for r in rows if r["descriptor"] != "lobe_contrast"], [mask])
    with pytest.raises(SystemExit, match="outside the population"):
        B.check_rows(rows + [dict(rows[0], history_id="1990-15N-09")], [mask])


def test_the_lobe_days_are_bound_to_the_lobe_position_reader(monkeypatch):
    h = {"days": [20, 21, 22, 23, 24], "lons": [60.0, 58.0, 56.0, 54.0, 52.0], "depths": [-3.0] * 5}
    b = rows_with([(d, x, -2.0) for d, x in zip(h["days"], h["lons"])])
    real = B.P.accumulate
    def extra_on_full(rows, hh, acc):                                         # a reader that counts one more day on the whole history
        real(rows, hh, acc)
        if len(hh["lons"]) == len(h["lons"]):
            acc["samples"] += 1
    monkeypatch.setattr(B.P, "accumulate", extra_on_full)
    with pytest.raises(SystemExit, match="lobe-position days differ from the reader's count"):
        B.history_participation("1990-15N-01", h, b, b, "15 to 19.5 N")


def test_the_matched_cohort_must_be_the_histories_read_on_every_day():
    desc = [{"name": "daily_profile_matched", "cohort": ["b", "a"]}]
    row = lambda hid, used: {"descriptor": "daily_profile", "history_id": hid, "days_used": used}
    assert B.check_cohort(desc, [row("a", 5), row("b", 5), row("c", 4), {"descriptor": "first_day", "history_id": "d", "days_used": 5}]) == "equal"
    for part in ([row("a", 5)], [row("a", 5), row("b", 5), row("c", 5)], [row("a", 5), row("b", 4)]):
        with pytest.raises(SystemExit, match="matched-profile cohort"):
            B.check_cohort(desc, part)
    with pytest.raises(SystemExit, match="matched-profile cohort"):
        B.check_cohort([{"name": "daily_profile_matched", "cohort": ["a", "a", "b"]}], [row("a", 5), row("b", 5)])


def test_the_daily_profile_mask_is_bound_to_the_joint_reader(monkeypatch):
    h = {"days": [20, 21, 22, 23, 24], "lons": [60.0, 58.0, 56.0, 54.0, 52.0], "depths": [-3.0] * 5}
    b = rows_with([(d, x, -2.0) for d, x in zip(h["days"], h["lons"])])
    monkeypatch.setattr(B.G, "profile", lambda b0, hs, bands: {"days": [1, 1, 1, 1, 0], "off_grid": [0] * 5, "by_day": []})
    with pytest.raises(SystemExit, match="differ from the joint reader's"):
        B.history_participation("1990-15N-01", h, b, b, "15 to 19.5 N")
    def refuses(b0, hs, bands):
        raise SystemExit("REFUSED: no first day enters the measure")
    monkeypatch.setattr(B.G, "profile", refuses)
    with pytest.raises(SystemExit, match="no first day enters"):
        B.history_participation("1990-15N-01", h, b, b, "15 to 19.5 N")              # day 0 is on the grid, so the reader must not refuse
    east = {"days": [20, 21, 22], "lons": [70.0, 67.0, 65.0], "depths": [-3.0] * 3}
    assert B.history_participation("1990-15N-02", east, b, b, "15 to 19.5 N")[1]["profile_days"][0] == 0     # off the grid, the refusal is expected


def test_the_pooled_checks_refuse_any_disagreement():
    rows = [{"descriptor": d, "history_id": "x", "participates": True, "days_used": u, "days_excluded": 0, "exclusion_reason": ""}
            for d, u in (("average_structure", 5), ("coherence_departure", 4), ("lobe_contrast", 2), ("first_day", 1), ("daily_profile", 5))]
    extras = [{"lobe_off_grid": 0, "profile_days": [1, 1, 1, 1, 1], "profile_off_grid": [0] * 5}]
    shape = {"center": -5.0, "east": 0.4, "west": 0.1, "members": 5, "histories": 1}
    rec = {"shape": {"results": {"composite": {"primary": dict(shape)}}}, "coherence": {"results": {"measure": {"primary": {"departure": -0.4}}}},
           "lobe": {"results": {"measure": {"primary": {"carried": 0.5, "pinned": 0.1, "contrast": 0.4, "samples": 2, "off_grid": 0}}}},
           "joint": {"results": {"measure": {"profile": [0.1] * 5, "profile_days": [1] * 5, "profile_off_grid": [0] * 5}}}}
    rc = {"shape": dict(shape), "coherence": {"departure": -0.4, "histories": 1},
          "lobe": {"carried": 0.5, "pinned": 0.1, "contrast": 0.4, "samples": 2, "off_grid": 0}, "profile": {"by_day": [0.1] * 5, "days": [1] * 5, "off_grid": [0] * 5}}
    assert set(B.check_pooled(rows, extras, rc, rec).values()) == {"equal"}
    for descriptor, used, name in (("daily_profile", 4, "daily_profile exported history-days"), ("coherence_departure", 3, "coherence_departure exported history-days")):
        bad = json.loads(json.dumps(rows)); next(r for r in bad if r["descriptor"] == descriptor)["days_used"] = used
        with pytest.raises(SystemExit, match=name):
            B.check_pooled(bad, extras, rc, rec)
    for path, value, name in ((("lobe", "samples"), 3, "lobe_contrast history-days"), (("coherence", "departure"), -0.41, "coherence record"),
                              (("profile", "off_grid"), [1, 0, 0, 0, 0], "daily_profile days off the grid by day"), (("shape", "members"), 6, "average_structure member-days")):
        bad = json.loads(json.dumps(rc)); bad[path[0]][path[1]] = value
        with pytest.raises(SystemExit, match=name):
            B.check_pooled(rows, extras, bad, rec)


def test_failure_modes_carry_the_net_sensitivity_from_the_record():
    stored = {"tables": {HI.combo(*W.HISTORIES): {"bands": {"15 to 19.5 N": {"westward": 34, "westward_at_net": {"-6": 50, "-10": 20}}}}}}
    f = B.failure_modes(stored, TRAVEL)
    net = [x for x in f if "net-displacement" in x["mode"]][0]
    assert net["value"] == {"15 to 19.5 N": {"-6": 50, "-8": 34, "-10": 20}}
    assert any(x["source"] == TRAVEL and "unresolved" in x["mode"] for x in f)


def test_main_writes_the_tables_and_manifest_and_never_overwrites(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    for p in (B.STORED, TRAVEL):
        fp = root / p; fp.parent.mkdir(parents=True, exist_ok=True); fp.write_text("{}")
    write_sources(str(root), digests={}, stored_sha=hashlib.sha256(b"{}").hexdigest(), matched=matched_record(cohort=["1990-15N-01"]))
    items = [item([3, 4, 5, 6], [60.0, 58.0, 55.0, 51.5]), item([8, 9, 10, 11], [65.0, 62.0, 60.0, 56.0], band="25 to 29.5 N")]
    monkeypatch.setattr(B, "load_records", lambda d: ({}, []))
    monkeypatch.setattr(B, "recompute", lambda by: items)
    monkeypatch.setattr(B, "check_identity", lambda its, st: None)
    monkeypatch.setattr(B, "check_inputs", lambda dg, st: None)
    monkeypatch.setattr(B, "available_days", lambda by: AVAIL)
    loaded = {"records": {}, "histories": {"path": "h", "sha256": hashlib.sha256(b"{}").hexdigest()}, **json.loads(json.dumps(LOADED))}
    monkeypatch.setattr(B.C, "load_inputs", lambda ns: ({}, None, {}, None, json.loads(json.dumps(loaded))))
    part = [{"descriptor": "first_day", "history_id": "1990-15N-01", "participates": True, "days_used": 1, "days_excluded": 0, "exclusion_reason": "",
             "used_offsets": "0"},
            {"descriptor": "daily_profile", "history_id": "1990-15N-01", "participates": True, "days_used": 5, "days_excluded": 0, "exclusion_reason": "",
             "used_offsets": "0 1 2 3 4"}]
    monkeypatch.setattr(B, "participation", lambda items, hist, by_x, records: (part, {"first_day starts": "equal"}))
    monkeypatch.setattr(B, "failure_modes", lambda st, tr: [])
    out = tmp_path / "out"
    assert B.main(["--out-dir", str(out), "--root", str(root), "--travel-source", TRAVEL]) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["counts"] == {"histories": 2, "by_band": {"15 to 19.5 N": 1, "25 to 29.5 N": 1}, "primary": 1, "members": 8,
                             "starts_on_first_available_day": 0, "ends_on_last_available_day": 0, "participation_rows": 2}
    assert man["participation_check"] == {"first_day starts": "equal", "daily_profile_matched cohort": "equal"}
    assert man["tables"]["participation"]["rows"] == 2 and man["product"].endswith("version 3") and man["tables"]["participation"]["columns"][-1] == "used_offsets"
    assert (man["built"], man["first_version_built"]) == ("2026-10-23", "2026-10-22")
    assert list(csv.DictReader(open(out / "participation.csv")))[1]["used_offsets"] == "0 1 2 3 4"
    assert man["tables"]["members"]["sha256"] == hashlib.sha256((out / "members.csv").read_bytes()).hexdigest()
    assert man["travel_status"]["vocabulary"] == ["unresolved"] and man["travel_status"]["source"]["path"] == TRAVEL
    assert man["definition"]["class_rule"] == {"min_days": 4, "max_net_deg": -8.0, "min_westward_step_fraction": 0.75} and man["definition"]["prominence_ms"] == A.PROMINENCE
    rows = list(csv.DictReader(open(out / "histories.csv")))
    assert [r["history_id"] for r in rows] == ["1990-15N-01", "1990-25N-01"] and list(rows[0]) == B.HIST_COLUMNS
    assert man["reference_check"] == {"path": None, "performed": False, "result": "not performed, no retained version was given"}
    assert man["member_digest"] == B.member_digest(B.history_rows(items, AVAIL)[1])
    with pytest.raises(SystemExit, match="never overwritten"):
        B.main(["--out-dir", str(out), "--root", str(root), "--travel-source", TRAVEL])
    out2 = tmp_path / "out2"
    assert B.main(["--out-dir", str(out2), "--root", str(root), "--travel-source", TRAVEL, "--reference", str(out), "--build-note", "a rebuild"]) == 0
    man2 = json.loads((out2 / "manifest.json").read_text())
    assert man2["reference_check"]["performed"] is True and man2["reference_check"]["result"] == "equal" and man2["build_note"] == "a rebuild"
    assert man2["member_digest"] == man["member_digest"] and (out2 / "members.csv").read_bytes() == (out / "members.csv").read_bytes()
    other = tmp_path / "other"
    monkeypatch.setattr(B, "recompute", lambda by: items[:1])
    with pytest.raises(SystemExit, match="differs from the retained reference"):
        B.main(["--out-dir", str(other), "--root", str(root), "--travel-source", TRAVEL, "--reference", str(out)])
    assert not other.exists()                                                  # nothing is written when the reference differs
    for k, changed in (("xband", {"eval60_xband_1990_07.json.gz": "3" * 64}), ("inject", {"path": "i", "sha256": "9" * 64})):
        monkeypatch.setattr(B.C, "load_inputs", lambda ns, k=k, changed=changed: ({}, None, {}, None, dict(loaded, **{k: changed})))
        with pytest.raises(SystemExit, match=f"the {k} input differs"):
            B.main(["--out-dir", str(other), "--root", str(root), "--travel-source", TRAVEL])
    monkeypatch.setattr(B.C, "load_inputs", lambda ns: ({}, None, {}, None, {k: v for k, v in loaded.items() if k != "shape"}))
    with pytest.raises(SystemExit, match="builder's load names no shape input"):
        B.main(["--out-dir", str(other), "--root", str(root), "--travel-source", TRAVEL])
    def refuse_inputs(dg, st):
        raise SystemExit("REFUSED: the month records differ")
    monkeypatch.setattr(B, "check_inputs", refuse_inputs)
    with pytest.raises(SystemExit, match="month records differ"):
        B.main(["--out-dir", str(other), "--root", str(root), "--travel-source", TRAVEL])
    with pytest.raises(SystemExit, match="travel-status source .* does not exist"):
        B.main(["--out-dir", str(other), "--root", str(root), "--travel-source", os.path.join("notes", "absent.md")])
