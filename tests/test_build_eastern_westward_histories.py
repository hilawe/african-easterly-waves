"""Tests for the eastern westward-history subset builder (`EASTERN_WESTWARD_HISTORIES_SUBSET_2026-10-22.md`)."""

import csv
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import build_eastern_westward_histories as B  # noqa: E402

HI, W, A = B.HI, B.W, B.A
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
    rows, members = B.history_rows(items)
    assert [r["history_id"] for r in rows] == ["1990-15N-01", "1990-15N-02", "1990-25N-01"]
    r = rows[0]
    assert (r["first_date"], r["last_date"], r["days"], r["net_deg"], r["deepest_anomaly_ms"]) == ("1990-06-04", "1990-06-07", 4, -8.0, -6.25)
    assert r["westward_step_fraction"] == 2 / 3 and r["speed_deg_per_day"] == -8.0 / 3 and r["primary"] is True and rows[2]["primary"] is False
    assert {x["travel_status"] for x in rows} == {"unresolved"} and r["rule_class"] == HI.classify(items[0][3])
    assert len(members) == 12 and members[1] == {"history_id": "1990-15N-01", "member": 1, "date": "1990-06-05", "day_of_season": 4, "lon": 58.0, "anomaly_ms": -6.25}
    still, _ = B.history_rows([item([3, 4, 5, 6], [60.0, 58.0, 58.0, 52.0])])
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


def write_sources(root, n=86, failures=5, digests=DIGESTS, stored_sha=STORED_SHA):
    recs = {"shape": {"results": {"composite": {"primary": {"center": -5.0, "east": 0.4, "west": 0.1, "members": 513, "histories": n}}}},
            "coherence": {"results": {"measure": {"primary": {"departure": -0.38, "histories": n}}}},
            "lobe": {"results": {"measure": {"primary": {"carried": 0.47, "pinned": 0.16, "contrast": 0.31, "samples": 154, "histories": n}}}},
            "joint": {"results": {"measure": {"first_day": 0.13, "profile": [0.13, 0.24], "profile_days": [77, 81], "histories": n}}},
            "prestart": {"verdict": {"adequate": False, "failures": ["x"] * failures}}}
    for k, v in recs.items():
        v["generated_by"] = {"records": dict(digests), "histories": {"path": B.STORED, "sha256": stored_sha}}
        p = os.path.join(root, B.SOURCES[k]); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(v, fh)


def test_descriptors_are_read_with_their_sources_and_population(tmp_path):
    write_sources(str(tmp_path))
    d = {x["name"]: x for x in B.descriptors(str(tmp_path), DIGESTS, STORED_SHA)}
    assert list(d) == ["average_structure", "coherence_departure", "lobe_contrast", "first_day", "daily_profile", "prestart"]
    assert d["coherence_departure"]["value"] == -0.38 and d["lobe_contrast"]["value"]["contrast"] == 0.31 and d["first_day"]["value"] == {"ratio": 0.13, "starts": 77}
    assert d["average_structure"]["value"]["member_days"] == 513 and d["daily_profile"]["value"]["history_days_by_day"] == [77, 81]
    assert d["lobe_contrast"]["value"]["history_days"] == 154
    assert d["prestart"]["value"] is None and d["prestart"]["status"] == "not measured" and "failed in 5 of its 16" in d["prestart"]["definition"]
    p = tmp_path / B.SOURCES["lobe"]
    assert d["lobe_contrast"]["source"] == {"path": B.SOURCES["lobe"], "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    assert all("not of any one history" in x["population"] for x in d.values())
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
    rows, members = B.history_rows(items)
    texts = {"histories": B.csv_text(B.HIST_COLUMNS, rows), "members": B.csv_text(B.MEMBER_COLUMNS, members)}
    ref = tmp_path / "ref"; ref.mkdir()
    for k, name in (("histories", "histories.csv"), ("members", "members.csv")):
        B.write_text(str(ref / name), texts[k])
    (ref / "manifest.json").write_text(json.dumps({"tables": {k: {"sha256": hashlib.sha256((ref / f"{k}.csv").read_bytes()).hexdigest()} for k in ("histories", "members")}}))
    assert B.check_reference(str(ref), texts)["result"] == "equal"
    for change in ("lon", "depth"):
        it = [item([3, 4, 5, 6, 7], [54.0, 52.0 if change == "lon" else 51.5, 51.5, 46.5, 42.5], [-2.76, -5.16 if change == "depth" else -4.16, -9.04, -6.18, -4.58])]
        assert B.check_identity(it, stored_for(items)) is None                 # the summaries cannot see an interior change
        r2, m2 = B.history_rows(it)
        assert B.member_digest(m2) != B.member_digest(members)
        t2 = {"histories": B.csv_text(B.HIST_COLUMNS, r2), "members": B.csv_text(B.MEMBER_COLUMNS, m2)}
        with pytest.raises(SystemExit, match="differs from the retained reference"):
            B.check_reference(str(ref), t2)
    (ref / "members.csv").write_text(texts["members"].replace("-4.16", "-4.17"))
    with pytest.raises(SystemExit, match="reference members.csv does not match its manifest"):
        B.check_reference(str(ref), texts)


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
    write_sources(str(root), digests={}, stored_sha=hashlib.sha256(b"{}").hexdigest())
    items = [item([3, 4, 5, 6], [60.0, 58.0, 55.0, 51.5]), item([8, 9, 10, 11], [65.0, 62.0, 60.0, 56.0], band="25 to 29.5 N")]
    monkeypatch.setattr(B, "load_records", lambda d: ({}, []))
    monkeypatch.setattr(B, "recompute", lambda by: items)
    monkeypatch.setattr(B, "check_identity", lambda its, st: None)
    monkeypatch.setattr(B, "check_inputs", lambda dg, st: None)
    monkeypatch.setattr(B, "failure_modes", lambda st, tr: [])
    out = tmp_path / "out"
    assert B.main(["--out-dir", str(out), "--root", str(root), "--travel-source", TRAVEL]) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["counts"] == {"histories": 2, "by_band": {"15 to 19.5 N": 1, "25 to 29.5 N": 1}, "primary": 1, "members": 8}
    assert man["tables"]["members"]["sha256"] == hashlib.sha256((out / "members.csv").read_bytes()).hexdigest()
    assert man["travel_status"]["vocabulary"] == ["unresolved"] and man["travel_status"]["source"]["path"] == TRAVEL
    assert man["definition"]["class_rule"] == {"min_days": 4, "max_net_deg": -8.0, "min_westward_step_fraction": 0.75} and man["definition"]["prominence_ms"] == A.PROMINENCE
    rows = list(csv.DictReader(open(out / "histories.csv")))
    assert [r["history_id"] for r in rows] == ["1990-15N-01", "1990-25N-01"] and list(rows[0]) == B.HIST_COLUMNS
    assert man["reference_check"] == {"path": None, "performed": False, "result": "not performed, no retained version was given"}
    assert man["member_digest"] == B.member_digest(B.history_rows(items)[1])
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
    def refuse_inputs(dg, st):
        raise SystemExit("REFUSED: the month records differ")
    monkeypatch.setattr(B, "check_inputs", refuse_inputs)
    with pytest.raises(SystemExit, match="month records differ"):
        B.main(["--out-dir", str(other), "--root", str(root), "--travel-source", TRAVEL])
    with pytest.raises(SystemExit, match="travel-status source .* does not exist"):
        B.main(["--out-dir", str(other), "--root", str(root), "--travel-source", os.path.join("notes", "absent.md")])
