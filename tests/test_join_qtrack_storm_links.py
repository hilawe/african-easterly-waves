"""The join of two storm-link artifacts into one resolution per key.

MUTATION LIST, written before the assertions:
  J1 a key matched in both files to DIFFERENT storms is resolved instead of refused;
  J2 a key-set mismatch is tolerated (the extra key silently dropped);
  J3 a flip (rejected in one file, matched in the other) is not recorded;
  J4 a key matched in neither file is counted as validated;
  J5 `from` names the wrong file for a single-file match.
Added after a review of the first version found two survivors and three contract gaps:
  J6 event-time equality dropped from the both-matched check (same SID, different
     event times, must not validate silently);
  J7 the single-match identifier is read from the FIRST artifact whichever file
     matched (a null SID with the right `from` label passed every count);
  J8 a match beside `ambiguous` or `shared_identifier` in the other file validates;
  J9 event_basin is dropped from a validated result, or the both-matched branch takes
     the first file's basin without comparing;
  J10 the compatibility check accepts differing rules, source hashes, population
      hashes or track membership;
  J11 two validated keys naming one storm identifier are accepted.
"""
import copy
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "join_qtrack_storm_links", os.path.join(HERE, "..", "scripts",
                                                "join_qtrack_storm_links.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["join_qtrack_storm_links"] = mod
    spec.loader.exec_module(mod)
    return mod


def row(name, genesis, status, sid=None, event_time=None, basin=None, subbasin=None,
        offset=None, km=None, nearest=None, systems=(1,)):
    return {"name": name, "qtrack_genesis": genesis, "status": status,
            "systems": list(systems), "event_basin": basin, "event_subbasin": subbasin,
            "sid": sid, "event_time": event_time, "offset_hours": offset,
            "wave_to_event_km": km, "nearest_sid": nearest}


RULE = {"max_offset_hours": 72.0, "max_distance_km": 500.0, "genesis_event": "first non-DB"}
SOURCES = {"src/aew/qtrack_storms.py": "s1", "src/aew/qtrack.py": "s2",
           "scripts/link_qtrack_storms.py": "s3"}


def artifact(rows_by_year, ibtracs="x", rule=RULE, sources=SOURCES, year_hash="h"):
    return {"years": {y: {"rows": rs, "input_sha256": f"{year_hash}{y}"}
                      for y, rs in rows_by_year.items()},
            "ibtracs_sha256": ibtracs, "rule": copy.deepcopy(rule),
            "source_sha256": dict(sources)}


def base_pair():
    # A: matched in NA only, negative in EP; B: matched in EP after NA rejected it on
    # distance (a flip); C: matched in BOTH to the same storm and event, a crosser
    # whose event basin is NA while its subbasin is CS; D: matched in neither.
    na = artifact({"1999": [
        row("ALPHA", "1999-08-01 00Z", "matched", "S1", "1999-08-01", "NA", "NA", 0.0, 100.0),
        row("UNNAMED", "1999-09-01 00Z", "rejected_distance", nearest="S9", offset=24.0,
            km=4000.0),
        row("CROSS", "1999-10-01 00Z", "matched", "S3", "1999-10-01", "NA", "CS", 0.0, 50.0),
        row("NONE", "1999-07-01 00Z", "no_candidate")]}, ibtracs="na")
    ep = artifact({"1999": [
        row("ALPHA", "1999-08-01 00Z", "no_candidate"),
        row("UNNAMED", "1999-09-01 00Z", "matched", "S2", "1999-09-01", "EP", "MM", 0.0, 200.0),
        row("CROSS", "1999-10-01 00Z", "matched", "S3", "1999-10-01", "NA", "CS", 0.0, 50.0),
        row("NONE", "1999-07-01 00Z", "no_candidate")]}, ibtracs="ep")
    return na, ep


def test_join_resolves_once_per_key_with_the_full_event_payload_and_records_flips():
    J = _load()
    na, ep = base_pair()
    j = J.join_links(na, ep, ("NA", "EP"))
    assert j["keys"] == 4 and j["validated"] == 3 and j["distinct_storms_validated"] == 3
    assert j["validated_by_source"] == {"NA": 1, "EP": 1, "both": 1}
    assert j["validated_by_event_basin"] == {"NA": 2, "EP": 1}
    r = j["resolutions"]
    assert r["1999|ALPHA|1999-08-01 00Z"] == {
        "state": "validated", "from": "NA", "sid": "S1", "event_time": "1999-08-01",
        "event_basin": "NA", "event_subbasin": "NA",
        "states": {"NA": "matched", "EP": "no_candidate"}}
    assert r["1999|UNNAMED|1999-09-01 00Z"] == {
        "state": "validated", "from": "EP", "sid": "S2", "event_time": "1999-09-01",
        "event_basin": "EP", "event_subbasin": "MM",
        "states": {"NA": "rejected_distance", "EP": "matched"}}
    assert r["1999|CROSS|1999-10-01 00Z"] == {
        "state": "validated", "from": "both", "sid": "S3", "event_time": "1999-10-01",
        "event_basin": "NA", "event_subbasin": "CS",
        "states": {"NA": "matched", "EP": "matched"}}
    assert r["1999|NONE|1999-07-01 00Z"]["state"] == "unresolved"
    assert [u["key"] for u in j["unresolved"]] == [["1999", "NONE", "1999-07-01 00Z"]]
    assert j["flips"] == [{"key": ["1999", "UNNAMED", "1999-09-01 00Z"], "validated_in": "EP",
                           "offset_hours": 0.0, "wave_to_event_km": 200.0,
                           "other_state": "rejected_distance", "other_nearest_sid": "S9",
                           "other_offset_hours": 24.0, "other_wave_to_event_km": 4000.0}]
    assert j["state_pairs"] == {"matched|matched": 1, "matched|no_candidate": 1,
                                "no_candidate|no_candidate": 1,
                                "rejected_distance|matched": 1}
    # the same pair in the other order resolves the same keys from the other labels
    k = J.join_links(ep, na, ("EP", "NA"))
    assert k["resolutions"]["1999|ALPHA|1999-08-01 00Z"]["from"] == "NA"
    assert k["resolutions"]["1999|ALPHA|1999-08-01 00Z"]["sid"] == "S1"
    assert k["resolutions"]["1999|UNNAMED|1999-09-01 00Z"]["from"] == "EP"
    assert k["resolutions"]["1999|UNNAMED|1999-09-01 00Z"]["sid"] == "S2"


def test_a_conflict_state_beside_a_match_stays_unresolved():
    J = _load()
    for conflict in ("ambiguous", "shared_identifier"):
        na, ep = base_pair()
        na["years"]["1999"]["rows"][0]["status"] = conflict       # ALPHA: NA conflict
        ep["years"]["1999"]["rows"][0] = row("ALPHA", "1999-08-01 00Z", "matched", "S1",
                                             "1999-08-01", "EP", "MM", 0.0, 10.0)
        j = J.join_links(na, ep, ("NA", "EP"))
        r = j["resolutions"]["1999|ALPHA|1999-08-01 00Z"]
        assert r["state"] == "unresolved" and conflict in r["reason"]
        assert j["validated"] == 2
        assert ["1999", "ALPHA", "1999-08-01 00Z"] in [u["key"] for u in j["unresolved"]]


def test_both_matched_requires_equal_event_metadata():
    J = _load()
    # same SID, different event TIMES: unresolved with the disagreement recorded
    na, ep = base_pair()
    ep["years"]["1999"]["rows"][2]["event_time"] = "1999-10-02"
    j = J.join_links(na, ep, ("NA", "EP"))
    r = j["resolutions"]["1999|CROSS|1999-10-01 00Z"]
    assert r["state"] == "unresolved" and "event_time" in r["reason"]
    assert r["disagreement"] == {"event_time": {"NA": "1999-10-01", "EP": "1999-10-02"}}
    # same SID and time, different event BASIN: also unresolved, never the first file's
    na, ep = base_pair()
    ep["years"]["1999"]["rows"][2]["event_basin"] = "EP"
    ep["years"]["1999"]["rows"][2]["event_subbasin"] = "MM"
    r = J.join_links(na, ep, ("NA", "EP"))["resolutions"]["1999|CROSS|1999-10-01 00Z"]
    assert r["state"] == "unresolved"
    assert r["disagreement"] == {"event_basin": {"NA": "NA", "EP": "EP"},
                                 "event_subbasin": {"NA": "CS", "EP": "MM"}}
    # different SIDs: refused outright
    na, ep = base_pair()
    ep["years"]["1999"]["rows"][2]["sid"] = "S4"
    with pytest.raises(ValueError, match="different storms"):
        J.join_links(na, ep, ("NA", "EP"))


def test_incompatible_runs_are_refused_each_for_its_own_reason():
    J = _load()
    na, ep = base_pair()
    ep["rule"]["max_distance_km"] = 5000.0
    with pytest.raises(ValueError, match=r"rules differ.*max_distance_km"):
        J.join_links(na, ep, ("NA", "EP"))
    na, ep = base_pair()
    ep["source_sha256"]["src/aew/qtrack_storms.py"] = "other"
    with pytest.raises(ValueError, match="source hash for src/aew/qtrack_storms.py"):
        J.join_links(na, ep, ("NA", "EP"))
    na, ep = base_pair()
    ep["years"]["1999"]["input_sha256"] = "other"
    with pytest.raises(ValueError, match="population hash for 1999"):
        J.join_links(na, ep, ("NA", "EP"))
    na, ep = base_pair()
    ep["years"]["1999"]["rows"][0]["systems"] = [7]
    with pytest.raises(ValueError, match="different track membership"):
        J.join_links(na, ep, ("NA", "EP"))
    na, ep = base_pair()
    del ep["rule"]
    with pytest.raises(ValueError, match="records no `rule`"):
        J.join_links(na, ep, ("NA", "EP"))
    # the two best-track hashes differ by design and are not compared
    na, ep = base_pair()
    assert na["ibtracs_sha256"] != ep["ibtracs_sha256"]
    assert J.join_links(na, ep, ("NA", "EP"))["validated"] == 3
    # differing key sets
    na, ep = base_pair()
    ep["years"]["1999"]["rows"].append(row("BETA", "1999-08-02 00Z", "no_candidate"))
    with pytest.raises(ValueError, match="key sets differ"):
        J.join_links(na, ep, ("NA", "EP"))


def test_one_storm_validated_for_two_keys_is_refused():
    J = _load()
    na, ep = base_pair()
    # ALPHA (NA-only) and UNNAMED (EP-only) both resolve to storm S1
    ep["years"]["1999"]["rows"][1]["sid"] = "S1"
    with pytest.raises(ValueError, match="validated for more than one key"):
        J.join_links(na, ep, ("NA", "EP"))


def test_the_cli_writes_the_resolution_and_refuses_with_nothing_written(tmp_path):
    J = _load()
    na, ep = base_pair()
    pa, pe, out = tmp_path / "na.json", tmp_path / "ep.json", tmp_path / "out.json"
    pa.write_text(json.dumps(na))
    pe.write_text(json.dumps(ep))
    assert J.main(["--first", str(pa), "--second", str(pe), "--out", str(out)]) == 0
    with open(out) as fh:
        v = json.load(fh)
    assert v["validated"] == 3 and v["validated_by_event_basin"] == {"NA": 2, "EP": 1}
    assert v["resolutions"]["1999|UNNAMED|1999-09-01 00Z"]["sid"] == "S2"
    assert v["inputs"]["NA"]["sha256"] == J._sha256(str(pa))
    assert v["inputs"]["EP"]["ibtracs_sha256"] == "ep"
    assert v["source_sha256"]["scripts/join_qtrack_storm_links.py"] == J._sha256(
        os.path.join(HERE, "..", "scripts", "join_qtrack_storm_links.py"))
    assert "ibtracs_sha256" in v["compatibility_checked"]
    # a refusal writes nothing
    ep["years"]["1999"]["rows"][2]["sid"] = "S4"
    pe.write_text(json.dumps(ep))
    out2 = tmp_path / "out2.json"
    assert J.main(["--first", str(pa), "--second", str(pe), "--out", str(out2)]) == 2
    assert not out2.exists()
