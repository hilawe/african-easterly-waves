"""The annual storm-link inventory, bound on synthetic joined and filter artifacts.

MUTATION LIST, written before the assertions:
  I1 an unresolved key is counted as an event in the first basin column;
  I2 the event basin is read from `from` (the file of origin) instead of `event_basin`;
  I3 a year present in the filter summary but absent from the join is tolerated;
  I4 a year whose key count differs between the two sources is tolerated;
  I5 per-year totals are not reconciled against the joined artifact's own counts;
  I6 the report forms a rate (any "/" or "percent" between keys and tracks).
Added after a review of the first version found its prose typed rather than rendered:
  I7 the closing prose states fixed counts (512, 13) instead of the inventory's totals;
  I8 the closing prose states one fixed reason instead of the recorded reasons;
  I9 the filter summary's storm-key total is not reconciled (513 beside 512 keys passes).
"""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "storm_link_inventory", os.path.join(HERE, "..", "scripts", "storm_link_inventory.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["storm_link_inventory"] = mod
    spec.loader.exec_module(mod)
    return mod


def joined_fixture():
    res = {"1999|A|g": {"state": "validated", "from": "NA", "event_basin": "NA"},
           "1999|B|g": {"state": "validated", "from": "EP", "event_basin": "EP"},
           "1999|C|g": {"state": "validated", "from": "both", "event_basin": "NA"},
           "1999|D|g": {"state": "unresolved"},
           "2000|E|g": {"state": "validated", "from": "EP", "event_basin": "NA"},  # crosser
           "2000|F|g": {"state": "unresolved"}}
    return {"resolutions": res, "keys": 6, "validated": 4,
            "unresolved": [{"key": ["1999", "D", "g"], "reason": "matched in neither file"},
                           {"key": ["2000", "F", "g"],
                            "reason": "matched in NA while EP holds an identity conflict "
                                      "(ambiguous)"}],
            "validated_by_event_basin": {"NA": 3, "EP": 1}}


def filter_fixture():
    return {"years": {"1999": {"n_kept": 90, "n_distinct_storms_kept": 4},
                      "2000": {"n_kept": 80, "n_distinct_storms_kept": 2}},
            "totals": {"n_kept": 170, "n_distinct_storms_kept": 6}}


def test_inventory_counts_events_by_event_basin_and_keeps_unresolved_explicit():
    M = _load()
    inv = M.inventory(joined_fixture(), filter_fixture())
    assert inv["years"] == {
        "1999": {"retained_tracks": 90, "storm_keys": 4, "events_NA": 2, "events_EP": 1,
                 "unresolved_keys": 1},
        "2000": {"retained_tracks": 80, "storm_keys": 2, "events_NA": 1, "events_EP": 0,
                 "unresolved_keys": 1}}
    assert inv["totals"] == {"retained_tracks": 170, "storm_keys": 6, "events_NA": 3,
                             "events_EP": 1, "unresolved_keys": 2}
    assert inv["reconciled_against"] == {"storm_keys": 6, "storm_keys_in_filter_summary": 6,
                                         "validated": 4, "unresolved_keys": 2,
                                         "retained_tracks": 170}
    assert inv["unresolved_reasons"] == {
        "matched in neither file": 1,
        "matched in NA while EP holds an identity conflict (ambiguous)": 1}


def test_inventory_refuses_disagreeing_sources():
    M = _load()
    j, f = joined_fixture(), filter_fixture()
    f["years"]["2001"] = {"n_kept": 1, "n_distinct_storms_kept": 0}
    f["totals"]["n_kept"] = 171
    with pytest.raises(ValueError, match="years differ"):
        M.inventory(j, f)
    j, f = joined_fixture(), filter_fixture()
    f["years"]["1999"]["n_distinct_storms_kept"] = 5
    with pytest.raises(ValueError, match="1999: 4 keys in the join but 5"):
        M.inventory(j, f)
    j, f = joined_fixture(), filter_fixture()
    j["validated"] = 5
    with pytest.raises(ValueError, match="do not sum to the joined artifact's validated"):
        M.inventory(j, f)
    j, f = joined_fixture(), filter_fixture()
    j["validated_by_event_basin"]["NA"] = 2
    with pytest.raises(ValueError, match="events in NA"):
        M.inventory(j, f)
    j, f = joined_fixture(), filter_fixture()
    f["totals"]["n_kept"] = 171
    with pytest.raises(ValueError, match="retained tracks do not sum"):
        M.inventory(j, f)
    j, f = joined_fixture(), filter_fixture()
    f["totals"]["n_distinct_storms_kept"] = 7          # the review's 513-beside-512 case
    with pytest.raises(ValueError, match="filter summary's storm-key total"):
        M.inventory(j, f)
    j, f = joined_fixture(), filter_fixture()
    j["resolutions"]["1999|A|g"]["event_basin"] = "WP"
    j["validated_by_event_basin"] = {"NA": 2, "EP": 1}
    with pytest.raises(ValueError, match="outside"):
        M.inventory(j, f)


def test_the_report_and_artifact_come_from_one_table_and_form_no_rate(tmp_path):
    M = _load()
    pj, pf = tmp_path / "joined.json", tmp_path / "filter.json"
    pj.write_text(json.dumps(joined_fixture()))
    pf.write_text(json.dumps(filter_fixture()))
    out, rep = tmp_path / "inv.json", tmp_path / "INV.md"
    assert M.main(["--joined", str(pj), "--filter-summary", str(pf), "--out", str(out),
                   "--report", str(rep)]) == 0
    with open(out) as fh:
        v = json.load(fh)
    assert v["totals"]["events_NA"] == 3 and v["inputs"]["joined"]["sha256"] == M._sha256(str(pj))
    text = rep.read_text()
    assert "| 1999 | 4 | 2 | 1 | 1 | 90 |" in text
    assert "| All years | 6 | 3 | 1 | 2 | 170 |" in text
    assert "no rate" in text and "percent" not in text and "%" not in text
    # the prose is rendered from the inventory: this fixture's counts and BOTH reasons
    assert "The 2 unresolved keys belong to the 6-key population." in text
    assert "does not establish whether the associated tracks developed" in text
    assert "neither developers" not in text and "no link state" not in text
    assert "\n- 1 matched in neither file\n" in text
    assert "\n- 1 matched in NA while EP holds an identity conflict (ambiguous)\n" in text
    assert "(IBTrACS)" in text
    assert "512" not in text and "13 unresolved" not in text
    assert chr(0x2014) not in text and chr(0x2013) not in text   # em and en dash
    # a refusal writes neither file
    j = joined_fixture()
    j["validated"] = 5
    pj.write_text(json.dumps(j))
    out2, rep2 = tmp_path / "inv2.json", tmp_path / "INV2.md"
    assert M.main(["--joined", str(pj), "--filter-summary", str(pf), "--out", str(out2),
                   "--report", str(rep2)]) == 2
    assert not out2.exists() and not rep2.exists()
