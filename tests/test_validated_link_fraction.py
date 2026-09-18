"""The validated-link fraction over shared-tail groups, bound on a synthetic pair, a
triple, singletons, and tagless groups.

MUTATION LIST, written before the assertions:
  G1 the denominator counts tracks instead of groups (a pair counts twice);
  G2 an unresolved key inside a group that also holds a validated link is dropped;
  G3 a group whose only tags are unresolved keys is counted as linked;
  G4 a group holding both basins is forced onto one basin;
  G5 a key whose tracks span two groups is tolerated instead of refused;
  G6 groups with no storm tags are not derived, so the three categories do not sum
     to the groups;
  G7 the report names a development probability;
  G8 an isolated tagged track is not a singleton group (its link is lost).
Added after a review of the first version (four findings):
  P1 the link file's digest is not checked against the joined artifact's inputs;
  P2 per-year population hashes (or the filter module hash) are not compared;
  P3 a mixed supported-and-unsupported basin set is reduced by iteration order
     instead of refused (bound by the per-key basin check; the fallback branch in
     the composition step is then unreachable BY EXHAUSTION, since a linked group's
     basin set is a nonempty subset of the two supported basins and all three such
     sets are named, so a mutation of that branch alone survives and is not counted);
  P4 the North Atlantic and eastern Pacific labels are swapped (the first fixture was
     symmetric and this survived);
  P5 the filter summary's pair count is labeled or reconciled as the group count.
"""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "validated_link_fraction",
        os.path.join(HERE, "..", "scripts", "validated_link_fraction.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validated_link_fraction"] = mod
    spec.loader.exec_module(mod)
    return mod


def fixtures():
    # 1999: six tracks; pair {1,2}, triple {3,4,5}, singleton {6} = three groups.
    #   A on [1,2] validated NA S1; B on [3] validated EP S2; C on [4] unresolved
    #   (inside the linked triple); D on [6] unresolved (a group with only an
    #   unresolved tag). No tagless group.
    # 2000: five tracks; pair {1,2}, singletons {3}, {4}, {5} = four groups.
    #   E on [1] validated NA S3 and F on [2] validated EP S4 (one group, BOTH
    #   basins); G on [3] validated NA S5 and H on [4] validated NA S6, TWO isolated
    #   tagged tracks that must stay two singleton groups (a first fixture had one
    #   per year and the mutation collapsing singletons survived); {5} tagless. The
    #   basin counts are ASYMMETRIC on purpose (2000 has two NA-only groups and no
    #   EP-only group), so swapping the two labels changes the expected rows.
    # Provenance: every year hash is shared by the filter summary and the link
    # artifact, both record the filter module hash "q", and the joined artifact
    # records the link digest "L" that the tests pass in.
    filt = {"years": {"1999": {"n_kept": 6, "shared_tail_groups": [[1, 2], [3, 4, 5]],
                               "n_shared_tail_pairs": 4, "input_sha256": "h1999"},
                      "2000": {"n_kept": 5, "shared_tail_groups": [[1, 2]],
                               "n_shared_tail_pairs": 1, "input_sha256": "h2000"}},
            "totals": {"n_kept": 11, "n_shared_tail_pairs": 5},
            "source_sha256": {"src/aew/qtrack.py": "q"}}
    links = {"source_sha256": {"src/aew/qtrack.py": "q"},
             "years": {"1999": {"input_sha256": "h1999", "rows": [
        {"name": "A", "qtrack_genesis": "g", "systems": [1, 2]},
        {"name": "B", "qtrack_genesis": "g", "systems": [3]},
        {"name": "C", "qtrack_genesis": "g", "systems": [4]},
        {"name": "D", "qtrack_genesis": "g", "systems": [6]}]},
        "2000": {"input_sha256": "h2000", "rows": [
            {"name": "E", "qtrack_genesis": "g", "systems": [1]},
            {"name": "F", "qtrack_genesis": "g", "systems": [2]},
            {"name": "G", "qtrack_genesis": "g", "systems": [3]},
            {"name": "H", "qtrack_genesis": "g", "systems": [4]}]}}}
    joined = {"resolutions": {
        "1999|A|g": {"state": "validated", "event_basin": "NA", "sid": "S1"},
        "1999|B|g": {"state": "validated", "event_basin": "EP", "sid": "S2"},
        "1999|C|g": {"state": "unresolved"},
        "1999|D|g": {"state": "unresolved"},
        "2000|E|g": {"state": "validated", "event_basin": "NA", "sid": "S3"},
        "2000|F|g": {"state": "validated", "event_basin": "EP", "sid": "S4"},
        "2000|G|g": {"state": "validated", "event_basin": "NA", "sid": "S5"},
        "2000|H|g": {"state": "validated", "event_basin": "NA", "sid": "S6"}},
        "validated": 6, "unresolved": [{"key": ["1999", "C", "g"]}, {"key": ["1999", "D", "g"]}],
        "distinct_storms_validated": 6, "inputs": {"NA": {"sha256": "L"}, "EP": {"sha256": "M"}}}
    return filt, links, joined


LINK_SHA = "L"


def test_groups_are_the_unit_and_every_rule_of_the_declaration_holds():
    M = _load()
    r = M.fraction(*fixtures(), links_sha256=LINK_SHA)
    assert r["years"]["1999"] == {
        "retained_tracks": 6, "shared_tail_groups": 2, "shared_tail_pairs": 4, "groups": 3,
        "groups_with_validated_links": 2, "groups_with_only_unresolved_tags": 1,
        "groups_with_no_storm_tags": 0,
        "linked_groups_by_basin": {"NA": 1, "EP": 1, "both": 0},
        "validated_keys": 2, "unresolved_keys": 2, "unresolved_keys_in_linked_groups": 1,
        "distinct_linked_storms": 2, "validated_link_fraction": 2 / 3}
    assert r["years"]["2000"] == {
        "retained_tracks": 5, "shared_tail_groups": 1, "shared_tail_pairs": 1, "groups": 4,
        "groups_with_validated_links": 3, "groups_with_only_unresolved_tags": 0,
        "groups_with_no_storm_tags": 1,
        "linked_groups_by_basin": {"NA": 2, "EP": 0, "both": 1},
        "validated_keys": 4, "unresolved_keys": 0, "unresolved_keys_in_linked_groups": 0,
        "distinct_linked_storms": 4, "validated_link_fraction": 3 / 4}
    t = r["totals"]
    assert t["groups"] == 7 and t["groups_with_validated_links"] == 5
    assert t["groups_with_only_unresolved_tags"] == 1 and t["groups_with_no_storm_tags"] == 1
    assert t["linked_groups_by_basin"] == {"NA": 3, "EP": 1, "both": 1}
    assert t["unresolved_keys"] == 2 and t["distinct_linked_storms"] == 6
    assert t["validated_link_fraction"] == 5 / 7
    # the pair count is reconciled under its own name and is not the group count
    assert t["shared_tail_pairs"] == 5 and t["shared_tail_groups"] == 3
    assert r["reconciled_against"]["shared_tail_pairs"] == 5
    assert "shared_tail_groups" not in r["reconciled_against"]


def test_refusals():
    M = _load()
    run = lambda f, l, j: M.fraction(f, l, j, links_sha256=LINK_SHA)
    filt, links, joined = fixtures()
    links["years"]["1999"]["rows"][0]["systems"] = [1, 3]        # spans pair and triple
    with pytest.raises(ValueError, match="span more than one group"):
        run(filt, links, joined)
    filt, links, joined = fixtures()
    joined["validated"] = 5
    with pytest.raises(ValueError, match="validated_keys sum to 6, the source says 5"):
        run(filt, links, joined)
    filt, links, joined = fixtures()
    filt["totals"]["n_shared_tail_pairs"] = 6
    with pytest.raises(ValueError, match="shared_tail_pairs sum to 5, the source says 6"):
        run(filt, links, joined)
    filt, links, joined = fixtures()
    del joined["resolutions"]["2000|H|g"]
    with pytest.raises(ValueError, match="different key sets"):
        run(filt, links, joined)
    # an unsupported basin is refused per key, before any group composition; here it
    # sits INSIDE a mixed group beside a supported basin, whichever is iterated first
    for bad in ("1999|A|g", "2000|E|g", "2000|F|g"):
        filt, links, joined = fixtures()
        joined["resolutions"][bad]["event_basin"] = "WP"
        with pytest.raises(ValueError, match="outside"):
            run(filt, links, joined)


def test_provenance_refusals():
    M = _load()
    filt, links, joined = fixtures()
    with pytest.raises(ValueError, match="digest is not an input the joined artifact"):
        M.fraction(filt, links, joined, links_sha256="X")
    with pytest.raises(ValueError, match="digest is not an input"):
        M.fraction(filt, links, joined)                       # no digest given at all
    filt, links, joined = fixtures()
    del joined["inputs"]
    with pytest.raises(ValueError, match="records no input digests"):
        M.fraction(filt, links, joined, links_sha256=LINK_SHA)
    filt, links, joined = fixtures()
    filt["years"]["2000"]["input_sha256"] = "other"
    with pytest.raises(ValueError, match="population hash for 2000"):
        M.fraction(filt, links, joined, links_sha256=LINK_SHA)
    filt, links, joined = fixtures()
    links["source_sha256"]["src/aew/qtrack.py"] = "other"
    with pytest.raises(ValueError, match="same filter module hash"):
        M.fraction(filt, links, joined, links_sha256=LINK_SHA)
    filt, links, joined = fixtures()
    filt["years"]["2001"] = dict(filt["years"]["2000"])
    with pytest.raises(ValueError, match="different years"):
        M.fraction(filt, links, joined, links_sha256=LINK_SHA)


def test_the_report_carries_the_table_and_names_no_probability(tmp_path):
    M = _load()
    filt, links, joined = fixtures()
    pf, pl, pj = tmp_path / "f.json", tmp_path / "l.json", tmp_path / "j.json"
    pf.write_text(json.dumps(filt))
    pl.write_text(json.dumps(links))
    joined["inputs"]["NA"]["sha256"] = M._sha256(str(pl))    # bind the real link file
    pj.write_text(json.dumps(joined))
    out, rep = tmp_path / "out.json", tmp_path / "REP.md"
    assert M.main(["--filter-summary", str(pf), "--links", str(pl), "--joined", str(pj),
                   "--out", str(out), "--report", str(rep)]) == 0
    with open(out) as fh:
        v = json.load(fh)
    assert v["totals"]["validated_link_fraction"] == 5 / 7
    assert v["inputs"]["joined"]["sha256"] == M._sha256(str(pj))
    text = rep.read_text()
    assert "| 1999 | 6 | 3 | 2 | 1 | 1 | 0 | 1 | 0 | 2 | 2 | 0.667 |" in text
    assert "| 2000 | 5 | 4 | 3 | 2 | 0 | 1 | 0 | 1 | 0 | 4 | 0.750 |" in text
    assert "| All years | 11 | 7 | 5 | 3 | 1 | 1 | 1 | 1 | 2 | 6 | 0.714 |" in text
    assert v["totals"]["linked_groups_by_basin"] == {"NA": 3, "EP": 1, "both": 1}
    assert "5 of 7 groups hold a validated link" in text
    assert "1 of them inside groups that also hold a validated link" in text
    assert "probability" not in text.lower()
    assert "not an identified physical wave" in text
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_the_cli_refuses_an_unbound_link_file_and_a_moved_population_and_writes_nothing(
        tmp_path):
    M = _load()
    filt, links, joined = fixtures()
    pf, pl, pj = tmp_path / "f.json", tmp_path / "l.json", tmp_path / "j.json"
    pf.write_text(json.dumps(filt))
    pl.write_text(json.dumps(links))
    pj.write_text(json.dumps(joined))                         # records "L", not this file
    out, rep = tmp_path / "out.json", tmp_path / "REP.md"
    assert M.main(["--filter-summary", str(pf), "--links", str(pl), "--joined", str(pj),
                   "--out", str(out), "--report", str(rep)]) == 2
    assert not out.exists() and not rep.exists()
    joined["inputs"]["NA"]["sha256"] = M._sha256(str(pl))
    pj.write_text(json.dumps(joined))
    filt["years"]["1999"]["input_sha256"] = "moved"
    pf.write_text(json.dumps(filt))
    assert M.main(["--filter-summary", str(pf), "--links", str(pl), "--joined", str(pj),
                   "--out", str(out), "--report", str(rep)]) == 2
    assert not out.exists() and not rep.exists()
