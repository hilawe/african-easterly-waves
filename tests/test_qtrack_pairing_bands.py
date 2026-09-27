"""The band tabulation counts the pilot's fixed assignments, binds the rebuilt populations
to the artifact, refuses a changed input or a tampered status list, and places band edges
where it says."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)


def _load():
    spec = importlib.util.spec_from_file_location("qtrack_pairing_bands", os.path.join(ROOT, "scripts", "qtrack_pairing_bands.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact(tmp_path):
    import test_qtrack_pairing_pilot as T
    M = T._load()
    evidence, qdir, adir, regions = T._world(tmp_path)
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    out_art = art_dir / "pairing_1990_2026-09-27.json"
    assert M.run(1990, evidence, qdir, adir, regions, str(out_art)) == 0
    return art_dir, out_art, evidence


def _sums(table):
    return {k: sum(r[k] for r in table.values()) for k in ("eligible", "paired", "no_candidate", "lost_assignment", "members", "subset")}


def test_bands_count_every_eligible_track_once_and_agree_with_the_artifact(tmp_path):
    B = _load()
    art_dir, out_art, _ = _artifact(tmp_path)
    out = tmp_path / "bands.json"
    assert B.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(out)]) == 0
    art = json.load(open(out_art))
    got = json.load(open(out))["seasons"][0]
    for tol in ("500", "350"):
        r = art["results"][tol]
        for side, table in (("a", "a_by_band"), ("b", "b_by_band"), ("a", "a_by_length"), ("b", "b_by_length")):
            s = _sums(got["by_tolerance"][tol][table])
            assert s["eligible"] == art["populations"][side]["eligible"]
            assert s["paired"] == r[side]["assigned"]
            assert s["no_candidate"] == len(r[side]["no_candidate"])
            assert s["lost_assignment"] == len(r[side]["lost_assignment"])
            assert s["paired"] + s["no_candidate"] + s["lost_assignment"] == s["eligible"]
        assert _sums(got["by_tolerance"][tol]["a_by_band"])["members"] == len(art["populations"]["a"]["duplicate_group_members"])
        assert _sums(got["by_tolerance"][tol]["a_by_band"])["subset"] == len(art["populations"]["a"]["africa_origin_in_season_ids"])
    # the synthetic world's eligible tracks all start at 12 N except one at 10 S, so the bands hold what they should
    a = got["by_tolerance"]["500"]["a_by_band"]
    assert a["10 to 15"]["eligible"] == art["populations"]["a"]["eligible"] - 1 and a["-20 to 0"]["eligible"] == 1
    assert a["-20 to 0"]["no_candidate"] == 1
    # the archive-rule populations are reported for both records, this record's equal to the artifact's subset
    pops = got["archive_rule_populations"]
    # the synthetic Africa polygon spans 10 W to 30 E and 0 to 30 N: this record's tracks 0, 3 and 7 and QTrack's
    # systems 1, 2 (on the polygon's eastern edge, which the archive's inpolygon counts as inside) and 3 start in it
    # in June; QTrack's system 4 starts at 50 N and this record's track 1 in January
    assert pops["this_record"]["ids"] == art["populations"]["a"]["africa_origin_in_season_ids"] == [0, 3, 7]
    assert pops["qtrack"]["ids"] == [1, 2, 3] and pops["qtrack"]["n"] == 3
    assert sum(pops["qtrack"]["first_latitude_by_band"].values()) == 3 and pops["qtrack"]["first_latitude_by_band"]["10 to 15"] == 3
    with pytest.raises(SystemExit):                                   # never overwritten
        B.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(out)])
    (art_dir / "pairing_2002_2026-09-27.json").write_text(out_art.read_text())   # a file named for 2002 holding 1990's artifact
    with pytest.raises(SystemExit, match="not the artifact for 2002"):
        B.main(["--artifacts", str(art_dir), "--years", "2002", "--out", str(tmp_path / "other.json")])


def test_a_changed_input_and_a_tampered_status_list_are_refused(tmp_path):
    B = _load()
    art_dir, out_art, evidence = _artifact(tmp_path)
    art = json.load(open(out_art))
    tampered = json.loads(json.dumps(art))
    tampered["results"]["500"]["pairs"].pop()                          # one eligible track now has no status
    with pytest.raises(SystemExit, match="statuses"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    moved = tampered["results"]["500"]["pairs"][0]
    moved["a_first"]["lat"] += 0.5                                    # the artifact says the pair starts elsewhere
    with pytest.raises(SystemExit, match="does not start"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    u = tampered["results"]["500"]["a"]["no_candidate"][0]
    tampered["results"]["500"]["a"]["lost_assignment"].append(dict(u))  # one identifier in two lists
    with pytest.raises(SystemExit, match="two status lists"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    tampered["populations"]["a"]["eligible"] += 1                     # the artifact claims one more eligible track than rebuilt
    with pytest.raises(SystemExit, match="rebuilt population"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    tampered["results"]["500"]["a"]["no_candidate"][0]["id"] = 9999   # same count, an identifier that is no eligible track
    with pytest.raises(SystemExit, match="rebuilt population"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    tampered["results"]["500"]["pairs"].append(dict(tampered["results"]["500"]["pairs"][0]))   # one identifier paired twice on both sides
    with pytest.raises(SystemExit, match="twice among the pairs"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    p0 = tampered["results"]["500"]["pairs"][0]
    tampered["results"]["500"]["a"]["no_candidate"].append({"id": p0["a"], "candidates": 0})   # a paired track also listed without a candidate
    with pytest.raises(SystemExit, match="two status lists"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    tampered["populations"]["a"]["africa_origin_in_season_ids"].pop()                          # the artifact's subset is not the archive rule's
    with pytest.raises(SystemExit, match="Africa-origin subset"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    del tampered["inputs"]["region_polygons"]["sha256"]["africa"]                              # a manifest missing one polygon
    with pytest.raises(SystemExit, match="incomplete"):
        B.tabulate(1990, tampered)
    tampered = json.loads(json.dumps(art))
    r = tampered["results"]["500"]
    moved = r["a"]["no_candidate"].pop()                                                         # side A stays a partition, side B repeats
    r["pairs"].append({**r["pairs"][0], "a": moved["id"]})
    with pytest.raises(SystemExit, match="b .* twice among the pairs"):
        B.tabulate(1990, tampered)
    with open(os.path.join(art["inputs"]["region_polygons"]["dir"], "africa.mat"), "ab") as fh:
        fh.write(b"\0")                                                                        # a polygon is no longer the artifact's
    with pytest.raises(SystemExit, match="region polygons"):
        B.tabulate(1990, art)
    with open(os.path.join(evidence, "era5_1990", "tracker_port.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="not the file"):
        B.tabulate(1990, art)


def test_band_edges_and_length_classes_are_where_they_say():
    B = _load()
    assert B.band_of(5.0) == "5 to 10" and B.band_of(4.999) == "0 to 5"
    assert B.band_of(20.0) == "20 to 25" and B.band_of(19.999) == "15 to 20"
    assert B.band_of(-20.0) == "-20 to 0" and B.band_of(35.0) == "25 to 35"
    with pytest.raises(SystemExit):
        B.band_of(35.001)
    with pytest.raises(SystemExit):
        B.band_of(-20.001)
    assert B.length_class(3) == "3 to 7" and B.length_class(7) == "3 to 7" and B.length_class(8) == "8 to 15"
    assert B.length_class(32) == "32 or more"
    # every latitude is counted once, with the overflow rows holding what lies outside the bands' span
    c = B.full_band_count([-25.0, -5.0, 12.0, 35.0, 36.0])
    assert c["south of -20"] == 1 and c["-20 to 0"] == 1 and c["10 to 15"] == 1 and c["25 to 35"] == 1 and c["north of 35"] == 1
    assert sum(c.values()) == 5
    with pytest.raises(SystemExit):
        B.length_class(2)
