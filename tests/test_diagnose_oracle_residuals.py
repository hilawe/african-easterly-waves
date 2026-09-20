"""The residual classifier over the identical-input comparison, bound on synthetic
exchange files.

MUTATION LIST, written before the assertions:
  D1 a pair with an extra step at one end and zero separation elsewhere is classed as
     displaced (the kind ignores the separation);
  D2 the side holding the extra steps is misattributed;
  D3 an unmatched track's nearest same-side track is not flagged as matched;
  D4 an interior extra step is reported as at an end.
Added after a review found the port's extras dropped whenever version 1 also had some
(the real pair 94/89: version 1 alone at 33034.50 and 33035.00, the port alone at
33034.75), and the worst pair mean quoted as if it bounded the largest single step:
  D5 the port's extras are dropped when version 1 has extras too;
  D6 the worst individual step is taken from pair means;
  D7 an unmatched track with an eligible counterpart under the matching rule is not
     distinguished from one with none.
Added after a review built the case that separates "eligible" from "nearest":
  D8 the eligibility flag is read from the NEAREST counterpart rather than from the
     matching graph's edges, so a leftover whose nearest neighbour is ineligible on
     overlap is reported as having no eligible counterpart at all.
"""
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.io import savemat

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
    spec = importlib.util.spec_from_file_location(
        "diagnose_oracle_residuals",
        os.path.join(HERE, "..", "scripts", "diagnose_oracle_residuals.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_oracle_residuals"] = mod
    spec.loader.exec_module(mod)
    return mod


def write_tracks(path, tracks, case_id, extra=None):
    payload = {"n": float(len(tracks)), "case_id": case_id}
    for i, (t, la, lo) in enumerate(tracks):
        payload[f"time{i}"] = np.asarray(t, dtype=float)
        payload[f"lat{i}"] = np.asarray(la, dtype=float)
        payload[f"lon{i}"] = np.asarray(lo, dtype=float)
    payload.update(extra or {})
    savemat(path, payload)


def test_classifier_reports_both_sides_extras_worst_step_and_eligibility(tmp_path):
    M = _load()
    d = tmp_path / "oracle"
    d.mkdir()
    savemat(str(d / "tracker_case.mat"), {"case_id": "c1"})
    # A: identical. B: version 1 has one extra step AFTER the port's span, shared steps
    #    equal. C: version 1 has an INTERIOR extra step and the port's neighbours are
    #    displaced by 0.3 degrees. G: extras on BOTH sides, the 94/89 shape: version 1
    #    alone holds the two steps after 6, the port alone holds an interior one at
    #    6.5, and one shared step is displaced by 2.0 degrees (the worst single step).
    #    D: version 1 only, 0.2 degrees from A for all three steps (an ELIGIBLE
    #    counterpart exists, so one-to-one assignment left it over). E: version 1
    #    only, sharing one step with A but 90 degrees away (no eligible counterpart).
    #    F: port only, no overlap with anything on the version 1 side.
    v1 = [([0, 1, 2], [10, 10, 10], [-20, -21, -22]),                     # A
          ([0, 1, 2, 3], [5, 5, 5, 5], [-30, -31, -32, -33]),             # B
          ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [-40, -41, -42, -43, -44]),  # C
          ([0, 1, 2], [10.2, 10.2, 10.2], [-20, -21, -22]),               # D
          ([2, 3, 4], [-30, -30, -30], [60, 61, 62]),                     # E
          ([5, 6, 7, 8], [-5, -5, -5, -5], [10, 11, 12, 13])]             # G
    port = [([0, 1, 2], [10, 10, 10], [-20, -21, -22]),                   # A
            ([0, 1, 2], [5, 5, 5], [-30, -31, -32]),                      # B
            ([0, 1, 3, 4], [0, 0.3, 0.3, 0], [-40, -41, -43, -44]),       # C, no step 2
            ([9, 10, 11], [25, 25, 25], [100, 101, 102]),                 # F, no overlap
            ([5, 6, 6.5], [-5, -7, -5], [10, 11, 11.5])]                  # G, 6 displaced
    write_tracks(str(d / "tracker_octave_instrumented.mat"), v1, "c1")
    write_tracks(str(d / "tracker_port.mat"), port, "c1", {"exclusive": False, "absorb": False})
    out = tmp_path / "res.json"
    assert M.main(["--oracle-dir", str(d), "--oracle", "tracker_octave_instrumented.mat",
                   "--out", str(out)]) == 0
    r = json.load(open(out))
    assert r["nonidentical_pairs"] == 3
    assert r["nonidentical_extra_kinds"] == {"extra_v1_only": 2, "extra_both_sides": 1}
    assert r["nonidentical_kinds"] == {"extra_v1_only": 1, "extra_v1_only_displaced": 1,
                                       "extra_both_sides_displaced": 1}
    b = next(p for p in r["pairs"] if p["v1_index"] == 1)
    assert b["kind"] == "extra_v1_only" and b["max_sep_deg"] == 0.0
    assert b["v1_extra"] == {"n": 1, "times": [3.0], "before": 0, "interior": 0, "after": 1}
    assert b["port_extra"] == {"n": 0, "times": [], "before": 0, "interior": 0, "after": 0}
    c = next(p for p in r["pairs"] if p["v1_index"] == 2)
    assert c["n_displaced_steps"] == 2
    assert c["v1_extra"] == {"n": 1, "times": [2.0], "before": 0, "interior": 1, "after": 0}
    g = next(p for p in r["pairs"] if p["v1_index"] == 5)
    assert g["kind"] == "extra_both_sides_displaced"
    assert g["v1_extra"] == {"n": 2, "times": [7.0, 8.0], "before": 0, "interior": 0, "after": 2}
    assert g["port_extra"] == {"n": 1, "times": [6.5], "before": 0, "interior": 1, "after": 0}
    assert abs(g["max_sep_deg"] - 2.0) < 1e-9
    assert abs(r["worst_individual_step_deg"] - 2.0) < 1e-9
    assert r["pairs_with_a_step_over_one_degree"] == 1
    un = {u["index"]: u for u in r["v1_unmatched"]}
    assert set(un) == {3, 4}
    assert un[3]["eligible_counterpart_exists"] is True
    assert un[3]["nearest_same_side"]["index"] == 0
    assert un[3]["nearest_same_side"]["that_track_matched"] is True
    assert un[4]["eligible_counterpart_exists"] is False
    assert un[4]["nearest_other_side"]["sep_deg"] > 5.0
    assert r["v1_unmatched_with_eligible_counterpart"] == 1
    assert [u["index"] for u in r["port_unmatched"]] == [3]
    assert r["port_unmatched"][0]["eligible_counterpart_exists"] is False
    assert r["input_sha256"]["tracker_port.mat"] == M._sha256(str(d / "tracker_port.mat"))
    assert "scripts/compare_tracker_oracle.py" in r["source_sha256"]


def test_eligibility_is_read_from_the_matching_graph_not_the_nearest_track(tmp_path):
    """D8: a review built the case that separates the two. An unmatched reference track
    whose NEAREST counterpart shares one step at zero separation, and another counterpart
    sharing enough steps within the tolerance. The nearest is ineligible on overlap, the
    other is eligible, and reading only the nearest reported no eligible counterpart."""
    M = _load()
    d = tmp_path / "oracle"
    d.mkdir()
    steps = [0.0, 0.25, 0.5, 0.75]
    # two identical reference tracks compete for one eligible port track, and a second
    # port track shares ONE step with the leftover at zero separation
    v1 = [(steps, [10.0] * 4, [0.0, 1.0, 2.0, 3.0]),
          (steps, [10.0] * 4, [0.0, 1.0, 2.0, 3.0])]
    port = [(steps, [10.5] * 4, [0.0, 1.0, 2.0, 3.0]),
            ([0.0], [10.0], [0.0])]
    write_tracks(str(d / "tracker_octave_instrumented.mat"), v1, "abc")
    write_tracks(str(d / "tracker_port.mat"), port, "abc")
    savemat(str(d / "tracker_case.mat"), {"case_id": "abc"})
    out = tmp_path / "residuals.json"
    assert M.main(["--oracle-dir", str(d), "--oracle", "tracker_octave_instrumented.mat",
                   "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    leftover = [u for u in r["v1_unmatched"]][0]
    # the nearest counterpart is the one-step track at zero separation, which is NOT
    # eligible, while port track 0 is
    assert leftover["nearest_other_side"]["shared_steps"] == 1
    assert leftover["nearest_other_side"]["sep_deg"] == 0.0
    assert leftover["eligible_counterpart_exists"] is True
    assert leftover["eligible_counterparts"] == [0]
    assert r["v1_unmatched_with_eligible_counterpart"] == 1
