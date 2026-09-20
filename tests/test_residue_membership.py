"""The residue membership accounting, bound on a synthetic residual artifact and case
artifacts whose claims it must check rather than trust.

MUTATION LIST, written before the assertions:
  R1 a claim naming an index outside the claimed category is accepted;
  R2 an index claimed by two cases is counted twice;
  R3 the remaining lists are not the complement of the explained lists;
  R4 the both-sides-extra pair is counted among the version-1-extra pairs.
Added after a review credited a copied case artifact from an unrelated window:
  R5 a case artifact naming another case identifier is credited;
  R6 a case artifact from a stale port run (different port-output digest) with valid
     category indices is credited;
  R7 a case artifact lacking the binding digests is credited.
Added after a review added an unwalked pair to a real case's claim and watched coverage
rise, then made one case claim every unmatched track:
  R8 an index no replay examined is credited;
  R9 an index examined by only one or two of the three replays is credited;
  R10 a case with no intervention record at all is credited for what it claims.
Added after two independent reviews found that requiring an EXAMINED index
was a better stand-in for an outcome rather than an outcome, so a case whose intervention
reproduced nothing was credited beside cases that reproduced their tracks exactly:
  R11 an index nothing reproduced and whose finished-track counts never move is credited;
  R12 an index a control reproduces as well as the intervention is credited;
  R13 the weaker track-survival outcome is reported in the same list as an exact
      reproduction.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "residue_membership", os.path.join(HERE, "..", "scripts", "residue_membership.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["residue_membership"] = mod
    spec.loader.exec_module(mod)
    return mod


CASE = "abc123"
HASHES = {"tracker_case.mat": "c" * 64, "tracker_port.mat": "p" * 64,
          "tracker_octave_instrumented.mat": "o1" * 32}


def walked(indices, outcome="reproduced"):
    """The intervention record a case carries for the indices it examined: the untouched
    replay, the intervention and its two controls, each recording an outcome for every
    one of them. `outcome` says which outcome the record shows."""
    def refs(exact):
        return [{"reference_index": i, "reproduced_exactly": exact} for i in indices]

    def west(n):
        return [{"steps": 9}] * n

    if outcome == "reproduced":
        exact = {"baseline": False, "intervention": True, "control": False,
                 "shape_control": False}
        counts = {"baseline": 1, "intervention": 1, "control": 1, "shape_control": 1}
    elif outcome == "track_survival":          # the Sahara shape: nothing reproduces it
        exact = dict.fromkeys(("baseline", "intervention", "control", "shape_control"), False)
        counts = {"baseline": 0, "intervention": 1, "control": 0, "shape_control": 0}
    else:                                       # examined, and nothing happened
        exact = dict.fromkeys(("baseline", "intervention", "control", "shape_control"), False)
        counts = {"baseline": 2, "intervention": 2, "control": 2, "shape_control": 2}
    steps = {"baseline": None, "intervention": 33035.25, "control": 33035.5,
             "shape_control": 33035.25}
    out = {}
    for label in exact:
        run = {"reference_tracks": refs(exact[label]),
               "finished_tracks_in_the_western_box": west(counts[label])}
        if steps[label] is not None:
            run.update({"injected_at": steps[label], "injections_applied": 1,
                        "applied_at": [steps[label]]})
        else:
            run.update({"injected_at": None, "injections_applied": 0, "applied_at": []})
        out[label] = run
    return out


PARAMETERS = {"divergence_step": 33035.25, "control_step": 33035.5}


def bound(explains, oracle_hash="o2" * 32, intervention=None, parameters=None):
    """A case artifact bound to the residual artifact's case and files, with a different
    reference output digest on purpose, since dump schedules differ, and the intervention
    record for whatever it claims."""
    claimed = list(explains.get("unmatched_v1_tracks", [])) \
        + list(explains.get("v1_extra_pairs", []))
    return {"case_id": CASE, "explains": explains,
            "parameters": dict(PARAMETERS if parameters is None else parameters),
            "intervention": walked(claimed) if intervention is None else intervention,
            "input_sha256": {"tracker_case.mat": HASHES["tracker_case.mat"],
                             "tracker_port.mat": HASHES["tracker_port.mat"],
                             "tracker_octave_instrumented.mat": oracle_hash}}


def residuals():
    return {"case_id": CASE, "input_sha256": dict(HASHES),
            "v1_unmatched": [{"index": 6, "eligible_counterpart_exists": False},
                             {"index": 19, "eligible_counterpart_exists": False},
                             {"index": 31, "eligible_counterpart_exists": True}],
            "pairs": [{"v1_index": 74, "extra_kind": "extra_v1_only"},
                      {"v1_index": 67, "extra_kind": "extra_v1_only"},
                      {"v1_index": 5, "extra_kind": "extra_v1_only"},
                      {"v1_index": 94, "extra_kind": "extra_both_sides"},
                      {"v1_index": 50, "extra_kind": "no_extra"}]}


def test_membership_checks_every_claim_and_derives_the_remaining_lists():
    M = _load()
    cases = {"a.json": bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}),
             "b.json": bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74, 67]}),
             "c.json": bound({})}                            # declares nothing
    r = M.membership(residuals(), cases)
    assert r["problems"] == []
    assert r["unmatched_v1_no_eligible_counterpart"] == [6, 19]
    assert r["unmatched_v1_with_eligible_counterpart"] == [31]
    assert r["pairs_by_extra_kind"]["extra_v1_only"] == [5, 67, 74]
    assert r["explained_unmatched"] == [19] and r["remaining_unmatched"] == [6]
    assert r["explained_v1_extra_pairs"] == [67, 74] and r["remaining_v1_extra_pairs"] == [5]
    assert r["counts"] == {"unmatched_no_eligible": 2, "unmatched_explained": 1,
                           "unmatched_explained_by_exact_reproduction": 1,
                           "v1_extra_pairs": 3, "v1_extra_pairs_explained": 2,
                           "v1_extra_pairs_explained_by_exact_reproduction": 2,
                           "both_sides_extra_pairs": 1}


def test_wrong_or_duplicate_claims_are_problems(tmp_path):
    M = _load()
    r = M.membership(residuals(), {"a.json": bound({"unmatched_v1_tracks": [31]})})
    assert any("not in the no-eligible-counterpart set" in p for p in r["problems"])
    r = M.membership(residuals(), {"a.json": bound({"v1_extra_pairs": [94]})})
    assert any("not a version-1-extra pair" in p for p in r["problems"])
    r = M.membership(residuals(), {"a.json": bound({"v1_extra_pairs": [74]}),
                                   "b.json": bound({"v1_extra_pairs": [74]})})
    assert any("claimed by more than one case" in p for p in r["problems"])
    assert r["counts"]["v1_extra_pairs_explained"] == 1
    # and the command refuses with nothing written
    pr, pc, out = tmp_path / "r.json", tmp_path / "c.json", tmp_path / "m.json"
    pr.write_text(json.dumps(residuals()))
    pc.write_text(json.dumps(bound({"unmatched_v1_tracks": [31]})))
    assert M.main(["--residuals", str(pr), "--cases", str(pc), "--out", str(out)]) == 2
    assert not out.exists()


def test_a_case_from_another_window_or_a_stale_port_run_is_refused(tmp_path):
    """The claimed indices are valid category members in every case below, so only
    the binding can refuse them."""
    M = _load()
    valid = {"unmatched_v1_tracks": [19], "v1_extra_pairs": [74]}
    other_case = dict(bound(valid), case_id="another-window")
    r = M.membership(residuals(), {"a.json": other_case})
    assert any("is not the residual artifact's" in p for p in r["problems"])
    assert "counts" not in r
    stale = bound(valid)
    stale["input_sha256"]["tracker_port.mat"] = "q" * 64
    r = M.membership(residuals(), {"a.json": stale})
    assert any("tracker_port.mat differs" in p for p in r["problems"])
    unbound = {"case_id": CASE, "explains": valid}
    r = M.membership(residuals(), {"a.json": unbound})
    assert any("digest of tracker_case.mat is missing" in p for p in r["problems"])
    # a different REFERENCE output digest is not a refusal (dump schedules differ)
    r = M.membership(residuals(), {"a.json": bound(valid, oracle_hash="z" * 64)})
    assert r["problems"] == [] and r["counts"]["unmatched_explained"] == 1
    # the command refuses the stale case with nothing written
    pr, pc, out = tmp_path / "r.json", tmp_path / "c.json", tmp_path / "m.json"
    pr.write_text(json.dumps(residuals()))
    pc.write_text(json.dumps(stale))
    assert M.main(["--residuals", str(pr), "--cases", str(pc), "--out", str(out)]) == 2
    assert not out.exists()


def test_a_claim_the_case_never_walked_is_refused():
    """A review added pair 106 to a real case's claim and watched coverage go from six of
    fifteen to seven with no problem reported, and made one case claim all five unmatched
    tracks the same way. The declaration was checked against the residual categories and
    against nothing the case had done."""
    M = _load()
    unwalked = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74, 67]},
                     intervention=walked([74]))
    out = M.membership(residuals(), {"a.json": unwalked})
    assert any("claims 67, which its replays" in w and "record no outcome for it" in w
               for w in out["problems"])
    # an index one replay does not record is not a walk either
    partial = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]})
    partial["intervention"]["control"]["reference_tracks"] = []
    out2 = M.membership(residuals(), {"a.json": partial})
    assert any("['control']" in w and "record no outcome for it" in w
               for w in out2["problems"])
    # and a case with no intervention at all claims nothing
    none = bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}, intervention={})
    out3 = M.membership(residuals(), {"a.json": none})
    assert any("records no intervention" in w for w in out3["problems"])
    # while a case that walked what it claims is credited
    good = bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": [74]})
    out4 = M.membership(residuals(), {"a.json": good})
    assert out4["problems"] == []
    assert out4["explained_unmatched"] == [19] and out4["explained_v1_extra_pairs"] == [74]


def test_a_credit_names_the_outcome_it_rests_on():
    """The Sahara case reproduces no track; what its intervention changes is the count of
    finished tracks in its western box. That is a real outcome and a weaker one, and the
    two must not be added into one number."""
    M = _load()
    cases = {"exact.json": bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
             "weaker.json": {**bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}),
                             "intervention": walked([19], outcome="track_survival")}}
    out = M.membership(residuals(), cases)
    assert out["problems"] == []
    assert out["explained_by_outcome"]["v1_extra_pairs"]["reproduced"] == [74]
    assert out["explained_by_outcome"]["unmatched"]["track_survival"] == [19]
    assert out["explained_by_outcome"]["unmatched"]["reproduced"] == []
    assert out["counts"]["unmatched_explained"] == 1
    assert out["counts"]["unmatched_explained_by_exact_reproduction"] == 0
    assert out["counts"]["v1_extra_pairs_explained_by_exact_reproduction"] == 1
    # an index nothing reproduced and whose counts never move is credited for nothing
    nothing = {"a.json": {**bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}),
                          "intervention": walked([19], outcome="nothing")}}
    out2 = M.membership(residuals(), nothing)
    assert any("neither reproduces it nor adds finished tracks" in w
               for w in out2["problems"])
    # A LOSS OF TRACKS IS NOT SURVIVAL. A review set a genuine artifact's baseline and
    # control counts to two and its intervention's to zero, and the first version credited
    # that under a label meaning the opposite.
    losing = {**bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}),
              "intervention": walked([19], outcome="track_survival")}
    for label, n in (("baseline", 2), ("control", 2), ("shape_control", 2),
                     ("intervention", 0)):
        losing["intervention"][label]["finished_tracks_in_the_western_box"] = [{"steps": 9}] * n
    out_loss = M.membership(residuals(), {"a.json": losing})
    assert any("REMOVES finished tracks" in w for w in out_loss["problems"])
    # A MISSING CONTROL is not a passed control, and an injection that never happened is
    # not an experiment: both were accepted with every credit intact.
    gone = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
            "intervention": walked([74])}
    del gone["intervention"]["control"]
    out_gone = M.membership(residuals(), {"a.json": gone})
    assert any("records no control replay" in w for w in out_gone["problems"])
    # and the shape control may not quietly disappear either, since the artifact declares
    # a four-run experiment
    no_shape = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
                "intervention": walked([74])}
    del no_shape["intervention"]["shape_control"]
    out_shape = M.membership(residuals(), {"a.json": no_shape})
    assert any("records no shape_control replay" in w for w in out_shape["problems"])
    unapplied = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
                 "intervention": walked([74])}
    unapplied["intervention"]["intervention"].update({"injections_applied": 0,
                                                      "applied_at": []})
    out_unapplied = M.membership(residuals(), {"a.json": unapplied})
    assert any("applied 0 of the 1 injection" in w for w in out_unapplied["problems"])
    lying = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
             "intervention": walked([74])}
    lying["intervention"]["control"]["applied_at"] = [33035.25]
    out_lying = M.membership(residuals(), {"a.json": lying})
    assert any("recorded an application at [33035.25]" in w for w in out_lying["problems"])
    same_step = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
                 "intervention": walked([74])}
    same_step["intervention"]["control"].update({"injected_at": 33035.25,
                                                 "applied_at": [33035.25]})
    out_same = M.membership(residuals(), {"a.json": same_step})
    assert any("injected at the divergence step" in w for w in out_same["problems"])
    # AN INJECTION AT THE WRONG TIMESTEP, which the trace refuses and the checker used to
    # credit: a review moved a genuine intervention ten hours away, left the declared
    # divergence step alone, and kept all four credits.
    shifted = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
               "intervention": walked([74])}
    for key in ("injected_at", "applied_at"):
        shifted["intervention"]["intervention"][key] = (
            33044.75 if key == "injected_at" else [33044.75])
    out_shifted = M.membership(residuals(), {"a.json": shifted})
    assert any("injected at 33044.75 and the case's divergence step is 33035.25" in w
               for w in out_shifted["problems"])
    assert out_shifted["explained_v1_extra_pairs"] == []
    # and a recorded control-step problem is not the checker's to ignore
    flagged = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
               "intervention": walked([74])}
    flagged["intervention"]["control_step_problems"] = ["the control step is not a timestep"]
    out_flagged = M.membership(residuals(), {"a.json": flagged})
    assert any("not a timestep" in w for w in out_flagged["problems"])
    # an index the TIME control reproduces as well is no evidence for the intervention
    both = {**bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}),
            "intervention": walked([19])}
    both["intervention"]["control"]["reference_tracks"] = [
        {"reference_index": 19, "reproduced_exactly": True}]
    out3 = M.membership(residuals(), {"a.json": both})
    assert any("the time control reproduces it as well as the intervention" in w
               for w in out3["problems"])
    # THE SHAPE CONTROL IS NOT A NULL CONTROL. It injects an axis where the port had none,
    # with the vertices moved off the crossing, and on three of the four real cases it
    # reproduces the tracks as well. That refines the claim to "the absence of an axis"
    # rather than "the absence of that line", and must not refuse the credit.
    refined = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
               "intervention": walked([74])}
    refined["intervention"]["shape_control"]["reference_tracks"] = [
        {"reference_index": 74, "reproduced_exactly": True}]
    out4 = M.membership(residuals(), {"a.json": refined})
    assert out4["problems"] == []
    assert out4["explained_by_outcome"]["v1_extra_pairs"]["reproduced"] == [74]
    # while the untouched replay reproducing it is no evidence at all
    already = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
               "intervention": walked([74])}
    already["intervention"]["baseline"]["reference_tracks"] = [
        {"reference_index": 74, "reproduced_exactly": True}]
    out5 = M.membership(residuals(), {"a.json": already})
    assert any("the untouched replay reproduces it" in w for w in out5["problems"])
