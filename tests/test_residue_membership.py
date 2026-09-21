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
Added after a review defeated the first axis check three ways through the command entry
point, each time keeping the credit for the pair:
  R18 a case with no dumped vertices to bind the injected axes to is credited;
  R19 a time control carrying the translated shape-control geometry is credited;
  R20 the same substitution is credited when a map declaring it accompanies it;
  R21 a time control carrying the OTHER declared step's geometry is credited.
Added after a second review defeated the derived comparison four more ways:
  R22 a not-a-number among the dumped reference coordinates is credited;
  R23 a not-a-number translation is credited;
  R24 a translation invisible at the comparison's own tolerance is credited;
  R25 an axis or an endpoint added to a run is credited.
Added after the same two families read the fold that answered them:
  R35 the effective selection region excludes version 1's own observation at a declared
      step;
  R36 an unmatched track's control step is one its finished track does not hold;
  R37 the credit basis does not say which grade applied to which index.
Added after a verification pass found the unmatched check reading a live list position as
a finished track index and the region guard checking a box that is not the selector:
  R38 an unmatched track is judged at a step its FINISHED track does not hold;
  R39 a disjoint axis box selects another region's axes while the case box passes;
  R40 a replay's reproduced_exactly flag disagrees with its own recorded tracks;
  R41 a replay's western-box list disagrees with its own recorded tracks;
  R42 a pair's control step, extra times or port counterpart are not the pinned outputs';
  R43 credit is granted with no finished output, or an unpinned one.
Added after the catalog for that fold left four behaviors unbound:
  R44 a port output declared as the reference output is used as one;
  R45 a case whose residual record agrees with it is judged at steps the outputs do not
      give, so the output-derived steps must be checked on their own;
  R46 a recorded track equal to the reference in time and latitude only is a reproduction;
  R47 a recorded track that is a prefix of the reference is a reproduction.
Added after a verification pass deleted and appended a latitude on a genuine replay track
and kept the credit, because the comparison checked one array's length and zipped the rest:
  R48 a recorded track with a missing or extra coordinate is counted or compared;
  R49 a recorded track with a coordinate that is not finite, or not a number, is counted;
  R50 a pinned output holding a ragged track is accepted as a reference.
Added after a review found a case that claims nothing escaping every check, so two null
results rested on control steps their pairs do not both hold:
  R51 a case that performed an experiment and claims nothing is not validated;
  R52 a case that declares no experiment at all is refused for having none.
Added after a review walked round that repair three ways on a null experiment:
  R53 a performed experiment declaring no reference tracks is validated;
  R54 an examined index the residual artifact does not know is filtered away rather than
      refused;
  R55 a case missing only its baseline reads as unperformed and skips every check;
  R56 the declared examined indices disagree with the ones the replays record;
  R57 a case is credited for an index its experiment does not declare as examined.
"""
import hashlib
import importlib.util
import json
import os
import sys

import pytest

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


POS = (-32.0, -60.3167)                # where every fixture track sits
REF_DIGEST = "o2" * 32                 # the reference output every case declares by default
PORT_DIGEST = "p" * 64                 # the port output, which is also the binding digest


def reference_track(steps):
    """Version 1's finished track for a fixture index: the declared steps plus the steps
    both sides share, at one position, so a port counterpart missing exactly the declared
    steps makes those steps the pair's extra times."""
    times = sorted(set(float(t) for t in steps) | {33035.0, 33035.5, 33036.0})
    return {"time": times, "lat": [POS[0]] * len(times), "lon": [POS[1]] * len(times)}


def port_counterpart(steps):
    ref = reference_track(steps)
    keep = [i for i, t in enumerate(ref["time"])
            if not any(abs(t - float(x)) < 1e-6 for x in steps)]
    return {k: [ref[k][i] for i in keep] for k in ("time", "lat", "lon")}


def outputs(steps=(33035.25,)):
    """The retained finished outputs, in the shape `verified_outputs` returns: every
    reference index holds the same track, every port index the same counterpart."""
    return {REF_DIGEST: {"path": "reference.mat", "kind": "reference",
                         "tracks": [reference_track(steps) for _ in range(120)]},
            PORT_DIGEST: {"path": "port.mat", "kind": "port",
                          "tracks": [port_counterpart(steps) for _ in range(6)]}}


def filler(steps, shift=1.0):
    """A finished track that is NOT the reference: same times, moved north."""
    ref = reference_track(steps)
    return {"time": list(ref["time"]), "lat": [la + shift for la in ref["lat"]],
            "lon": list(ref["lon"])}


def walked(indices, outcome="reproduced", steps=(33035.25,)):
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
    injected = {"baseline": [], "intervention": [33035.25], "control": [33035.5],
                "shape_control": [33035.25]}
    # EVERY RUN CARRIES ITS COMPLETE FINISHED TRACKS, from which the checker recomputes
    # both outcomes: the reference copy where the flag says reproduced, fillers elsewhere,
    # as many in the western box as the recorded list says
    def tracks_for(label):
        n = counts[label]
        if exact[label]:
            return [reference_track(steps)] + [filler(steps, 1.0 + k) for k in range(n - 1)]
        return [filler(steps, 1.0 + k) for k in range(n)]
    axes = {"33035.2500": [{"lat": [-32.0], "lon": [-60.3167]}]}
    moved = {"33035.2500": [{"lat": [-32.0], "lon": [-64.3167]}]}
    control_axes = {"33035.5000": axes["33035.2500"]}
    out = {"partial_runs": []}
    out["shape_control_offset"] = 4.0
    for label in exact:
        by_step = ({} if label == "baseline" else
                   moved if label == "shape_control" else
                   control_axes if label == "control" else axes)
        run = {"reference_tracks": refs(exact[label]),
               "finished_tracks_in_the_western_box": west(counts[label]),
               "requested_times": list(injected[label]),
               "injections_applied": len(injected[label]),
               "applied_at": list(injected[label]),
               "finished_tracks": tracks_for(label),
               "axes_by_requested_step": dict(by_step)}
        out[label] = run
    out["shape_control"]["longitude_offset_deg"] = out.pop("shape_control_offset")
    return out


DUMPED = {"33035.2500": [{"kind": "AXISPTS", "lat": [-32.0], "lon": [-60.3167]}]}


def dumped(steps, lat, lon):
    """Version 1's own recorded vertices at each declared step, which is what a run's
    injected axes are now checked against. A review deleted the expectation maps the
    artifact used to carry beside its runs and the comparison stopped happening, so the
    expectation is derived from these instead."""
    return {f"{float(t):.4f}": [{"kind": "AXISPTS", "lat": list(lat), "lon": list(lon)}]
            for t in steps}


LOG_DIGEST = "1" * 64


def log_text(vertices, tracks=(6, 19, 31)):
    """A reference log holding exactly those axis vertices, in version 1's dump format.

    The tests give ONE source of geometry to both the case artifact and the log, so a
    test that wants them to disagree has to say so. That is the shape of the defect this
    check exists for: an artifact whose own fields all agree with each other."""
    lines = []
    for key, groups in sorted(vertices.items()):
        for g in groups:
            vals = [str(x) for pair in zip(g["lat"], g["lon"]) for x in pair]
            lines.append(f"AXISPTS {key} {len(g['lat'])} " + " ".join(vals))
        # and version 1's own track lines at that step for the fixture's unmatched
        # tracks, which is what bounds the steps a case may declare for one
        for index in tracks:
            lines.append(f"TRACK {key} {index} 5 -32.0 -60.3167 33030.0")
    return "\n".join(lines) + "\n"


def logs(vertices=None):
    """The verified reference runs, in the shape `verified_logs` returns."""
    return {LOG_DIGEST: {"path": "fixture.log",
                         "text": log_text(DUMPED if vertices is None else vertices)}}


def _member(M, cases, vertices=None, steps=None):
    """The accounting over a synthetic residue, with the reference log the credit path
    now requires and a residual artifact whose pair times are the cases' declared steps
    unless the test says otherwise."""
    st = _steps_of(cases) if steps is None else steps
    return M.membership(residuals(st), cases, logs(vertices), (), outputs(st))


PARAMETERS = {"divergence_steps": [33035.25], "control_steps": [33035.5],
              "window": [33030.0, 33040.0],
              "box": {"lat": [-40.0, -10.0], "lon": [-70.0, 45.0]},
              "west": {"lat": [-40.0, -10.0], "lon": [-70.0, 45.0]},
              "feature_life": [33030.0, 33040.0]}


def bound(explains, oracle_hash="o2" * 32, intervention=None, parameters=None,
          vertices=None):
    """A case artifact bound to the residual artifact's case and files, with a different
    reference output digest on purpose, since dump schedules differ, and the intervention
    record for whatever it claims."""
    claimed = list(explains.get("unmatched_v1_tracks", [])) \
        + list(explains.get("v1_extra_pairs", []))
    # the window and the box are how the log's axes are SELECTED the way the case
    # selected them, so every case carries them whatever else it declares
    declared = dict(PARAMETERS)
    declared.update(parameters or {})
    # a performed experiment must say which indices it examined, and the declaration must
    # agree with the indices its replays record outcomes for
    declared.setdefault("reference_tracks", list(claimed))
    return {"case_id": CASE, "explains": explains,
            "parameters": declared,
            "at_divergence": {"v1": dict(DUMPED if vertices is None else vertices)},
            "intervention": (walked(claimed, steps=declared["divergence_steps"])
                             if intervention is None else intervention),
            "input_sha256": {"octave1.log": LOG_DIGEST,
                             "tracker_case.mat": HASHES["tracker_case.mat"],
                             "tracker_port.mat": HASHES["tracker_port.mat"],
                             "tracker_octave_instrumented.mat": oracle_hash}}


def residuals(steps=(33035.25,)):
    """The residual artifact a case is bound to. It records, for each version-1-extra
    pair, the times at which version 1 holds an observation and the port does not, and
    for each unmatched track its span, because those are what the scope check reads.
    `steps` is what the pairs' extra times are, so a test wanting a case to disagree
    with its pair says so here."""
    extra = {"n": len(steps), "times": [float(t) for t in steps]}
    # every record carries its mean location, which the case box must contain, and the
    # fixture's location is the default axis so the default box holds it
    where = {"mean_lat": -32.0, "mean_lon": -60.3167}
    return {"case_id": CASE, "input_sha256": dict(HASHES),
            "v1_unmatched": [{"index": 6, "eligible_counterpart_exists": False,
                              "first_time": 33030.0, "last_time": 33040.0, **where},
                             {"index": 19, "eligible_counterpart_exists": False,
                              "first_time": 33030.0, "last_time": 33040.0, **where},
                             {"index": 31, "eligible_counterpart_exists": True,
                              "first_time": 33030.0, "last_time": 33040.0, **where}],
            "pairs": [{"v1_index": 74, "port_index": 3, "extra_kind": "extra_v1_only",
                       "v1_extra": extra, **where},
                      {"v1_index": 67, "port_index": 4, "extra_kind": "extra_v1_only",
                       "v1_extra": extra, **where},
                      {"v1_index": 5, "port_index": 5, "extra_kind": "extra_v1_only",
                       "v1_extra": extra, **where},
                      {"v1_index": 94, "extra_kind": "extra_both_sides"},
                      {"v1_index": 50, "extra_kind": "no_extra"}]}


def _steps_of(cases):
    """The divergence steps the fixture's cases declare, which is what the residual pairs
    record in every test that is not about the two disagreeing."""
    for c in cases.values():
        steps = (c.get("parameters") or {}).get("divergence_steps")
        if steps:
            return tuple(steps)
    return (33035.25,)


def test_membership_checks_every_claim_and_derives_the_remaining_lists():
    M = _load()
    cases = {"a.json": bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}),
             "b.json": bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74, 67]}),
             "c.json": bound({}, intervention={"attempted": "an injection",
                                               "reason": "version 1 dumps no axis there"})}
    r = _member(M, cases)
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
    r = _member(M, {"a.json": bound({"unmatched_v1_tracks": [31]})})
    assert any("not in the no-eligible-counterpart set" in p for p in r["problems"])
    r = _member(M, {"a.json": bound({"v1_extra_pairs": [94]})})
    assert any("not a version-1-extra pair" in p for p in r["problems"])
    r = _member(M, {"a.json": bound({"v1_extra_pairs": [74]}),
                                   "b.json": bound({"v1_extra_pairs": [74]})})
    assert any("claimed by more than one case" in p for p in r["problems"])
    assert r["counts"]["v1_extra_pairs_explained"] == 1
    # and the command refuses with nothing written
    pr, pc, out = tmp_path / "r.json", tmp_path / "c.json", tmp_path / "m.json"
    pr.write_text(json.dumps(residuals()))
    pc.write_text(json.dumps(bound({"unmatched_v1_tracks": [31]})))
    assert _refuses(M, tmp_path, json.loads(pc.read_text()), "cmd") == (2, False)


def test_a_case_from_another_window_or_a_stale_port_run_is_refused(tmp_path):
    """The claimed indices are valid category members in every case below, so only
    the binding can refuse them."""
    M = _load()
    valid = {"unmatched_v1_tracks": [19], "v1_extra_pairs": [74]}
    other_case = dict(bound(valid), case_id="another-window")
    r = _member(M, {"a.json": other_case})
    assert any("is not the residual artifact's" in p for p in r["problems"])
    assert "counts" not in r
    stale = bound(valid)
    stale["input_sha256"]["tracker_port.mat"] = "q" * 64
    r = _member(M, {"a.json": stale})
    assert any("tracker_port.mat differs" in p for p in r["problems"])
    unbound = {"case_id": CASE, "explains": valid}
    r = _member(M, {"a.json": unbound})
    assert any("digest of tracker_case.mat is missing" in p for p in r["problems"])
    # A DIFFERENT REFERENCE OUTPUT DIGEST USED NOT TO BE A REFUSAL, since dump schedules
    # differ between instrumented runs and the output was not read. It is read now, by
    # final index, so a case must declare a reference output that was supplied and pinned.
    r = _member(M, {"a.json": bound(valid, oracle_hash="z" * 64)})
    assert any("declares reference output zzzzzzzzzzzz, which was not supplied to this "
               "command" in p for p in r["problems"])
    assert "counts" not in r or r["counts"]["unmatched_explained"] == 0
    # the command refuses a stale case with nothing written. The command fixture binds
    # the port output it writes, so the staleness here is in the exported window's own
    # digest, which the fixture does not touch.
    stale_window = bound(valid)
    stale_window["input_sha256"]["tracker_case.mat"] = "q" * 64
    assert _refuses(M, tmp_path, stale_window, "cmd") == (2, False)


def test_a_claim_the_case_never_walked_is_refused():
    """A review added pair 106 to a real case's claim and watched coverage go from six of
    fifteen to seven with no problem reported, and made one case claim all five unmatched
    tracks the same way. The declaration was checked against the residual categories and
    against nothing the case had done."""
    M = _load()
    unwalked = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74, 67]},
                     intervention=walked([74]))
    out = _member(M, {"a.json": unwalked})
    # the claim is refused where it first contradicts the evidence: the case declares it
    # examined 67 and 74 and its replays record outcomes for 74 alone
    assert any("declares reference tracks [67, 74] and its replays record outcomes for "
               "[74]" in w for w in out["problems"])
    assert out["explained_v1_extra_pairs"] == []
    # the outcome path, where ONE replay lacks the index while the declaration and the
    # other replays agree, is the `partial` case immediately below
    # an index one replay does not record is not a walk either
    partial = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]})
    partial["intervention"]["control"]["reference_tracks"] = []
    out2 = _member(M, {"a.json": partial})
    assert any("the control replay records no outcome for 74" in w
               for w in out2["problems"])
    # and a case with no intervention at all claims nothing
    none = bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}, intervention={})
    out3 = _member(M, {"a.json": none})
    assert any("records no intervention" in w for w in out3["problems"])
    # while a case that walked what it claims is credited
    good = bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": [74]})
    out4 = _member(M, {"a.json": good})
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
    out = _member(M, cases)
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
    out2 = _member(M, nothing)
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
        losing["intervention"][label]["finished_tracks"] = [filler((33035.25,), 1.0 + k)
                                                            for k in range(n)]
    out_loss = _member(M, {"a.json": losing})
    assert any("REMOVES finished tracks" in w for w in out_loss["problems"])
    # A MISSING CONTROL is not a passed control, and an injection that never happened is
    # not an experiment: both were accepted with every credit intact.
    gone = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
            "intervention": walked([74])}
    del gone["intervention"]["control"]
    out_gone = _member(M, {"a.json": gone})
    assert any("records no control replay" in w for w in out_gone["problems"])
    # and the shape control may not quietly disappear either, since the artifact declares
    # a four-run experiment
    no_shape = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
                "intervention": walked([74])}
    del no_shape["intervention"]["shape_control"]
    out_shape = _member(M, {"a.json": no_shape})
    assert any("records no shape_control replay" in w for w in out_shape["problems"])
    unapplied = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
                 "intervention": walked([74])}
    unapplied["intervention"]["intervention"].update({"injections_applied": 0,
                                                      "applied_at": []})
    out_unapplied = _member(M, {"a.json": unapplied})
    assert any("applied 0 of the 1 injections" in w for w in out_unapplied["problems"])
    lying = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
             "intervention": walked([74])}
    lying["intervention"]["control"]["applied_at"] = [33035.25]
    out_lying = _member(M, {"a.json": lying})
    assert any("recorded applications at [33035.25]" in w for w in out_lying["problems"])
    same_step = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
                 "intervention": walked([74])}
    same_step["intervention"]["control"].update({"requested_times": [33035.25],
                                                 "applied_at": [33035.25]})
    out_same = _member(M, {"a.json": same_step})
    assert any("injected at a declared divergence step" in w for w in out_same["problems"])
    # AN INJECTION AT THE WRONG TIMESTEP, which the trace refuses and the checker used to
    # credit: a review moved a genuine intervention ten hours away, left the declared
    # divergence step alone, and kept all four credits.
    shifted = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
               "intervention": walked([74])}
    shifted["intervention"]["intervention"].update(
        {"requested_times": [33044.75], "applied_at": [33044.75]})
    out_shifted = _member(M, {"a.json": shifted})
    assert any("injected at [33044.75] and the case declares [33035.25]" in w
               for w in out_shifted["problems"])
    assert out_shifted["explained_v1_extra_pairs"] == []
    # and a recorded control-step problem is not the checker's to ignore
    flagged = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
               "intervention": walked([74])}
    flagged["intervention"]["control_step_problems"] = ["the control step is not a timestep"]
    out_flagged = _member(M, {"a.json": flagged})
    # THE ARTIFACT'S OWN VERDICT ON ITS CONTROL STEPS IS NOT READ HERE. It could only ever
    # add refusals, so nothing was hidden by it, but it is the judged object's own
    # statement and has no place in the contract the checker applies. The trace computes
    # and records it for its own refusal, and the trace's tests bind that.
    assert not any("not a timestep" in w for w in out_flagged["problems"])
    assert out_flagged["problems"] == _member(M, {"a.json": bound(
        {"unmatched_v1_tracks": [19], "v1_extra_pairs": []})})["problems"]
    # an index the TIME control reproduces as well is no evidence for the intervention
    both = {**bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []}),
            "intervention": walked([19])}
    both["intervention"]["control"]["reference_tracks"] = [
        {"reference_index": 19, "reproduced_exactly": True}]
    # the flag alone would be refused as disagreeing with the tracks, so the control's
    # recorded tracks carry the reference too, and the recomputation agrees with it
    both["intervention"]["control"]["finished_tracks"].append(reference_track((33035.25,)))
    both["intervention"]["control"]["finished_tracks_in_the_western_box"].append({"steps": 4})
    out3 = _member(M, {"a.json": both})
    assert any("the time control reproduces it as well as the intervention" in w
               for w in out3["problems"])
    # THE SHAPE CONTROL IS NOT A NULL CONTROL. It injects an axis where the port had none,
    # with the vertices moved off the crossing, and on three of the four performed
    # experiments it reproduces the tracks as well. What that shows is that the reference
    # vertices are not necessary among the placements tested, and it must not refuse the
    # credit. The reproduction is put in the TRACKS, since a flag without the track behind
    # it is refused as disagreeing with the recomputation.
    refined = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
               "intervention": walked([74])}
    refined["intervention"]["shape_control"]["reference_tracks"] = [
        {"reference_index": 74, "reproduced_exactly": True}]
    refined["intervention"]["shape_control"]["finished_tracks"].append(
        reference_track((33035.25,)))
    refined["intervention"]["shape_control"]["finished_tracks_in_the_western_box"].append(
        {"steps": 4})
    out4 = _member(M, {"a.json": refined})
    assert out4["problems"] == []
    assert out4["explained_by_outcome"]["v1_extra_pairs"]["reproduced"] == [74]
    # while the untouched replay reproducing it is no evidence at all
    already = {**bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]}),
               "intervention": walked([74])}
    already["intervention"]["baseline"]["reference_tracks"] = [
        {"reference_index": 74, "reproduced_exactly": True}]
    already["intervention"]["baseline"]["finished_tracks"].append(reference_track((33035.25,)))
    already["intervention"]["baseline"]["finished_tracks_in_the_western_box"].append(
        {"steps": 4})
    out5 = _member(M, {"a.json": already})
    assert any("the untouched replay reproduces it" in w for w in out5["problems"])


def test_a_control_must_carry_as_many_injections_as_the_experiment():
    """R14 and R15. A two-injection experiment compared against a one-injection control is
    not a comparison, and a partial run naming a step the case does not declare is not
    part of the experiment it claims to belong to."""
    M = _load()
    two = [33035.25, 33035.75]
    V = dumped(two, [-19.4], [39.0])
    case = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]},
                 parameters={"divergence_steps": two, "control_steps": [33035.5, 33036.0]},
                 vertices=V)
    run = case["intervention"]
    axis = [{"lat": [-19.4], "lon": [39.0]}]
    moved = [{"lat": [-19.4], "lon": [35.0]}]
    run["partial_runs"] = ["partial_at_33035.2500", "partial_at_33035.7500"]
    for label, times in (("intervention", two), ("shape_control", two),
                         ("control", [33035.5, 33036.0])):
        by_step = {f"{t:.4f}": (moved if label == "shape_control" else axis) for t in times}
        run[label].update({"requested_times": list(times), "applied_at": list(times),
                           "injections_applied": len(times),
                           "axes_by_requested_step": by_step})
    for t in two:
        run[f"partial_at_{t:.4f}"] = {
            "requested_times": [t], "applied_at": [t], "injections_applied": 1,
            "axes_by_requested_step": {f"{t:.4f}": axis},
            "reference_tracks": [{"reference_index": 74, "reproduced_exactly": False}],
            "finished_tracks": [], "finished_tracks_in_the_western_box": []}
    assert _member(M, {"a.json": case}, V)["problems"] == []
    # a control with ONE injection against a two-step experiment
    short = json.loads(json.dumps(case))
    short["intervention"]["control"].update({"requested_times": [33035.5],
                                             "applied_at": [33035.5],
                                             "injections_applied": 1})
    out = _member(M, {"a.json": short}, V)
    assert any("carries 1 injections against the experiment's 2" in w for w in out["problems"])
    # a partial run at a step the case does not declare
    # A SINGLE-STEP RUN THE CASE DOES NOT DECLARE, in place of one it does
    stray = json.loads(json.dumps(case))
    stray["intervention"]["partial_runs"] = ["partial_at_33035.2500",
                                             "partial_at_33099.0000"]
    stray["intervention"]["partial_at_33099.0000"] = stray["intervention"].pop(
        "partial_at_33035.7500")
    out2 = _member(M, {"a.json": stray}, V)
    assert any("names single-step replays" in w for w in out2["problems"])
    # BOTH SINGLE-STEP RUNS DELETED, which removes the comparison the case rests on
    missing = json.loads(json.dumps(case))
    missing["intervention"]["partial_runs"] = []
    for t in two:
        missing["intervention"].pop(f"partial_at_{t:.4f}")
    out3 = _member(M, {"a.json": missing}, V)
    assert any("names single-step replays [] and the case declares" in w
               for w in out3["problems"])
    # ONE SINGLE-STEP RUN TESTING THE OTHER'S STEP
    swapped = json.loads(json.dumps(case))
    swapped["intervention"]["partial_at_33035.7500"].update(
        {"requested_times": [33035.25], "applied_at": [33035.25],
         "axes_by_requested_step": {"33035.2500": axis}})
    out4 = _member(M, {"a.json": swapped}, V)
    assert any("injected at 33035.25 and is named for 33035.75" in w
               for w in out4["problems"])
    # A CONTROL TIME REPEATED instead of covering the declared schedule
    repeated = json.loads(json.dumps(case))
    repeated["intervention"]["control"].update(
        {"requested_times": [33035.5, 33035.5], "applied_at": [33035.5, 33035.5],
         "axes_by_requested_step": {"33035.5000": axis}})
    out5 = _member(M, {"a.json": repeated}, V)
    assert any("repeats a step" in w for w in out5["problems"])
    # EMPTY AXES, which a review left behind while keeping the receipts and outcomes
    hollow = json.loads(json.dumps(case))
    for label in ("intervention", "control", "shape_control"):
        hollow["intervention"][label]["axes_by_requested_step"] = {
            k: [] for k in hollow["intervention"][label]["axes_by_requested_step"]}
    out6 = _member(M, {"a.json": hollow}, V)
    assert any("records no axis at" in w for w in out6["problems"])
    # THE TRANSLATED AXES SUBSTITUTED INTO THE TIME CONTROL
    swapped_axes = json.loads(json.dumps(case))
    swapped_axes["intervention"]["control"]["axes_by_requested_step"] = {
        "33035.5000": moved, "33036.0000": moved}
    out7 = _member(M, {"a.json": swapped_axes}, V)
    assert any("are not the ones the case's dumped vertices give for it" in w
               for w in out7["problems"])


def test_the_time_control_must_cover_the_declared_schedule_not_merely_belong_to_it():
    """R16: membership in the declared control set is weaker than equality with it. A
    control that injects twice at one declared step belongs to the set and never tests the
    other, which is the shape a review reproduced."""
    M = _load()
    two = [33035.25, 33035.75]
    V = dumped(two, [-19.4], [39.0])
    axis = [{"lat": [-19.4], "lon": [39.0]}]
    moved = [{"lat": [-19.4], "lon": [35.0]}]
    case = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]},
                 parameters={"divergence_steps": two,
                             "control_steps": [33035.5, 33036.0]},
                 vertices=V)
    run = case["intervention"]
    run["partial_runs"] = ["partial_at_33035.2500", "partial_at_33035.7500"]
    for label, times in (("intervention", two), ("shape_control", two),
                         ("control", [33035.5, 33036.0])):
        run[label].update({"requested_times": list(times), "applied_at": list(times),
                           "injections_applied": len(times),
                           "axes_by_requested_step":
                               {f"{t:.4f}": (moved if label == "shape_control" else axis)
                                for t in times}})
    for t in two:
        run[f"partial_at_{t:.4f}"] = {
            "requested_times": [t], "applied_at": [t], "injections_applied": 1,
            "axes_by_requested_step": {f"{t:.4f}": axis},
            "reference_tracks": [{"reference_index": 74, "reproduced_exactly": False}],
            "finished_tracks": [], "finished_tracks_in_the_western_box": []}
    assert _member(M, {"a.json": case}, V)["problems"] == []
    # a control that covers only ONE of the two declared steps, twice over, is a member of
    # the declared set at every injection and still never tests 33036.0
    partial_cover = json.loads(json.dumps(case))
    partial_cover["intervention"]["control"].update(
        {"requested_times": [33035.5, 33035.5], "applied_at": [33035.5, 33035.5],
         "axes_by_requested_step": {"33035.5000": axis}})
    out = _member(M, {"a.json": partial_cover}, V)
    assert any("repeats a step" in w for w in out["problems"])
    # and one that injects at two steps, only one of them declared
    half = json.loads(json.dumps(case))
    half["intervention"]["control"].update(
        {"requested_times": [33035.5, 33037.0], "applied_at": [33035.5, 33037.0],
         "axes_by_requested_step": {"33035.5000": axis, "33037.0000": axis}})
    out2 = _member(M, {"a.json": half}, V)
    assert any("declares its control steps as" in w for w in out2["problems"])


def test_a_control_covering_a_subset_of_a_longer_declared_schedule_is_refused():
    """R17: membership and equality come apart only when the declared control list is
    longer than the experiment, so that case is both refused for its shape AND checked
    against the schedule rather than against mere membership in it."""
    M = _load()
    two = [33035.25, 33035.75]
    V = dumped(two, [-19.4], [39.0])
    axis = [{"lat": [-19.4], "lon": [39.0]}]
    moved = [{"lat": [-19.4], "lon": [35.0]}]
    case = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]},
                 parameters={"divergence_steps": two,
                             "control_steps": [33035.5, 33036.0, 33036.5]},
                 vertices=V)
    run = case["intervention"]
    run["partial_runs"] = ["partial_at_33035.2500", "partial_at_33035.7500"]
    for label, times in (("intervention", two), ("shape_control", two),
                         ("control", [33035.5, 33036.5])):
        run[label].update({"requested_times": list(times), "applied_at": list(times),
                           "injections_applied": len(times),
                           "axes_by_requested_step":
                               {f"{t:.4f}": (moved if label == "shape_control" else axis)
                                for t in times}})
    for t in two:
        run[f"partial_at_{t:.4f}"] = {
            "requested_times": [t], "applied_at": [t], "injections_applied": 1,
            "axes_by_requested_step": {f"{t:.4f}": axis},
            "reference_tracks": [{"reference_index": 74, "reproduced_exactly": False}],
            "finished_tracks": [], "finished_tracks_in_the_western_box": []}
    out = _member(M, {"a.json": case}, V)
    # the evidence check reaches it first: the third declared control step is one at which
    # the two sides do not both hold an observation
    assert any("declares control step 33036.5000, at which the two sides do not both "
               "hold an observation" in w for w in out["problems"])
    assert out["explained_v1_extra_pairs"] == []
    # and the contract, asked on its own, still refuses the shape of the schedule: every
    # injected time is a MEMBER of the declared set, and the schedule is still wrong
    why = M.experiment_problems(run, two, [33035.5, 33036.0, 33036.5], (),
                                M.dumped_vertices(case["at_divergence"]["v1"]))
    assert any("declares 3 control steps for 2 divergence steps" in w for w in why)
    assert any("declares its control steps as" in w for w in why)


def _mozambique_shaped():
    """A two-step case in the Mozambique shape, accepted as it stands, which the checks
    below alter one way at a time.

    THE TWO STEPS CARRY DIFFERENT VERTICES, which the real pair does. A fixture whose two
    steps hold the same geometry cannot tell a correct step pairing from a reversed one,
    and a mutation reversing it survived exactly such a fixture."""
    two = [33035.25, 33035.75]
    # TWO AXES OF TWO VERTICES EACH, not one axis of one. A review removed both cardinality
    # refusals from the comparison and the suite stayed green, because a fixture holding a
    # single vertex in a single group cannot tell a missing axis or a missing endpoint from
    # a present one.
    axes = {"33035.2500": [{"lat": [-19.4, -19.5], "lon": [39.0, 38.9]},
                           {"lat": [-19.7, -19.8], "lon": [38.7, 38.6]}],
            "33035.7500": [{"lat": [-21.6, -21.7], "lon": [39.0, 38.8]},
                           {"lat": [-22.0, -22.1], "lon": [38.5, 38.4]}]}
    moved = {k: [{"lat": list(g["lat"]), "lon": [x - 4.0 for x in g["lon"]]} for g in v]
             for k, v in axes.items()}
    controls = [33035.5, 33036.0]
    vertices = {k: [{"kind": "AXISPTS", "lat": list(g["lat"]), "lon": list(g["lon"])}
                    for g in v] for k, v in axes.items()}
    case = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]},
                 parameters={"divergence_steps": two, "control_steps": controls},
                 vertices=vertices)
    run = case["intervention"]
    run["partial_runs"] = ["partial_at_33035.2500", "partial_at_33035.7500"]
    at_control = {f"{c:.4f}": axes[f"{t:.4f}"] for t, c in zip(two, controls)}
    for label, times, by_step in (("intervention", two, axes),
                                  ("shape_control", two, moved),
                                  ("control", controls, at_control)):
        run[label].update({"requested_times": list(times), "applied_at": list(times),
                           "injections_applied": len(times),
                           "axes_by_requested_step": dict(by_step)})
    run["shape_control"]["longitude_offset_deg"] = 4.0
    for t in two:
        key = f"{t:.4f}"
        run[f"partial_at_{key}"] = {
            "requested_times": [t], "applied_at": [t], "injections_applied": 1,
            "axes_by_requested_step": {key: axes[key]},
            "reference_tracks": [{"reference_index": 74, "reproduced_exactly": False}],
            "finished_tracks": [], "finished_tracks_in_the_western_box": []}
    return case, axes, moved, vertices


def write_output(path, tracks):
    """A finished output in the layout both harness sides write, with a case id."""
    from scipy.io import savemat
    payload = {"n": len(tracks), "case_id": CASE}
    for i, tr in enumerate(tracks):
        payload[f"lat{i}"] = tr["lat"]
        payload[f"lon{i}"] = tr["lon"]
        payload[f"time{i}"] = tr["time"]
    savemat(str(path), payload)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _command(M, tmp_path, case, name, vertices=None, steps=None):
    """Run the command on one case, with the reference log, the two finished outputs and
    the manifest it requires, and report its exit status, whether it wrote anything, and
    where. THE CASE MUST DECLARE THE EVIDENCE IT IS CHECKED AGAINST, or every one of these
    would be refused for the wrong reason, which is a hollow test."""
    st = _steps_of({"c": case}) if steps is None else steps
    pr, pc, out = (tmp_path / f"{name}-r.json", tmp_path / f"{name}-c.json",
                   tmp_path / f"{name}-m.json")
    log, manifest = tmp_path / f"{name}.log", tmp_path / f"{name}-manifest.json"
    ref_mat, port_mat = tmp_path / f"{name}-ref.mat", tmp_path / f"{name}-port.mat"
    residual = residuals(st)
    text = log_text(DUMPED if vertices is None else vertices)
    log.write_text(text)
    log_digest = hashlib.sha256(text.encode()).hexdigest()
    ref_digest = write_output(ref_mat, outputs(st)[REF_DIGEST]["tracks"])
    port_digest = write_output(port_mat, outputs(st)[PORT_DIGEST]["tracks"])
    case = json.loads(json.dumps(case))
    case.setdefault("input_sha256", {}).update(
        {"octave1.log": log_digest, "tracker_octave_instrumented.mat": ref_digest,
         "tracker_port.mat": port_digest})
    residual["input_sha256"]["tracker_port.mat"] = port_digest
    pr.write_text(json.dumps(residual))
    pc.write_text(json.dumps(case))
    manifest.write_text(json.dumps(
        {"logs": [{"sha256": log_digest, "file": str(log), "bytes": len(text)}],
         "outputs": [{"sha256": ref_digest, "file": str(ref_mat), "kind": "reference"},
                     {"sha256": port_digest, "file": str(port_mat), "kind": "port"}]}))
    code = M.main(["--residuals", str(pr), "--cases", str(pc), "--out", str(out),
                   "--reference-log", str(log), "--reference-output", str(ref_mat),
                   str(port_mat), "--manifest", str(manifest)])
    return code, out.exists(), out


def _refuses(M, tmp_path, case, name, vertices=None, steps=None):
    """The command's own answer on an altered record: a refusal, with nothing written."""
    code, exists, _out = _command(M, tmp_path, case, name, vertices, steps)
    return code, exists


def test_the_expected_axes_are_derived_from_the_case_evidence_not_read_beside_the_runs(
        tmp_path):
    """R18, R19 and R20. A review defeated the first version of this check three ways, all
    of them through the command entry point and all of them keeping the credit for the
    pair. The expectation was a map the artifact carried beside its runs, so deleting the
    map switched the comparison off rather than refusing the record, and writing the same
    wrong geometry into both the run and its map made the two agree with each other while
    agreeing with nothing else. The expectation is now DERIVED from the case's dumped
    vertices, the declared control pairing and the recorded translation.

    R18 a case with no dumped vertices to bind the injected axes to is credited;
    R19 a time control carrying the translated shape-control geometry is credited;
    R20 the same substitution is credited when a map declaring it accompanies it;
    R21 a time control carrying the other declared step's geometry is credited."""
    M = _load()
    case, axes, moved, V = _mozambique_shaped()
    assert _member(M, {"a.json": case}, V)["problems"] == []

    # R18: THE EVIDENCE ITSELF REMOVED. A case that records no vertices at a step its
    # reference log holds one at, and declares as a divergence step, is refused by the
    # SOURCE check, which runs before the experiment is looked at.
    blind = json.loads(json.dumps(case))
    blind.pop("at_divergence")
    out = _member(M, {"a.json": blind}, V)
    assert any("records no axis vertices at 33035.2500, which its reference log holds"
               in w for w in out["problems"])
    assert out["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, blind, "blind", V, (33035.25, 33035.75)) == (2, False)
    half = json.loads(json.dumps(case))
    half["at_divergence"]["v1"].pop("33035.7500")
    out2 = _member(M, {"a.json": half}, V)
    assert any("records no axis vertices at 33035.7500" in w for w in out2["problems"])
    # and the derivation refuses the same absence on its own, where a reader that never
    # reached the source check would meet it
    why = M.experiment_problems(case["intervention"], [33035.25, 33035.75],
                                [33035.5, 33036.0], (), {})
    assert any("no dumped reference vertices at 33035.2500" in w for w in why)

    # R19: THE TRANSLATED AXES SUBSTITUTED INTO THE TIME CONTROL. The control must carry
    # the reference geometry at a step the two sides agree on, since the control that
    # isolates the step is not also a control on the shape.
    swapped = json.loads(json.dumps(case))
    swapped["intervention"]["control"]["axes_by_requested_step"] = {
        "33035.5000": moved["33035.2500"], "33036.0000": moved["33035.7500"]}
    out3 = _member(M, {"a.json": swapped}, V)
    assert any("control replay's axes at 33035.5000 are not the ones the case's dumped "
               "vertices give for it" in w for w in out3["problems"])
    assert out3["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, swapped, "swapped", V, (33035.25, 33035.75)) == (2, False)

    # R20: THE SAME SUBSTITUTION WITH A MAP AGREEING WITH IT. A declaration beside the run
    # cannot revive the bypass, because nothing reads one.
    declared = json.loads(json.dumps(swapped))
    declared["intervention"]["control_axes_by_step"] = {
        "33035.5000": moved["33035.2500"], "33036.0000": moved["33035.7500"]}
    declared["intervention"]["reference_axes_by_step"] = dict(axes)
    out4 = _member(M, {"a.json": declared}, V)
    assert any("are not the ones the case's dumped vertices give for it" in w
               for w in out4["problems"])
    assert out4["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, declared, "declared", V, (33035.25, 33035.75)) == (2, False)

    # R21: THE CONTROL PAIRED WITH THE WRONG STEP. Each control time is paired with one
    # divergence step, and it must carry THAT step's geometry, which only a case whose
    # steps hold different vertices can tell apart from the reverse pairing.
    crossed = json.loads(json.dumps(case))
    crossed["intervention"]["control"]["axes_by_requested_step"] = {
        "33035.5000": axes["33035.7500"], "33036.0000": axes["33035.2500"]}
    out5 = _member(M, {"a.json": crossed}, V)
    assert any("control replay's axes at 33035.5000 are not the ones" in w
               for w in out5["problems"])
    assert out5["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, crossed, "crossed", V, (33035.25, 33035.75)) == (2, False)

    # and an expectation that does not cover the steps a run recorded is refused on that
    # alone, rather than only on the schedule check that follows it
    elsewhere = json.loads(json.dumps(case))
    elsewhere["intervention"]["control"].update(
        {"requested_times": [33035.5, 33037.0], "applied_at": [33035.5, 33037.0],
         "axes_by_requested_step": {"33035.5000": axes["33035.2500"],
                                    "33037.0000": axes["33035.7500"]}})
    out6 = _member(M, {"a.json": elsewhere}, V)
    assert any("the case's own vertices give" in w for w in out6["problems"])


def test_a_geometry_that_cannot_be_compared_is_refused_rather_than_agreed_with(tmp_path):
    """R22 through R25. A second review defeated the first derived comparison four ways,
    all through the command and all keeping the credit for pair 106.

    R22 a not-a-number in the dumped reference coordinates is credited, because a
        difference from one never EXCEEDS a tolerance and only the run's own coordinates
        were checked for finiteness;
    R23 a not-a-number translation is credited, and the prose describes it;
    R24 a translation of 1e-12, finite and nonzero and invisible at the comparison's own
        tolerance, leaves the shape control carrying the intervention's geometry;
    R25 an axis or an endpoint added to a run is credited, which a fixture holding one
        vertex in one group cannot tell from the correct record."""
    M = _load()
    case, axes, moved, V = _mozambique_shaped()
    assert _member(M, {"a.json": case}, V)["problems"] == []

    # R22: an unusable expectation. NaN is not written bare here, since that is not valid
    # JSON. A review wrote the string, which floats to a not-a-number on conversion.
    # THROUGH THE COMMAND THE SOURCE CHECK REACHES IT FIRST, and it must: a coordinate that
    # is not a number can never be shown to be the one the reference log holds, whatever
    # the artifact's other fields say. The log the record is checked against here is built
    # from the spoiled record itself, so the refusal is not merely a disagreement.
    for bad in ("NaN", None, float("inf"), [39.0], []):
        spoiled = json.loads(json.dumps(case))
        spoiled["at_divergence"]["v1"]["33035.2500"][0]["lon"] = (
            bad if isinstance(bad, list) else [bad, 38.9])
        out = _member(M, {"a.json": spoiled}, spoiled["at_divergence"]["v1"])
        assert any("at 33035.2500 are not the ones its reference log holds" in w
                   or "records no axis vertices at 33035.2500" in w
                   for w in out["problems"]), bad
        assert out["explained_v1_extra_pairs"] == []
        assert _refuses(M, tmp_path, spoiled, f"unusable-{len(out['problems'])}",
                        spoiled["at_divergence"]["v1"]) == (2, False)
    # AND THE DERIVATION REFUSES IT ON ITS OWN, which is where a reader that has already
    # passed the source check meets it: the trace builds its experiment from the log and
    # never runs the source check, so this guard is the one standing there.
    for lat, lon in ((["NaN", -19.5], [39.0, 38.9]), ([-19.4, -19.5], [39.0]),
                     ([], []), ([-19.4], [float("inf")])):
        why = M.experiment_problems(
            case["intervention"], [33035.25, 33035.75], [33035.5, 33036.0], (),
            {"33035.2500": [{"lat": lat, "lon": lon}],
             "33035.7500": [{"lat": [-21.6], "lon": [39.0]}]})
        assert any("empty, ragged or not finite" in w for w in why), (lat, lon)

    # R23 and R24: the translation.
    for value, expected in (("NaN", "zero or not a finite number"),
                            (0, "zero or not a finite number"),
                            (1e-12, "leaves its vertices equal to the intervention's")):
        tiny = json.loads(json.dumps(case))
        tiny["intervention"]["shape_control"]["longitude_offset_deg"] = value
        tiny["intervention"]["shape_control"]["axes_by_requested_step"] = dict(axes)
        out = _member(M, {"a.json": tiny}, V)
        assert any(expected in w for w in out["problems"]), value
        assert out["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, tiny, "invisible", V, (33035.25, 33035.75)) == (2, False)

    # R25: cardinality, which needs more than one axis and more than one vertex to test.
    extra_axis = json.loads(json.dumps(case))
    extra_axis["intervention"]["control"]["axes_by_requested_step"]["33035.5000"] = (
        list(axes["33035.2500"]) + [dict(axes["33035.2500"][0])])
    out2 = _member(M, {"a.json": extra_axis}, V)
    assert any("control replay's axes at 33035.5000 are not the ones" in w
               for w in out2["problems"])
    assert _refuses(M, tmp_path, extra_axis, "extra-axis", V, (33035.25, 33035.75)) == (2, False)
    extra_point = json.loads(json.dumps(case))
    grew = json.loads(json.dumps(axes["33035.2500"]))
    grew[0]["lat"].append(-19.6)
    grew[0]["lon"].append(38.8)
    extra_point["intervention"]["control"]["axes_by_requested_step"]["33035.5000"] = grew
    out3 = _member(M, {"a.json": extra_point}, V)
    assert any("control replay's axes at 33035.5000 are not the ones" in w
               for w in out3["problems"])
    assert _refuses(M, tmp_path, extra_point, "extra-point", V, (33035.25, 33035.75)) == (2, False)
    # and a coordinate changed beyond the FIRST vertex of the SECOND axis
    late = json.loads(json.dumps(case))
    moved_late = json.loads(json.dumps(axes["33035.2500"]))
    moved_late[1]["lat"][1] = -19.9
    late["intervention"]["control"]["axes_by_requested_step"]["33035.5000"] = moved_late
    out4 = _member(M, {"a.json": late}, V)
    assert any("control replay's axes at 33035.5000 are not the ones" in w
               for w in out4["problems"])


def test_credit_requires_the_original_log_and_not_the_artifact_s_own_copy(tmp_path):
    """R26 through R31. A review translated a case's embedded vertices AND every replay's
    axes together by one degree, left the recorded digests and outcomes alone, and the
    accounting credited the pair: every field the checker observed still agreed with every
    other one. The vertices are now re-read from the reference log the case declares.

    R26 a case whose embedded vertices and replays are translated together is credited;
    R27 credit is granted with no reference log supplied at all;
    R28 credit is granted from a log the retained manifest does not name;
    R29 credit is granted from a log whose contents do not match the digest it is filed
        under;
    R30 credit is granted when the case declares a log that was not supplied;
    R31 credit is granted when the log holds no axis at a declared divergence step."""
    M = _load()
    case, axes, moved, V = _mozambique_shaped()
    assert _member(M, {"a.json": case}, V)["problems"] == []

    # R26: the coordinated translation, which is the reproduction that forced this check.
    together = json.loads(json.dumps(case))
    for recs in together["at_divergence"]["v1"].values():
        for r in recs:
            r["lon"] = [x - 1.0 for x in r["lon"]]
    for label, run in together["intervention"].items():
        if isinstance(run, dict) and isinstance(run.get("axes_by_requested_step"), dict):
            for groups in run["axes_by_requested_step"].values():
                for g in groups:
                    g["lon"] = [x - 1.0 for x in g["lon"]]
    out = _member(M, {"a.json": together}, V)
    assert any("are not the ones its reference log holds there" in w
               for w in out["problems"])
    assert out["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, together, "together", V, (33035.25, 33035.75)) == (2, False)
    # every field the checker used to observe still agrees with every other one, which is
    # the whole point: the experiment checks alone report nothing wrong with it
    assert M.experiment_problems(together["intervention"], [33035.25, 33035.75],
                                 [33035.5, 33036.0], (),
                                 M.dumped_vertices(together["at_divergence"]["v1"])) == []

    # R27: no log at all. Nothing that skipped the check may report the result of one.
    two_step = residuals((33035.25, 33035.75))
    assert M.membership(two_step, {"a.json": case}, {})["problems"] == [
        "no reference log was supplied, so no case's recorded axis vertices can be traced "
        "to the run that produced them"]
    assert "explained_v1_extra_pairs" not in M.membership(two_step, {"a.json": case}, {})

    # R28 and R29: the manifest is what says a log is one this project accepts, and a
    # digest the candidate supplies names its claimed input rather than authorizing it.
    text = log_text(V)
    log, manifest = tmp_path / "ref.log", tmp_path / "manifest.json"
    log.write_text(text)
    real = hashlib.sha256(text.encode()).hexdigest()
    manifest.write_text(json.dumps({"logs": [{"sha256": real, "file": str(log)}]}))
    got, why = M.verified_logs([str(log)], str(manifest))
    assert why == [] and list(got) == [real]
    unpinned = tmp_path / "unpinned.json"
    unpinned.write_text(json.dumps({"logs": [{"sha256": "9" * 64, "file": str(log)}]}))
    _got, why2 = M.verified_logs([str(log)], str(unpinned))
    assert any("the manifest does not name as one this project accepts" in w for w in why2)
    log.write_text(text + "AXISPTS 33035.2500 1 -19.4 39.0\n")   # filed under the old digest
    _got2, why3 = M.verified_logs([str(log)], str(manifest))
    assert any("the manifest does not name" in w for w in why3)
    _got3, why4 = M.verified_logs([str(tmp_path / "absent.log")], str(manifest))
    assert any("is not present" in w for w in why4)
    _got4, why5 = M.verified_logs([str(log)], str(tmp_path / "no-manifest.json"))
    assert any("manifest" in w and "is not present" in w for w in why5)

    # R30: a case declaring a log nobody supplied.
    elsewhere = json.loads(json.dumps(case))
    elsewhere["input_sha256"]["octave1.log"] = "a" * 64
    out2 = _member(M, {"a.json": elsewhere}, V)
    assert any("which was not supplied to this command" in w for w in out2["problems"])
    assert out2["explained_v1_extra_pairs"] == []
    missing = json.loads(json.dumps(case))
    missing["input_sha256"].pop("octave1.log")
    out3 = _member(M, {"a.json": missing}, V)
    assert any("declares no reference-log digest" in w for w in out3["problems"])

    # R31: a log that holds no axis where the case says version 1 dumped one.
    thin = {k: v for k, v in V.items() if k != "33035.7500"}
    out4 = _member(M, {"a.json": case}, thin)
    assert any("at 33035.7500 and its reference log holds none there" in w
               for w in out4["problems"])
    assert out4["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, case, "thin", thin) == (2, False)

    # THE SELECTION IS REPRODUCED, NOT ASSUMED, so a log dump outside the case's own
    # declared box or outside its window is not one of the case's axes.
    far = json.loads(json.dumps(case))
    far["parameters"]["box"] = {"lat": [0.0, 10.0], "lon": [0.0, 10.0]}
    out5 = _member(M, {"a.json": far}, V)
    assert any("its reference log holds none there" in w for w in out5["problems"])
    narrow = json.loads(json.dumps(case))
    narrow["parameters"]["window"] = [33035.0, 33035.5]
    out6 = _member(M, {"a.json": narrow}, V)
    assert any("at 33035.7500 and its reference log holds none there" in w
               for w in out6["problems"])
    assert out6["explained_v1_extra_pairs"] == []

    # A LOG LINE THAT WILL NOT PARSE, or that holds an odd or nonfinite set of
    # coordinates, is a refusal rather than a line quietly passed over. A parser that
    # skips what it cannot read reports "the log holds nothing here", which is a different
    # and much weaker statement than "the log could not be read".
    for line, why in (("AXISPTS 33035.2500 2 -19.4 39.0 -19.5\n",
                       "odd or nonfinite set of coordinates"),
                      ("AXISPTS 33035.2500 1 -19.4 nope\n",
                       "not a readable axis dump"),
                      ("AXISPTS notatime 1 -19.4 39.0\n",
                       "not a readable axis dump")):
        text = log_text(V) + line
        got, why_log = M.axes_from_log(text, [33030.0, 33040.0],
                                       {"lat": [-40.0, -10.0], "lon": [-70.0, 45.0]}, None)
        assert any(why in w for w in why_log), line
        logs = {"1" * 64: {"path": "fixture.log", "text": text}}
        out = M.membership(residuals((33035.25, 33035.75)), {"a.json": case}, logs, (),
                           outputs((33035.25, 33035.75)))
        assert any(why in w for w in out["problems"]), line
        assert out["explained_v1_extra_pairs"] == []


EVIDENCE = os.path.join(HERE, "..", "docs", "aewc_v2", "evidence")
ARTIFACTS = os.path.join(HERE, "..", "docs", "aewc_v2", "artifacts")


def _retained_evidence_present():
    """Whether every retained file the manifest names is on disk. A copy of the tracked
    tree that lacks one (a mutation run, a public export) SKIPS the tests that read the
    genuine evidence rather than failing them, and a skip is reported as one."""
    manifest = os.path.join(ARTIFACTS, "reference_logs.json")
    if not os.path.exists(manifest):
        return False
    m = json.load(open(manifest))
    return all(os.path.exists(os.path.join(HERE, "..", e["file"]))
               for e in list(m.get("logs") or ()) + list(m.get("outputs") or ()))


RETAINED = _retained_evidence_present()


@pytest.mark.skipif(not RETAINED, reason="retained evidence not present")
def test_the_two_log_readers_agree_on_the_retained_logs_under_every_case_s_gate(tmp_path):
    """The checker parses the reference log with a reader of its own, deliberately not the
    trace's, so that a defect in one cannot authorize the other. Two readers of one format
    can drift, and an outside review named the places it expected them to: the box
    fallback, the time key, the treatment of a bad line. THIS BINDS THEM: both readers are
    run over both retained logs under every committed case's own window and box, and the
    axis maps they produce must be identical. A reader that drifted would show up here as
    a disagreement, not as two bugs quietly filtering the same line."""
    import gzip
    import importlib.util
    M = _load()
    spec = importlib.util.spec_from_file_location(
        "trace_sahara_case", os.path.join(HERE, "..", "scripts", "trace_sahara_case.py"))
    T = importlib.util.module_from_spec(spec)
    sys.modules["trace_sahara_case"] = T
    spec.loader.exec_module(T)
    manifest = json.load(open(os.path.join(ARTIFACTS, "reference_logs.json")))
    plain = {}
    for e in manifest["logs"]:
        raw = gzip.open(os.path.join(HERE, "..", e["file"]), "rb").read()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"]
        path = tmp_path / (e["sha256"][:12] + ".log")
        path.write_bytes(raw)
        plain[e["sha256"]] = (str(path), raw.decode())
    cases = sorted(f for f in os.listdir(ARTIFACTS) if "_case_" in f and f.endswith(".json"))
    assert len(cases) >= 5
    compared = 0
    for name in cases:
        case = json.load(open(os.path.join(ARTIFACTS, name)))
        p = case["parameters"]
        digest = case["input_sha256"]["octave1.log"]
        path, text = plain[digest]
        # the checker's reader, from the case's declared gate
        mine, why = M.axes_from_log(text, p["window"], p["box"], p.get("axis_box"))
        assert why == [], (name, why)
        # the trace's reader, under the same gate set through its module globals
        T.WINDOW = tuple(p["window"])
        T.BOX = {"lat": tuple(p["box"]["lat"]), "lon": tuple(p["box"]["lon"])}
        T.AXIS_BOX = (None if p.get("axis_box") is None else
                      {"lat": tuple(p["axis_box"]["lat"]), "lon": tuple(p["axis_box"]["lon"])})
        theirs = {}
        for r in T.read_v1_dumps(path):
            if r["kind"] == "AXISPTS":
                theirs.setdefault(T.step_key(r["time"]), []).append(
                    {"lat": r["lat"], "lon": r["lon"]})
        assert sorted(mine) == sorted(theirs), (name, sorted(mine), sorted(theirs))
        for key in mine:
            assert M._axes_equal(mine[key], theirs[key]), (name, key)
        # and what the case recorded is what BOTH readers give at its declared steps
        recorded = M.dumped_vertices(case["at_divergence"]["v1"])
        for key in recorded:
            assert M._axes_equal(recorded[key], mine[key]), (name, key)
        compared += 1
    assert compared == len(cases)


def test_the_steps_a_case_is_judged_at_are_the_residual_pair_s_own(tmp_path):
    """R32 and R33. Two outside reviews, reading cold, found the same thing from the
    packet alone: the checker verified the case's vertices against the log inside a gate
    the case drew itself. Shrink the window to one of a pair's two missing observations,
    declare that one step, inject once, and be credited for the whole pair. The residual
    artifact records each pair's version-1-extra times, and the case must declare exactly
    those. An unmatched track's record carries only its span, so that check is the weaker
    one and is bound as such."""
    M = _load()
    case, axes, moved, V = _mozambique_shaped()
    assert _member(M, {"a.json": case}, V)["problems"] == []

    # the reviews' attack: a two-observation pair explained at one of its steps
    Vsub = {"33035.2500": V["33035.2500"]}
    half = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]},
                 parameters={"divergence_steps": [33035.25], "control_steps": [33035.5],
                             "window": [33035.0, 33035.5]}, vertices=Vsub)
    iv = half["intervention"]
    for label in ("intervention", "shape_control"):
        iv[label]["axes_by_requested_step"] = {
            "33035.2500": (moved if label == "shape_control" else axes)["33035.2500"]}
    iv["control"]["axes_by_requested_step"] = {"33035.5000": axes["33035.2500"]}
    # as a ONE-step case it is internally consistent: the pair's own times are what refuse it
    assert _member(M, {"a.json": half}, Vsub, steps=(33035.25,))["problems"] == []
    out = _member(M, {"a.json": half}, Vsub, steps=(33035.25, 33035.75))
    assert any("declares divergence steps ['33035.2500'], while the pair's version-1-extra "
               "observations are at ['33035.2500', '33035.7500']" in w
               for w in out["problems"])
    assert out["explained_v1_extra_pairs"] == []
    pr, pc, o = tmp_path / "r.json", tmp_path / "c.json", tmp_path / "m.json"
    pr.write_text(json.dumps(residuals((33035.25, 33035.75))))
    text = log_text(V)
    log, manifest = tmp_path / "ref.log", tmp_path / "manifest.json"
    log.write_text(text)
    d = hashlib.sha256(text.encode()).hexdigest()
    half["input_sha256"]["octave1.log"] = d
    pc.write_text(json.dumps(half))
    manifest.write_text(json.dumps({"logs": [{"sha256": d, "file": str(log)}]}))
    assert _refuses(M, tmp_path, half, "half", Vsub, (33035.25, 33035.75)) == (2, False)
    # a pair the residual artifact records no extra times for cannot be claimed at all
    bare = residuals((33035.25, 33035.75))
    for pair in bare["pairs"]:
        pair.pop("v1_extra", None)
    out2 = M.membership(bare, {"a.json": case}, logs(V), (), outputs((33035.25, 33035.75)))
    assert any("records no version-1-extra times" in w for w in out2["problems"])

    # R33 (revised): an unmatched track is judged at steps its FINISHED track holds, read
    # from the pinned reference output. The span check that stood here admitted steps the
    # track was never observed at, and a log TRACK-line check that replaced it read a live
    # list position as a finished index.
    lone = bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []})
    assert _member(M, {"a.json": lone})["problems"] == []
    off = json.loads(json.dumps(lone))
    off["parameters"]["divergence_steps"] = [33035.75]          # inside the span, never held
    out3 = _member(M, {"a.json": off}, steps=(33035.25,))      # the outputs stay genuine
    assert any("claims unmatched track 19 and declares step 33035.7500, at which version "
               "1's finished track holds no observation" in w for w in out3["problems"])
    assert out3["explained_unmatched"] == []


def test_an_absent_time_control_is_not_a_passed_one():
    """R34. `not any([])` is true, so an experiment record with no time control read as
    one whose control failed to reproduce the track. The command never reaches this, since
    the experiment contract refuses an absent replay first, but the function is public."""
    M = _load()
    run = walked([74])
    assert M.outcome_for(run, 74) == ("reproduced", None)
    run.pop("control")
    outcome, why = M.outcome_for(run, 74)
    assert outcome is None and "no time control was recorded" in why


def test_the_axis_box_selects_the_log_s_axes_when_declared_and_the_case_box_otherwise():
    """No committed case declares an axis box, so the branch that prefers it to the case
    box was never exercised, and a mutation collapsing it survived. Here an axis box is
    declared narrower than the case box, an axis lies inside the case box and outside the
    axis box, and the reader must NOT select it. Without the axis box it must."""
    M = _load()
    box = {"lat": [-40.0, -10.0], "lon": [-70.0, 45.0]}
    axis_box = {"lat": [-20.0, -19.0], "lon": [38.0, 40.0]}
    text = ("AXISPTS 33035.2500 1 -19.4 39.0\n"          # inside both
            "AXISPTS 33035.2500 1 -32.0 -60.3\n"         # inside the case box only
            "AXISPTS 33035.7500 2 -21.6 39.0 -21.7 38.8\n")  # outside the axis box
    narrow, why = M.axes_from_log(text, [33030.0, 33040.0], box, axis_box)
    assert why == []
    assert sorted(narrow) == ["33035.2500"]
    assert narrow["33035.2500"] == [{"lat": [-19.4], "lon": [39.0]}]
    wide, _ = M.axes_from_log(text, [33030.0, 33040.0], box, None)
    assert sorted(wide) == ["33035.2500", "33035.7500"]
    assert len(wide["33035.2500"]) == 2
    # and the trace's reader gives the same answer under the same gate, which is the
    # binding the two-reader design rests on
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trace_sahara_case", os.path.join(HERE, "..", "scripts", "trace_sahara_case.py"))
    T = importlib.util.module_from_spec(spec)
    sys.modules["trace_sahara_case"] = T
    spec.loader.exec_module(T)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(text)
    T.WINDOW = (33030.0, 33040.0)
    T.BOX = {"lat": tuple(box["lat"]), "lon": tuple(box["lon"])}
    for their_axis_box, mine in ((axis_box, narrow), (None, wide)):
        T.AXIS_BOX = (None if their_axis_box is None else
                      {"lat": tuple(their_axis_box["lat"]), "lon": tuple(their_axis_box["lon"])})
        theirs = {}
        for r in T.read_v1_dumps(fh.name):
            if r["kind"] == "AXISPTS":
                theirs.setdefault(T.step_key(r["time"]), []).append(
                    {"lat": r["lat"], "lon": r["lon"]})
        assert sorted(theirs) == sorted(mine)
        assert all(M._axes_equal(theirs[k], mine[k]) for k in mine)
    os.unlink(fh.name)


def test_the_written_artifact_states_what_a_credit_rests_on(tmp_path):
    """The field names say "explained", and a reader will take the word at its width. The
    artifact must carry the statement of what is independently verified and what is the
    case's own report, and the command must print it."""
    M = _load()
    case = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]})
    code, exists, out = _command(M, tmp_path, case, "basis")
    assert (code, exists) == (0, True)
    written = json.loads(out.read_text())
    basis = written["credit_basis"]
    assert set(basis) == {"read_from_retained_evidence", "derived_under_a_declared_region",
                          "claimed_by_the_replay_records",
                          "recomputed_from_the_replay_records_against_the_pinned_reference",
                          "what_a_credit_therefore_means"}
    assert any("produced the finished tracks it records" in line
               for line in basis["claimed_by_the_replay_records"])
    assert any("reproduced_exactly" in line for line in
               basis["recomputed_from_the_replay_records_against_the_pinned_reference"])
    assert "does not establish that the replay produced those tracks" in \
        basis["what_a_credit_therefore_means"]
    assert written["explained_v1_extra_pairs"] == [74]
    assert len(written["reference_outputs_sha256"]) == 2


def test_the_box_must_hold_the_claimed_index_s_own_location_and_unmatched_steps_the_log_s(
        tmp_path):
    """R35 through R37. Pair TIMES are fixed by the residual record, but the box is still
    the case's, and a box can be drawn around a convenient contemporaneous axis and away
    from the one the pair is about (on the Mozambique log there are 117 axis groups at the
    first step, one of them in the case box). The residual record carries each index's own
    mean location, which the box must contain. For an unmatched track the span check
    admitted steps at which the track was not observed at all; the pinned log's own TRACK
    lines say when it was."""
    M = _load()
    case = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]})
    assert _member(M, {"a.json": case})["problems"] == []
    # R35 (revised): THE EFFECTIVE SELECTOR must contain version 1's own observation at
    # every declared step. A review changed only the axis box, selected nine South
    # American axes for the Mozambique pair, and a guard on the case box was satisfied
    # throughout, so the guard is on whichever region actually selects.
    away = json.loads(json.dumps(case))
    away["parameters"]["box"] = {"lat": [-40.0, -35.0], "lon": [-70.0, -65.0]}
    out = _member(M, {"a.json": away})
    assert any("claims pair 74 and its effective selection region does not contain "
               "version 1's own observation at 33035.2500 (-32.00, -60.32)" in w
               for w in out["problems"])
    assert out["explained_v1_extra_pairs"] == []
    # R36 (revised): an unmatched track whose declared CONTROL step its finished track
    # does not hold
    lone = bound({"unmatched_v1_tracks": [19], "v1_extra_pairs": []})
    assert _member(M, {"a.json": lone})["problems"] == []
    late = json.loads(json.dumps(lone))
    late["parameters"]["control_steps"] = [33035.75]
    late["intervention"]["control"].update(
        {"requested_times": [33035.75], "applied_at": [33035.75],
         "axes_by_requested_step": {"33035.7500": DUMPED["33035.2500"]}})
    out4 = _member(M, {"a.json": late})
    assert any("declares control step 33035.7500, at which version 1's finished track "
               "holds no observation" in w for w in out4["problems"])
    assert out4["explained_unmatched"] == []
    # R37: the basis names the grade per index
    both = {"a.json": case, "b.json": lone}
    out5 = _member(M, both)
    assert out5["problems"] == []
    assert out5["credit_basis_by_index"]["74"].startswith("pair:")
    assert out5["credit_basis_by_index"]["19"].startswith("unmatched track:")
    assert "the case's own" in out5["credit_basis_by_index"]["19"]


def test_outcomes_are_recomputed_from_recorded_tracks_and_scope_from_the_pinned_outputs(
        tmp_path):
    """R38 through R43. The credit rule used to read each replay's reproduced_exactly flag
    and its western-box list as the replay wrote them. Every replay now records its
    complete finished tracks, the checker recomputes both outcomes against the pinned
    reference by final index, and a recorded value that disagrees with the recomputation
    is a refusal. Scope, control-step agreement and the effective region are read from
    the two pinned outputs rather than from the residual record or the case alone."""
    M = _load()
    two = (33035.25, 33035.75)
    case, axes, moved, V = _mozambique_shaped()
    assert _member(M, {"a.json": case}, V)["problems"] == []
    assert _command(M, tmp_path, case, "genuine", V, two)[:2] == (0, True)

    # R40: the flag, flipped and unflipped, against the recorded tracks
    lying = json.loads(json.dumps(case))
    lying["intervention"]["baseline"]["reference_tracks"][0]["reproduced_exactly"] = True
    out = _member(M, {"a.json": lying}, V)
    assert any("baseline replay records reproduced_exactly=True for 74 and its own "
               "finished tracks give False" in w for w in out["problems"])
    assert out["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, lying, "lying", V, two) == (2, False)
    hollow = json.loads(json.dumps(case))
    hollow["intervention"]["intervention"]["finished_tracks"] = []      # flag still True
    out2 = _member(M, {"a.json": hollow}, V)
    assert any("intervention replay records reproduced_exactly=True for 74 and its own "
               "finished tracks give False" in w for w in out2["problems"])
    gone = json.loads(json.dumps(case))
    gone["intervention"]["control"].pop("finished_tracks")
    out3 = _member(M, {"a.json": gone}, V)
    assert any("control replay records no finished tracks" in w for w in out3["problems"])
    assert _refuses(M, tmp_path, gone, "gone", V, two) == (2, False)
    # and the credit follows the RECOMPUTED value, never the flag: a track equal to the
    # reference smuggled into the time control with the flag left False is refused, since
    # the recomputation says the control reproduced it
    smuggled = json.loads(json.dumps(case))
    smuggled["intervention"]["control"]["finished_tracks"].append(reference_track(two))
    out4 = _member(M, {"a.json": smuggled}, V)
    assert any("control replay records reproduced_exactly=False for 74 and its own "
               "finished tracks give True" in w for w in out4["problems"])

    # R41: the western list against the tracks
    miscounted = json.loads(json.dumps(case))
    miscounted["intervention"]["intervention"]["finished_tracks_in_the_western_box"] = []
    out5 = _member(M, {"a.json": miscounted}, V)
    assert any("intervention replay records 0 finished tracks in the western box and its "
               "own finished tracks give 1" in w for w in out5["problems"])

    # R42: scope from the outputs, not the residual record and not the case
    agree = json.loads(json.dumps(case))
    agree["parameters"]["control_steps"] = [33035.5, 33035.75]     # port lacks 33035.75
    agree["intervention"]["control"].update(
        {"requested_times": [33035.5, 33035.75], "applied_at": [33035.5, 33035.75],
         "axes_by_requested_step": {"33035.5000": axes["33035.2500"],
                                    "33035.7500": axes["33035.7500"]}})
    out6 = _member(M, {"a.json": agree}, V)
    assert any("declares control step 33035.7500, at which the two sides do not both "
               "hold an observation" in w for w in out6["problems"])
    r = residuals(two)
    r["pairs"][0]["v1_extra"]["times"] = [33035.25]                 # residual disagrees
    out7 = M.membership(r, {"a.json": case}, logs(V), (), outputs(two))
    assert any("whose residual record says version 1's extra observations are at "
               "['33035.2500'] and the retained outputs say ['33035.2500', '33035.7500']"
               in w for w in out7["problems"])
    r2 = residuals(two)
    r2["pairs"][0]["port_index"] = 99                                # no such port track
    out8 = M.membership(r2, {"a.json": case}, logs(V), (), outputs(two))
    assert any("pair 74's port counterpart, which the retained output does not hold" in w
               for w in out8["problems"])

    # R39: the axis box, disjoint from the pair, with the case box untouched
    elsewhere = json.loads(json.dumps(case))
    elsewhere["parameters"]["axis_box"] = {"lat": [-25.0, -15.0], "lon": [35.0, 45.0]}
    out9 = _member(M, {"a.json": elsewhere}, V)
    assert any("effective selection region does not contain version 1's own observation "
               "at 33035.2500" in w for w in out9["problems"])
    assert out9["explained_v1_extra_pairs"] == []
    assert _refuses(M, tmp_path, elsewhere, "elsewhere", V, two) == (2, False)

    # R43: no outputs, and an unpinned one
    assert M.membership(residuals(two), {"a.json": case}, logs(V), (), {})["problems"] == [
        "no finished output was supplied, so no case's scope or outcome can be read from "
        "the retained runs"]
    ref_mat = tmp_path / "unpinned.mat"
    write_output(ref_mat, outputs(two)[REF_DIGEST]["tracks"])
    manifest = tmp_path / "manifest-without-outputs.json"
    manifest.write_text(json.dumps({"logs": [], "outputs": []}))
    got, why = M.verified_outputs([str(ref_mat)], str(manifest))
    assert got == {} and any("the manifest does not name" in w for w in why)
    _got, why2 = M.verified_outputs([str(tmp_path / "absent.mat")], str(manifest))
    assert any("is not present" in w for w in why2)
    other = json.loads(json.dumps(case))
    other["input_sha256"]["tracker_octave_instrumented.mat"] = "9" * 64
    out10 = _member(M, {"a.json": other}, V)
    assert any("declares reference output 999999999999, which was not supplied" in w
               for w in out10["problems"])


@pytest.mark.skipif(not RETAINED, reason="retained evidence not present")
def test_the_genuine_sahara_track_is_read_by_final_index_not_live_position():
    """The defect, on the retained evidence. The log line `TRACK 33025.0000 19 8 -13.0
    -54.0` names live list position 19 at that loop time; finished track 19 is at (18.7,
    -8.05) then, and holds no observation at 33024.25, 33025.25, 33025.75 or 33027.00,
    all inside its lifetime. A check built on the log accepted all four."""
    M = _load()
    manifest = os.path.join(ARTIFACTS, "reference_logs.json")
    entries = json.load(open(manifest))["outputs"]
    paths = [os.path.join(HERE, "..", e["file"]) for e in entries]
    got, why = M.verified_outputs(paths, manifest)
    assert why == [] and len(got) == len(entries)
    case = json.load(open(os.path.join(ARTIFACTS, "sahara_case_2026-09-18.json")))
    ref = got[case["input_sha256"]["tracker_octave_instrumented.mat"]]
    assert ref["kind"] == "reference"
    t19 = ref["tracks"][19]
    assert M._at(t19, "33025.0000") == (18.7, -8.05)
    residual = json.load(open(os.path.join(ARTIFACTS,
                                           "tracker_oracle_residuals_2026-09-18.json")))
    for never in ("33024.2500", "33025.2500", "33025.7500", "33027.0000"):
        assert never not in M._times(t19)
        off = json.loads(json.dumps(case))
        off["parameters"]["divergence_steps"] = [float(never)]
        why = M.evidence_problems("sahara", off, residual, case["explains"], got)
        assert any(f"declares step {never}, at which version 1's finished track holds no "
                   f"observation" in w for w in why), never
    assert M.evidence_problems("sahara", case, residual, case["explains"], got) == []


def test_the_reference_output_kind_the_output_derived_steps_and_exact_equality_are_bound():
    """R44 through R47, the four behaviors a mutation run found unbound."""
    M = _load()
    two = (33035.25, 33035.75)
    case, axes, moved, V = _mozambique_shaped()
    assert _member(M, {"a.json": case}, V)["problems"] == []

    # R44: the port output, declared as the reference. Its kind is what refuses it, since
    # a port output with enough tracks would otherwise be read as version 1's.
    swapped = json.loads(json.dumps(case))
    swapped["input_sha256"]["tracker_octave_instrumented.mat"] = PORT_DIGEST
    big_port = outputs(two)
    big_port[PORT_DIGEST]["tracks"] = [port_counterpart(two) for _ in range(120)]
    out = M.membership(residuals(two), {"a.json": swapped}, logs(V), (), big_port)
    assert any("declares reference output pppppppppppp, which was not supplied" in w
               for w in out["problems"])
    assert out["explained_v1_extra_pairs"] == []

    # R45: the residual record AGREES with the case, and the retained outputs do not. The
    # residual check is silent, and the output-derived steps must refuse on their own.
    Vsub = {"33035.2500": V["33035.2500"]}
    one = bound({"unmatched_v1_tracks": [], "v1_extra_pairs": [74]},
                parameters={"divergence_steps": [33035.25], "control_steps": [33035.5],
                            "window": [33035.0, 33035.5]}, vertices=Vsub)
    for label in ("intervention", "shape_control"):
        one["intervention"][label]["axes_by_requested_step"] = {
            "33035.2500": (moved if label == "shape_control" else axes)["33035.2500"]}
    one["intervention"]["control"]["axes_by_requested_step"] = {"33035.5000": axes["33035.2500"]}
    assert _member(M, {"a.json": one}, Vsub, steps=(33035.25,))["problems"] == []
    out2 = M.membership(residuals((33035.25,)), {"a.json": one}, logs(Vsub), (),
                        outputs(two))
    assert any("declares divergence steps ['33035.2500'], while the retained outputs put "
               "version 1's extra observations at ['33035.2500', '33035.7500']" in w
               for w in out2["problems"])
    # the residual record disagrees with the outputs here as well, and that is reported
    # too; what this binds is that the case's DECLARED steps are checked against the
    # outputs on their own, in a message of their own
    assert out2["explained_v1_extra_pairs"] == []

    # R46 and R47: exact equality means every coordinate and the whole length. A track
    # that matches in time and latitude and not longitude, and one that is a prefix of the
    # reference, are each put in the intervention with the flag left True, so the correct
    # recomputation (False) disagrees with the flag and refuses.
    ref = reference_track(two)
    east = {"time": list(ref["time"]), "lat": list(ref["lat"]),
            "lon": [x + 0.5 for x in ref["lon"]]}
    prefix = {k: ref[k][:-1] for k in ("time", "lat", "lon")}
    for name, near in (("east", east), ("prefix", prefix)):
        almost = json.loads(json.dumps(case))
        almost["intervention"]["intervention"]["finished_tracks"] = [near]
        out3 = _member(M, {"a.json": almost}, V)
        assert any("intervention replay records reproduced_exactly=True for 74 and its own "
                   "finished tracks give False" in w for w in out3["problems"]), name
        assert out3["explained_v1_extra_pairs"] == []


def test_a_recorded_track_is_validated_before_it_is_counted_or_compared(tmp_path):
    """R48 through R50. The review's reproductions: delete the final latitude of the
    intervention's reproducing track, or append one, and the earlier comparison still
    called it an exact reproduction. Every recorded track's three arrays must now be the
    same nonzero length and every coordinate finite, or the replay is refused before
    anything is counted, and the pinned reference is held to the same shape."""
    M = _load()
    two = (33035.25, 33035.75)
    case, axes, moved, V = _mozambique_shaped()
    assert _member(M, {"a.json": case}, V)["problems"] == []
    ref = reference_track(two)
    shapes = {
        "final latitude deleted": {"time": list(ref["time"]), "lat": ref["lat"][:-1],
                                   "lon": list(ref["lon"])},
        "latitude appended": {"time": list(ref["time"]), "lat": ref["lat"] + [-20.0],
                              "lon": list(ref["lon"])},
        "longitude deleted": {"time": list(ref["time"]), "lat": list(ref["lat"]),
                              "lon": ref["lon"][:-1]},
        "time appended": {"time": ref["time"] + [33036.25], "lat": list(ref["lat"]),
                          "lon": list(ref["lon"])},
        "empty": {"time": [], "lat": [], "lon": []},
        "not finite": {"time": list(ref["time"]), "lat": ["NaN"] + ref["lat"][1:],
                       "lon": list(ref["lon"])},
        "not a number": {"time": list(ref["time"]), "lat": [None] + ref["lat"][1:],
                         "lon": list(ref["lon"])},
    }
    for name, near in shapes.items():
        almost = json.loads(json.dumps(case))
        almost["intervention"]["intervention"]["finished_tracks"] = [near]
        out = _member(M, {"a.json": almost}, V)
        assert any("intervention replay records a finished track that cannot be compared"
                   in w for w in out["problems"]), name
        assert out["explained_v1_extra_pairs"] == [], name
        # and it is refused even when the replay's flag is lowered to match, since the
        # track is refused before any comparison is made of it
        lowered = json.loads(json.dumps(almost))
        lowered["intervention"]["intervention"]["reference_tracks"][0]["reproduced_exactly"] = False
        assert any("cannot be compared" in w
                   for w in _member(M, {"a.json": lowered}, V)["problems"]), name
    # the two reproductions through the command, with nothing written
    for name in ("final latitude deleted", "latitude appended"):
        almost = json.loads(json.dumps(case))
        almost["intervention"]["intervention"]["finished_tracks"] = [shapes[name]]
        assert _refuses(M, tmp_path, almost, name.replace(" ", "-"), V, two) == (2, False)
    # a ragged track anywhere in a replay refuses that replay, not only the reproducing one
    elsewhere = json.loads(json.dumps(case))
    elsewhere["intervention"]["control"]["finished_tracks"].append(shapes["longitude deleted"])
    out2 = _member(M, {"a.json": elsewhere}, V)
    assert any("control replay records a finished track that cannot be compared" in w
               for w in out2["problems"])
    # R50: the pinned side is held to the same shape
    ragged = tmp_path / "ragged.mat"
    tracks = outputs(two)[REF_DIGEST]["tracks"]
    tracks[74] = shapes["final latitude deleted"]
    digest = write_output(ragged, tracks)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"logs": [], "outputs": [
        {"sha256": digest, "file": str(ragged), "kind": "reference"}]}))
    got, why = M.verified_outputs([str(ragged)], str(manifest))
    assert got == {} and any("holds a track that cannot be compared" in w for w in why)
    # and _same_track itself never compares a prefix
    assert not M._same_track(shapes["final latitude deleted"], ref)
    assert not M._same_track(ref, shapes["final latitude deleted"])
    assert not M._same_track(shapes["latitude appended"], ref)
    assert M._same_track(ref, json.loads(json.dumps(ref)))


def test_a_performed_experiment_is_validated_whether_or_not_it_claims_anything(tmp_path):
    """R51 and R52. Every check used to run inside `if claimed`, so a null-result case was
    never validated at all: two of them carried time controls at steps their pairs do not
    both hold, and the command still succeeded. A NEGATIVE RESULT RESTS ON THE SAME
    CONTROLS A POSITIVE ONE DOES. A case that declares no experiment stays exempt, since
    there is nothing to validate."""
    M = _load()
    two = (33035.25, 33035.75)
    case, axes, moved, V = _mozambique_shaped()
    silent = json.loads(json.dumps(case))
    silent["explains"] = {"unmatched_v1_tracks": [], "v1_extra_pairs": []}
    silent["parameters"]["reference_tracks"] = [74]
    assert _member(M, {"a.json": silent}, V)["problems"] == []
    assert _member(M, {"a.json": silent}, V)["explained_v1_extra_pairs"] == []
    # R51: an invalid control step is now reported, though nothing is claimed
    bad = json.loads(json.dumps(silent))
    bad["parameters"]["control_steps"] = [33035.5, 33037.0]        # the port holds neither
    bad["intervention"]["control"].update(
        {"requested_times": [33035.5, 33037.0], "applied_at": [33035.5, 33037.0],
         "axes_by_requested_step": {"33035.5000": axes["33035.2500"],
                                    "33037.0000": axes["33035.7500"]}})
    out = _member(M, {"a.json": bad}, V)
    assert any("declares control step 33037.0000, at which the two sides do not both "
               "hold an observation" in w for w in out["problems"])
    assert _refuses(M, tmp_path, bad, "silent-bad", V, two) == (2, False)
    # and the experiment contract too, with nothing claimed
    hollow = json.loads(json.dumps(silent))
    hollow["intervention"].pop("shape_control")
    assert any("records no shape_control replay" in w
               for w in _member(M, {"a.json": hollow}, V)["problems"])
    # R52: a case that declares NO experiment is exempt, not refused
    none = json.loads(json.dumps(silent))
    none["intervention"] = {}
    assert _member(M, {"a.json": none}, V)["problems"] == []


def test_an_incomplete_or_undeclared_experiment_cannot_escape_validation(tmp_path):
    """R53 through R57. The repair that validated unclaimed experiments keyed "performed"
    on the baseline alone and silently filtered examined indices it could not classify, so
    a review walked round it three ways on the same invalid null case: by deleting the
    examined list, by setting it to an unknown index, and by deleting only the baseline
    while leaving every other replay in place. An experiment that records ANY replay is
    validated as a performed or incomplete one."""
    M = _load()
    two = (33035.25, 33035.75)
    case, axes, moved, V = _mozambique_shaped()
    silent = json.loads(json.dumps(case))                  # examines 74, claims nothing
    silent["explains"] = {"unmatched_v1_tracks": [], "v1_extra_pairs": []}
    silent["parameters"]["reference_tracks"] = [74]
    assert _member(M, {"a.json": silent}, V)["problems"] == []
    # the invalid control this is all built on
    invalid = json.loads(json.dumps(silent))
    invalid["parameters"]["control_steps"] = [33035.5, 33037.0]
    invalid["intervention"]["control"].update(
        {"requested_times": [33035.5, 33037.0], "applied_at": [33035.5, 33037.0],
         "axes_by_requested_step": {"33035.5000": axes["33035.2500"],
                                    "33037.0000": axes["33035.7500"]}})
    assert any("do not both hold an observation" in w
               for w in _member(M, {"a.json": invalid}, V)["problems"])

    # R53 and the empty list: no declaration at all
    for spoil in (lambda c: c["parameters"].pop("reference_tracks"),
                  lambda c: c["parameters"].__setitem__("reference_tracks", [])):
        blind = json.loads(json.dumps(invalid)); spoil(blind)
        out = _member(M, {"a.json": blind}, V)
        assert any("records replays and declares no reference tracks" in w
                   for w in out["problems"])
        assert out["explained_v1_extra_pairs"] == []
        assert _refuses(M, tmp_path, blind, "blind-examined", V, two) == (2, False)

    # R54: an index the residual artifact does not classify is REFUSED, not filtered
    unknown = json.loads(json.dumps(invalid))
    unknown["parameters"]["reference_tracks"] = [9999]
    out2 = _member(M, {"a.json": unknown}, V)
    assert any("declares reference track 9999, which the residual artifact classifies as "
               "neither a version-1-extra pair nor an unmatched track" in w
               for w in out2["problems"])
    assert _refuses(M, tmp_path, unknown, "unknown-examined", V, two) == (2, False)

    # R55: every replay is required, the baseline included
    for label in ("baseline", "intervention", "control", "shape_control"):
        partial = json.loads(json.dumps(silent)); partial["intervention"].pop(label)
        out3 = _member(M, {"a.json": partial}, V)
        assert any(f"records no {label} replay" in w for w in out3["problems"]), label
        assert sum(1 for w in out3["problems"] if f"records no {label} replay" in w) == 1
        assert _refuses(M, tmp_path, partial, f"no-{label}", V, two) == (2, False)

    # R56: the declaration must agree with what the replays recorded outcomes for
    disagree = json.loads(json.dumps(silent))
    disagree["parameters"]["reference_tracks"] = [67]          # the replays record 74
    out4 = _member(M, {"a.json": disagree}, V)
    assert any("declares reference tracks [67] and its replays record outcomes for [74]"
               in w for w in out4["problems"])
    assert _refuses(M, tmp_path, disagree, "disagreeing", V, two) == (2, False)

    # R57: a claim outside the examined set
    overclaim = json.loads(json.dumps(case))                   # claims 74
    overclaim["parameters"]["reference_tracks"] = []
    out5 = _member(M, {"a.json": overclaim}, V)
    assert any("claims [74], which its experiment does not declare among the reference "
               "tracks it examined" in w for w in out5["problems"])
    assert out5["explained_v1_extra_pairs"] == []

    # and a genuinely unperformed case, which records no replay at all, stays exempt
    unperformed = json.loads(json.dumps(silent))
    unperformed["intervention"] = {"attempted": "an injection at the divergence step",
                                   "reason": "version 1 dumps no axis there"}
    unperformed["parameters"]["reference_tracks"] = []
    assert _member(M, {"a.json": unperformed}, V)["problems"] == []
