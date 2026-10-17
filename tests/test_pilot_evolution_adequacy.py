"""The adequacy harness's scorer must refuse views that violate the declared challenges, not
only accept the reader's true outputs. Each fabricated view below was accepted by the first
scorer (the independent review of 2026-10-16 passed them to it), so the tests are written
against those counterexamples: one switching pair somewhere is not switching at every
alternation, a replacement at the wrong pair is not the constructed crossing, one ambiguous
pair anywhere is not the four constructed ones, an eastward phase is not standing, and two
standing intervals on one side of a phase do not bracket it."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("adequacy_under_test", os.path.join(ROOT, "scripts", "pilot_evolution_adequacy.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _view(rows, pairs, phases=(), standing=(), motion=None):
    """A view in the harness's common shape: per row the pair classes by index, phases and
    standing intervals with step indices, and per-step motion labels."""
    n = len(pairs) + 1
    return {"pairs": {r: list(pairs) for r in rows},
            "phases": {r: [dict(p) for p in phases] for r in rows},
            "standing": {r: [dict(s) for s in standing] for r in rows},
            "motion": {r: list(motion) if motion is not None else [(i, "standing") for i in range(n)] for r in rows},
            "n_steps": n}


def test_a2_requires_switching_at_every_alternation_and_standing_between():
    H = _load()
    rows = [20]
    n = 16
    switch_at = [1, 3, 5, 7, 9, 11, 13]
    good = ["switching" if k in switch_at else "continuous_unique" for k in range(n - 1)]
    assert H.score("A2", _view(rows, good), {"rows": rows, "switch_pairs": switch_at})[0]
    one_only = ["switching" if k == 3 else "unassessable" for k in range(n - 1)]
    assert not H.score("A2", _view(rows, one_only), {"rows": rows, "switch_pairs": switch_at})[0], "one switching pair with everything else unassessable is not the challenge"
    moving = [(i, "west") for i in range(n)]
    assert not H.score("A2", _view(rows, good, motion=moving), {"rows": rows, "switch_pairs": switch_at})[0], "the steps between switches must be standing"


def test_a3_requires_the_replacement_at_the_constructed_crossing():
    H = _load()
    rows = [20]
    n = 12
    good = ["replacement" if k == 2 else "continuous_unique" for k in range(n - 1)]
    assert H.score("A3", _view(rows, good), {"rows": rows, "replacement_pair": 2})[0]
    wrong = ["replacement" if k == 0 else "unassessable" for k in range(n - 1)]
    assert not H.score("A3", _view(rows, wrong), {"rows": rows, "replacement_pair": 2})[0]


def test_a4_requires_exactly_the_constructed_ambiguous_pairs_inside_one_westward_phase():
    H = _load()
    rows = [20]
    n = 12
    amb = [2, 3, 6, 7]
    good = ["continuous_ambiguous" if k in amb else "continuous_unique" for k in range(n - 1)]
    phase = [{"direction": "west", "i_start": 0, "i_end": n - 1, "hours": (n - 1) * 6, "net_move": -11.0, "ambiguous_pairs": 4}]
    assert H.score("A4", _view(rows, good, phases=phase), {"rows": rows, "ambiguous_pairs": amb})[0]
    wrong = ["continuous_ambiguous" if k == 0 else "continuous_unique" for k in range(n - 1)]
    phase1 = [{"direction": "west", "i_start": 0, "i_end": n - 1, "hours": (n - 1) * 6, "net_move": -11.0, "ambiguous_pairs": 1}]
    assert not H.score("A4", _view(rows, wrong, phases=phase1), {"rows": rows, "ambiguous_pairs": amb})[0]


def test_a6_slow_case_forbids_phases_in_either_direction_and_a1_requires_bracketing():
    H = _load()
    rows = [20]
    n = 16
    cont = ["continuous_unique"] * (n - 1)
    east = [{"direction": "east", "i_start": 0, "i_end": n - 1, "hours": 90, "net_move": 2.0, "ambiguous_pairs": 0}]
    assert not H.score("A6", _view(rows, cont, phases=east), {"rows": rows, "rate": 0.5})[0], "an eastward phase is not standing"
    stand = [{"i_start": 0, "i_end": n - 1, "hours": 90}]
    assert H.score("A6", _view(rows, cont, standing=stand), {"rows": rows, "rate": 0.5})[0]
    # A1: a westward phase with two standing intervals both BEFORE it is not bracketed
    n = 28
    cont = ["continuous_unique"] * (n - 1)
    phase = [{"direction": "west", "i_start": 7, "i_end": 20, "hours": 78, "net_move": -6.0, "ambiguous_pairs": 0}]
    both_before = [{"i_start": 0, "i_end": 3, "hours": 18}, {"i_start": 4, "i_end": 7, "hours": 18}]
    assert not H.score("A1", _view(rows, cont, phases=phase, standing=both_before), {"rows": rows})[0]
    bracket = [{"i_start": 0, "i_end": 7, "hours": 42}, {"i_start": 20, "i_end": 27, "hours": 42}]
    assert H.score("A1", _view(rows, cont, phases=phase, standing=bracket), {"rows": rows})[0]


# ---------------------------------------------------------------- part three of the second package (2026-10-17)

def test_b1_requires_every_pair_continuous_and_the_constructed_net_move():
    """B1 is A1 with two additions, the net move must sit within 1.5 degrees of the
    constructed one and no assessable pair may be a jump, a replacement or a switch."""
    H = _load()
    rows = [20]
    n = 28
    cont = ["continuous_unique"] * (n - 1)
    phase = [{"direction": "west", "i_start": 7, "i_end": 20, "hours": 78, "net_move": -9.0, "ambiguous_pairs": 0}]
    bracket = [{"i_start": 0, "i_end": 7, "hours": 42}, {"i_start": 20, "i_end": 27, "hours": 42}]
    key = {"rows": rows, "net_move": -9.0, "expected_hours": 78}
    assert H.score("B1", _view(rows, cont, phases=phase, standing=bracket), key)[0]
    one_jump = list(cont); one_jump[12] = "jump"
    assert not H.score("B1", _view(rows, one_jump, phases=phase, standing=bracket), key)[0], "a jump inside the moving phase is not recoverable displacement"
    off = [{**phase[0], "net_move": -5.0}]
    assert not H.score("B1", _view(rows, cont, phases=off, standing=bracket), key)[0], "a net move four degrees short of the construction is not the constructed displacement"
    short = [{**phase[0], "i_end": 14, "hours": 42}]
    assert not H.score("B1", _view(rows, cont, phases=short, standing=bracket), key)[0], "a phase 36 hours shorter than the geometry's is not the constructed displacement"


def test_b3_forbids_unique_continuity_at_an_alternation_and_requires_standing():
    H = _load()
    rows = [20]
    n = 16
    alt = [1, 3, 5, 7, 9, 11, 13]
    key = {"rows": rows, "switch_pairs": alt}
    switching = ["switching" if k in alt else "continuous_unique" for k in range(n - 1)]
    assert H.score("B3", _view(rows, switching), key)[0]
    ambiguous = ["continuous_ambiguous" if k in alt else "continuous_ambiguous" for k in range(n - 1)]
    assert H.score("B3", _view(rows, ambiguous), key)[0], "within the bound the exchange reads as ambiguous continuity, which retains the ambiguity"
    unique_once = list(switching); unique_once[5] = "continuous_unique"
    assert not H.score("B3", _view(rows, unique_once), key)[0], "an exchange read as unique continuity is false continuity"
    moving = [(i, "west") for i in range(n)]
    assert not H.score("B3", _view(rows, switching, motion=moving), key)[0]


def test_b4_to_b6_require_ambiguity_where_the_geometry_puts_a_competitor_within_the_bound():
    H = _load()
    rows = [20]
    n = 16
    need = [7, 8, 9]
    key = {"rows": rows, "ambiguous_required": need, "no_phase": False}
    good = ["continuous_ambiguous" if k in need else "continuous_unique" for k in range(n - 1)]
    assert H.score("B4", _view(rows, good), key)[0]
    abstain = list(good); abstain[8] = "unassessable"
    assert H.score("B4", _view(rows, abstain), key)[0], "abstaining at a required pair is allowed"
    unique_there = list(good); unique_there[8] = "continuous_unique"
    assert not H.score("B4", _view(rows, unique_there), key)[0], "unique continuity where a competitor sits within the bound is false continuity"
    jump_elsewhere = list(good); jump_elsewhere[2] = "jump"
    assert not H.score("B4", _view(rows, jump_elsewhere), key)[0]
    extra = list(good); extra[3] = "continuous_ambiguous"
    ok, notes = H.score("B4", _view(rows, extra), key)
    assert ok, "ambiguity at a pair the geometry does not require is reported, not refused"
    key6 = {"rows": rows, "ambiguous_required": need, "no_phase": True}
    west = [{"direction": "west", "i_start": 4, "i_end": 10, "hours": 36, "net_move": -3.0, "ambiguous_pairs": 0}]
    assert not H.score("B6", _view(rows, good, phases=west), key6)[0], "the stationary selection must not read as a phase"


def test_the_declared_a1_duration_rule_is_reported_beside_the_implemented_b1_criterion():
    """From the independent review of 2026-10-17 (F1): section 6 declared B1 'exactly as
    A1', whose duration rule is 48 to 90 elapsed hours, while the implemented scorer uses
    the reader's window rule on the ideal centers within 12 hours. Both are reported for
    every B1 and B2 run, so a 96-hour phase at 4 degrees per day passes the implemented
    criterion and is shown failing the declared one."""
    H = _load()
    rows = [20]
    n = 36
    cont = ["continuous_unique"] * (n - 1)
    long_phase = [{"direction": "west", "i_start": 9, "i_end": 25, "hours": 96, "net_move": -12.0, "ambiguous_pairs": 0}]
    short_phase = [{"direction": "west", "i_start": 11, "i_end": 24, "hours": 78, "net_move": -6.0, "ambiguous_pairs": 0}]
    assert H.declared_a1_duration_pass(_view(rows, cont, phases=long_phase), rows) is False
    assert H.declared_a1_duration_pass(_view(rows, cont, phases=short_phase), rows) is True
    two = [dict(long_phase[0]), {"direction": "west", "i_start": 27, "i_end": 35, "hours": 48, "net_move": -3.0, "ambiguous_pairs": 0}]
    assert H.declared_a1_duration_pass(_view(rows, cont, phases=two), rows) is False, "A1 requires exactly one westward phase of 48 hours or more"
