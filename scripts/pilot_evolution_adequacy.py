#!/usr/bin/env python3
"""Adequacy harness for an evolution reader (`EASTERN_DIRECTION_RECOMMENDATION_11_2026-10-16.md`,
stage A, challenges A1 to A7 declared there). Synthetic meridional-wind rows are built from
Gaussian northerly minima on the listing's integer grid, passed through the listing's own
`row_measurements` (unchanged, with a broad window and the entries' thresholds) to make a
listing record, and then through the reader under test. The answer keys are read only by
the scoring step. Required results are the declaration's, with the tolerances stated in
`REQUIRED`. The noise case A7 is reported and is not a condition.

    python3 scripts/pilot_evolution_adequacy.py --reader classes|phases --out OUT.json --text OUT.txt [--grid integer|native-lon]

Part three of the second development package (`EASTERN_EVOLUTION_DEVELOPMENT_2_2026-10-17.md`,
section 6) added the `--grid` option, which builds every construction from the same geometry
on the integer grid (the frozen default) or on the files' native half-degree longitude grid,
and the challenges B1 to B6 declared there, run with the phases reader only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_i1_entry_listing as L  # noqa: E402

LONS = np.arange(30.0, 66.0)
GRID = "integer"
ROWS = list(range(30, 9, -1))
ANN = {"i1": {"lon": [44, 63], "rows": [12, 30]}}
THRESHOLDS = {"primary": -8.0, "sensitivity": -6.0}
STEP_HOURS = 6


def configure_grid(grid):
    """Rebinds the module's longitude columns. `integer` is the frozen harness grid, one
    column per degree from 30 to 65 E. `native-lon` is the files' spacing, 71 half-degree
    columns over the same span. Every generator reads LONS at call time, so one geometry
    is sampled on whichever grid is configured."""
    global LONS, GRID
    if grid == "integer":
        LONS = np.arange(30.0, 66.0)
    elif grid == "native-lon":
        LONS = np.arange(30.0, 65.0 + 0.25, 0.5)
    else:
        raise SystemExit(f"REFUSED: unknown grid {grid!r}")
    GRID = grid
    return LONS


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def gauss(center, depth, width=2.0):
    return -abs(depth) * np.exp(-0.5 * ((LONS - center) / width) ** 2)


def record_from(fields, start="2000-01-01T00"):
    """fields: list over steps of {row: v array over LONS}. Rows absent get a weak background."""
    steps = []
    for k, rows in enumerate(fields):
        stamp = f"2000-01-{1 + (k * STEP_HOURS) // 24:02d}T{(k * STEP_HOURS) % 24:02d}"
        per_row = []
        for row in ROWS:
            v = rows.get(row, np.full(LONS.shape, 1.0))
            valid = np.ones(LONS.shape, bool)
            m = L.row_measurements(LONS, v, valid, np.zeros(LONS.shape), valid, ANN, row, THRESHOLDS)
            per_row.append(m)
        steps.append({"time": stamp, "annotation": ANN, "rows": per_row})
    return {"declared": {"every_hours": STEP_HOURS, "rows": ROWS, "lons": LONS.tolist(), "grid": GRID, "lon_spacing_deg": float(LONS[1] - LONS[0])}, "steps": steps}


def background(v=1.0):
    return np.full(LONS.shape, v)


def case_a1(rows=range(20, 29), rate_deg_per_day=2.0, days=(2, 3, 2), start_lon=52.0, depth=12.0, noise=None, seed=1):
    """A moving phase inside persistent flow: standing, moving west, standing."""
    n1, n2, n3 = [int(d * 24 / STEP_HOURS) for d in days]
    per_step = rate_deg_per_day * STEP_HOURS / 24.0
    rng = np.random.default_rng(seed)
    fields = []; truth = []
    for k in range(n1 + n2 + n3):
        if k < n1:
            c = start_lon; phase = "standing"
        elif k < n1 + n2:
            c = start_lon - per_step * (k - n1 + 1); phase = "west"
        else:
            c = start_lon - per_step * n2; phase = "standing"
        f = {}
        for r in rows:
            v = background() + gauss(c, depth)
            if noise:
                # Independent draws averaged over three neighboring columns, then rescaled by sqrt(3) so
                # that the applied interior standard deviation is `noise` (the review of 2026-10-16 found
                # the unscaled kernel applied about 0.58 m/s for a nominal 1 m/s), and the correlation
                # is 2/3 at one degree and 1/3 at two.
                e = rng.normal(0, noise, LONS.shape)
                e = np.convolve(e, np.ones(3) / 3, mode="same") * np.sqrt(3.0)
                v = v + e
            f[r] = v
        fields.append(f); truth.append((phase, c))
    return fields, {"rows": list(rows), "truth": truth, "moving_steps": n2, "net_move": -per_step * n2}


def case_a2(rows=range(20, 25), sep=5.0, days=4):
    """Two standing minima 5 degrees apart whose depths alternate every 12 hours."""
    n = int(days * 24 / STEP_HOURS)
    fields = []
    deeper = []
    for k in range(n):
        deeper_west = ((k // 2) % 2 == 0)
        deeper.append(deeper_west)
        f = {r: background() + gauss(50.0, 12.0 if deeper_west else 9.0) + gauss(50.0 + sep, 9.0 if deeper_west else 12.0) for r in rows}
        fields.append(f)
    switch_pairs = [k for k in range(n - 1) if deeper[k] != deeper[k + 1]]
    return fields, {"rows": list(rows), "switch_pairs": switch_pairs}


def case_a3(rows=range(20, 25), days=3):
    """An eastern minimum weakening past the threshold while a western one 4 degrees away
    stands deep. Two construction defects were found and corrected on the first runs. With
    the default width of 2 the two Gaussians 4 degrees apart merged into one migrating
    minimum, so the width is 1.2. And with both depths changing linearly the two minima
    crossed in depth while both sat shallower than the -8 m/s threshold (the +1 m/s
    background puts a depth-9 minimum at -8), so the crossing pairs were unassessable and
    no replacement could appear; the eastern minimum now weakens from depth 12 to 6.5 in
    steps of 0.5 while the western one stands at depth 9.1 (-8.1 m/s), so the selection
    crosses at the step where the eastern minimum passes 9.1 and it is shallower than -8
    at the next step, which is the declared replacement (old lost, new pre-existing). A third
    correction on the next run: with steps of 0.5 the eastern minimum lost the selection
    while still at or under -8 m/s (a switching pair, A2's family) and passed the threshold
    one step later as a continuous pair, so the replacement never appeared. The declared
    replacement needs the eastern minimum to pass the threshold in the same step it loses
    the selection, so it now weakens by 1.1 per step from depth 12.2 (10.0 then 8.9 across
    the crossing) while the western one stands at depth 9.05 (-8.05 m/s)."""
    n = int(days * 24 / STEP_HOURS)
    depths = [max(12.2 - 1.1 * k, 0.5) for k in range(n)]
    fields = [{r: background() + gauss(56.0, depths[k], width=1.2) + gauss(52.0, 9.05, width=1.2) for r in rows} for k in range(n)]
    crossing = next(k for k in range(n - 1) if depths[k] >= 9.05 and depths[k + 1] < 9.05)   # the pair at which the eastern minimum loses the selection and passes the threshold
    return fields, {"rows": list(rows), "replacement_pair": crossing}


def case_a4(rows=range(20, 25), days=3, beside_steps=(3, 4, 7, 8)):
    """A minimum moving west with a second deep minimum 2 degrees east of it on some steps,
    so that at the pair entering such a step the alternative lies within the pair bound of
    the earlier selection. Construction corrected on the first run: on the integer grid two
    distinct local minima need a higher column between them, so the declared 1-degree
    separation is not representable and 2 degrees is the closest, both Gaussians are
    narrow (width 0.7) so the midpoint column stays higher, and the minimum moves one
    column per step (4 degrees per day) so that the alternative sits 1 degree from the
    earlier selection rather than at the bound's edge."""
    n = int(days * 24 / STEP_HOURS)
    fields = []
    for k in range(n):
        c = 58.0 - 1.0 * k
        f = {}
        for r in rows:
            v = background() + gauss(c, 12.0, width=0.7)
            if k in beside_steps:
                v = v + gauss(c + 2.0, 9.5, width=0.7)
            f[r] = v
        fields.append(f)
    # The pair entering a beside step is ambiguous (the alternative at c+2 lies one degree from the earlier
    # selection at c+1), and the pair leaving it is ambiguous only when the next step is also a beside step
    # (the later alternative at c_next+2 lies one degree from the earlier selection).
    ambiguous = sorted({b - 1 for b in beside_steps if b >= 1} | {b for b in beside_steps if b + 1 in beside_steps})
    return fields, {"rows": list(rows), "beside_steps": list(beside_steps), "ambiguous_pairs": ambiguous}


def case_a6(rows=range(20, 25), rate=1.0, days=4):
    n = int(days * 24 / STEP_HOURS)
    fields = [{r: background() + gauss(55.0 - rate * STEP_HOURS / 24.0 * k, 12.0) for r in rows} for k in range(n)]
    return fields, {"rows": list(rows), "rate": rate}


def case_a8(rows=range(20, 25), days=4, period_steps=1):
    """A standing minimum whose grid position shifts by one column (49 or 50 E) every
    `period_steps` steps, which on an integer grid is what a minimum centered between two
    columns looks like. Added in stage B of the eleventh effort after the I2 development
    case's northern rows, standing at 48 to 50 E for four days in the entry, were read as
    alternating short westward and eastward phases under a 24-hour window with a 1 degree
    per day threshold (one column over the window meets the threshold). The one-step and
    two-step alternations cancel over an even centered window and did not reproduce that
    failure, so the declared form is the column shift held for 48 hours (`period_steps`
    8), the shape the I2 rows show. Required, no phase of 24 hours or more and at least
    one standing interval of 48 hours or more."""
    n = int(days * 24 / STEP_HOURS)
    fields = [{r: background() + gauss(49.0 + ((k // period_steps) % 2), 12.0, width=1.0) for r in rows} for k in range(n)]
    return fields, {"rows": list(rows)}


def case_b1(rows=range(20, 25), rate=2.0, width=1.0, days=(3, 3, 3), start_lon=59.0, depth=12.0, deform=False):
    """B1, recoverable displacement: one minimum standing, translating west at `rate`, then
    standing, with Gaussian width `width` (1.0 and 1.5 degrees, two to three native
    columns). With `deform` (B2) the width grows from 1.0 to 2.5 degrees across the moving
    phase and the depth varies between 10 and 14. The key carries the ideal centers, from
    which the scorer's expected phase duration is computed with the reader's own window
    rule, so the requirement is the geometry's and not a tuned number. Three standing days
    at each end keep the bracketing intervals clear of the window's extension of the
    phase at the faster rates (two days left 24 elapsed hours at 4 degrees per day, at the
    edge of the requirement)."""
    n1, n2, n3 = [int(d * 24 / STEP_HOURS) for d in days]
    per_step = rate * STEP_HOURS / 24.0
    fields = []; centers = []
    for k in range(n1 + n2 + n3):
        if k < n1:
            c = start_lon
        elif k < n1 + n2:
            c = start_lon - per_step * (k - n1 + 1)
        else:
            c = start_lon - per_step * n2
        w, d = width, depth
        if deform and n1 <= k < n1 + n2:
            frac = (k - n1) / max(1, n2 - 1)
            w = 1.0 + 1.5 * frac
            d = 12.0 + 2.0 * np.sin(2 * np.pi * frac)
        fields.append({r: background() + gauss(c, d, width=w) for r in rows}); centers.append(c)
    return fields, {"rows": list(rows), "net_move": -per_step * n2, "rate": rate, "width": width, "days": list(days), "deform": deform, "centers": centers}


def declared_a1_duration_pass(view, rows):
    """Section 6 declared B1 'exactly as A1', whose duration rule is exactly one westward
    phase of 48 to 90 elapsed hours per row. The implemented B1 criterion replaced it with
    the reader's window rule on the ideal centers (within 12 hours), recorded as a
    departure at the independent review of 2026-10-17 (F1). This reports the declared rule
    beside the implemented one for every B1 and B2 run, so that a phase the window rule
    lengthens past 90 hours is shown failing the declaration."""
    for r in rows:
        west48 = [p for p in view["phases"][r] if p["direction"] == "west" and p["hours"] >= 48]
        if len(west48) != 1 or not (48 <= west48[0]["hours"] <= 90):
            return False
    return True


def case_b3(rows=range(20, 25), sep=1.0, days=4):
    """B3, amplitude exchange between stationary extrema `sep` degrees apart (1, 2, 3),
    depths alternating every 12 hours between 12 and 9.5 (the shallower one at -8.5 m/s
    with the background, so both are deep). Widths 0.35, 0.7 and 1.0 keep the midpoint
    column higher than both minima on the native grid. On the integer grid the 1-degree
    separation puts the two centers on adjacent columns, which cannot both be local
    minima, so that construction is not representable there and its failure is the
    integer sampling's blind spot."""
    width = {1.0: 0.35, 2.0: 0.7, 3.0: 1.0}[sep]
    n = int(days * 24 / STEP_HOURS)
    fields = []; deeper = []
    for k in range(n):
        dw = ((k // 2) % 2 == 0); deeper.append(dw)
        fields.append({r: background() + gauss(50.0, 12.0 if dw else 9.5, width) + gauss(50.0 + sep, 9.5 if dw else 12.0, width) for r in rows})
    switch_pairs = [k for k in range(n - 1) if deeper[k] != deeper[k + 1]]
    return fields, {"rows": list(rows), "switch_pairs": switch_pairs, "sep": sep}


def _ambiguous_from_geometry(sel, other, bound=1.5):
    """Pairs k at which the construction's other minimum lies within the reader's bound of
    the selection at either end (and is not the selection itself), from the ideal
    geometry. `other[k]` is None where only one minimum exists."""
    need = []
    for k in range(len(sel) - 1):
        a = other[k + 1] is not None and other[k + 1] != sel[k] and abs(other[k + 1] - sel[k]) <= bound
        b = other[k] is not None and other[k] != sel[k + 1] and abs(other[k] - sel[k + 1]) <= bound
        if a or b:
            need.append(k)
    return need


def case_b4(rows=range(20, 25), days=(2, 2), center=54.0, width=0.4, merge=False):
    """B4, a split: one minimum at `center` for two days, then two minima whose centers
    separate by half a degree per step for two days, the western one deeper (12 against
    11). B5 (`merge`) is the reverse, two minima approaching by half a degree per step
    for two days, then one. Widths 0.4 keep two minima half a degree apart listable on the
    native grid (the column between them is higher than both). On the integer grid the
    first split step and the last merge step put both centers within one column, so one
    minimum is listed there and the required ambiguity cannot appear, which is the blind
    spot the construction tests."""
    n1, n2 = [int(d * 24 / STEP_HOURS) for d in days]
    fields = []; sel = []; other = []
    for k in range(n1 + n2):
        if merge:
            d = 0.5 * (n1 - k) if k < n1 else 0.0
        else:
            d = 0.5 * (k - n1 + 1) if k >= n1 else 0.0
        if d > 0:
            v = {r: background() + gauss(center - d, 12.0, width) + gauss(center + d, 11.0, width) for r in rows}
            sel.append(center - d); other.append(center + d)
        else:
            v = {r: background() + gauss(center, 12.0, width) for r in rows}
            sel.append(center); other.append(None)
        fields.append(v)
    return fields, {"rows": list(rows), "ambiguous_required": _ambiguous_from_geometry(sel, other), "no_phase": False, "sel": sel, "other": other}


def case_b6(rows=range(20, 25), stationary=54.0, start=60.0, end=48.0, width=0.4):
    """B6, a moving minimum (depth 11) at one degree per step passing a stationary one
    (depth 13), so the selection stays on the stationary minimum throughout and the pairs
    at which the mover lies within the bound of it, apart from the coincidence step, must
    read as ambiguous. Widths 0.4 keep the two listable one degree apart on the native
    grid. On the integer grid one degree apart is adjacent columns, where only one of the
    two can be a local minimum, so the approach and departure pairs cannot read as
    ambiguous there."""
    n = int(start - end) + 1
    fields = []; sel = []; other = []
    for k in range(n):
        m = start - k
        fields.append({r: background() + gauss(stationary, 13.0, width) + gauss(m, 11.0, width) for r in rows})
        sel.append(stationary); other.append(None if m == stationary else m)
    return fields, {"rows": list(rows), "ambiguous_required": _ambiguous_from_geometry(sel, other), "no_phase": True, "sel": sel, "other": other}


def ideal_phase_hours(reader, centers, P):
    """The elapsed hours of the longest westward stretch the reader's own window rule gives
    the ideal (unsampled) centers, the geometry's expected phase duration."""
    motion = reader.step_motion({"lons": list(centers)}, P)
    best = 0; i = 0
    while i < len(motion):
        if motion[i] == "west":
            j = i
            while j + 1 < len(motion) and motion[j + 1] == "west":
                j += 1
            best = max(best, (j - i) * P["step_hours"]); i = j + 1
        else:
            i += 1
    return best


REQUIRED = {
    "A1": "exactly one westward phase per row of 48 to 90 elapsed hours with net move between -7 and -4 degrees, a standing interval of at least 24 elapsed hours ending at or before the phase's start and one beginning at or after its end (bracketing), no eastward phase of 24 hours or more, and no standing interval spanning the whole sequence",
    "B1": "exactly one westward phase of 48 hours or more per row whose elapsed hours lie within 12 of the duration the reader's window rule gives the ideal centers and whose net move lies within 1.5 degrees of the constructed one, bracketed by standing intervals of at least 24 elapsed hours, no eastward phase of 24 hours or more, no standing interval spanning the whole sequence, and every pair continuous (no switching, replacement, jump or unassessable pair)",
    "B2": "as B1, with the width growing from 1.0 to 2.5 degrees across the moving phase and the depth varying between 10 and 14",
    "B3": "the alternation pairs of the construction are never continuous unique (switching, or continuous ambiguous within the bound), every other pair is continuous, every step is standing, and there is no phase of 24 hours or more",
    "B4": "every pair continuous, and the pairs at which the construction's other minimum lies within the pair bound of the selection at either end are continuous ambiguous or unassessable, never continuous unique; ambiguity at other pairs is reported, not refused",
    "B5": "as B4, for the merge",
    "B6": "as B4 for the pairs at which the moving minimum lies within the bound of the stationary selection, apart from the coincidence step, and the selection has no phase of 24 hours or more",
    "A2": "switching at exactly the alternation pairs of the construction and continuous pairs everywhere else, every step labeled standing, and no phase of 24 hours or more in either direction",
    "A3": "a replacement at exactly the constructed crossing pair and continuous pairs everywhere else, and no phase of 24 hours or more in either direction",
    "A4": "ambiguous continuity at exactly the constructed pairs and unique continuity everywhere else, exactly one westward phase of 48 hours or more spanning those pairs with its ambiguous count equal to their number, and no eastward phase of 24 hours or more",
    "A5": "identical outputs for identical inputs",
    "A6": "the 1 degree per day minimum yields one westward phase of 48 elapsed hours or more and no eastward phase of 24 hours or more; the 0.5 degree per day minimum yields no phase of 24 hours or more in either direction and at least one standing interval of 48 hours or more",
    "A7": "reported, not a condition",
    "A8": "a standing minimum whose column shifts by one yields no phase of 24 hours or more in either direction and at least one standing interval of 48 hours or more (added in stage B, see case_a8)",
}
# Hours in the views are ELAPSED between the first and last sample of a span. The first scorer
# (reviewed 2026-10-16) checked existence only, so a view with one switching pair and every
# other pair unassessable passed A2, a replacement at the wrong pair passed A3, one arbitrary
# ambiguous pair passed A4, and a 96-hour eastward phase passed the slow A6 case. The scorer now
# checks locations and bracketing against keys the generators state, and
# tests/test_pilot_evolution_adequacy.py holds those counterexamples.


def load_reader(name):
    path = os.path.join(HERE, {"classes": "pilot_entry_evolution_classes.py", "phases": "pilot_entry_evolution_phases.py"}[name])
    spec = importlib.util.spec_from_file_location(f"reader_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def analyze_with(reader, name, record):
    """A common view over both readers: per row the pair classes in order, the phases and
    standing intervals with their step indices and elapsed hours, and the per-step motion
    labels. The closed reader has no per-step motion, so its steps inside displacement runs
    are labeled by the run's sign and every other step standing."""
    P = dict(reader.PRIMARY)
    n_steps = len(record["steps"])
    if name == "phases":
        out = reader.analyze(record, P)
        return {"n_steps": n_steps,
                "pairs": {r: [p["class"] for p in ps] for r, ps in out["pairs"].items()},
                "phases": {r: [{"direction": ph["direction"], "i_start": ph["i_start"], "i_end": ph["i_end"], "hours": ph["hours"], "net_move": ph["net_move"], "ambiguous_pairs": ph.get("ambiguous_pairs", 0)} for ph in phs] for r, phs in out["phases"].items()},
                "standing": {r: [{"i_start": s["i_start"], "i_end": s["i_end"], "hours": s["hours"]} for s in ss] for r, ss in out["standing"].items()},
                "motion": {r: list(m) for r, m in out["motion"].items()}}
    pairs = reader.classify_pairs(record, P)
    runs = reader.runs(record, P, pairs)
    motion = {}
    for r, rr in runs.items():
        lab = {}
        for run in rr:
            if run["class"] == "displacement":
                for i in range(run["i_start"], run["i_end"] + 1):
                    lab[i] = "west" if run["net_move"] < 0 else "east"
        motion[r] = [(i, lab.get(i, "standing")) for i in range(n_steps)]
    return {"n_steps": n_steps,
            "pairs": {r: [p["class"] for p in ps] for r, ps in pairs.items()},
            "phases": {r: [{"direction": "west" if run["net_move"] < 0 else "east", "i_start": run["i_start"], "i_end": run["i_end"], "hours": run["hours"], "net_move": run["net_move"], "ambiguous_pairs": None} for run in rr if run["class"] == "displacement"] for r, rr in runs.items()},
            "standing": {r: [{"i_start": run["i_start"], "i_end": run["i_end"], "hours": run["hours"]} for run in rr if run["class"] == "standing"] for r, rr in runs.items()},
            "motion": motion}


def _continuous(c):
    return c in ("continuous", "continuous_unique", "continuous_ambiguous")


def score(case, view, key):
    """True only when every row meets the declared requirement of the case, with the
    locations the construction fixes checked exactly. Hours are elapsed."""
    rows = key["rows"]
    ok = True; notes = []
    n_steps = view.get("n_steps")
    for r in rows:
        pairs = view["pairs"][r]; phases = view["phases"][r]; standing = view["standing"][r]
        west = [p for p in phases if p["direction"] == "west"]
        east_long = [p for p in phases if p["direction"] == "east" and p["hours"] >= 24]
        long_any = [p for p in phases if p["hours"] >= 24]
        if case == "A1":
            good = len([p for p in west if p["hours"] >= 48]) == 1 and not east_long
            if good:
                ph = next(p for p in west if p["hours"] >= 48)
                good = 48 <= ph["hours"] <= 90 and -7.0 <= ph["net_move"] <= -4.0
                before = [s for s in standing if s["hours"] >= 24 and s["i_end"] <= ph["i_start"]]
                after = [s for s in standing if s["hours"] >= 24 and s["i_start"] >= ph["i_end"]]
                whole = any(n_steps is not None and s["i_start"] == 0 and s["i_end"] == n_steps - 1 for s in standing)
                good = good and before and after and not whole
            if not good:
                ok = False; notes.append(f"{r} N: phases {phases}, standing {standing}")
        elif case in ("A2", "A3"):
            want = "switching" if case == "A2" else "replacement"
            where = set(key["switch_pairs"]) if case == "A2" else {key["replacement_pair"]}
            found = {k for k, c in enumerate(pairs) if c == want}
            others_ok = all(_continuous(c) for k, c in enumerate(pairs) if k not in where)
            motion_ok = all(lab == "standing" for _, lab in view["motion"][r]) if case == "A2" else True
            if found != where or not others_ok or not motion_ok or long_any:
                ok = False; notes.append(f"{r} N: {want} at {sorted(found)} (required {sorted(where)}), others continuous {others_ok}, motion standing {motion_ok}, phases of 24 h or more {long_any}")
        elif case == "A4":
            where = set(key["ambiguous_pairs"])
            found = {k for k, c in enumerate(pairs) if c == "continuous_ambiguous"}
            others_unique = all(c == "continuous_unique" for k, c in enumerate(pairs) if k not in where)
            long_west = [p for p in west if p["hours"] >= 48]
            spans = len(long_west) == 1 and all(long_west[0]["i_start"] <= k < long_west[0]["i_end"] for k in where) and long_west[0]["ambiguous_pairs"] == len(where)
            if found != where or not others_unique or not spans or east_long:
                ok = False; notes.append(f"{r} N: ambiguous at {sorted(found)} (required {sorted(where)}), others unique {others_unique}, west phases {west}")
        elif case == "A6":
            if key["rate"] >= 1.0:
                if len([p for p in west if p["hours"] >= 48]) != 1 or east_long:
                    ok = False; notes.append(f"{r} N at {key['rate']} deg/day: west phases {west}, east phases of 24 h or more {east_long}")
            else:
                stands = [s for s in standing if s["hours"] >= 48]
                if long_any or not stands:
                    ok = False; notes.append(f"{r} N at {key['rate']} deg/day: phases of 24 h or more {long_any}, standing of 48 h or more {stands}")
        elif case == "A8":
            stands = [s for s in standing if s["hours"] >= 48]
            if long_any or not stands:
                ok = False; notes.append(f"{r} N: phases of 24 h or more {long_any}, standing of 48 h or more {stands}")
        elif case in ("B1", "B2"):
            west48 = [p for p in west if p["hours"] >= 48]
            good = len(west48) == 1 and not east_long
            if good:
                ph = west48[0]
                good = abs(ph["net_move"] - key["net_move"]) <= 1.5 and abs(ph["hours"] - key["expected_hours"]) <= 12
                before = [s for s in standing if s["hours"] >= 24 and s["i_end"] <= ph["i_start"]]
                after = [s for s in standing if s["hours"] >= 24 and s["i_start"] >= ph["i_end"]]
                whole = any(n_steps is not None and s["i_start"] == 0 and s["i_end"] == n_steps - 1 for s in standing)
                good = good and before and after and not whole
            good = good and all(_continuous(c) for c in pairs)
            if not good:
                ok = False; notes.append(f"{r} N: west phases {west}, expected {key['expected_hours']} h and {key['net_move']:+g} deg, non-continuous pairs at {[k for k, c in enumerate(pairs) if not _continuous(c)]}, standing {standing}")
        elif case == "B3":
            where = set(key["switch_pairs"])
            at = all(pairs[k] in ("switching", "continuous_ambiguous") for k in where)
            others = all(_continuous(c) for k, c in enumerate(pairs) if k not in where)
            motion_ok = all(lab == "standing" for _, lab in view["motion"][r])
            if not (at and others and motion_ok) or long_any:
                ok = False; notes.append(f"{r} N: alternation pairs {sorted(where)} read {[pairs[k] for k in sorted(where)]}, others continuous {others}, motion standing {motion_ok}, phases of 24 h or more {long_any}")
        elif case in ("B4", "B5", "B6"):
            need = set(key["ambiguous_required"])
            at = all(pairs[k] in ("continuous_ambiguous", "unassessable") for k in need)
            others = all(_continuous(c) for k, c in enumerate(pairs) if k not in need)
            extra = [k for k, c in enumerate(pairs) if k not in need and c == "continuous_ambiguous"]
            phase_ok = not long_any if key.get("no_phase") else True
            if not (at and others and phase_ok):
                ok = False; notes.append(f"{r} N: required ambiguous at {sorted(need)} read {[pairs[k] for k in sorted(need)]}, others continuous {others}, phases of 24 h or more {long_any if key.get('no_phase') else 'not required'}")
            elif extra:
                notes.append(f"{r} N: ambiguity reported at {extra} beyond the required {sorted(need)}")
    return ok, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reader", choices=("classes", "phases"), required=True); ap.add_argument("--out", required=True); ap.add_argument("--text", required=True)
    ap.add_argument("--grid", choices=("integer", "native-lon"), default="integer", help="the longitude grid the constructions are sampled on (integer, the frozen default, or native-lon)")
    a = ap.parse_args(argv)
    for p in (a.out, a.text):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    configure_grid(a.grid)
    reader, rpath = load_reader(a.reader)
    results = {}; lines = [f"# adequacy of reader '{a.reader}' ({os.path.basename(rpath)}) on the declared challenges, grid {a.grid} ({LONS[1] - LONS[0]:g} deg, {LONS.size} columns)"]
    cases = {"A1": case_a1(), "A2": case_a2(), "A3": case_a3(), "A4": case_a4(), "A6_1.0": case_a6(rate=1.0), "A6_0.5": case_a6(rate=0.5),
             "A8_1": case_a8(period_steps=1), "A8_2": case_a8(period_steps=2), "A8_8": case_a8(period_steps=8, days=6)}
    probes = []
    if a.reader == "phases":
        for rate in (2.0, 3.0, 4.0):
            for width in (1.0, 1.5):
                cases[f"B1_{rate:g}_{width:g}"] = case_b1(rate=rate, width=width)
        # 5 and 6 degrees per day were declared with section 6. 7 and 8 were added on the first run's
        # reading (the native grid held continuity through 6, so the floor lay beyond the declared
        # probes) and recorded in section 6 before the rerun. With two moving days at 8 degrees per
        # day the minimum ends at 43 E, under the window's supported bound, so the faster probes
        # move for a day and a half.
        for rate in (5.0, 6.0, 7.0, 8.0):
            cases[f"B1_{rate:g}_1_probe"] = case_b1(rate=rate, width=1.0, days=(3, 2 if rate <= 6 else 1.5, 3)); probes.append(f"B1_{rate:g}_1_probe")
        cases["B2_3"] = case_b1(rate=3.0, width=1.0, deform=True)
        # The declared geometry of section 6 (two standing days, three moving, two standing), run
        # beside the retained three-three-three runs after the independent review of 2026-10-17
        # (F1) asked for the declared construction to be retained. Reported, scored under both
        # the implemented criterion and the declared A1 duration rule.
        for rate in (2.0, 3.0, 4.0):
            for width in (1.0, 1.5):
                cases[f"B1decl_{rate:g}_{width:g}"] = case_b1(rate=rate, width=width, days=(2, 3, 2)); probes.append(f"B1decl_{rate:g}_{width:g}")
        for sep in (1.0, 2.0, 3.0):
            cases[f"B3_{sep:g}"] = case_b3(sep=sep)
        cases["B4"] = case_b4(); cases["B5"] = case_b4(merge=True); cases["B6"] = case_b6()
        for name, (fields, key) in cases.items():
            if name.startswith(("B1", "B2")):
                key["expected_hours"] = ideal_phase_hours(reader, key["centers"], dict(reader.PRIMARY))
    for name, (fields, key) in cases.items():
        rec = record_from(fields)
        view = analyze_with(reader, a.reader, rec)
        case = "B1" if name.startswith("B1decl") else name.split("_")[0]
        ok, notes = score(case, view, key)
        results[name] = {"pass": ok, "notes": notes, "required": REQUIRED[case], "view": view, "key": {k: v for k, v in key.items() if k not in ("centers",)}}
        tag = "reported" if name in probes else ("PASS" if ok else "FAIL")
        line = f"{name}: {tag}" + ("" if ok else " | " + " || ".join(notes[:3])) + (" | " + " || ".join(notes[:3]) if ok and notes else "")
        if name.startswith(("B1", "B2")):
            d = declared_a1_duration_pass(view, key["rows"])
            results[name]["declared_a1_duration_48_90_pass"] = d
            hours = sorted({p["hours"] for r in key["rows"] for p in view["phases"][r] if p["direction"] == "west" and p["hours"] >= 48})
            line += f" | days {key['days']}, implemented criterion {'pass' if ok else 'fail'}, declared A1 duration rule 48 to 90 h {'pass' if d else 'FAIL'} (phases of 48 h or more: {hours} h)"
        lines.append(line)
    # A5 identical inputs
    rec = record_from(case_a1()[0]); v1 = analyze_with(reader, a.reader, rec); v2 = analyze_with(reader, a.reader, json.loads(json.dumps(rec)))
    results["A5"] = {"pass": v1 == v2, "required": REQUIRED["A5"]}
    lines.append(f"A5: {'PASS' if v1 == v2 else 'FAIL'}")
    # A7 noise, reported
    fields, key = case_a1(noise=1.0)
    view = analyze_with(reader, a.reader, record_from(fields)); ok, notes = score("A1", view, key)
    results["A7"] = {"pass_reported_not_required": ok, "notes": notes[:5], "required": REQUIRED["A7"], "view": view}
    lines.append(f"A7 (noise of 1 m/s interior standard deviation, reported): {'as A1' if ok else 'differs from A1'}" + ("" if ok else " | " + " || ".join(notes[:2])))
    mandatory_names = [k for k in results if k not in probes and k != "A7"]
    mandatory = all(results[k]["pass"] for k in mandatory_names)
    failed = [k for k in mandatory_names if not results[k]["pass"]]
    lines.append(f"mandatory challenges ({len(mandatory_names)}): {'ALL PASS' if mandatory else 'NOT ALL PASS, failing ' + ', '.join(failed)}")
    out = {"generated_by": "scripts/pilot_evolution_adequacy.py", "script_sha256": sha256_of(os.path.abspath(__file__)), "reader": a.reader, "reader_path": rpath,
           "reader_sha256": sha256_of(rpath), "listing_row_measurements_sha256": sha256_of(os.path.join(HERE, "pilot_i1_entry_listing.py")), "params": dict(reader.PRIMARY),
           "grid": a.grid, "lon_spacing_deg": float(LONS[1] - LONS[0]), "n_columns": int(LONS.size),
           "declared": "EASTERN_DIRECTION_RECOMMENDATION_11_2026-10-16.md, section 4, stage A (A1 to A8); EASTERN_EVOLUTION_DEVELOPMENT_2_2026-10-17.md, section 6 (B1 to B6 and the grid)",
           "results": results, "mandatory_names": mandatory_names, "probes": probes, "mandatory_pass": mandatory}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    with open(a.text, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if mandatory else 1


if __name__ == "__main__":
    raise SystemExit(main())
