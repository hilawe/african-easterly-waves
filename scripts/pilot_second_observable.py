#!/usr/bin/env python3
"""The second-observable development package (`EASTERN_SECOND_OBSERVABLE_DEVELOPMENT_2026-10-17.md`,
sections 2 to 4): the four cheap candidate measurements of pattern translation around a
selected northerly minimum (the round's two designs as written, A0 and C0, and their declared
repairs, A1 and C1), the constructions with known answers, the scoring against the declared
adequacy rule, the diagnoses on the four development intervals from their saved listings and
the retained fields, and the runtime measurement. Every result is a table with denominators.
A supported step is graded as consistent with westward translation of the surrounding
pattern and no more.

    python3 scripts/pilot_second_observable.py constructions --out OUT.json --table OUT.md
    python3 scripts/pilot_second_observable.py diagnose --interval 1987_september --out OUT.json --table OUT.md
    python3 scripts/pilot_second_observable.py runtime --interval 1987_september --out OUT.md
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_i1_entry_listing as L  # noqa: E402

STEP_HOURS = 6
DX = 0.5
LON = np.arange(30.0, 65.0 + 1e-9, DX)
LAT = np.arange(10.0, 30.0 + 1e-9, DX)
ROWS = [22, 21, 20, 19, 18]
ANN = {"i1": {"lon": [32, 63], "rows": [10, 30]}}
THRESHOLDS = {"primary": -8.0, "sensitivity": -6.0}
DEPTH_MS = -8.0
PAIR_BOUND = 1.5
# The declared parameters of section 2.
A_HALF = (8, 6)            # cells, 4 degrees by 3
A_MIN_CELLS = 15
A_COND = 1000.0
A_GRAD_FLOOR = 0.5         # m/s per degree
A_R2_MIN, A_S_MIN = 0.4, 1.0
C_HALF = (6, 4)            # cells, 3 degrees by 2
C_SEARCH = (4, 2)          # cells, 2 degrees by 1
C_MASK_MAX = 0.2
C_COMMON_MIN = 0.8
C_STD_MIN = 1.5
C_SIGMA = 1.0
C_S_MIN = 0.5
C_WEST_MIN = -0.25         # degrees per step, 1 degree per day
EXT = "docs/aewc_v2/evidence/validation/eastern_pilot_2026-10-05/reports/extent"
INTERVALS = {
    "1983_june": {"listing": f"{EXT}/eval60_native_entry_listing_1983.json", "phases": f"{EXT}/eval60_native_phases_1983.json"},
    "1987_september": {"listing": f"{EXT}/eval60_stageE_entry_listing_1987.json", "phases": f"{EXT}/eval60_stageE_phases_1987.json"},
    "1990_august": {"listing": f"{EXT}/eval60_native_entry_listing_i1wide_1990.json", "phases": f"{EXT}/eval60_native_phases_i1wide_1990.json"},
    "2006_june": {"listing": f"{EXT}/eval60_native_entry_listing_i2_2006.json", "phases": f"{EXT}/eval60_native_phases_i2_2006.json"},
}
DATA_DIR = "data/era5/region6h"
FIELD_LAT = np.arange(5.0, 35.0 + 1e-9, DX)
FIELD_LON = np.arange(20.0, 75.0 + 1e-9, DX)


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------------------------ fields and blocks

def _index_of(axis, value, what):
    j = int(round((value - axis[0]) / DX))
    if j < 0 or j >= len(axis) or abs(axis[j] - value) > 1e-6:
        raise SystemExit(f"REFUSED: {what} {value} is not a column of the half-degree grid")
    return j


def _block(arr, iy, ix, fill):
    """arr at rows iy and columns ix, with out-of-range indices filled."""
    iy = np.asarray(iy); ix = np.asarray(ix)
    out = np.full((len(iy), len(ix)), fill, dtype=arr.dtype)
    oky = (iy >= 0) & (iy < arr.shape[0]); okx = (ix >= 0) & (ix < arr.shape[1])
    out[np.ix_(oky, okx)] = arr[np.ix_(iy[oky], ix[okx])]
    return out


def _result(outcome, reason="", **kw):
    d = {"outcome": outcome, "reason": reason}
    d.update(kw)
    return d


# ------------------------------------------------------------------ candidate A, the advection fit

def advection_fit(field, k, x0, y0, growth=False, both_flanks=False, half=A_HALF):
    """A0 (as designed) with growth=False and both_flanks=False, A1 (repaired) with both True.
    Speeds in degrees per day, tendency over the centered 12 hours, gradients over 1 degree.
    `half` is the neighborhood's half-width in cells (zonal, meridional), the declared
    value by default and the scale sensitivity of section 5 otherwise."""
    v, valid = field["v"], field["valid"]
    nt = v.shape[0]
    if k - 1 < 0 or k + 1 >= nt:
        return _result("abstained", "window end")
    ix0 = _index_of(field["lon"], x0, "longitude"); iy0 = _index_of(field["lat"], y0, "latitude")
    vk = v[k]
    ok = valid[k - 1] & valid[k] & valid[k + 1]
    st = np.zeros_like(ok); st[1:-1, 1:-1] = valid[k][:-2, 1:-1] & valid[k][2:, 1:-1] & valid[k][1:-1, :-2] & valid[k][1:-1, 2:]
    ok = ok & st
    vx = np.full(vk.shape, np.nan); vy = np.full(vk.shape, np.nan); vt = np.full(vk.shape, np.nan)
    vx[:, 1:-1] = (vk[:, 2:] - vk[:, :-2]) / (2 * DX)
    vy[1:-1, :] = (vk[2:, :] - vk[:-2, :]) / (2 * DX)
    vt[:] = (v[k + 1] - v[k - 1]) / (2 * STEP_HOURS / 24.0)
    iy = np.arange(iy0 - half[1], iy0 + half[1] + 1); ix = np.arange(ix0 - half[0], ix0 + half[0] + 1)
    use = _block(ok, iy, ix, False)
    n_used = int(use.sum())
    if n_used < A_MIN_CELLS:
        return _result("abstained", "few cells", n_used=n_used)
    if both_flanks:
        row = use[half[1], :]
        west = np.flatnonzero(row[:half[0]]); east = np.flatnonzero(row[half[0] + 1:])
        if len(west) < 3 or len(east) < 3 or (half[0] - west.min()) < 3 or (east.max() + 1) < 3:
            return _result("abstained", "one flank", n_used=n_used)
    gx = _block(vx, iy, ix, np.nan)[use]; gy = _block(vy, iy, ix, np.nan)[use]; gt = _block(vt, iy, ix, np.nan)[use]; gv = _block(vk, iy, ix, np.nan)[use]
    grad_rms = float(np.sqrt(np.mean(gx ** 2 + gy ** 2)))
    if grad_rms < A_GRAD_FLOOR:
        return _result("abstained", "flat", n_used=n_used, grad_rms=grad_rms)
    cols = [-gx, -gy] + ([gv] if growth else [])
    X = np.column_stack(cols)
    norms = np.sqrt(np.mean(X ** 2, axis=0))
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        # A zero column (a field constant along one axis) is the extreme of ill conditioning;
        # found crashing the scaled decomposition at the review of 2026-10-18 (F1).
        return _result("abstained", "ill-conditioned", n_used=n_used, cond=float("inf"))
    Xc = X / norms if growth else X
    cond = float(np.linalg.cond(Xc))
    if not np.isfinite(cond) or cond > A_COND:
        return _result("abstained", "ill-conditioned", n_used=n_used, cond=cond)
    sst = float(np.sum(gt ** 2))
    if sst == 0.0:
        return _result("rejected", "no tendency", n_used=n_used, c_x=0.0, c_y=0.0, g=0.0, r2=0.0, S=0.0, cond=cond)
    beta, *_ = np.linalg.lstsq(X, gt, rcond=None)
    ssr = float(np.sum((gt - X @ beta) ** 2))
    r2 = 1.0 - ssr / sst
    c_x, c_y = float(beta[0]), float(beta[1])
    g = float(beta[2]) if growth else 0.0
    S = r2 * max(-c_x, 0.0)
    outcome = "supported" if (r2 >= A_R2_MIN and S >= A_S_MIN) else "rejected"
    return _result(outcome, "", n_used=n_used, c_x=c_x, c_y=c_y, g=g, r2=float(r2), S=float(S), cond=cond)


# ------------------------------------------------------------------ candidate C, the shifted correlation

def shift_correlation(field, k, x0, y0, x1, refine=False, westward=False, half=C_HALF):
    """C0 (as designed) with refine=False and westward=False, C1 (repaired) with both True.
    `half` is the patch's half-width in cells, the declared value by default."""
    v, valid = field["v"], field["valid"]
    if k + 1 >= v.shape[0]:
        return _result("abstained", "window end")
    ix0 = _index_of(field["lon"], x0, "longitude"); iy0 = _index_of(field["lat"], y0, "latitude")
    iy = np.arange(iy0 - half[1], iy0 + half[1] + 1); ix = np.arange(ix0 - half[0], ix0 + half[0] + 1)
    n_patch = len(iy) * len(ix)
    V0 = _block(v[k], iy, ix, np.nan); ok0 = _block(valid[k], iy, ix, False)
    if 1.0 - ok0.sum() / n_patch > C_MASK_MAX:
        return _result("abstained", "masked at t")
    ok1u = _block(valid[k + 1], iy, ix, False)
    if 1.0 - ok1u.sum() / n_patch > C_MASK_MAX:
        return _result("abstained", "masked at t+1")
    sd0 = float(np.std(V0[ok0]))
    if sd0 < C_STD_MIN:
        return _result("abstained", "flat", std=sd0)
    corr = {}
    for sy in range(-C_SEARCH[1], C_SEARCH[1] + 1):
        for sx in range(-C_SEARCH[0], C_SEARCH[0] + 1):
            V1 = _block(v[k + 1], iy + sy, ix + sx, np.nan); ok1 = _block(valid[k + 1], iy + sy, ix + sx, False)
            both = ok0 & ok1
            if both.sum() < C_COMMON_MIN * n_patch:
                continue
            a = V0[both]; b = V1[both]
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            corr[(sx, sy)] = float(np.corrcoef(a, b)[0, 1])
    if not corr:
        return _result("abstained", "no searchable shift")
    (sx, sy), rmax = sorted(corr.items(), key=lambda kv: (-kv[1], abs(kv[0][0]), abs(kv[0][1])))[0]
    dx = sx * DX; dy = sy * DX
    dx_used = dx
    if refine and (sx - 1, sy) in corr and (sx + 1, sy) in corr:
        ym, y0v, yp = corr[(sx - 1, sy)], rmax, corr[(sx + 1, sy)]
        den = ym - 2 * y0v + yp
        if den < 0:
            p = 0.5 * (ym - yp) / den
            dx_used = dx + float(np.clip(p, -0.5, 0.5)) * DX
    move = x1 - x0
    S = rmax * float(np.exp(-((move - dx_used) ** 2) / (2 * C_SIGMA ** 2)))
    out = dict(rmax=rmax, dx=dx, dy=dy, dx_refined=dx_used, move=move, std=sd0, n_patch=n_patch)
    if westward and dx_used > C_WEST_MIN:
        return _result("rejected", "pattern not westward", S=0.0, **out)
    return _result("supported" if S >= C_S_MIN else "rejected", "", S=float(S), **out)


def smoothed(field):
    """The field after a 3 by 3 mean at every cell whose nine cells are valid, invalid
    elsewhere, cached on the field. The anticipated repair A2 of section 5 (the advection
    fit's tendency and gradients are finite differences, which noise of 1 m/s swamps)."""
    if "v_smooth" not in field:
        v, valid = field["v"], field["valid"]
        s = np.zeros_like(v); ok = np.ones_like(valid)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s += np.nan_to_num(np.roll(np.roll(v, dy, axis=1), dx, axis=2), nan=0.0)
                ok &= np.roll(np.roll(valid, dy, axis=1), dx, axis=2)
        ok[:, [0, -1], :] = False; ok[:, :, [0, -1]] = False
        field["v_smooth"] = np.where(ok, s / 9.0, np.nan); field["valid_smooth"] = ok
    return {"lon": field["lon"], "lat": field["lat"], "v": field["v_smooth"], "valid": field["valid_smooth"], "step_hours": field["step_hours"]}


CANDIDATES = {
    "A0": ("step", lambda f, k, x0, y0, x1=None: advection_fit(f, k, x0, y0)),
    "A1": ("step", lambda f, k, x0, y0, x1=None: advection_fit(f, k, x0, y0, growth=True, both_flanks=True)),
    "A2": ("step", lambda f, k, x0, y0, x1=None: advection_fit(smoothed(f), k, x0, y0, growth=True, both_flanks=True)),
    "C0": ("pair", lambda f, k, x0, y0, x1: shift_correlation(f, k, x0, y0, x1)),
    "C1": ("pair", lambda f, k, x0, y0, x1: shift_correlation(f, k, x0, y0, x1, refine=True, westward=True)),
}


# ------------------------------------------------------------------ constructions

def gauss2(X, Y, xc, yc, d, sx, sy):
    return d * np.exp(-0.5 * ((X - xc) / sx) ** 2 - 0.5 * ((Y - yc) / sy) ** 2)


def sample(fn, nsteps=13, mask=None, noise=None, smooth=False, seed=0, lon=LON, lat=LAT):
    """A field dict sampled from fn(X, Y, t_days) on the grid at the cadence, with an
    optional validity mask(X, Y, t) and optional noise of the stated standard deviation,
    independent per cell and step, or smoothed by a 3 by 3 mean and rescaled to it."""
    X, Y = np.meshgrid(lon, lat)
    rng = np.random.default_rng(seed)
    v = np.empty((nsteps, len(lat), len(lon))); valid = np.ones(v.shape, bool)
    for k in range(nsteps):
        t = k * STEP_HOURS / 24.0
        f = np.asarray(fn(X, Y, t), float)
        if noise:
            e = rng.normal(0.0, noise, f.shape)
            if smooth:
                s = np.zeros_like(e)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        s += np.roll(np.roll(e, dy, axis=0), dx, axis=1)
                e = s / 9.0 * 3.0
            f = f + e
        if mask is not None:
            valid[k] = np.asarray(mask(X, Y, t), bool)
        v[k] = np.where(valid[k], f, np.nan)
    return {"lon": lon, "lat": lat, "v": v, "valid": valid, "step_hours": STEP_HOURS}


def _tilted(X, Y, t, a1, a2, along=3.5):
    u = np.array([-0.8, 1.0]); u = u / np.linalg.norm(u)
    s = (X - 54.0) * u[0] + (Y - 20.0) * u[1]
    n = -(X - 54.0) * u[1] + (Y - 20.0) * u[0]
    across = np.exp(-0.5 * (n / 1.5) ** 2)
    return across * (a1(t) * np.exp(-0.5 * ((s - 4.0) / along) ** 2) + a2(t) * np.exp(-0.5 * ((s + 4.0) / along) ** 2))


def _exchange(t, deep=-12.0, shallow=-4.0):
    f = min(t / 2.0, 1.0)
    return deep + (shallow - deep) * f, shallow + (deep - shallow) * f


def constructions():
    """The declared constructions K1 to K15 as (id, field, truth). truth: answer in
    {translation, none}, speed in degrees per day or None, kind in {positive, negative,
    limit, property}."""
    out = []
    def k1(rate):
        return lambda X, Y, t: gauss2(X, Y, 56.0 - rate * t, 20.0, -12.0, 3.0, 2.0)
    out.append(("K1", sample(k1(2.5)), {"answer": "translation", "speed": 2.5, "kind": "positive"}))
    out.append(("K2", sample(k1(0.0)), {"answer": "none", "speed": None, "kind": "negative"}))
    out.append(("K3", sample(lambda X, Y, t: gauss2(X, Y, 54.0, 20.0, -10.0, 6.0, 3.0) + gauss2(X, Y, 57.0 - 2.5 * t, 20.0, -3.0, 1.0, 1.0)),
                {"answer": "none", "speed": None, "kind": "negative"}))
    out.append(("K4", sample(lambda X, Y, t: gauss2(X, Y, 56.0 - 2.5 * t, 20.0, -12.0, 3.0, 2.0) + gauss2(X, Y, 50.0, 20.0, -8.0, 8.0, 4.0)),
                {"answer": "translation", "speed": 2.5, "kind": "positive"}))
    # K5 as declared merged the two troughs (scales 2.5 at 5 degrees apart), so the deepest
    # point slid east instead of switching; it is kept as K5m and K5 uses scales of 1.5 at 6
    # degrees apart so that the selection switches (section 5 of the record).
    out.append(("K5", sample(lambda X, Y, t: gauss2(X, Y, 52.0, 20.0, -10.0, 1.5, 2.0) + gauss2(X, Y, 58.0, 20.0, -2.0 - 12.0 * t / 3.0, 1.5, 2.0)),
                {"answer": "none", "speed": None, "kind": "negative"}))
    out.append(("K5m", sample(lambda X, Y, t: gauss2(X, Y, 52.0, 20.0, -10.0, 2.5, 2.0) + gauss2(X, Y, 57.0, 20.0, -2.0 - 12.0 * t / 3.0, 2.5, 2.0)),
                {"answer": "none", "speed": None, "kind": "negative"}))
    out.append(("K6", sample(lambda X, Y, t: gauss2(X, Y, 56.0, 20.0, -6.0 - 8.0 * t / 3.0, 3.0, 2.0)),
                {"answer": "none", "speed": None, "kind": "negative"}))
    out.append(("K7", sample(lambda X, Y, t: gauss2(X, Y, 56.0 - 2.5 * t, 20.0, -8.0 - 8.0 * t / 3.0, 3.0, 2.0)),
                {"answer": "translation", "speed": 2.5, "kind": "positive"}))
    out.append(("K8", sample(lambda X, Y, t: gauss2(X, Y, 51.0, 20.0, _exchange(t)[0], 2.0, 2.0) + gauss2(X, Y, 57.0, 20.0, _exchange(t)[1], 2.0, 2.0)),
                {"answer": "none", "speed": None, "kind": "negative"}))
    # K9 as declared (along-band scale 2, lobes of -12 and -4) left supported minima on 10
    # row-steps only (3 on row 22, 7 on row 18, none on rows 19 to 21); the lobes are -14 and
    # -6 with an along-band scale of 3.5, 59 supported row-steps (section 5, corrected at the
    # review of 2026-10-18).
    out.append(("K9", sample(lambda X, Y, t: _tilted(X, Y, t, lambda s: _exchange(s, -14.0, -6.0)[0], lambda s: _exchange(s, -14.0, -6.0)[1])),
                {"answer": "none", "speed": None, "kind": "negative"}))
    for tag, rate, smooth in (("K10_K1_noise", 2.5, False), ("K10_K2_noise", 0.0, False), ("K10_K1_smooth", 2.5, True), ("K10_K2_smooth", 0.0, True)):
        out.append((tag, sample(k1(rate), noise=1.0, smooth=smooth, seed=7), {"answer": "translation" if rate else "none", "speed": rate or None, "kind": "limit"}))
    out.append(("K11a", sample(k1(2.5), mask=lambda X, Y, t: X <= 56.0 - 2.5 * t + 1.0), {"answer": "translation", "speed": 2.5, "kind": "limit"}))
    # K11b as declared (cells east of 55 E invalid) left the trough's center masked and no
    # interior minimum to select; the mask is east of 57 E, one degree past the center
    # (section 5).
    out.append(("K11b", sample(lambda X, Y, t: gauss2(X, Y, 56.0, 20.0, -6.0 - 8.0 * t / 3.0, 3.0, 2.0), mask=lambda X, Y, t: X <= 57.0),
                {"answer": "none", "speed": None, "kind": "negative"}))
    f1 = sample(k1(2.5)); f2 = sample(k1(0.0))
    v = f2["v"].copy(); v[4:] = f1["v"][4:]
    out.append(("K12", {**f2, "v": v, "valid": f2["valid"].copy()}, {"answer": "property", "speed": None, "kind": "property"}))
    out.append(("K13", sample(lambda X, Y, t: -10.0 * np.cos(2 * np.pi * (X - 56.0 + 2.5 * t) / 20.0) * np.exp(-((Y - 20.0) ** 2) / 32.0)),
                {"answer": "translation", "speed": 2.5, "kind": "positive"}))
    out.append(("K14", sample(k1(6.0)), {"answer": "translation", "speed": 6.0, "kind": "limit"}))
    out.append(("K15_1.5", sample(k1(1.5)), {"answer": "translation", "speed": 1.5, "kind": "limit"}))
    out.append(("K15_1.0", sample(k1(1.0)), {"answer": "translation", "speed": 1.0, "kind": "limit"}))
    return out


def selections(field, rows=ROWS):
    """{row: [(step, lon, v) or None]} by the listing's own row measurement, supported
    (assigned, at or below the depth) or None."""
    out = {}
    lon = field["lon"]
    for row in rows:
        iy = _index_of(field["lat"], float(row), "latitude")
        sel = []
        for k in range(field["v"].shape[0]):
            vr = np.where(field["valid"][k][iy], field["v"][k][iy], np.nan)
            m = L.row_measurements(lon, np.nan_to_num(vr, nan=0.0), field["valid"][k][iy], np.zeros(lon.shape), np.zeros(lon.shape, bool), ANN, row, THRESHOLDS)
            if m["status"] == "assigned" and m["i1"]["v"] <= DEPTH_MS:
                sel.append((k, float(m["i1"]["lon"]), float(m["i1"]["v"])))
            else:
                sel.append(None)
        out[row] = sel
    return out


def continuous_pairs(sel):
    """[(k, x0, x1)] for consecutive supported selections within the pair bound."""
    out = []
    for k in range(len(sel) - 1):
        a, b = sel[k], sel[k + 1]
        if a is not None and b is not None and abs(b[1] - a[1]) <= PAIR_BOUND:
            out.append((k, a[1], b[1]))
    return out


def run_candidates(field, sel, rows=ROWS):
    """Every candidate asked at every supported row-step (A) or continuous pair (C)."""
    res = {name: [] for name in CANDIDATES}
    for row in rows:
        s = sel[row]
        for name, (unit, fn) in CANDIDATES.items():
            if unit == "step":
                for item in s:
                    if item is None:
                        continue
                    k, x0, _ = item
                    r = fn(field, k, x0, float(row)); r.update(row=row, k=k, x0=x0)
                    res[name].append(r)
            else:
                for k, x0, x1 in continuous_pairs(s):
                    r = fn(field, k, x0, float(row), x1); r.update(row=row, k=k, x0=x0, x1=x1)
                    res[name].append(r)
    return res


def summarize(results, truth):
    """Counts with denominators, medians over assessed results, and the declared adequacy
    verdict for pass-or-fail constructions."""
    asked = len(results)
    sup = [r for r in results if r["outcome"] == "supported"]; rej = [r for r in results if r["outcome"] == "rejected"]; abst = [r for r in results if r["outcome"] == "abstained"]
    assessed = sup + rej
    reasons = {}
    for r in abst + rej:
        if r["reason"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    med = lambda key: (statistics.median([r[key] for r in assessed if key in r]) if any(key in r for r in assessed) else None)
    out = {"asked": asked, "supported": len(sup), "rejected": len(rej), "abstained": len(abst), "assessed": len(assessed), "reasons": reasons,
           "support_fraction_of_assessed": (len(sup) / len(assessed)) if assessed else None,
           "abstention_fraction_of_asked": (len(abst) / asked) if asked else None,
           "median_S": med("S"), "p90_S": (float(np.percentile([r["S"] for r in assessed], 90)) if assessed else None),
           "median_c_x": med("c_x"), "median_r2": med("r2"), "median_dx": med("dx_refined"), "median_rmax": med("rmax")}
    if truth["speed"] is not None and any("c_x" in r for r in assessed):
        errs = [abs(r["c_x"] + truth["speed"]) for r in assessed if "c_x" in r]
        out["median_abs_speed_error"] = statistics.median(errs) if errs else None
    out["verdict"] = adequacy(out, truth)
    return out


def adequacy(s, truth):
    """The declared rule of section 1. Returns adequate, not adequate, not assessed, or
    reported (operating limits and the property)."""
    if truth["kind"] not in ("positive", "negative"):
        return "reported"
    if s["assessed"] == 0:
        return "not assessed"
    if truth["kind"] == "negative":
        return "adequate" if s["support_fraction_of_assessed"] <= 0.05 else "not adequate"
    ok = s["support_fraction_of_assessed"] >= 0.80 and s["abstention_fraction_of_asked"] <= 0.50
    if s.get("median_abs_speed_error") is not None:
        ok = ok and s["median_abs_speed_error"] <= 0.5
    return "adequate" if ok else "not adequate"


def run_constructions():
    out = {"constructions": []}
    k1_results = None
    for cid, field, truth in constructions():
        sel = selections(field)
        res = run_candidates(field, sel)
        entry = {"id": cid, "truth": truth, "selections": {str(r): [None if s is None else [s[0], s[1], round(s[2], 3)] for s in sel[r]] for r in ROWS},
                 "candidates": {name: {"summary": summarize(res[name], truth), "results": res[name]} for name in CANDIDATES}}
        if cid == "K1":
            k1_results = res
        if cid == "K12":
            # The sampled fields equal K1's from step 4. A step candidate's centered stencil lies
            # inside the replaced fields from step 5, a pair candidate's from the pair at step 4.
            # Complete keyed results compared for exact equality (review of 2026-10-18, F4).
            same = {}
            for name in CANDIDATES:
                start = K12_EQUAL_FROM["step" if CANDIDATES[name][0] == "step" else "pair"]
                later = [r for r in res[name] if r["k"] >= start]
                ref = {(r["row"], r["k"]): r for r in k1_results[name]}
                same[name] = len(later) > 0 and all((r["row"], r["k"]) in ref and _same(r, ref[(r["row"], r["k"])]) for r in later)
            entry["property_outputs_equal_k1_over_identical_stencils"] = {"A_from_step": K12_EQUAL_FROM["step"], "C_from_pair": K12_EQUAL_FROM["pair"], "equal": same}
        out["constructions"].append(entry)
    return out


K12_EQUAL_FROM = {"step": 5, "pair": 4}


def _same(a, b):
    """Complete keyed results, exactly equal (the same keys and the same values)."""
    return set(a) == set(b) and all(a[key] == b[key] for key in a)


def overall_verdicts(out):
    """Per candidate, the verdict counts over the pass-or-fail constructions and the overall
    verdict of section 1 (adequate overall only when adequate on every one). Derived from the
    retained summaries, so the account cannot contradict the table (review of 2026-10-18)."""
    res = {}
    for name in CANDIDATES:
        vs = [e["candidates"][name]["summary"]["verdict"] for e in out["constructions"] if e["truth"]["kind"] in ("positive", "negative")]
        counts = {v: vs.count(v) for v in ("adequate", "not adequate", "not assessed")}
        res[name] = {"constructions": len(vs), **counts, "overall": "adequate overall" if counts["adequate"] == len(vs) else "not adequate overall"}
    return res


def scale_pass_sets(rows, ids=("K3", "K4")):
    """Per candidate, the scales (half-widths in degrees) at which every one of the named
    constructions is adequate."""
    out = {}
    for cand in SCALES:
        for r in rows:
            if r["candidate"] != cand:
                continue
            key = tuple(r["half_degrees"])
            out.setdefault(cand, {}).setdefault(key, [])
            if r["id"] in ids:
                out[cand][key].append(r["summary"]["verdict"] == "adequate")
    return {cand: [list(k) for k, v in d.items() if v and all(v)] for cand, d in out.items()}


def construction_table(out):
    L_ = ["# the candidates on the declared constructions (section 3), counts with denominators",
          "asked is every supported row-step on rows 18 to 22 N (A0, A1) or every continuous pair there (C0, C1); assessed is supported plus rejected; the verdict is the declared adequacy rule of section 1 on pass-or-fail constructions and reported otherwise", ""]
    L_.append("| construction | known answer | candidate | asked | supported | rejected | abstained | support of assessed | abstention of asked | median S | S at the 90th percentile | median c_x or dx | speed error | verdict | reasons |")
    L_.append("| --- |" + " --- |" * 14)
    f = lambda x, n=2: "" if x is None else f"{x:.{n}f}"
    for e in out["constructions"]:
        for name in CANDIDATES:
            s = e["candidates"][name]["summary"]
            mc = s["median_c_x"] if name.startswith("A") else s["median_dx"]
            L_.append(f"| {e['id']} | {e['truth']['answer']}{'' if e['truth']['speed'] is None else ' at ' + format(e['truth']['speed'], 'g')} | {name} | {s['asked']} | {s['supported']} | {s['rejected']} | {s['abstained']} | {f(s['support_fraction_of_assessed'])} | {f(s['abstention_fraction_of_asked'])} | {f(s['median_S'])} | {f(s['p90_S'])} | {f(mc)} | {f(s.get('median_abs_speed_error'))} | {s['verdict']} | {', '.join(f'{k} {v}' for k, v in sorted(s['reasons'].items()))} |")
    L_.append("")
    L_.append("## selections on each construction (row 20 N), the listing's own deepest supported minimum by step")
    L_.append("")
    for e in out["constructions"]:
        s20 = e["selections"]["20"]
        L_.append(f"- {e['id']}: " + ", ".join("none" if s is None else f"{s[1]:g}" for s in s20))
    k12 = next(e for e in out["constructions"] if e["id"] == "K12")
    pr = k12["property_outputs_equal_k1_over_identical_stencils"]
    L_.append("")
    L_.append(f"K12, complete outputs equal K1's over identical input stencils (A from step {pr['A_from_step']}, C from the pair at step {pr['C_from_pair']}), per candidate: " + ", ".join(f"{k} {v}" for k, v in pr["equal"].items()))
    L_.append("")
    L_.append("## overall verdicts under the declared rule of section 1, over the pass-or-fail constructions")
    L_.append("")
    L_.append("| candidate | pass-or-fail constructions | adequate | not adequate | not assessed | overall |")
    L_.append("| --- |" + " --- |" * 5)
    for name, v in overall_verdicts(out).items():
        L_.append(f"| {name} | {v['constructions']} | {v['adequate']} | {v['not adequate']} | {v['not assessed']} | {v['overall']} |")
    return "\n".join(L_) + "\n"


SCALES = {"A1": [(8, 6), (12, 9), (16, 12)], "C1": [(6, 4), (9, 6), (12, 8)]}
SCALE_IDS = ("K1", "K3", "K4", "K13")


def run_scales(ids=SCALE_IDS):
    """The anticipated comparison of section 5: the repaired forms at the declared
    neighborhood and at 1.5 and 2 times it, on the clean translation, the standing
    envelope with a moving internal feature, the translation on a standing background,
    and the broad wave. Asked exactly as the constructions run is."""
    out = []
    by_id = {cid: (field, truth) for cid, field, truth in constructions() if cid in ids}
    for cid in ids:
        field, truth = by_id[cid]
        sel = selections(field)
        for cand, halves in SCALES.items():
            for half in halves:
                res = []
                for row in ROWS:
                    if cand == "A1":
                        for item in sel[row]:
                            if item is not None:
                                r = advection_fit(field, item[0], item[1], float(row), growth=True, both_flanks=True, half=half); r.update(row=row, k=item[0]); res.append(r)
                    else:
                        for k, x0, x1 in continuous_pairs(sel[row]):
                            r = shift_correlation(field, k, x0, float(row), x1, refine=True, westward=True, half=half); r.update(row=row, k=k); res.append(r)
                out.append({"id": cid, "truth": truth, "candidate": cand, "half_cells": list(half), "half_degrees": [half[0] * DX, half[1] * DX], "summary": summarize(res, truth)})
    return out


def scale_table(rows):
    f = lambda x, n=2: "" if x is None else f"{x:.{n}f}"
    L_ = ["# the repaired forms at three neighborhood scales (section 5), counts with denominators",
          "the declared scale first (A1 4 by 3 degrees, C1 3 by 2), then 1.5 and 2 times it; asked, assessed and the verdict exactly as in the constructions table", "",
          "| construction | known answer | candidate | half-width, degrees | asked | supported | rejected | abstained | support of assessed | median S | S at the 90th percentile | median c_x or dx | speed error | verdict |",
          "| --- |" + " --- |" * 13]
    for r in rows:
        s = r["summary"]; mc = s["median_c_x"] if r["candidate"].startswith("A") else s["median_dx"]
        L_.append(f"| {r['id']} | {r['truth']['answer']}{'' if r['truth']['speed'] is None else ' at ' + format(r['truth']['speed'], 'g')} | {r['candidate']} | {r['half_degrees'][0]:g} by {r['half_degrees'][1]:g} | {s['asked']} | {s['supported']} | {s['rejected']} | {s['abstained']} | {f(s['support_fraction_of_assessed'])} | {f(s['median_S'])} | {f(s['p90_S'])} | {f(mc)} | {f(s.get('median_abs_speed_error'))} | {s['verdict']} |")
    L_.append("")
    for cand, scales in scale_pass_sets(rows).items():
        L_.append(f"- {cand}, scales adequate on K3 and K4 together: " + (", ".join(f"{a:g} by {b:g}" for a, b in scales) if scales else "none"))
    return "\n".join(L_) + "\n"


# ------------------------------------------------------------------ the four intervals

def load_interval(name):
    import pilot_field_sequence as F
    import pilot_vorticity_beside as VB
    paths = INTERVALS[name]
    rec = json.load(open(paths["listing"])); ph = json.load(open(paths["phases"]))
    if ph["record"]["sha256"] != sha256_of(paths["listing"]):
        raise SystemExit(f"REFUSED: {paths['phases']} records a listing digest that does not match {paths['listing']}")
    year = int(rec["year"]); level = float(rec["level_hpa"])
    stamps = [s["time"] for s in rec["steps"]]
    t0 = F.to_day(stamps[0]) - STEP_HOURS / 24.0; t1 = F.to_day(stamps[-1]) + STEP_HOURS / 24.0
    files = {n: os.path.join(DATA_DIR, f"era5_{n}_{year}_6h_region.nc") for n in (f"v{level:g}", "sp")}
    grid = F.grid_from_level_file(files[f"v{level:g}"], year)
    steps = F.sequence_steps(grid["time"], t0, t1, STEP_HOURS)
    step_days = [float(grid["time"][k]) for k in steps]
    if len(step_days) != len(stamps) + 2:
        raise SystemExit(f"REFUSED: the files hold {len(step_days)} steps for the extended window, not {len(stamps) + 2}")
    t_read = time.perf_counter()
    v = F.field_on(files[f"v{level:g}"], "v", "m s**-1", FIELD_LAT, FIELD_LON, step_days, level)
    sp = F.surface_pressure_on(files["sp"], FIELD_LAT, FIELD_LON, step_days)
    t_read = time.perf_counter() - t_read
    valid = np.stack([F.terrain_classes(sp[k], v[k], level) == "valid" for k in range(len(step_days))])
    field = {"lon": FIELD_LON, "lat": FIELD_LAT, "v": np.where(valid, v, np.nan), "valid": valid, "step_hours": STEP_HOURS,
             "stamps": [(F.EPOCH + dt.timedelta(days=d)).strftime("%Y-%m-%dT%H") for d in step_days]}
    assert field["stamps"][1:-1] == stamps, "the loaded steps do not bracket the listing's steps by one"
    return rec, ph, field, {"read_seconds": t_read, "files": {n: {"path": p, "sha256": sha256_of(p)} for n, p in files.items()}}, VB


def diagnose(name, set_name="primary"):
    rec, ph, field, meta, VB = load_interval(name)
    res = ph["results"][set_name]
    rows = sorted({int(r) for r in res["pairs"]}, reverse=True)
    t_run = time.perf_counter()
    per = {n: [] for n in CANDIDATES}
    t_cand = {n: 0.0 for n in CANDIDATES}
    n = len(rec["steps"])
    for row in rows:
        cls = VB.classify_steps(res, rec, row)
        pairs = {p["k"]: p for p in res["pairs"].get(str(row), res["pairs"].get(row, []))}
        motion = {i: m for i, m in res["motion"].get(str(row), res["motion"].get(row, []))}
        for i in range(n):
            m = next((m for m in rec["steps"][i]["rows"] if int(m["row_lat"]) == row), None)
            if m is None or m.get("status") != "assigned":
                continue
            x0 = float(m["i1"]["lon"])
            for cand, (unit, fn) in CANDIDATES.items():
                if unit == "step":
                    t0 = time.perf_counter(); r = fn(field, i + 1, x0, float(row)); t_cand[cand] += time.perf_counter() - t0
                    r.update(row=row, k=i, x0=x0, cls=cls[i], label=motion.get(i))
                    per[cand].append(r)
            p = pairs.get(i)
            if p is not None and str(p["class"]).startswith("continuous"):
                for cand, (unit, fn) in CANDIDATES.items():
                    if unit == "pair":
                        t0 = time.perf_counter(); r = fn(field, i + 1, x0, float(row), float(p["lon_to"])); t_cand[cand] += time.perf_counter() - t0
                        r.update(row=row, k=i, x0=x0, x1=float(p["lon_to"]), cls=cls[i], label=motion.get(i), pair_class=p["class"])
                        per[cand].append(r)
    t_run = time.perf_counter() - t_run
    tables = {cand: by_class(per[cand]) for cand in CANDIDATES}
    return {"interval": name, "set": set_name, "listing": {"path": INTERVALS[name]["listing"], "sha256": ph["record"]["sha256"]}, "phases_path": INTERVALS[name]["phases"],
            "fields": meta["files"], "field_box": {"lat": [float(FIELD_LAT[0]), float(FIELD_LAT[-1])], "lon": [float(FIELD_LON[0]), float(FIELD_LON[-1])]},
            "steps": n, "rows": rows,
            "timing_seconds": {"read": meta["read_seconds"], "candidates": t_run, "per_candidate": t_cand, "read_inclusive_total": meta["read_seconds"] + t_run,
                               "boundaries": "read is the two field_on calls only (grid discovery and file hashing excluded); candidates is the loop over row-steps and pairs; per_candidate sums each candidate's own calls, the 3 by 3 smoothing charged to A2 at its first call; the total is read plus candidates"},
            "by_class": tables, "results": per}


CLASS_ORDER = ("west", "standing", "east", "break", "unassessable")


def by_class(results):
    out = []
    for c in CLASS_ORDER + ("all",):
        rs = [r for r in results if c == "all" or r["cls"] == c]
        sup = [r for r in rs if r["outcome"] == "supported"]; rej = [r for r in rs if r["outcome"] == "rejected"]; abst = [r for r in rs if r["outcome"] == "abstained"]
        assessed = sup + rej
        row = {"class": c, "asked": len(rs), "supported": len(sup), "rejected": len(rej), "abstained": len(abst), "assessed": len(assessed),
               "median_S": statistics.median([r["S"] for r in assessed]) if assessed else None}
        if any("c_x" in r for r in assessed):
            cx = [r["c_x"] for r in assessed]
            row["median_c_x"] = statistics.median(cx)
            labeled = [r for r in assessed if r.get("label") in ("west", "standing", "east")]
            agree = [r for r in labeled if (r["label"] == "west" and r["c_x"] < -1.0) or (r["label"] == "standing" and abs(r["c_x"]) <= 1.0) or (r["label"] == "east" and r["c_x"] > 1.0)]
            row["labeled"] = len(labeled); row["label_agreement"] = len(agree)
        if any("dx_refined" in r for r in assessed):
            row["median_dx"] = statistics.median([r["dx_refined"] for r in assessed])
            row["median_rmax"] = statistics.median([r["rmax"] for r in assessed])
        reasons = {}
        for r in abst + rej:
            if r["reason"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        row["reasons"] = reasons
        out.append(row)
    return out


def diagnosis_table(d):
    f = lambda x, n=2: "" if x is None else f"{x:.{n}f}"
    L_ = [f"# the candidates on {d['interval']}, listing {os.path.basename(d['listing']['path'])} (sha256 {d['listing']['sha256'][:16]}), set {d['set']}, rows {d['rows'][0]} to {d['rows'][-1]} N",
          "A0, A1 and A2 asked at every assigned row-step, C0 and C1 at every pair the frozen reader classes continuous; row-steps classed by the reader's labels (west, standing, east inside chains; break, unassessable outside) exactly as the history-layer diagnosis classed them; the labels are not truth, and no count here is a pass or a fail; label agreement for A counts assessed steps inside chains whose estimated zonal speed has the label's sign at 1 degree per day", ""]
    for cand in CANDIDATES:
        L_.append(f"## {cand}")
        L_.append("| class | asked | supported | rejected | abstained | median S | median c_x or dx | median R2 or Rmax | label agreement | reasons |")
        L_.append("| --- |" + " --- |" * 9)
        for row in d["by_class"][cand]:
            mc = row.get("median_c_x", row.get("median_dx"))
            mr = row.get("median_rmax")
            if "median_c_x" in row:
                rs = [r for r in d["results"][cand] if (row["class"] == "all" or r["cls"] == row["class"]) and "r2" in r]
                mr = statistics.median([r["r2"] for r in rs]) if rs else None
            la = "" if "labeled" not in row else f"{row['label_agreement']} of {row['labeled']}"
            L_.append(f"| {row['class']} | {row['asked']} | {row['supported']} | {row['rejected']} | {row['abstained']} | {f(row['median_S'])} | {f(mc)} | {f(mr)} | {la} | {', '.join(f'{k} {v}' for k, v in sorted(row['reasons'].items()))} |")
        L_.append("")
    L_.append(f"timing, seconds: read {d['timing_seconds']['read']:.2f}, candidates {d['timing_seconds']['candidates']:.2f} ({d['steps']} steps, {len(d['rows'])} rows, single process)")
    return "\n".join(L_) + "\n"


def runtime_table(d):
    n_season = 488
    T = d["timing_seconds"]
    per_step = T["candidates"] / d["steps"]
    return "\n".join([f"# runtime of the complete diagnosis of {d['interval']} on this laptop, single process",
                      f"- file reads of the two retained variables over the extended window: {T['read']:.2f} s ({T['boundaries']})",
                      f"- all five candidates at every assigned row-step and continuous pair, {d['steps']} steps by {len(d['rows'])} rows: {T['candidates']:.2f} s, {per_step:.3f} s per step",
                      "- per candidate, its own calls: " + ", ".join(f"{c} {t:.2f} s" for c, t in T["per_candidate"].items()),
                      f"- read-inclusive total: {T['read_inclusive_total']:.2f} s",
                      f"- scaled to one season of {n_season} steps as arithmetic, not a measurement: {per_step * n_season:.0f} s for the five candidates",
                      "- development shortcuts: no one-sided tendency at the window ends, no resampling, no uncertainty calculation beyond the condition number, the search in C over integer cells with one parabola, pure numpy without vectorization across rows",
                      ""])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "diagnose", "runtime", "scales"))
    ap.add_argument("--interval", choices=sorted(INTERVALS), default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--table", default=None)
    a = ap.parse_args(argv)
    for p in (a.out, a.table):
        if p and os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    here = {"script": "scripts/pilot_second_observable.py", "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_SECOND_OBSERVABLE_DEVELOPMENT_2026-10-17.md"}
    if a.mode == "constructions":
        out = run_constructions(); out["generated_by"] = here
        table = construction_table(out)
    elif a.mode == "scales":
        out = {"scales": run_scales(), "generated_by": here}
        table = scale_table(out["scales"])
    elif a.mode == "diagnose":
        if not a.interval:
            raise SystemExit("REFUSED: --interval is required")
        out = diagnose(a.interval); out["generated_by"] = here
        table = diagnosis_table(out)
    else:
        if not a.interval:
            raise SystemExit("REFUSED: --interval is required")
        d = diagnose(a.interval)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(runtime_table(d))
        print(runtime_table(d))
        return 0
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    if a.table:
        with open(a.table, "w", encoding="utf-8") as fh:
            fh.write(table)
    print("\n".join(table.splitlines()[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
