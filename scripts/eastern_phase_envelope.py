#!/usr/bin/env python3
"""Phase progression and envelope evolution of the eastern 2 to 10 day signal
(`EASTERN_PHASE_ENVELOPE_2026-10-26.md`). The 700 hPa band means on 10 W to 75 E are
band-passed to 2 to 10 days per season, turned into analytic signals along time, and for
base longitudes across the band the complex lagged covariance with every longitude is
pooled over the 24 retained seasons. A signed density of its phase tilt in longitude
separates covariance whose phase progresses west or east from covariance of one phase,
such as a standing pattern's. The density's centroid is followed as the lag grows (the
envelope of the population's phase-tilted activity), and the local wavenumber and
frequency around the followed positions give the phase speed. The readings say what moves
and how fast, for the population, and not why. Noise-free packets that form and decay in
one region give an envelope slower than their own (section 7), so a difference between the
envelope and phase speeds is not by itself dispersion. A standing envelope class describes
the measured covariance centroid, and half-day packets moving east read standing under the
matched noise while their noise-free envelope moves east (section 12), so it does not show
by itself that the activity stays in place. Nothing here identifies a feature.

    python3 scripts/eastern_phase_envelope.py constructions --out-dir DIR
    python3 scripts/eastern_phase_envelope.py archive --xband DIR --constructions PATH --out-dir DIR
    python3 scripts/eastern_phase_envelope.py archive --level 600 --xband DIR --constructions PATH --out-dir DIR
"""

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import sys

import numpy as np
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_anomaly_minima as A  # noqa: E402
import eastern_crossing_test as X  # noqa: E402
import eastern_lag_regression as LR  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402

YEARS = LR.YEARS
LONS = LR.LONS
BAND_LAT = LR.BAND_LAT
OBS_PER_DAY = LR.OBS_PER_DAY
DT_DAYS = 1.0 / OBS_PER_DAY
LAGS = LR.LAGS
BASES = LR.BASES
TRIM = 2 * OBS_PER_DAY            # samples dropped at each end of a season's analytic block
EDGE_DEG = 1.0
N_MIN = 0.3
LONG_WAVE_DEG = 60.0
K_MIN = 2.0 * math.pi / LONG_WAVE_DEG
N_BOOT = 1000
BOOT_SEED = 20261026
UNDEFINED_MAX = 0.05
SLOW = 1.0
MIN_POS_LAG_DAYS = 1.0
SCORED_BASES = (50.0, 55.0)


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ analytic signals and sums

def analytic(S):
    """The season's 2 to 10 day band-pass (the lag regression's filter), its finite block, and
    the analytic signal along time over that block with TRIM samples dropped at each end."""
    F = LR.filtered(np.asarray(S, float))
    e0, e1 = LR.valid_block(F)
    Z = hilbert(F[e0:e1], axis=0)
    if Z.shape[0] <= 2 * TRIM + 2 * len(LAGS):
        raise SystemExit("REFUSED: a season's analytic block is too short for the lags")
    return Z[TRIM:Z.shape[0] - TRIM]


def season_sums(Z, base_col):
    """The sums one season contributes at each lag, over the times t with t and t + lag both
    in the block: count, sum and sum of squares of the base, sum and sum of squares at every
    longitude, and the sum of field(t + lag) times the conjugate of base(t)."""
    T, nx = Z.shape
    nl = len(LAGS)
    n = np.zeros(nl); sa = np.zeros(nl, complex); saa = np.zeros(nl)
    sb = np.zeros((nl, nx), complex); sab = np.zeros((nl, nx), complex); sbb = np.zeros((nl, nx))
    for li, L in enumerate(LAGS):
        L = int(L)
        t0, t1 = max(0, -L), min(T, T - L)
        a = Z[t0:t1, base_col]; B = Z[t0 + L:t1 + L]
        n[li] = t1 - t0; sa[li] = a.sum(); saa[li] = float((a.real ** 2 + a.imag ** 2).sum())
        sb[li] = B.sum(axis=0); sab[li] = (B * np.conj(a)[:, None]).sum(axis=0); sbb[li] = (B.real ** 2 + B.imag ** 2).sum(axis=0)
    return {"n": n, "sa": sa, "saa": saa, "sb": sb, "sab": sab, "sbb": sbb}


def stack(per_season):
    return {k: np.stack([s[k] for s in per_season]) for k in per_season[0]}


def maps(st, w=None):
    """The pooled complex covariance C, the activity envelope E = |C| and the coherence
    envelope N = |C| over the two standard deviations, from stacked per-season sums weighted
    by `w` (season counts of a resample, ones by default)."""
    w = np.ones(st["n"].shape[0]) if w is None else np.asarray(w, float)
    n = w @ st["n"]
    sa = w @ st["sa"]; saa = w @ st["saa"]
    sb = np.tensordot(w, st["sb"], axes=1); sab = np.tensordot(w, st["sab"], axes=1); sbb = np.tensordot(w, st["sbb"], axes=1)
    ma = sa / n; mb = sb / n[:, None]
    C = sab / n[:, None] - mb * np.conj(ma)[:, None]
    va = saa / n - np.abs(ma) ** 2
    vb = sbb / n[:, None] - np.abs(mb) ** 2
    E = np.abs(C)
    with np.errstate(invalid="ignore", divide="ignore"):
        N = np.where((va[:, None] > 0) & (vb > 0), E / np.sqrt(va[:, None] * vb), np.nan)
    return C, E, N


# ------------------------------------------------------------------ following the envelope

TRACK_HALF_DEG = 15.0       # section 7, development change 2
PHASE_HALF_DEG = 6.0        # section 7, development change 3


def density(C):
    """The signed phase-tilt density Im(conj(C[x - 0.5]) C[x + 0.5]) at each lag and interior
    column, about |C|^2 sin(-k), so positive where the phase progresses west (with the analytic
    signal's positive frequencies) and near zero where the covariance holds one phase across
    longitude, as a standing pattern does (section 7, development change 1)."""
    W = np.full(C.shape, np.nan)
    W[:, 1:-1] = np.imag(np.conj(C[:, :-2]) * C[:, 2:])
    return W


def coherent_density(C, N):
    """The density where N is at least N_MIN at the column and at both neighbors, NaN elsewhere."""
    W = density(C)
    ok = np.zeros(C.shape, bool)
    good = np.isfinite(N) & (N >= N_MIN)
    ok[:, 1:-1] = good[:, :-2] & good[:, 1:-1] & good[:, 2:]
    return np.where(ok, W, np.nan)


def direction(W, base_col, lons=LONS):
    """+1 when the signed density at lag 0 within TRACK_HALF_DEG of the base sums to zero or
    more (westward phase progression dominant), -1 otherwise, and the westward share of the
    density's positive and negative parts there."""
    zero = int(np.flatnonzero(LAGS == 0)[0])
    win = np.flatnonzero(np.abs(lons - lons[base_col]) <= TRACK_HALF_DEG + 1e-9)
    d = W[zero, win]; d = d[np.isfinite(d)]
    pos, neg = float(np.clip(d, 0, None).sum()), float(np.clip(-d, 0, None).sum())
    return (1 if d.sum() >= 0 else -1), (pos / (pos + neg) if pos + neg > 0 else None)


def follow(W, N, base_col, sign, lons=LONS):
    """The centroid of the density in the chosen direction (sign times W, signed) followed
    from lag 0, within TRACK_HALF_DEG of the base and then of the previous centroid, lag by lag
    in both directions, while the windowed density sums above zero, the coherence N at the
    nearest column is at least N_MIN, and the centroid lies more than EDGE_DEG inside the grid.
    Returns {lag in samples: longitude}, empty when lag 0 itself fails."""
    zero = int(np.flatnonzero(LAGS == 0)[0])
    D = sign * W

    def centroid(li, x0):
        win = np.flatnonzero(np.abs(lons - x0) <= TRACK_HALF_DEG + 1e-9)
        d = D[li, win]; ok = np.isfinite(d)
        tot = float(d[ok].sum()) if ok.any() else 0.0
        if tot <= 0:
            return None
        x = float((d[ok] * lons[win][ok]).sum() / tot)
        j = int(np.argmin(np.abs(lons - x)))
        if not (np.isfinite(N[li, j]) and N[li, j] >= N_MIN) or x <= lons[0] + EDGE_DEG or x >= lons[-1] - EDGE_DEG:
            return None
        return x

    x = centroid(zero, float(lons[base_col]))
    if x is None:
        return {}
    pos = {0: x}
    for d in (1, -1):
        xp, li = x, zero + d
        while 0 <= li < len(LAGS):
            xn = centroid(li, xp)
            if xn is None:
                break
            pos[int(LAGS[li])] = xn
            xp, li = xn, li + d
    return pos


def slope(pos):
    """The least-squares slope of followed position against lag in days, at least 3 lags."""
    if len(pos) < 3:
        return None
    lags = np.array(sorted(pos), float) * DT_DAYS
    x = np.array([pos[k] for k in sorted(pos)], float)
    return float(np.polyfit(lags, x, 1)[0])


def phase_speed(C, W, pos, lons=LONS):
    """The phase speed over windows of PHASE_HALF_DEG around the followed positions: the local
    wavenumber (from the phase one column either side) and frequency (from the phase one step
    either side in lag) averaged with weights W^2 / |C|^2, which favor covariance whose phase
    tilts in either direction over covariance of one phase and do not depend on the direction
    chosen for the envelope, so that the speed's sign is measured and not selected (section 7,
    development changes 3 and 5), and their ratio in degrees per day. Returns (speed or None,
    mean wavenumber in radians per degree, mean frequency in radians per day); the speed is
    None when fewer than 3 lags are followed or the wavelength exceeds LONG_WAVE_DEG."""
    zero = int(np.flatnonzero(LAGS == 0)[0])
    if len(pos) < 3:
        return None, None, None
    K = np.full(C.shape, np.nan); Om = np.full(C.shape, np.nan)
    K[:, 1:-1] = -np.angle(C[:, 2:] * np.conj(C[:, :-2])) / (lons[2:] - lons[:-2])
    Om[1:-1, :] = np.angle(C[2:, :] * np.conj(C[:-2, :])) / (2.0 * DT_DAYS)
    A2 = np.abs(C) ** 2
    sk = so = sw = 0.0
    for L, x in pos.items():
        li = zero + int(L)
        if li - 1 < 0 or li + 1 >= len(LAGS):
            continue
        win = np.flatnonzero(np.abs(lons - x) <= PHASE_HALF_DEG + 1e-9)
        win = win[(win > 0) & (win < len(lons) - 1)]
        d = W[li, win]
        with np.errstate(invalid="ignore", divide="ignore"):
            wt = np.where(A2[li, win] > 0, d * d / A2[li, win], 0.0)
        ok = np.isfinite(K[li, win]) & np.isfinite(Om[li, win]) & np.isfinite(wt)
        sk += float((wt[ok] * K[li, win][ok]).sum()); so += float((wt[ok] * Om[li, win][ok]).sum()); sw += float(wt[ok].sum())
    if sw <= 0:
        return None, None, None
    kb, ob = sk / sw, so / sw
    if abs(kb) < K_MIN:
        return None, kb, ob
    return ob / kb, kb, ob


def readings(C, N, base_col, sign=None, lons=LONS):
    """One map's readings: the phase-progression direction (fixed by `sign` when given, for the
    resamples), the envelope of that direction's density followed and its speed, and the phase
    speed at its positions. The density is kept only where the coherence is at least N_MIN at
    the column and both neighbors it is formed from, since the phase where the covariance is
    incoherent is sampling error (section 7, development change 6)."""
    W = coherent_density(C, N)
    d0, share = direction(W, base_col, lons)
    s = d0 if sign is None else sign
    pos = follow(W, N, base_col, s, lons)
    cp, kb, ob = phase_speed(C, W, pos, lons)
    return {"direction": "westward phase tilt" if s > 0 else "eastward phase tilt", "sign": s, "west_share": share,
            "envelope_speed": slope(pos), "phase_speed": cp, "k_mean": kb, "omega_mean": ob,
            "no_zonal_progression": bool(kb is not None and abs(kb) < K_MIN and len(pos) >= 3),
            "lags": sorted(pos), "positions": [pos[k] for k in sorted(pos)],
            "pos_lag_days": (max(pos) * DT_DAYS) if pos else 0.0, "neg_lag_days": (-min(pos) * DT_DAYS) if pos else 0.0}


# ------------------------------------------------------------------ intervals and classes

def boot_weights(n_seasons, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seasons, size=(n_boot, n_seasons))
    return np.stack([np.bincount(r, minlength=n_seasons) for r in idx]).astype(float)


def interval(vals):
    """The 95 percent percentile interval of the defined values and the undefined fraction."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    ok = np.isfinite(v)
    frac = int(np.count_nonzero(~ok)) / len(v)             # a count, so 50 of 1,000 is exactly 0.05 (review F2)
    if not ok.any():
        return None, None, float(frac)
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), float(frac)


def classify(point, lo, hi, undefined, pos_lag_days):
    """Section 4's classes: westward when the point is at most -SLOW and the interval lies below
    zero, eastward symmetrically, both needing a day of positive lag; standing when the interval
    lies within -SLOW to +SLOW; unresolved otherwise or when more than UNDEFINED_MAX of the
    resamples give no value."""
    if point is None or lo is None or undefined > UNDEFINED_MAX:
        return "unresolved"
    if point <= -SLOW and hi < 0 and pos_lag_days >= MIN_POS_LAG_DAYS:
        return "westward"
    if point >= SLOW and lo > 0 and pos_lag_days >= MIN_POS_LAG_DAYS:
        return "eastward"
    if lo >= -SLOW and hi <= SLOW:
        return "standing"
    return "unresolved"


def phase_class(r, lo, hi, undefined, no_zonal_share):
    """The phase class, or no zonal phase progression when the point and at least 1 -
    UNDEFINED_MAX of the resamples read a wavelength longer than LONG_WAVE_DEG."""
    if r["no_zonal_progression"] and no_zonal_share >= 1.0 - UNDEFINED_MAX:
        return "no zonal phase progression"
    return classify(r["phase_speed"], lo, hi, undefined, r["pos_lag_days"])


def base_result(st, base_col, W, lons=LONS):
    """The point readings from the pooled sums and their intervals from the resamples, each
    resample read in the point estimate's direction of phase tilt."""
    C, E, N = maps(st)
    r = readings(C, N, base_col, None, lons)
    env, ph, diff, nz = [], [], [], []
    for w in W:
        Cb, Eb, Nb = maps(st, w)
        b = readings(Cb, Nb, base_col, r["sign"], lons)
        env.append(b["envelope_speed"]); ph.append(b["phase_speed"])
        diff.append(None if b["envelope_speed"] is None or b["phase_speed"] is None else b["envelope_speed"] - b["phase_speed"])
        nz.append(b["no_zonal_progression"])
    out = dict(r)
    for name, vals in (("envelope", env), ("phase", ph), ("difference", diff)):
        lo, hi, und = interval(vals)
        out[f"{name}_interval"] = [lo, hi]; out[f"{name}_undefined"] = und
    out["no_zonal_share"] = float(np.mean(nz))
    out["envelope_class"] = classify(r["envelope_speed"], *out["envelope_interval"], out["envelope_undefined"], r["pos_lag_days"])
    out["phase_class"] = phase_class(r, *out["phase_interval"], out["phase_undefined"], out["no_zonal_share"])
    d0, d1 = out["difference_interval"]
    out["speeds_differ"] = bool(d0 is not None and out["difference_undefined"] <= UNDEFINED_MAX and not (d0 <= 0.0 <= d1))
    return out


def point_class(v):
    """The class of a noise-free reading's point alone, used as the reference in the revised
    rule (section 7, development change 4)."""
    if v is None:
        return "none"
    return "westward" if v <= -SLOW else "eastward" if v >= SLOW else "standing"


def summarize(seasons_Z, bases=BASES, n_boot=N_BOOT):
    W = boot_weights(len(seasons_Z), n_boot)
    out = {}
    for b in bases:
        col = int(np.flatnonzero(np.isclose(LONS, b))[0])
        st = stack([season_sums(Z, col) for Z in seasons_Z])
        out[f"{b:g}"] = base_result(st, col, W)
    return out


# ------------------------------------------------------------------ constructions

CONSTRUCTION_SEED = 20261026
RATE_DAYS = 4.0
SPARSE_RATE_DAYS = 16.0
PACKET_SX, PACKET_ST = 6.0, 2.0
CENTER_RANGE = (47.0, 57.0)
U1 = {"center": 50.0, "width": 4.0}
CASES = {
    "ND7": {"kind": "packets", "wavelength": 30.0, "c_phase": -7.0, "c_env": -7.0},
    "ND3": {"kind": "packets", "wavelength": 12.0, "c_phase": -3.0, "c_env": -3.0},
    "D07": {"kind": "packets", "wavelength": 30.0, "c_phase": -7.0, "c_env": 0.0},
    "D03": {"kind": "packets", "wavelength": 12.0, "c_phase": -3.0, "c_env": 0.0},
    "DE7": {"kind": "packets", "wavelength": 30.0, "c_phase": -7.0, "c_env": 3.0},
    "DE3": {"kind": "packets", "wavelength": 12.0, "c_phase": -3.0, "c_env": 3.0},
    "U1": {"kind": "fixed", "wavelength": 30.0, "c_phase": -7.0},
    "U2": {"kind": "driver", "wavelength": 30.0, "c_phase": -7.0, "c_env": 3.0},
    "S": {"kind": "R1"},
    "N": {"kind": "R6"},
    "ND7w": {"kind": "packets", "wavelength": 30.0, "c_phase": -7.0, "c_env": -7.0, "amp_of": "ND7", "amp_factor": 0.5},
    "ND7s": {"kind": "packets", "wavelength": 30.0, "c_phase": -7.0, "c_env": -7.0, "amp_of": "ND7", "rate_days": SPARSE_RATE_DAYS},
    "DE7q": {"kind": "packets", "wavelength": 30.0, "c_phase": -7.0, "c_env": 3.0, "st": 0.5, "seed_of": "DE7"},   # the review's half-day challenge (section 12)
}
SCORED_CASES = ("ND7", "ND3", "D07", "D03", "DE7", "DE3", "U1", "U2")


def _events(rng, n_steps, rate_days, st=PACKET_ST):
    T = n_steps * DT_DAYS
    n = int(rng.poisson((T + 6.0 * st) / rate_days))
    t0 = rng.uniform(-3.0 * st, T + 3.0 * st, n)
    x0 = rng.uniform(*CENTER_RANGE, n)
    ph = rng.uniform(0.0, 2.0 * math.pi, n)
    return t0, x0, ph


def signal(case, i, amp=1.0, lons=LONS):
    """One season of a construction's signal alone, six-hourly, before noise and rounding."""
    p = CASES[case]
    n_steps = A.SEASON_DAYS * OBS_PER_DAY
    t = (np.arange(n_steps) * DT_DAYS)[:, None]
    x = lons[None, :]
    rng = np.random.default_rng([CONSTRUCTION_SEED, list(CASES).index(p.get("seed_of", case)), i])
    st = p.get("st", PACKET_ST)
    k = 2.0 * math.pi / p["wavelength"]
    V = np.zeros((n_steps, len(lons)))
    if p["kind"] == "packets":
        t0, x0, ph = _events(rng, n_steps, p.get("rate_days", RATE_DAYS), st)
        for a, b, c in zip(t0, x0, ph):
            tau = t - a
            V += np.exp(-((x - b - p["c_env"] * tau) ** 2) / (2 * PACKET_SX ** 2) - tau ** 2 / (2 * st ** 2)) * np.cos(k * (x - b - p["c_phase"] * tau) + c)
    elif p["kind"] == "fixed":
        ph = float(rng.uniform(0.0, 2.0 * math.pi))
        V += np.exp(-((x - U1["center"]) ** 2) / (2 * U1["width"] ** 2)) * np.cos(k * (x - p["c_phase"] * t) + ph)
    elif p["kind"] == "driver":
        ph = float(rng.uniform(0.0, 2.0 * math.pi))
        t0, x0, _ = _events(rng, n_steps, RATE_DAYS, st)
        env = np.zeros_like(V)
        for a, b in zip(t0, x0):
            tau = t - a
            env += np.exp(-((x - b - p["c_env"] * tau) ** 2) / (2 * PACKET_SX ** 2) - tau ** 2 / (2 * st ** 2))
        V += env * np.cos(k * (x - p["c_phase"] * t) + ph)
    else:
        raise SystemExit(f"REFUSED: {case} has no signal of its own")
    return amp * V


def filtered_std_at(seasons, lon=50.0):
    col = int(np.flatnonzero(np.isclose(LONS, lon))[0])
    vals = []
    for S in seasons:
        F = LR.filtered(S)
        e0, e1 = LR.valid_block(F)
        vals.append(F[e0:e1, col])
    return float(np.std(np.concatenate(vals)))


def noise(i):
    return A.noise_field(LR.CONSTRUCTION_NOISE_STD, seed=HI.NOISE_SEEDS[i], n_steps=A.SEASON_DAYS * OBS_PER_DAY, lons=LONS)


def amplitudes():
    """Each signal's amplitude, set so that its filtered standard deviation at 50 E equals the
    filtered noise's (section 3), the weak and sparse cases derived from ND7's."""
    ns = filtered_std_at([noise(i) for i in range(len(YEARS))])
    amp = {}
    for c, p in CASES.items():
        if p["kind"] in ("R1", "R6") or "amp_of" in p:
            continue
        amp[c] = ns / filtered_std_at([signal(c, i) for i in range(len(YEARS))])
    for c, p in CASES.items():
        if "amp_of" in p:
            amp[c] = amp[p["amp_of"]] * p.get("amp_factor", 1.0)
    return {"noise_filtered_std_50E": ns, "amplitude": amp}


def construction_samples(case, i, amp):
    p = CASES[case]
    if p["kind"] == "R1":
        return LR.construction_samples("R1", i)
    if p["kind"] == "R6":
        return LR.construction_samples("R6", i)
    return np.round(signal(case, i, amp[case]) + noise(i), 2)


def clean_readings(case, amp, bases=SCORED_BASES):
    """The point readings of a construction's signal alone (no noise, rounded like the noisy
    samples), the population values the noisy readings are scored against (section 7,
    development change 4)."""
    Zs = [analytic(np.round(signal(case, i, amp[case]), 2)) for i in range(len(YEARS))]
    out = {}
    for b in bases:
        col = int(np.flatnonzero(np.isclose(LONS, b))[0])
        C, E, N = maps(stack([season_sums(Z, col) for Z in Zs]))
        r = readings(C, N, col)
        out[f"{b:g}"] = {"envelope_speed": r["envelope_speed"], "phase_speed": r["phase_speed"], "envelope_point_class": point_class(r["envelope_speed"]),
                         "direction": r["direction"], "lags": r["lags"], "positions": r["positions"]}
    return out


def run_constructions(n_boot=N_BOOT, cases=None):
    cal = amplitudes()
    out, clean = {}, {}
    for c in (cases or CASES):
        Zs = [analytic(construction_samples(c, i, cal["amplitude"])) for i in range(len(YEARS))]
        out[c] = summarize(Zs, n_boot=n_boot)
        if CASES[c]["kind"] not in ("R1", "R6"):
            clean[c] = clean_readings(c, cal["amplitude"])
    return {"calibration": cal, "cases": out, "clean": clean}


CONTRAST_CASES = ("ND7", "ND3", "D07", "D03", "DE7", "DE3", "U1")     # whose envelope class must be resolved


def apply_rule(results):
    """The adequacy rule as revised in development (section 7, changes 4 and 7): in the scored
    cases at 50 and 55 E the phase is westward within 25 percent of the constructed phase speed;
    the envelope's class equals the noise-free reading's point class in the contrast cases and
    is that class or unresolved in U2, so that no case makes a false class claim; and in S and
    N no base reads a moving phase or envelope. Envelope magnitudes are not scored."""
    fails = []
    cases, clean = results["cases"], results["clean"]
    for c in SCORED_CASES:
        cp = CASES[c]["c_phase"]
        for b in SCORED_BASES:
            r, q = cases[c][f"{b:g}"], clean[c][f"{b:g}"]
            if r["phase_class"] != "westward" or r["phase_speed"] is None or abs(r["phase_speed"] - cp) > 0.25 * abs(cp):
                fails.append(f"{c} {b:g} E phase {r['phase_class']} {r['phase_speed']}")
            allowed = {q["envelope_point_class"]} if c in CONTRAST_CASES else {q["envelope_point_class"], "unresolved"}
            if q["envelope_speed"] is None or r["envelope_class"] not in allowed:
                fails.append(f"{c} {b:g} E envelope {r['envelope_class']} {r['envelope_speed']} against noise-free {q['envelope_point_class']} {q['envelope_speed']}")
    for c in ("S", "N"):
        for b, r in cases[c].items():
            moving = [k for k in ("phase_class", "envelope_class") if r[k] in ("westward", "eastward")]
            if moving:
                fails.append(f"{c} {b} E moving in {moving}")
    return {"adequate": not fails, "failures": fails}


def check_constructions(path, rerun=None):
    """The archive runs only on constructions made by this exact script, equal to a fresh run,
    and adequate under the rule recomputed from them."""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps((rerun or run_constructions)()))
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if not verdict["adequate"]:
        raise SystemExit(f"REFUSED: the constructions do not pass the rule of section 4 ({verdict['failures']})")
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored verdict disagrees with the rule applied to the stored readings")
    return rec


# ------------------------------------------------------------------ the archive and the 600 hPa band means

def level_band_means(year, level):
    """A season's band means at `level` on 10 W to 75 E by the band-mean rule of the extended
    records (ten half-degree rows, a cell valid where the surface pressure exceeds the level
    strictly, a sample kept where 8 of 10 rows are valid), stored to two decimals, as
    {band: (samples, longitudes)}. The reader refuses the held-out season."""
    import netCDF4
    import pilot_eastern_propagation as P
    import eastern_regime_views as V
    if year == P.HELD_OUT:
        raise SystemExit("REFUSED: the held-out season is never opened")
    d = netCDF4.Dataset(os.path.join(P.DATA_DIR, f"era5_v{level}_{year}_6h_region.nc"))
    try:
        lat = np.asarray(d.variables["latitude"][:], float); lon = np.asarray(d.variables["longitude"][:], float)
    finally:
        d.close()
    rows = np.flatnonzero((lat >= 10.0 - 1e-6) & (lat <= 29.5 + 1e-6))
    cols = np.flatnonzero((lon >= X.XLON[0] - 1e-6) & (lon <= X.XLON[1] + 1e-6))
    blk = P.read_season(year, level, rows, cols)
    blat, blon = lat[rows.min():rows.max() + 1], lon[cols.min():cols.max() + 1]
    if not np.allclose(blon, LONS):
        raise SystemExit("REFUSED: the longitudes are not 10 W to 75 E at half a degree")
    if list(V.BANDS) != list(BAND_LAT):
        raise SystemExit("REFUSED: the band-mean rule's bands are not the declared bands")
    out = {}
    for b, r in V.band_rows(blat).items():
        series, _ = P.band_series(blk["v"][:, r, :], blk["sp"][:, r, :], level)
        out[b] = np.round(series, 2)
    return out


def load_xband(xband_dir):
    files = sorted(glob.glob(os.path.join(xband_dir, "eval60_xband_????_??.json.gz")))
    recs = []
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            recs.append(json.load(fh))
    X.check_xband_identity(files, recs)
    LR.check_inputs(recs)
    by = {}
    for r in recs:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    return files, by


def check_level_builder(by, years=YEARS):
    """The builder reproduces the stored 700 hPa band means exactly for every season and band
    before its output at another level is used (section 6)."""
    for y in years:
        got = level_band_means(y, 700)
        for b in BAND_LAT:
            want = LR.season_samples(by[y], b)
            if not np.array_equal(got[b], want):
                raise SystemExit(f"REFUSED: the band-mean builder does not reproduce the stored 700 hPa record for {y} {b}")
    return True


def archive_results(seasons_by_band, n_boot=N_BOOT):
    out = {}
    for band in BAND_LAT:
        Zs = [analytic(S) for S in seasons_by_band[band]]
        out[band] = summarize(Zs, n_boot=n_boot)
    return out


def _f(v, iv):
    return "" if v is None else f"{v:+.2f} ({'' if iv[0] is None else format(iv[0], '+.2f')} to {'' if iv[1] is None else format(iv[1], '+.2f')})"


def reading_text(r):
    """The generated reading of one band and base, naming what moves and at what speed, with no
    mechanism words (section 5). A difference between the two speeds is stated as a
    difference only, since noise-free non-dispersive packets show one (section 7)."""
    s = (f"{r['direction']} (westward share {'' if r['west_share'] is None else format(r['west_share'], '.2f')}); "
         f"phases {r['phase_class']} at {_f(r['phase_speed'], r['phase_interval']) or 'no speed'} deg/day; "
         f"envelope {r['envelope_class']} at {_f(r['envelope_speed'], r['envelope_interval']) or 'no speed'} deg/day")
    if r["phase_class"] in ("westward", "eastward") and r["envelope_speed"] is not None and r["phase_speed"] is not None:
        s += f"; envelope minus phase {_f(r['envelope_speed'] - r['phase_speed'], r['difference_interval'])} deg/day"
    return s


# ------------------------------------------------------------------ text, figure and main

HEAD = ("| case or band | base (deg E) | phase tilt (westward share) | phase class | phase speed (interval) | envelope class | envelope speed (interval) | speeds differ | lags followed (days, - and +) | mean wavelength (deg) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def rows(name, res):
    out = []
    for b, r in res.items():
        wl = "" if not r["k_mean"] else f"{2 * math.pi / abs(r['k_mean']):.1f}"
        sh = "" if r["west_share"] is None else f"{r['west_share']:.2f}"
        out.append(f"| {name} | {b} | {r['direction']} ({sh}) | {r['phase_class']} | {_f(r['phase_speed'], r['phase_interval'])} | {r['envelope_class']} | "
                   f"{_f(r['envelope_speed'], r['envelope_interval'])} | {'yes' if r['speeds_differ'] else 'no'} | "
                   f"{r['neg_lag_days']:.2f}, {r['pos_lag_days']:.2f} | {wl} |")
    return out


def constructions_text(results, verdict):
    cal = results["calibration"]
    L = ["# phase and envelope, the constructions (section 3, rule as revised in section 7)", "",
         f"filtered noise standard deviation at 50 E {cal['noise_filtered_std_50E']:.3f} m/s; amplitudes "
         + ", ".join(f"{c} {a:.3f}" for c, a in cal["amplitude"].items()), "", HEAD]
    for c in results["cases"]:
        L += rows(c, results["cases"][c])
    L += ["", "## noise-free readings (the reference of the revised rule)", "",
          "| case | base (deg E) | phase speed | envelope speed | envelope point class |", "| --- | --- | --- | --- | --- |"]
    for c, v in results["clean"].items():
        for b, q in v.items():
            L.append(f"| {c} | {b} | {'' if q['phase_speed'] is None else format(q['phase_speed'], '+.2f')} | "
                     f"{'' if q['envelope_speed'] is None else format(q['envelope_speed'], '+.2f')} | {q['envelope_point_class']} |")
    L += ["", f"adequate under the revised rule, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(L) + "\n"


def archive_text(res, level):
    L = [f"# phase and envelope, the archive at {level} hPa (section 8)", "", HEAD]
    for band, v in res.items():
        L += rows(band, v)
    L += ["", "## generated readings", ""]
    for band, v in res.items():
        for b, r in v.items():
            L.append(f"- {band}, {b} E, {reading_text(r)}")
    return "\n".join(L) + "\n"


def draw(path, seasons_by_band, res, bases=(10.0, 45.0, 50.0, 55.0, 60.0, 65.0), level=700):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(BAND_LAT), len(bases), figsize=(3.0 * len(bases), 2.7 * len(BAND_LAT)), sharex=True, sharey=True, constrained_layout=True)
    lagd = LAGS * DT_DAYS
    for r_i, band in enumerate(BAND_LAT):
        Zs = [analytic(S) for S in seasons_by_band[band]]
        for c_i, b in enumerate(bases):
            col = int(np.flatnonzero(np.isclose(LONS, b))[0])
            C, E, N = maps(stack([season_sums(Z, col) for Z in Zs]))
            ax = axes[r_i, c_i]
            v0 = float(np.abs(C[int(np.flatnonzero(LAGS == 0)[0]), col]))
            cs = ax.contourf(LONS, lagd, C.real / v0, levels=np.arange(-1.0, 1.05, 0.1), cmap="RdBu_r", extend="both")
            ax.contour(LONS, lagd, N, levels=[N_MIN], colors="k", linewidths=0.6)
            rr = res[band][f"{b:g}"]
            ax.plot(rr["positions"], np.array(rr["lags"]) * DT_DAYS, "k.", ms=2.5)
            if rr["phase_speed"] is not None and rr["positions"]:
                x0 = rr["positions"][rr["lags"].index(0)]
                ax.plot([x0 - 4 * rr["phase_speed"], x0 + 4 * rr["phase_speed"]], [-4, 4], color="m", lw=0.8)
            ax.set_title(f"{band}, base {b:g} E", fontsize=8)
            ax.set_xlim(LONS[0], LONS[-1])
            if c_i == 0:
                ax.set_ylabel("lag (days)")
            if r_i == len(BAND_LAT) - 1:
                ax.set_xlabel("longitude (deg E)")
    fig.colorbar(cs, ax=axes, shrink=0.6, label="real part of the lagged covariance over the base variance")
    fig.suptitle(f"{level} hPa, 24 seasons. Black dots: the activity envelope followed. Magenta: the phase speed through it. Black contour: coherence {N_MIN}.", fontsize=8)
    fig.savefig(path, dpi=90); plt.close(fig)


def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--xband", default=None)
    ap.add_argument("--constructions", default=None)
    ap.add_argument("--level", type=int, default=700, choices=(700, 600))
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    gen = {"script": "scripts/eastern_phase_envelope.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_PHASE_ENVELOPE_2026-10-26.md",
           "filtering_sha256": sha256_of(os.path.join(os.path.dirname(HERE), "src", "aew", "filtering.py")),
           "lag_regression_sha256": LR.SCRIPT_SHA256_AT_LOAD}
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_phase_env_constructions{a.tag}.{e}") for e in ("json", "md"))
        refuse_existing([pj, pm])
        results = json.loads(json.dumps(run_constructions()))
        verdict = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict}, fh)
        text = constructions_text(results, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.xband and a.constructions):
        raise SystemExit("REFUSED: --xband and --constructions are required for the archive")
    check_constructions(a.constructions)
    stem = f"eval60_phase_env_{a.level}"
    paths = {k: os.path.join(a.out_dir, f"{stem}_{k}{a.tag}.{e}") for k, e in (("archive", "json"), ("tables", "md"), ("maps", "png"))}
    refuse_existing(paths.values())
    files, by = load_xband(a.xband)
    if a.level == 700:
        seasons = {band: [LR.season_samples(by[y], band) for y in YEARS] for band in BAND_LAT}
    else:
        check_level_builder(by)
        built = {y: level_band_means(y, a.level) for y in YEARS}
        for y in YEARS:
            for band in BAND_LAT:
                if not np.isfinite(built[y][band]).all():
                    raise SystemExit(f"REFUSED: {y} {band} at {a.level} hPa holds a missing sample")
        seasons = {band: [built[y][band] for y in YEARS] for band in BAND_LAT}
    res = archive_results(seasons)
    draw(paths["maps"], seasons, res, level=a.level)
    gen.update({"level": a.level, "xband": {os.path.basename(p): sha256_of(p) for p in files},
                "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}})
    with open(paths["archive"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res, "readings": {band: {b: reading_text(r) for b, r in v.items()} for band, v in res.items()},
                   "figure": {"path": paths["maps"], "sha256": sha256_of(paths["maps"])}}, fh)
    text = archive_text(res, a.level)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
