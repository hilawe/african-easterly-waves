#!/usr/bin/env python3
"""The injection-recovery test (`EASTERN_INJECTION_RECOVERY_2026-10-21.md`): westward signals
of known form and strength are added to the 700 hPa band means on 10 W to 75 E, the
lag-regression diagnostic is rerun unchanged, and its readings are tabulated against what was
added. Three forms are added, continuous wave trains, the same trains in bursts, and Gaussian
troughs on the recorded tracks of the westward histories, each on the archive's own background
and on noise matched to the archive's filtered variability. The outcome measures what the
diagnostic can see on these backgrounds. It identifies no feature and bounds no event count.

    python3 scripts/eastern_injection_recovery.py constructions --records DIR --histories PATH --out-dir DIR
    python3 scripts/eastern_injection_recovery.py archive --xband DIR --records DIR --histories PATH \\
        --lagreg PATH --constructions PATH --out-dir DIR
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_anomaly_minima as A  # noqa: E402
import eastern_crossing_test as X  # noqa: E402
import eastern_lag_regression as L  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_standing_climatology as SC  # noqa: E402
import eastern_wobble_climatology as W  # noqa: E402

YEARS = L.YEARS
LONS = L.LONS
N_SAMPLES = A.SEASON_DAYS * L.OBS_PER_DAY
BASES = (45.0, 50.0, 55.0, 60.0, 65.0, 70.0)
SCALE_BASES = (50.0, 55.0, 60.0, 65.0, 70.0)
PLACEMENTS = 5
DETECT_MIN = 4
INJ_SEED = 2026102101
MATCHED_SEED_STEP = 1000
TRAIN_PERIOD = 4.0
TRAIN_SPEEDS = (5.0, 7.0, 10.0)
TRAIN_AMPS = (0.5, 1.0, 1.5, 2.0, 3.0)
BURST_SPEED = 7.0
BURST_DAYS = 10
BURST_COUNTS = (3, 6)
BURST_AMPS = (1.0, 2.0, 3.0, 4.0)
PACKETS = {"P1": (1, 1, 2.0), "P2": (2, 1, 2.0), "P3": (3, 1, 2.0), "P4": (4, 1, 2.0),
           "C2": (1, 2, 2.0), "C4": (1, 4, 2.0), "C8": (1, 8, 2.0), "W1": (1, 1, 4.0)}
CAL_DEPTH = 6.0
CAL_START = 30
MOVING_SLOPE = -1.0
QUIET_STD = 0.3


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the exact injection

def hundredths(S):
    """The samples as integer hundredths, refused unless they lie exactly on the hundredth lattice."""
    H = np.rint(np.asarray(S, float) * 100.0)
    if not np.array_equal(H / 100.0, S):
        raise SystemExit("REFUSED: a background sample is not an exact hundredth")
    return H


def inject(H, V):
    """The background in hundredths plus the injected field rounded to hundredths, back in m/s,
    so a zero field returns the background exactly."""
    return (H + np.rint(np.asarray(V, float) * 100.0)) / 100.0


def times():
    return (np.arange(N_SAMPLES) / L.OBS_PER_DAY)[:, None]


def train_field(amp, speed, phase, period=TRAIN_PERIOD, envelope=None):
    """A westward train a sin(2 pi (x + c t) / (c P) + phase) on the whole grid, times an
    envelope in time when one is given."""
    t = times(); x = LONS[None, :]
    V = amp * np.sin(2.0 * np.pi * (x + speed * t) / (speed * period) + phase)
    return V if envelope is None else V * envelope[:, None]


def burst_envelope(rng, k, days=BURST_DAYS):
    """k Hann windows of `days` days, one starting at a whole day drawn uniformly inside each of k
    equal segments of the season so that the windows never overlap."""
    t = times()[:, 0]; env = np.zeros_like(t); seg = A.SEASON_DAYS / k
    for j in range(k):
        lo = int(np.ceil(j * seg)); hi = int(np.floor((j + 1) * seg)) - days
        s = int(rng.integers(lo, hi + 1))
        inside = (t >= s) & (t < s + days)
        env[inside] += np.sin(np.pi * (t[inside] - s) / days) ** 2
    return env


def track_of(h):
    """First longitude, speed in degrees per day, length in days and recorded depth in m/s (the
    negative of the mean member depth) of a recorded history."""
    x = h["lons"]; n = len(x)
    return {"first_lon": float(x[0]), "speed": float(x[-1] - x[0]) / (n - 1), "days": n, "recorded": -float(np.mean(h["depths"]))}


def calibration_ratio(track, width, month_of=None):
    """The recorded depth over the injected depth for this track alone: a trough of CAL_DEPTH m/s
    and `width` degrees moving on the track from day CAL_START, no noise, through the daily path
    and the histories' declared representation. On each day of the track the deepest minimum
    within TOL_DEG of the trough's daily-mean center is taken, the depths the history members
    carry, and the ratio is their mean over CAL_DEPTH. None unless at least half of the days hold
    such a minimum, which happens for slow, long tracks at the wider width, whose trough the
    nine-day background largely absorbs."""
    month_of = A.month_index() if month_of is None else month_of
    t = np.arange(A.SEASON_DAYS * 4) / 4.0
    V = np.zeros((len(t), len(A.LONS))); pos = np.full(len(t), np.nan)
    for k, tk in enumerate(t):
        if CAL_START <= tk < CAL_START + track["days"]:
            pos[k] = track["first_lon"] + track["speed"] * (tk - CAL_START)
            V[k] = A.gauss(A.LONS, pos[k], CAL_DEPTH, width)
    H = A.hundredths_from_samples(np.round(V, 2))
    mins = HI.rep_minima(W.HISTORIES[0], H, month_of, A.LONS)
    centers = pos.reshape(A.SEASON_DAYS, 4).mean(axis=1)
    got = []
    for d in range(CAL_START, CAL_START + track["days"]):
        near = [m["depth"] for m in (mins[d] or []) if abs(m["lon"] - centers[d]) <= HI.TOL_DEG]
        if near:
            got.append(-min(near))
    if 2 * len(got) < track["days"]:
        return None
    return float(np.mean(got)) / CAL_DEPTH


def packet_field(tracks, rng, depth_mult, copies, width):
    """Gaussian troughs on the given tracks, each copied `copies` times, each copy starting on a
    whole day t0 drawn uniformly from 0 to the season's length minus its own, moving from its
    first longitude at its speed for its length in days, at depth_mult times its injected depth."""
    t = times()[:, 0]; V = np.zeros((N_SAMPLES, len(LONS)))
    for tr in tracks:
        for _ in range(copies):
            t0 = int(rng.integers(0, A.SEASON_DAYS - tr["days"] + 1))
            k0, k1 = t0 * L.OBS_PER_DAY, (t0 + tr["days"]) * L.OBS_PER_DAY
            d = depth_mult * tr["injected"][f"{width:g}"]
            V[k0:k1] += np.array([A.gauss(LONS, tr["first_lon"] + tr["speed"] * (tk - t0), d, width) for tk in t[k0:k1]])
    return V


# ------------------------------------------------------------------ backgrounds and the reading

def pooled_filtered_std(seasons_f, cols):
    """The standard deviation of the filtered field at each column, pooled over the seasons'
    finite blocks."""
    F = np.concatenate([F[slice(*L.valid_block(F))] for F in seasons_f])
    return [float(F[:, c].std()) for c in cols]


def col(b):
    return int(np.flatnonzero(np.isclose(LONS, b))[0])


def unit_noise_scale():
    """The mean over SCALE_BASES of the pooled filtered std of construction noise of unit std."""
    fs = [L.filtered(np.round(A.noise_field(1.0, seed=HI.NOISE_SEEDS[i], n_steps=N_SAMPLES, lons=LONS), 2)) for i in range(len(YEARS))]
    return float(np.mean(pooled_filtered_std(fs, [col(b) for b in SCALE_BASES])))


def matched_background(std, p):
    """Placement p's matched-noise background, the construction noise at `std` with seeds offset
    by MATCHED_SEED_STEP times (p + 1), rounded to hundredths, as integer hundredths per season."""
    return [hundredths(np.round(A.noise_field(std, seed=HI.NOISE_SEEDS[i] + MATCHED_SEED_STEP * (p + 1), n_steps=N_SAMPLES, lons=LONS), 2))
            for i in range(len(YEARS))]


def read(seasons, lat):
    """The lag-regression diagnostic, unchanged, at BASES on filtered seasons, reduced to the
    readers declared in section 2.4."""
    res = L.annotate(L.summarize([L.filtered(S) for S in seasons], bases=BASES, lat=lat, null_bases=()))
    out = {}
    for b, v in res.items():
        s = v["slope_deg_per_day"]
        out[b] = {"class": v["class"], "slope": s, "westmost": v["westmost"], "positive_lag_days": v["positive_lag_days"],
                  "reaches": v["reaches_s2_west"], "moving": bool(s is not None and s <= MOVING_SLOPE),
                  "reaches_025": v["sensitivity"]["0.25"]["reaches_s2_west"], "reaches_02": v["sensitivity"]["0.2"]["reaches_s2_west"],
                  "class_02": v["sensitivity"]["0.2"]["class"]}
    return out


def passes(r):
    """The lag-regression reading of section 2.5 of its record, from the bases at 60, 65 and 70 E."""
    return {k: any(r[f"{b:g}"][f] for b in L.S2_EAST_BASES) for k, f in (("0.3", "reaches"), ("0.25", "reaches_025"), ("0.2", "reaches_02"))}


# ------------------------------------------------------------------ the configurations

def configurations():
    out = []
    for c in TRAIN_SPEEDS:
        for a in TRAIN_AMPS:
            out.append({"part": "A", "name": f"train {c:g} deg/day, {a:g} m/s", "speed": c, "amp": a})
    for k in BURST_COUNTS:
        for a in BURST_AMPS:
            out.append({"part": "A2", "name": f"{k} bursts, {a:g} m/s", "speed": BURST_SPEED, "amp": a, "bursts": k})
    for name, (m, n, w) in PACKETS.items():
        out.append({"part": "B", "name": name, "depth_mult": m, "copies": n, "width": w})
    return out


def injected_field(cfg, rng, tracks):
    if cfg["part"] == "A":
        return train_field(cfg["amp"], cfg["speed"], float(rng.uniform(0.0, 2.0 * np.pi)))
    if cfg["part"] == "A2":
        ph = float(rng.uniform(0.0, 2.0 * np.pi))
        return train_field(cfg["amp"], cfg["speed"], ph, envelope=burst_envelope(rng, cfg["bursts"]))
    return packet_field(tracks, rng, cfg["depth_mult"], cfg["copies"], cfg["width"])


def run_config(cfg_i, cfg, backgrounds, tracks_by_year, band_i, lat, bg_i):
    """Five placements on their backgrounds (a list of per-season hundredths for each placement),
    each with its own random draws keyed by the configuration, background, band, placement and
    season."""
    reads = []
    for p in range(PLACEMENTS):
        seasons = []
        for i, y in enumerate(YEARS):
            rng = np.random.default_rng([INJ_SEED, cfg_i, bg_i, band_i, p, i])
            seasons.append(inject(backgrounds[p][i], injected_field(cfg, rng, tracks_by_year.get(y, []))))
        r = read(seasons, lat)
        reads.append({"bases": r, "passes": passes(r)})
    return reads


def summarize_reads(reads):
    """Placement counts of the reading at each threshold and, per base, of the westward class and
    of a moving ridge at 0.3, with the median slope and westmost peak over placements."""
    out = {"reading_passes": {k: sum(r["passes"][k] for r in reads) for k in ("0.3", "0.25", "0.2")}, "bases": {}}
    for b in reads[0]["bases"]:
        vs = [r["bases"][b] for r in reads]
        sl = [v["slope"] for v in vs if v["slope"] is not None]
        out["bases"][b] = {"westward": sum(v["class"] == "westward" for v in vs), "moving": sum(v["moving"] for v in vs),
                           "median_slope": float(np.median(sl)) if sl else None, "median_westmost": float(np.median([v["westmost"] for v in vs]))}
    return out


def limit(values, counts):
    """The smallest grid value at which the count reaches DETECT_MIN with every larger value also
    reaching it, None when no value does, 'not monotone' when a larger value falls back."""
    ok = [c >= DETECT_MIN for c in counts]
    if not any(ok):
        return None
    first = ok.index(True)
    return values[first] if all(ok[first:]) else "not monotone"


# ------------------------------------------------------------------ the tracks

def band_tracks(hist, month_of=None):
    """{band: {year: [track]}} with each track's calibrated injected depth at both widths. At the
    declared width of 2 degrees every track's ratio must be defined. At 4 degrees a track without
    one takes the smallest ratio defined for any track, the deepest trough and so the one most
    easily seen, and is marked."""
    month_of = A.month_index() if month_of is None else month_of
    out, flat = {}, []
    for (y, b), hs in hist.items():
        for h in hs:
            tr = track_of(h)
            tr["ratio"] = {f"{w:g}": calibration_ratio(tr, w, month_of) for w in (2.0, 4.0)}
            if tr["ratio"]["2"] is None:
                raise SystemExit(f"REFUSED: the track {tr} has no calibration at the declared width")
            out.setdefault(b, {}).setdefault(y, []).append(tr); flat.append(tr)
    floor4 = min(t["ratio"]["4"] for t in flat if t["ratio"]["4"] is not None)
    for tr in flat:
        tr["ratio_4_fallback"] = tr["ratio"]["4"] is None
        if tr["ratio_4_fallback"]:
            tr["ratio"]["4"] = floor4
        tr["injected"] = {w: tr["recorded"] / r for w, r in tr["ratio"].items()}
    return out


def load_histories(records_dir, histories_path):
    files = sorted(glob.glob(os.path.join(records_dir, "eval60_regime_????_??.json.gz")))
    recs = [SC.load_record(p) for p in files]
    SC.check_archive_identity(files, recs)
    by = {}
    for r in recs:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    with open(histories_path, encoding="utf-8") as fh:
        expected = W.stored_history_count(json.load(fh))
    return W.recompute_histories(by, expected), files


# ------------------------------------------------------------------ constructions

REFERENCE_TRACK = {"first_lon": 58.0, "speed": -2.3, "days": 6, "recorded": None}
REVIEW_TRAIN = {"wavelength": 30.0, "speed": 7.0}


def run_constructions(tracks):
    """K1, the review's train at 1 and 2 m/s on R6 through the injection path; K2, zero injected on
    R6; K3, packets P1 on a quiet background in the two bands holding most histories; K4, the
    reference track's calibration at both widths."""
    out = {}
    bg = [hundredths(L.construction_samples("R6", i)) for i in range(len(YEARS))]
    t = times(); x = LONS[None, :]
    k1 = {}
    for amp in (1.0, 2.0):
        seasons = []
        for i in range(len(YEARS)):
            ph = float(np.random.default_rng(L.PHASE_SEED + i).uniform(0.0, 2.0 * np.pi))
            seasons.append(inject(bg[i], amp * np.sin(2.0 * np.pi * (x + REVIEW_TRAIN["speed"] * t) / REVIEW_TRAIN["wavelength"] + ph)))
        k1[f"{amp:g}"] = passes(read(seasons, 17.25))
    out["K1"] = k1
    zero = read([inject(H, np.zeros(H.shape)) for H in bg], 17.25)
    plain = read([H / 100.0 for H in bg], 17.25)
    out["K2"] = {"identical": zero == plain}
    quiet_std = QUIET_STD / unit_noise_scale()
    k3 = {}
    for bi, band in enumerate(("15 to 19.5 N", "20 to 24.5 N")):
        cfg = {"part": "B", "name": "P1", "depth_mult": 1, "copies": 1, "width": 2.0}
        reads = []
        for p in range(PLACEMENTS):
            qb = matched_background(quiet_std, p)
            seasons = []
            for i, y in enumerate(YEARS):
                rng = np.random.default_rng([INJ_SEED, 999, bi, p, i])
                seasons.append(inject(qb[i], injected_field(cfg, rng, tracks.get(band, {}).get(y, []))))
            r = read(seasons, L.BAND_LAT[band])
            reads.append({"bases": r, "passes": passes(r)})
        k3[band] = summarize_reads(reads)
    out["K3"] = k3
    out["K4"] = {f"{w:g}": calibration_ratio(REFERENCE_TRACK, w) for w in (2.0, 4.0)}
    return out


def apply_rule(c):
    """Section 4: K1 reads no reach at any threshold at 1 m/s and reach at all three at 2 m/s; K2 is
    identical; K3 gives a moving ridge in at least DETECT_MIN placements at four or more of the six
    bases and a westward class in at least DETECT_MIN placements at one or more, in each band; K4's
    ratios lie strictly between 0.5 and 1."""
    fails = []
    if any(c["K1"]["1"].values()):
        fails.append("K1 reaches at 1 m/s")
    if not all(c["K1"]["2"].values()):
        fails.append("K1 does not reach at every threshold at 2 m/s")
    if not c["K2"]["identical"]:
        fails.append("K2 zero injection differs from the background")
    for band, s in c["K3"].items():
        mv = sum(v["moving"] >= DETECT_MIN for v in s["bases"].values())
        ww = sum(v["westward"] >= DETECT_MIN for v in s["bases"].values())
        if mv < 4 or ww < 1:
            fails.append(f"K3 {band} moving at {mv} bases, westward at {ww}")
    for w, r in c["K4"].items():
        if not 0.5 < r < 1.0:
            fails.append(f"K4 ratio at width {w} is {r}")
    return {"adequate": not fails, "failures": fails}


def check_constructions(path, tracks, rerun=None):
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps((rerun or run_constructions)(tracks)))
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if not verdict["adequate"]:
        raise SystemExit(f"REFUSED: the constructions do not pass the rule of section 4 ({verdict['failures']})")
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored summary disagrees with the rule applied to the stored cases")
    return rec


# ------------------------------------------------------------------ the archive

def check_zero_injection(by, lagreg):
    """The archive's background with nothing injected reads exactly as the lag-regression record
    at every base of BASES in every band, apart from the null, which is not computed here, and
    the record was made by the lag-regression script as loaded."""
    if lagreg["generated_by"]["sha256"] != L.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the lag-regression record was made by a different version of its script")
    for band, lat in L.BAND_LAT.items():
        seasons = [inject(hundredths(L.season_samples(by[y], band)), 0.0) for y in YEARS]
        res = L.annotate(L.summarize([L.filtered(S) for S in seasons], bases=BASES, lat=lat, null_bases=()))
        res = json.loads(json.dumps(res))
        for b in res:
            stored = {k: v for k, v in lagreg["results"][band]["bases"][b].items() if k != "shift_max_r"}
            if res[b] != stored:
                raise SystemExit(f"REFUSED: {band} base {b} with nothing injected differs from the lag-regression record")


def archive_results(by, tracks):
    """Every configuration on both backgrounds in every band, with the background scales."""
    u = unit_noise_scale()
    out = {"unit_noise_filtered_std": u, "bands": {}}
    cfgs = configurations()
    for band_i, (band, lat) in enumerate(L.BAND_LAT.items()):
        arch = [hundredths(L.season_samples(by[y], band)) for y in YEARS]
        fstd = pooled_filtered_std([L.filtered(H / 100.0) for H in arch], [col(b) for b in BASES])
        scale = float(np.mean([s for b, s in zip(BASES, fstd) if b in SCALE_BASES]))
        mstd = scale / u
        bgs = {"archive": [arch] * PLACEMENTS, "matched": [matched_background(mstd, p) for p in range(PLACEMENTS)]}
        tr_by_year = tracks.get(band, {})
        entry = {"filtered_std": dict(zip((f"{b:g}" for b in BASES), fstd)), "scale": scale, "matched_noise_std": mstd,
                 "packets": sum(len(v) for v in tr_by_year.values()), "backgrounds": {}}
        for bg_i, (bg_name, bg) in enumerate(bgs.items()):
            base_read = read([H / 100.0 for H in bg[0]], lat)
            rows = []
            for cfg_i, cfg in enumerate(cfgs):
                rows.append({"config": cfg, "summary": summarize_reads(run_config(cfg_i, cfg, bg, tr_by_year, band_i, lat, bg_i))})
            entry["backgrounds"][bg_name] = {"nothing_injected": base_read, "rows": rows, "limits": limits_of(rows)}
        out["bands"][band] = entry
    return out


def limits_of(rows):
    """Section 2.5's limits: per train speed and burst count, the amplitude at which the reading
    passes at 0.3; for the packets, the depth multiple and the count multiple at which some base
    is westward and at which some base is moving."""
    out = {}
    for c in TRAIN_SPEEDS:
        rs = [r for r in rows if r["config"]["part"] == "A" and r["config"]["speed"] == c]
        out[f"train {c:g}"] = limit([r["config"]["amp"] for r in rs], [r["summary"]["reading_passes"]["0.3"] for r in rs])
    for k in BURST_COUNTS:
        rs = [r for r in rows if r["config"]["part"] == "A2" and r["config"]["bursts"] == k]
        out[f"bursts {k}"] = limit([r["config"]["amp"] for r in rs], [r["summary"]["reading_passes"]["0.3"] for r in rs])
    pk = {r["config"]["name"]: r["summary"] for r in rows if r["config"]["part"] == "B"}
    for what in ("westward", "moving"):
        best = lambda s: max(v[what] for v in s["bases"].values())
        out[f"packets depth, {what}"] = limit([1, 2, 3, 4], [best(pk[n]) for n in ("P1", "P2", "P3", "P4")])
        out[f"packets count, {what}"] = limit([1, 2, 4, 8], [best(pk[n]) for n in ("P1", "C2", "C4", "C8")])
        out[f"packets wide, {what}"] = best(pk["W1"]) >= DETECT_MIN
    return out


# ------------------------------------------------------------------ readers added after the archive was read (section 7)

TRAIN_GAIN_4D = 0.99926   # the filter's amplitude gain at a 4-day period, as measured on the lag-regression review


def change_from_nothing(g):
    """For each packet configuration on one background, the bases whose westward class or moving
    ridge (each in at least DETECT_MIN placements) differs from the read with nothing injected,
    and the largest change of median slope at any base. Added after the archive was read,
    because on the archive the limits of section 2.5 are already met with nothing injected."""
    ref = g["nothing_injected"]
    out = {}
    for r in g["rows"]:
        c = r["config"]
        if c["part"] != "B":
            continue
        bs = r["summary"]["bases"]; d = {"new_westward": [], "lost_westward": [], "new_moving": [], "lost_moving": [], "max_slope_change": None}
        for b, v in bs.items():
            w0, m0 = ref[b]["class"] == "westward", ref[b]["moving"]
            w1, m1 = v["westward"] >= DETECT_MIN, v["moving"] >= DETECT_MIN
            for key, before, after in (("westward", w0, w1), ("moving", m0, m1)):
                if after and not before:
                    d[f"new_{key}"].append(b)
                if before and not after:
                    d[f"lost_{key}"].append(b)
            if v["median_slope"] is not None and ref[b]["slope"] is not None:
                ch = abs(v["median_slope"] - ref[b]["slope"])
                d["max_slope_change"] = ch if d["max_slope_change"] is None else max(d["max_slope_change"], ch)
        out[c["name"]] = d
    return out


def train_variance_fraction(a, scale):
    """The filtered variance of a continuous train of amplitude a over the background's filtered
    variance at 50 to 70 E."""
    return (a * TRAIN_GAIN_4D) ** 2 / 2.0 / scale ** 2


def post_hoc(res):
    out = {}
    for band, e in res["bands"].items():
        out[band] = {}
        for bg, g in e["backgrounds"].items():
            fr = {}
            for c in TRAIN_SPEEDS:
                v = g["limits"][f"train {c:g}"]
                fr[f"train {c:g}"] = train_variance_fraction(v, e["scale"]) if isinstance(v, float) else None
            out[band][bg] = {"packet_change": change_from_nothing(g), "train_variance_fraction": fr}
    return out


def post_hoc_sentences(ph):
    out = {}
    for band, d in ph.items():
        s = []
        for bg, g in d.items():
            p1 = g["packet_change"]["P1"]
            moved = [k for k in ("new_westward", "lost_westward", "new_moving", "lost_moving") if p1[k]]
            s.append(f"on the {bg} background, P1 changes " + ("no base's westward class or moving ridge" if not moved
                     else "; ".join(f"{k.replace('_', ' ')} at {', '.join(p1[k])} E" for k in moved))
                     + f" from the read with nothing injected, the median slope moving by at most {fmt(p1['max_slope_change'])} degrees per day")
        out[band] = s
    return out


# ------------------------------------------------------------------ text and main

def fmt(v, f=".2f"):
    return "" if v is None else format(v, f)


def class_and_slope(b, v):
    return f"{b} {v['class']} {fmt(v['slope'])}"


def readings(res):
    """Section 2.5's generated sentences, one set per band."""
    out = {}
    for band, e in res["bands"].items():
        s = []
        for bg in ("archive", "matched"):
            lim = e["backgrounds"][bg]["limits"]
            for c in TRAIN_SPEEDS:
                v = lim[f"train {c:g}"]
                s.append(f"on the {bg} background, a continuous train at {c:g} degrees per day "
                         + (f"fires the reading in at least {DETECT_MIN} of {PLACEMENTS} placements from {v:g} m/s" if isinstance(v, float)
                            else ("fires the reading at no amplitude tested" if v is None else "fires the reading, but not monotonically in amplitude")))
            p1 = next(r["summary"] for r in e["backgrounds"][bg]["rows"] if r["config"]["name"] == "P1")
            ww = [b for b, v in p1["bases"].items() if v["westward"] >= DETECT_MIN]
            mv = [b for b, v in p1["bases"].items() if v["moving"] >= DETECT_MIN]
            s.append(f"on the {bg} background, troughs on the band's {e['packets']} recorded history tracks at their recorded depths and count give "
                     f"a westward class at {', '.join(ww) + ' E' if ww else 'no base'} and a moving ridge at {', '.join(mv) + ' E' if mv else 'no base'} "
                     f"in at least {DETECT_MIN} of {PLACEMENTS} placements")
        out[band] = s
    return out


TRAIN_HEAD = ("| band | background | form | amplitude (m/s) | reading passes at 0.3, 0.25, 0.2 (of 5) | westward at 60, 65, 70 E (of 5) |\n"
              "| --- | --- | --- | --- | --- | --- |")
PACKET_HEAD = ("| band | background | packets (depth x, count x, width) | westward class by base 45 to 70 E (of 5) | moving ridge by base (of 5) | median slope by base (deg per day) |\n"
               "| --- | --- | --- | --- | --- | --- |")


def archive_text(res):
    L_ = ["# injection recovery, the archive (section 7)", "", f"unit construction noise, filtered std over 50 to 70 E, {res['unit_noise_filtered_std']:.4f} m/s", "",
          "| band | filtered std at 45, 50, 55, 60, 65, 70 E (m/s) | scale over 50 to 70 E | matched noise std | history tracks |", "| --- | --- | --- | --- | --- |"]
    for band, e in res["bands"].items():
        L_.append(f"| {band} | {', '.join(format(v, '.2f') for v in e['filtered_std'].values())} | {e['scale']:.3f} | {e['matched_noise_std']:.3f} | {e['packets']} |")
    L_ += ["", TRAIN_HEAD]
    for band, e in res["bands"].items():
        for bg, g in e["backgrounds"].items():
            for r in g["rows"]:
                c = r["config"]
                if c["part"] == "B":
                    continue
                form = f"train {c['speed']:g} deg/day" if c["part"] == "A" else f"{c['bursts']} bursts of {BURST_DAYS} days, {c['speed']:g} deg/day"
                rp = r["summary"]["reading_passes"]; bs = r["summary"]["bases"]
                L_.append(f"| {band} | {bg} | {form} | {c['amp']:g} | {rp['0.3']}, {rp['0.25']}, {rp['0.2']} | {bs['60']['westward']}, {bs['65']['westward']}, {bs['70']['westward']} |")
    L_ += ["", PACKET_HEAD]
    for band, e in res["bands"].items():
        for bg, g in e["backgrounds"].items():
            for r in g["rows"]:
                c = r["config"]
                if c["part"] != "B":
                    continue
                bs = r["summary"]["bases"]
                L_.append(f"| {band} | {bg} | {c['name']} ({c['depth_mult']}, {c['copies']}, {c['width']:g}) | {', '.join(str(v['westward']) for v in bs.values())} | "
                          f"{', '.join(str(v['moving']) for v in bs.values())} | {', '.join(fmt(v['median_slope']) for v in bs.values())} |")
    L_ += ["", "| band | background | limit | value |", "| --- | --- | --- | --- |"]
    for band, e in res["bands"].items():
        for bg, g in e["backgrounds"].items():
            for k, v in g["limits"].items():
                L_.append(f"| {band} | {bg} | {k} | {'none tested' if v is None else v} |")
    L_ += ["", "| band | background | nothing injected, class and slope by base 45 to 70 E |", "| --- | --- | --- |"]
    for band, e in res["bands"].items():
        for bg, g in e["backgrounds"].items():
            cells = "; ".join(class_and_slope(b, v) for b, v in g["nothing_injected"].items())
            L_.append(f"| {band} | {bg} | {cells} |")
    L_ += [""] + [f"- {band}, {s}" for band, ss in res["readings"].items() for s in ss]
    if "post_hoc" in res:
        L_ += ["", "## readers added after the archive was read (section 7)", "",
               "| band | background | packets | new westward | lost westward | new moving | lost moving | largest median slope change (deg per day) |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for band, d in res["post_hoc"].items():
            for bg, g in d.items():
                for name, c in g["packet_change"].items():
                    L_.append(f"| {band} | {bg} | {name} | {', '.join(c['new_westward'])} | {', '.join(c['lost_westward'])} | {', '.join(c['new_moving'])} | "
                              f"{', '.join(c['lost_moving'])} | {fmt(c['max_slope_change'])} |")
        L_ += ["", "| band | background | train speed (deg per day) | limit (m/s) | train filtered variance over the background's |", "| --- | --- | --- | --- | --- |"]
        for band, d in res["post_hoc"].items():
            for bg, g in d.items():
                for k, v in g["train_variance_fraction"].items():
                    lim = res["bands"][band]["backgrounds"][bg]["limits"][k]
                    L_.append(f"| {band} | {bg} | {k.split()[1]} | {'none tested' if lim is None else lim} | {fmt(v)} |")
        L_ += [""] + [f"- {band}, {s}" for band, ss in res["post_hoc_readings"].items() for s in ss]
    return "\n".join(L_) + "\n"


def constructions_text(results, verdict):
    L_ = ["# injection recovery, the constructions (section 3)", ""]
    L_.append(f"- K1, the review's train on R6, reading passes at 0.3, 0.25, 0.2: 1 m/s {list(results['K1']['1'].values())}, 2 m/s {list(results['K1']['2'].values())}")
    L_.append(f"- K2, nothing injected on R6 reads as R6: {results['K2']['identical']}")
    for band, s in results["K3"].items():
        bs = s["bases"]
        L_.append(f"- K3, {band}, quiet background, westward by base {', '.join(str(v['westward']) for v in bs.values())}, moving {', '.join(str(v['moving']) for v in bs.values())}, "
                  f"median slope {', '.join(fmt(v['median_slope']) for v in bs.values())}")
    L_.append(f"- K4, reference track calibration ratio at widths 2 and 4: {results['K4']['2']:.4f}, {results['K4']['4']:.4f}")
    L_ += ["", f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(L_) + "\n"


def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--xband", "--records", "--histories", "--lagreg", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if not (a.records and a.histories):
        raise SystemExit("REFUSED: --records and --histories are required")
    gen = {"script": "scripts/eastern_injection_recovery.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_INJECTION_RECOVERY_2026-10-21.md",
           "lagreg_script_sha256": L.SCRIPT_SHA256_AT_LOAD}
    hist, rfiles = load_histories(a.records, a.histories)
    tracks = band_tracks(hist)
    gen.update({"records": {os.path.basename(p): sha256_of(p) for p in rfiles}, "histories": {"path": a.histories, "sha256": sha256_of(a.histories)}})
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_inject_constructions{a.tag}.{e}") for e in ("json", "md"))
        refuse_existing([pj, pm])
        results = json.loads(json.dumps(run_constructions(tracks))); verdict = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict}, fh)
        text = constructions_text(results, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.xband and a.lagreg and a.constructions):
        raise SystemExit("REFUSED: --xband, --lagreg and --constructions are required for the archive")
    check_constructions(a.constructions, tracks)
    pj, pm = (os.path.join(a.out_dir, f"eval60_inject_{k}{a.tag}.{e}") for k, e in (("archive", "json"), ("tables", "md")))
    refuse_existing([pj, pm])
    files = sorted(glob.glob(os.path.join(a.xband, "eval60_xband_????_??.json.gz")))
    recs = []
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            recs.append(json.load(fh))
    X.check_xband_identity(files, recs)
    L.check_inputs(recs)
    by = {}
    for r in recs:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    with open(a.lagreg, encoding="utf-8") as fh:
        lagreg = json.load(fh)
    check_zero_injection(by, lagreg)
    res = archive_results(by, tracks)
    res["readings"] = readings(res)
    res["post_hoc"] = post_hoc(res)
    res["post_hoc_readings"] = post_hoc_sentences(res["post_hoc"])
    res["tracks"] = {b: {str(y): v for y, v in d.items()} for b, d in tracks.items()}
    gen.update({"xband": {os.path.basename(p): sha256_of(p) for p in files}, "lagreg": {"path": a.lagreg, "sha256": sha256_of(a.lagreg)},
                "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    text = archive_text(res)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
