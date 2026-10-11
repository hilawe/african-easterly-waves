#!/usr/bin/env python3
"""The synthetic stage of `EASTERN_TEMPORAL_IDENTITY_STAGE1_2026-10-11.md`, section 15,
with the 30-day window fixed and every rule of section 2 unchanged.

Step A runs the declared object-level checks with noise, using the feasibility check's
own generator, objects and measures. Step B, run only after step A passes, adds the
minimal linker for the declared continuity and ambiguity tests.

    python3 scripts/pilot_temporal_identity_stage1.py --step A --out <json>
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_temporal_identity_feasibility as F  # noqa: E402

HALF = F.WINDOWS["30-day"]
PHI, PAD, SIGMA_CELLS = 0.7, 8, 1.0
LEVELS = {"base": 0.3e-5, "stress": 0.5e-5}
REALIZATIONS, NEED = 20, 18
STEPS = range(F.SCORED.start - HALF, F.SCORED.stop + HALF)


def kernel_sd():
    d = np.zeros((41, 41)); d[20, 20] = 1.0
    return float(np.sqrt((ndimage.gaussian_filter(d, SIGMA_CELLS, mode="constant") ** 2).sum()))


KERNEL_SD = kernel_sd()


def unit_field(rng):
    w = rng.standard_normal((F.LAT.size + 2 * PAD, F.LON.size + 2 * PAD))
    return ndimage.gaussian_filter(w, SIGMA_CELLS, mode="nearest")[PAD:-PAD, PAD:-PAD] / KERNEL_SD


def noise(seed, sd):
    """Spatially smoothed, lag-one autoregressive noise at every step, stationary from the start."""
    rng = np.random.default_rng(seed)
    out, n = {}, None
    for k in STEPS:
        e = unit_field(rng)
        n = e if n is None else PHI * n + math.sqrt(1.0 - PHI ** 2) * e
        out[k] = sd * n
    return out


def seed_of(level, r):
    return 1000 * list(LEVELS).index(level) + r


def cases():
    """name -> (field function, centers, required_from, scored structures)."""
    steady = 2e-5 * np.exp(-(((F.LONG - 40) / 7.5) ** 2 + ((F.LATG - 10) / 3) ** 2))
    amp2 = lambda k: 0.0 if k < 36 else (3e-5 if k >= 44 else 3e-5 * (k - 36) / 8.0)  # noqa: E731
    d5amp = lambda k, v: 3e-5 * (1 - 0.5 * math.exp(-((F.track(k, v) - 40.0) / 4.0) ** 2))  # noqa: E731
    out = {}
    for v in (0.5, 1.75):
        c = lambda k, v=v: (10.0, F.track(k, v))  # noqa: E731
        out[f"D1 v={v}"] = (lambda k, v=v: F.vortex(10.0, F.track(k, v), 3e-5), {"vortex": c}, None)
        out[f"D5 v={v}"] = (lambda k, v=v: F.vortex(10.0, F.track(k, v), d5amp(k, v)), {"vortex": c}, None)
        out[f"D3a v={v}"] = (lambda k, v=v: F.vortex(10.0, F.track(k, v), 3e-5) + steady, {"vortex": c}, None)
        out[f"D3b v={v}"] = (lambda k, v=v: F.vortex(10.0, F.track(k, v), 3e-5) + F.vortex(20.0, 45.0, 3e-5 * (0.5 + 0.5 * math.sin(2 * math.pi * k / 16))),
                             {"vortex": c, "pulsing": lambda k: (20.0, 45.0)}, None)
        out[f"D7 v={v}"] = (lambda k, v=v: F.vortex(10.0, F.track(k, v), 3e-5) + steady * (1 + 0.5 * math.sin(2 * math.pi * k / 80)), {"vortex": c}, None)
    out["D2 v=1.75"] = (lambda k: F.vortex(10.0, F.track(k, 1.75), 3e-5) + F.vortex(10.0, F.track(k, 1.75) + 25.0, 3e-5),
                        {"vortex": lambda k: (10.0, F.track(k, 1.75)), "following": lambda k: (10.0, F.track(k, 1.75) + 25.0)}, None)
    out["D4 v=0.5"] = (lambda k: F.vortex(8.0, F.track(k, 0.5), 3e-5) + F.vortex(14.0, F.track(k, 0.5), amp2(k)),
                       {"persisting": lambda k: (8.0, F.track(k, 0.5)), "second": lambda k: (14.0, F.track(k, 0.5))}, {"second": 44})
    out["D6 v=0.5"] = (lambda k: F.vortex(6.0, F.track(k, 0.5), 3e-5) + F.vortex(14.0, F.track(k, 0.5), amp2(k)),
                       {"persisting": lambda k: (6.0, F.track(k, 0.5)), "second": lambda k: (14.0, F.track(k, 0.5))}, {"second": 44})
    return out


def moving_ok(s):
    return s["steps_with_object"] == s["eligible_steps"] and s["pairs_below_quarter"] == 0 and s["pairs_above_3deg"] == 0


def step_a_case(name, res):
    """Whether one realization meets its object-level requirement, and its brief record."""
    st = res["structures"]
    brief = {k: {x: s[x] for x in ("steps_with_object", "eligible_steps", "pairs_below_quarter", "pairs_above_3deg", "min_overlap", "max_displacement_deg")}
             for k, s in st.items()}
    if name.startswith("D4"):
        sep = F.separate(res, "persisting", "second")
        return None, {**brief, "separation": sep}
    if name.startswith("D6"):
        sep = F.separate(res, "persisting", "second")
        ok = moving_ok(st["persisting"]) and moving_ok(st["second"]) and sep["steps_both_exist"] == sep["steps_both_eligible"] == sep["steps_distinct"]
        return ok, {**brief, "separation": sep}
    if name.startswith("D3b"):
        sep = F.separate(res, "vortex", "pulsing")
        return moving_ok(st["vortex"]) and sep["steps_distinct"] == sep["steps_both_exist"], {**brief, "separation": sep}
    if name.startswith("D2"):
        return moving_ok(st["vortex"]) and moving_ok(st["following"]), brief
    return moving_ok(st["vortex"]), brief


def paired_d7(r7, r1):
    """D7's vortex object beside D1's in the same realization: centroid offsets, size changes,
    connections by the 5-degree rule and background objects."""
    a, b = r7["structures"]["vortex"]["rows"], r1["structures"]["vortex"]["rows"]
    off, dsize = [], []
    for x, y in zip(a, b):
        if x["eligible"] and x["anomaly_centroid"] and y["anomaly_centroid"]:
            off.append(math.hypot(x["anomaly_centroid"][0] - y["anomaly_centroid"][0], x["anomaly_centroid"][1] - y["anomaly_centroid"][1]))
            dsize.append(x["anomaly_cells"] - y["anomaly_cells"])
    return {"max_centroid_offset_deg": round(max(off), 3) if off else None, "mean_centroid_offset_deg": round(float(np.mean(off)), 3) if off else None,
            "size_change_cells": [min(dsize), max(dsize)] if dsize else None, "connection_steps_5deg": F.unwanted(r7), "background": F.background_objects(r7)}


def noise_check(seed=999):
    """The generator against its declaration on one long draw: standard deviation, lag-one
    correlation, and spatial correlation at 2 cells (exp(-1) for a one-cell kernel)."""
    nz = noise(seed, 1.0)
    x = np.stack([nz[k] for k in STEPS])
    lag = float(np.mean([np.corrcoef(x[:-1, i, j], x[1:, i, j])[0, 1] for i in range(5, 26, 5) for j in range(5, 38, 5)]))
    sp = float(np.corrcoef(x[:, 10:21, 10:31].ravel(), x[:, 12:23, 10:31].ravel())[0, 1])
    return {"sd": round(float(x.std()), 3), "lag_one": round(lag, 3), "corr_at_2_cells": round(sp, 3), "expected": {"sd": 1.0, "lag_one": PHI, "corr_at_2_cells": round(math.exp(-1), 3)}}


def step_a():
    out = {"levels": {}, "noise_check": noise_check()}
    cs = cases()
    for level, sd in LEVELS.items():
        per = {n: {"met": 0, "realizations": []} for n in cs}
        d7pairs = {"D7 v=0.5": [], "D7 v=1.75": []}
        for r in range(REALIZATIONS):
            nz = noise(seed_of(level, r), sd)
            results = {}
            for name, (fn, centers, req) in cs.items():
                f = {k: fn(k) + nz[k] for k in STEPS}
                res = F.run_case(name, f, centers, req, anom=F.anomaly_w(f, HALF))
                ok, brief = step_a_case(name, res)
                per[name]["realizations"].append({"r": r, "met": ok, **brief})
                per[name]["met"] += int(bool(ok))
                results[name] = res
            for v in ("0.5", "1.75"):
                d7pairs[f"D7 v={v}"].append({"r": r, **paired_d7(results[f"D7 v={v}"], results[f"D1 v={v}"])})
        scored = [n for n in cs if not n.startswith("D4")]
        out["levels"][level] = {"met_of_20": {n: per[n]["met"] for n in scored}, "D4_one_component_steps": [x["separation"]["steps_both_exist"] - x["separation"]["steps_distinct"]
                                                                                                         for x in per["D4 v=0.5"]["realizations"]],
                                "passes": all(per[n]["met"] >= NEED for n in scored), "cases": per, "D7_beside_D1": d7pairs}
    out["step_a_passes"] = out["levels"]["base"]["passes"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", choices=("A",), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    t0 = time.time()
    res = step_a()
    res.update({"generated_by": "scripts/pilot_temporal_identity_stage1.py --step A", "window_half_steps": HALF, "noise": {"phi": PHI, "sigma_cells": SIGMA_CELLS,
                "pad": PAD, "kernel_sd": KERNEL_SD, "levels": LEVELS}, "seconds": round(time.time() - t0, 1)})
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({lv: {k: v for k, v in x.items() if k in ("met_of_20", "D4_one_component_steps", "passes")} for lv, x in res["levels"].items()}, indent=1))
    print("step A passes:", res["step_a_passes"], "in", res["seconds"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
