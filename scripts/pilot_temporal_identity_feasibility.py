#!/usr/bin/env python3
"""The noise-free feasibility check of `EASTERN_TEMPORAL_IDENTITY_STAGE1_2026-10-11.md`,
section 9. It generates the declared synthetic fields, forms the centered 41-step mean
anomaly, finds objects blind by eight-connected components, and only then locates the
component nearest each prescribed center to report peaks, cell counts, consecutive
overlaps, displacements and separation. No linker, no noise and no real data.

    python3 scripts/pilot_temporal_identity_feasibility.py --out <json>
"""
import argparse
import json
import math
import os

import numpy as np
from scipy import ndimage

LAT = np.arange(0.0, 30.5, 1.0)
LON = np.arange(20.0, 62.5, 1.0)
LATG, LONG = np.meshgrid(LAT, LON, indexing="ij")
STEPS, SCORED, HALF = 81, range(20, 61), 20
THR, MIN_CELLS, RADIUS = 1.0e-5, 4, 2.5
INTERIOR = {"lat": (3.0, 27.0), "lon": (23.0, 59.0)}


def vortex(lat, lon, amp):
    return amp * np.exp(-(((LATG - lat) ** 2 + (LONG - lon) ** 2) / RADIUS ** 2))


def track(k, v):
    return 40.0 - v * (k - 40)


def anomaly(fields):
    """Field minus its centered 41-step mean, at the scored steps."""
    return {k: fields[k] - fields[k - HALF:k + HALF + 1].mean(axis=0) for k in SCORED}


def components(field):
    """Blind objects: eight-connected cells above the threshold, at least MIN_CELLS."""
    lab, n = ndimage.label(field > THR, structure=np.ones((3, 3), int))
    sizes = ndimage.sum(np.ones_like(field), lab, index=range(1, n + 1))
    keep = {i + 1 for i, c in enumerate(sizes) if c >= MIN_CELLS}
    return np.where(np.isin(lab, list(keep)), lab, 0)


def nearest_cell(lat, lon):
    return int(np.argmin(np.abs(LAT - lat))), int(np.argmin(np.abs(LON - lon)))


def interior(lat, lon):
    return INTERIOR["lat"][0] <= lat <= INTERIOR["lat"][1] and INTERIOR["lon"][0] <= lon <= INTERIOR["lon"][1]


def peak_near(field, lat, lon, within=3.0):
    m = (np.abs(LATG - lat) <= within) & (np.abs(LONG - lon) <= within)
    return float(field[m].max()) if m.any() else None


def describe(field, lab, lat, lon):
    i, j = nearest_cell(lat, lon)
    c = int(lab[i, j])
    out = {"peak_near": peak_near(field, lat, lon), "component": c}
    if c:
        mask = lab == c
        out.update({"cells": int(mask.sum()), "centroid": [round(float(LATG[mask].mean()), 3), round(float(LONG[mask].mean()), 3)], "mask": mask})
    else:
        out.update({"cells": 0, "centroid": None, "mask": None})
    return out


def run_case(name, fields, centers, required_from=None, anom=None):
    """centers: {label: function k -> (lat, lon)}; required_from: {label: first required step}.
    `fields` is indexed by step; `anom`, when given, is the anomaly at the scored steps."""
    anom = anomaly(fields) if anom is None else anom
    res = {"case": name, "structures": {}}
    labs = {k: (components(fields[k]), components(anom[k])) for k in SCORED}
    for label, cen in centers.items():
        rows, prev = [], None
        start = (required_from or {}).get(label, 0)
        for k in SCORED:
            lat, lon = cen(k)
            elig = interior(lat, lon) and k >= start
            raw, an = describe(fields[k], labs[k][0], lat, lon), describe(anom[k], labs[k][1], lat, lon)
            row = {"step": k, "center": [lat, round(lon, 3)], "eligible": bool(elig), "raw_peak": raw["peak_near"], "raw_cells": raw["cells"],
                   "anomaly_peak": an["peak_near"], "anomaly_cells": an["cells"], "anomaly_centroid": an["centroid"]}
            if elig and prev is not None and prev["mask"] is not None and an["mask"] is not None:
                shared = int((prev["mask"] & an["mask"]).sum())
                row["overlap_fraction_of_smaller"] = round(shared / min(int(prev["mask"].sum()), int(an["mask"].sum())), 3)
                row["centroid_displacement_deg"] = round(math.hypot(an["centroid"][0] - prev["centroid"][0], an["centroid"][1] - prev["centroid"][1]), 3)
            row["_component"] = an["component"]
            row["_far_cell_deg"] = None if an["mask"] is None else float(np.sqrt((LATG[an["mask"]] - lat) ** 2 + (LONG[an["mask"]] - lon) ** 2).max())
            rows.append(row)
            prev = an if elig else None
        e = [r for r in rows if r["eligible"]]
        ov = [r["overlap_fraction_of_smaller"] for r in e if "overlap_fraction_of_smaller" in r]
        dp = [r["centroid_displacement_deg"] for r in e if "centroid_displacement_deg" in r]
        res["structures"][label] = {
            "eligible_steps": len(e), "censored_steps": len(rows) - len(e),
            "eligible_interval": [e[0]["step"], e[-1]["step"]] if e else None,
            "steps_with_object": sum(1 for r in e if r["anomaly_cells"] >= MIN_CELLS),
            "raw_peak_range": [min(r["raw_peak"] for r in e), max(r["raw_peak"] for r in e)] if e else None,
            "anomaly_peak_range": [min(r["anomaly_peak"] for r in e), max(r["anomaly_peak"] for r in e)] if e else None,
            "raw_cells_range": [min(r["raw_cells"] for r in e), max(r["raw_cells"] for r in e)] if e else None,
            "anomaly_cells_range": [min(r["anomaly_cells"] for r in e), max(r["anomaly_cells"] for r in e)] if e else None,
            "min_overlap": min(ov) if ov else None, "consecutive_pairs": len(ov), "pairs_below_quarter": sum(1 for x in ov if x < 0.25),
            "max_displacement_deg": max(dp) if dp else None, "pairs_above_3deg": sum(1 for x in dp if x > 3.0),
            "rows": rows}
    res["_labs"] = labs
    return res


def separate(res, a, b, use="anomaly"):
    """Steps (eligible for both) where a's and b's objects exist and are distinct components."""
    ra, rb = res["structures"][a]["rows"], res["structures"][b]["rows"]
    both = [(x, y) for x, y in zip(ra, rb) if x["eligible"] and y["eligible"]]
    exist = [(x, y) for x, y in both if x["_component"] and y["_component"]]
    return {"steps_both_eligible": len(both), "steps_both_exist": len(exist), "steps_distinct": sum(1 for x, y in exist if x["_component"] != y["_component"])}


def raw_separation(fields, lat_a, lat_b, lon_of_k, steps):
    """Whether the case's own raw field has below-threshold cells between two vortices on one
    meridian: the minimum raw value along the meridian between the centers, per step."""
    out = []
    for k in steps:
        j = nearest_cell(lat_a, lon_of_k(k))[1]
        i0, i1 = nearest_cell(lat_a, 0)[0], nearest_cell(lat_b, 0)[0]
        out.append(float(fields[k][min(i0, i1):max(i0, i1) + 1, j].min()))
    return out


def gen(fn):
    return np.stack([fn(k) for k in range(STEPS)])


def above_cells(field, lat, lon):
    """Cells of the eight-connected above-threshold region holding the cell nearest the
    center, with no minimum size, so a region the 4-cell rule drops is still counted."""
    lab, _ = ndimage.label(field > THR, structure=np.ones((3, 3), int))
    c = int(lab[nearest_cell(lat, lon)])
    return (lab == c) if c else None


def mechanism():
    """The D5 failures, read without the minimum size: the above-threshold cells near the
    dip at the slow speed, and raw against anomaly overlaps at the fast speed's dip."""
    out = {}
    for v, steps in ((0.5, range(36, 45)), (1.75, range(38, 43))):
        f = gen(lambda k, v=v: vortex(10.0, track(k, v), 3e-5 * (1 - 0.5 * math.exp(-((track(k, v) - 40.0) / 4.0) ** 2))))
        an = anomaly(f)
        rows, prev = [], None
        for k in steps:
            lon = track(k, v)
            mr, ma = above_cells(f[k], 10.0, lon), above_cells(an[k], 10.0, lon)
            row = {"step": k, "center_lon": lon, "raw_peak": round(peak_near(f[k], 10.0, lon) / 1e-5, 3), "anomaly_peak": round(peak_near(an[k], 10.0, lon) / 1e-5, 3),
                   "raw_above_cells": int(mr.sum()) if mr is not None else 0, "anomaly_above_cells": int(ma.sum()) if ma is not None else 0}
            if prev is not None:
                for key, m, pm in (("raw", mr, prev[0]), ("anomaly", ma, prev[1])):
                    row[f"{key}_overlap_fraction_of_smaller"] = None if m is None or pm is None else round(int((m & pm).sum()) / min(int(m.sum()), int(pm.sum())), 3)
            rows.append(row)
            prev = (mr, ma)
        out[f"D5 v={v}"] = rows
    return out


WINDOWS = {"10-day": 20, "30-day": 60}                               # half-widths in six-hourly steps
FAR_DEG = 5.0


def seq(fn, half):
    """The analytic field at every step a complete centered window needs for the scored steps."""
    return {k: fn(k) for k in range(SCORED.start - half, SCORED.stop + half)}


def anomaly_w(f, half):
    return {k: f[k] - np.stack([f[j] for j in range(k - half, k + half + 1)]).mean(axis=0) for k in SCORED}


def background_objects(res, label="vortex"):
    """Objects other than the moving vortex's, per scored step: how many steps hold one, the
    largest, and the largest centroid shift of the largest one between consecutive steps."""
    labs, rows = res["_labs"], res["structures"][label]["rows"]
    steps, biggest, prev, shift = 0, 0, None, 0.0
    for row in rows:
        lab = labs[row["step"]][1]
        others = [c for c in np.unique(lab) if c and c != row["_component"]]
        if not others:
            prev = None
            continue
        steps += 1
        big = max(others, key=lambda c: int((lab == c).sum()))
        m = lab == big
        biggest = max(biggest, int(m.sum()))
        cen = (float(LATG[m].mean()), float(LONG[m].mean()))
        if prev is not None:
            shift = max(shift, math.hypot(cen[0] - prev[0], cen[1] - prev[1]))
        prev = cen
    return {"steps_with_background_object": steps, "largest_background_cells": biggest, "largest_background_shift_deg": round(shift, 3)}


def unwanted(res, label="vortex"):
    rows = [r for r in res["structures"][label]["rows"] if r["eligible"]]
    return sum(1 for r in rows if r["_far_cell_deg"] is not None and r["_far_cell_deg"] > FAR_DEG)


def summary_of(res):
    out = {}
    for k, s in res["structures"].items():
        out[k] = {x: s[x] for x in ("eligible_interval", "eligible_steps", "steps_with_object", "raw_peak_range", "anomaly_peak_range", "raw_cells_range",
                                    "anomaly_cells_range", "min_overlap", "consecutive_pairs", "pairs_below_quarter", "max_displacement_deg", "pairs_above_3deg")}
        out[k]["unwanted_connection_steps"] = unwanted(res, k)
    return out


def background_response(periods_days=(5, 10, 15, 20, 30, 60)):
    """Fraction of a stationary sinusoidal background left in the anomaly by each window."""
    out = {}
    for name, half in WINDOWS.items():
        n = 2 * half + 1
        out[name] = {str(p): round(1.0 - math.sin(math.pi * n / (4 * p)) / (n * math.sin(math.pi / (4 * p))), 3) for p in periods_days}
    return out


def windows(previous):
    """The section 12 comparison: every case under both windows at the same scored steps."""
    steady_f = 2e-5 * np.exp(-(((LONG - 40) / 7.5) ** 2 + ((LATG - 10) / 3) ** 2))
    amp2 = lambda k: 0.0 if k < 36 else (3e-5 if k >= 44 else 3e-5 * (k - 36) / 8.0)  # noqa: E731
    d5amp = lambda k, v: 3e-5 * (1 - 0.5 * math.exp(-((track(k, v) - 40.0) / 4.0) ** 2))  # noqa: E731
    specs = []
    for v in (0.5, 1.75):
        c = lambda k, v=v: (10.0, track(k, v))  # noqa: E731
        specs += [(f"D1 v={v}", lambda k, v=v: vortex(10.0, track(k, v), 3e-5), {"vortex": c}, None),
                  (f"D5 v={v}", lambda k, v=v: vortex(10.0, track(k, v), d5amp(k, v)), {"vortex": c}, None),
                  (f"D3a v={v}", lambda k, v=v: vortex(10.0, track(k, v), 3e-5) + steady_f, {"vortex": c}, None),
                  (f"D3b v={v}", lambda k, v=v: vortex(10.0, track(k, v), 3e-5) + vortex(20.0, 45.0, 3e-5 * (0.5 + 0.5 * math.sin(2 * math.pi * k / 16))),
                   {"vortex": c, "pulsing": lambda k: (20.0, 45.0)}, None),
                  (f"D7 v={v}", lambda k, v=v: vortex(10.0, track(k, v), 3e-5) + steady_f * (1 + 0.5 * math.sin(2 * math.pi * k / 80)), {"vortex": c}, None)]
    specs += [("D4 v=0.5", lambda k: vortex(8.0, track(k, 0.5), 3e-5) + vortex(14.0, track(k, 0.5), amp2(k)),
               {"persisting": lambda k: (8.0, track(k, 0.5)), "second": lambda k: (14.0, track(k, 0.5))}, {"second": 44}),
              ("D6 v=0.5", lambda k: vortex(6.0, track(k, 0.5), 3e-5) + vortex(14.0, track(k, 0.5), amp2(k)),
               {"persisting": lambda k: (6.0, track(k, 0.5)), "second": lambda k: (14.0, track(k, 0.5))}, {"second": 44})]
    prev = {c["case"]: c for c in previous["cases"]}
    out = {"background_response": background_response(), "cases": {}}
    raw6 = seq(specs[-1][1], 0)
    gap = [float(raw_separation(raw6, 6.0, 14.0, lambda k: track(k, 0.5), [k])[0]) for k in range(44, 61)]
    raw_distinct = 0
    for k in range(44, 61):
        lab = components(raw6[k])
        a, b = lab[nearest_cell(6.0, track(k, 0.5))], lab[nearest_cell(14.0, track(k, 0.5))]
        raw_distinct += int(bool(a) and bool(b) and a != b)
    out["D6_raw_check"] = {"min_between_by_step": [round(x / 1e-5, 3) for x in gap], "all_below_threshold": bool(max(gap) < THR), "raw_distinct_steps": raw_distinct, "required_steps": 17}
    for name, fn, centers, req in specs:
        out["cases"][name] = {}
        for wname, half in WINDOWS.items():
            f = seq(fn, half)
            res = run_case(name, f, centers, req, anom=anomaly_w(f, half))
            if wname == "10-day" and name in prev:                       # the closed section 10 result, reproduced
                for lab_, st in res["structures"].items():
                    old = prev[name]["structures"][lab_]["rows"]
                    new = [{k2: x for k2, x in r.items() if not k2.startswith("_")} for r in st["rows"]]
                    if json.loads(json.dumps(new)) != old:
                        raise SystemExit(f"REFUSED: the 10-day {name} {lab_} does not reproduce section 10")
            entry = {"structures": summary_of(res)}
            if "pulsing" in centers or "second" in centers:
                a_, b_ = list(centers)[:2]
                entry["separation"] = separate(res, a_, b_)
            if name.startswith(("D3a", "D7")):
                entry["background"] = background_objects(res)
            out["cases"][name][wname] = entry
    s30 = {n: out["cases"][n]["30-day"] for n in out["cases"]}
    def ok_moving(e):
        s = e["structures"]["vortex"]
        return s["steps_with_object"] == s["eligible_steps"] and s["pairs_below_quarter"] == 0 and s["pairs_above_3deg"] == 0
    checks = {}
    for v in ("0.5", "1.75"):
        checks[f"D1 v={v}"] = ok_moving(s30[f"D1 v={v}"])
        checks[f"D5 v={v}"] = ok_moving(s30[f"D5 v={v}"])
        checks[f"D3a v={v}"] = s30[f"D3a v={v}"]["background"]["steps_with_background_object"] == 0
        sep = s30[f"D3b v={v}"]["separation"]
        checks[f"D3b v={v}"] = sep["steps_distinct"] == sep["steps_both_exist"]
        checks[f"D7 v={v}"] = ok_moving(s30[f"D7 v={v}"]) and s30[f"D7 v={v}"]["structures"]["vortex"]["unwanted_connection_steps"] == 0
    if out["D6_raw_check"]["all_below_threshold"]:
        sep = s30["D6 v=0.5"]["separation"]
        checks["D6 v=0.5"] = sep["steps_both_exist"] == sep["steps_both_eligible"] and sep["steps_distinct"] == sep["steps_both_eligible"]
    else:
        checks["D6 v=0.5"] = "inconsistent, not scored"
    out["checks_30_day"] = checks
    out["clears_prerequisites_30_day"] = all(v is True for v in checks.values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mechanism", action="store_true", help="write only the D5 mechanism reading")
    ap.add_argument("--windows", help="the section 10 record; write the section 12 window comparison")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    if args.windows:
        prev = json.load(open(args.windows))
        w = windows(prev)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": "scripts/pilot_temporal_identity_feasibility.py --windows", "previous": args.windows, "threshold": THR,
                       "min_cells": MIN_CELLS, "windows_half_steps": WINDOWS, "far_cell_deg": FAR_DEG, **w}, fh, indent=1)
        print(json.dumps({k: w[k] for k in ("background_response", "D6_raw_check", "checks_30_day", "clears_prerequisites_30_day")}, indent=1))
        return 0
    if args.mechanism:
        m = mechanism()
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": "scripts/pilot_temporal_identity_feasibility.py --mechanism", "threshold": THR, "rows": m}, fh, indent=1)
        print(json.dumps(m, indent=0))
        return 0
    # internal checks of the transform: a steady field has zero anomaly
    steady = gen(lambda k: 2e-5 * np.exp(-(((LONG - 40) / 7.5) ** 2 + ((LATG - 10) / 3) ** 2)))
    assert max(float(np.abs(a).max()) for a in anomaly(steady).values()) < 1e-18
    cases = []
    for v in (0.5, 1.75):
        c = lambda k, v=v: (10.0, track(k, v))  # noqa: E731
        d1 = gen(lambda k, v=v: vortex(10.0, track(k, v), 3e-5))
        cases.append(run_case(f"D1 v={v}", d1, {"vortex": c}))
        d5 = gen(lambda k, v=v: vortex(10.0, track(k, v), 3e-5 * (1 - 0.5 * math.exp(-((track(k, v) - 40.0) / 4.0) ** 2))))
        cases.append(run_case(f"D5 v={v}", d5, {"vortex": c}))
        d3a = gen(lambda k, v=v: vortex(10.0, track(k, v), 3e-5)) + steady
        r = run_case(f"D3a v={v}", d3a, {"vortex": c})
        anom_d1, anom_d3a = anomaly(d1), anomaly(d3a)
        r["anomaly_equals_D1_max_abs_diff"] = max(float(np.abs(anom_d1[k] - anom_d3a[k]).max()) for k in SCORED)
        r["objects_apart_from_vortex"] = sum(1 for k in SCORED
                                            for cmp in set(np.unique(r["_labs"][k][1])) - {0, r["structures"]["vortex"]["rows"][k - 20]["_component"]})
        r["raw_objects_apart_from_vortex"] = sum(1 for k in SCORED
                                                for cmp in set(np.unique(r["_labs"][k][0])) - {0} if not r["_labs"][k][0][nearest_cell(10.0, track(k, v))] == cmp)
        cases.append(r)
        d3b = gen(lambda k, v=v: vortex(10.0, track(k, v), 3e-5) + vortex(20.0, 45.0, 3e-5 * (0.5 + 0.5 * math.sin(2 * math.pi * k / 16))))
        r = run_case(f"D3b v={v}", d3b, {"vortex": c, "pulsing": lambda k: (20.0, 45.0)})
        r["separation"] = separate(r, "vortex", "pulsing")
        cases.append(r)
    amp2 = lambda k: 0.0 if k < 36 else (3e-5 if k >= 44 else 3e-5 * (k - 36) / 8.0)  # noqa: E731
    d4 = gen(lambda k: vortex(8.0, track(k, 0.5), 3e-5) + vortex(14.0, track(k, 0.5), amp2(k)))
    r = run_case("D4 v=0.5", d4, {"persisting": lambda k: (8.0, track(k, 0.5)), "second": lambda k: (14.0, track(k, 0.5))}, {"second": 44})
    r["separation"] = separate(r, "persisting", "second")
    r["raw_min_between_from_44"] = [round(x / 1e-5, 3) for x in raw_separation(d4, 8.0, 14.0, lambda k: track(k, 0.5), range(44, 61))]
    anom4 = anomaly(d4)
    r["anomaly_min_between_from_44"] = [round(x / 1e-5, 3) for x in raw_separation({k: anom4[k] for k in range(44, 61)}, 8.0, 14.0, lambda k: track(k, 0.5), range(44, 61))]
    cases.append(r)
    for r in cases:
        r.pop("_labs")
        for s in r["structures"].values():
            for row in s["rows"]:
                row.pop("_component", None)
                row.pop("_far_cell_deg", None)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": "scripts/pilot_temporal_identity_feasibility.py", "threshold": THR, "min_cells": MIN_CELLS, "radius_deg": RADIUS,
                   "cases": cases}, fh, indent=1)
    for r in cases:
        print(r["case"], {k: {x: s[x] for x in ("eligible_interval", "steps_with_object", "eligible_steps", "raw_peak_range", "anomaly_peak_range", "raw_cells_range",
                                                 "anomaly_cells_range", "min_overlap", "pairs_below_quarter", "max_displacement_deg")}
                          for k, s in r["structures"].items()},
              {k: r[k] for k in r if k in ("separation", "anomaly_equals_D1_max_abs_diff", "objects_apart_from_vortex", "raw_objects_apart_from_vortex",
                                            "raw_min_between_from_44", "anomaly_min_between_from_44")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
