#!/usr/bin/env python3
"""The stage D selection rule of the displacement effort, mechanical and declared before
any inspection (`EASTERN_STAGE_D_EVALUATION_2026-10-17.md`, section 2). It reads the saved
season filtered-view records (`pilot_season_filtered_view.py`) and nothing else.

The season: among the candidate records, the largest absolute value of the filtered
northern-band mean at longitudes from 40 to 60 E. The streak: for each sign, a maximal run
of consecutive supported times at which that sign's extremum within 40 to 60 E has
magnitude at least 1.0 m/s and moves by at most 2.5 degrees between consecutive times; it
qualifies when it spans at least 48 elapsed hours and the least-squares slope of its
longitude against time is westward by at least 1 degree per day; the qualifying streak
with the largest amplitude is selected, the earlier on a tie. The window: one day either
side of the streak, capped at eight days centered on it. A season with no qualifying
streak falls through to the next season by amplitude, and the fallback is recorded.

    python3 scripts/pilot_stage_d_select.py RECORD... --exclude 1990 2005 2006 --out OUT.json

Corrected for stage E (`EASTERN_STAGE_E_EVALUATION_2026-10-17.md`, section 2) on 2026-10-17,
after the stage D record (section 6, corrected at section 15) recorded two latent defects:
the slope's time axis assumed a quarter day per sample without reading the record's
cadence, and the eight-day cap centered on a streak could put a window bound between the
record's time steps. The cadence is now read from the record and a non-uniform one is
refused, the slope uses real elapsed hours, the window is chosen among the record's own
time steps (the cap is a whole number of steps centered on the streak's middle sample), and
the output names the band's latitude bounds from the record it read. The rule is otherwise
unchanged. The stage D selection record (script digest 3fa6ab051c7df6c4) stands as produced.
"""
import argparse
import datetime as dt
import hashlib
import json
import os

import numpy as np

LON_RANGE = (40.0, 60.0)
MIN_MAGNITUDE = 1.0
MAX_STEP_DEG = 2.5
MIN_HOURS = 48
MIN_WEST_RATE = 1.0
PAD_HOURS = 24
MAX_WINDOW_HOURS = 8 * 24
BAND = "northern"


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _parse(t):
    return dt.datetime.strptime(t, "%Y-%m-%dT%H")


def hours_between(a, b):
    return (_parse(b) - _parse(a)).total_seconds() / 3600.0


def cadence_hours(record):
    """The one uniform step of the record's times, in hours. A record whose consecutive
    times are not all the same positive step is refused rather than assumed six-hourly."""
    times = record["times"]
    steps = sorted({round(hours_between(a, b), 6) for a, b in zip(times, times[1:])})
    if len(steps) != 1 or steps[0] <= 0:
        raise SystemExit(f"REFUSED: the record's times are not on one uniform cadence (steps {steps} h)")
    return float(steps[0])


def _array(record):
    lons = np.asarray(record["lons"], float)
    series = np.array([[np.nan if x is None else x for x in row] for row in record["bands"][BAND]["series"]], float)
    inside = (lons >= LON_RANGE[0]) & (lons <= LON_RANGE[1])
    return lons, series, inside


def season_amplitude(record):
    lons, series, inside = _array(record)
    sub = series[:, inside]
    return float(np.nanmax(np.abs(sub))) if np.isfinite(sub).any() else 0.0


def streaks(record):
    """Every maximal run for each sign, qualifying or not, in time order."""
    lons, series, inside = _array(record)
    times = record["times"]
    cadence_hours(record)
    sub = series[:, inside]; sub_lons = lons[inside]
    out = []
    for sign in (1, -1):
        ext_lon, ext_val = [], []
        for row in sub:
            if not np.isfinite(row).any():
                ext_lon.append(None); ext_val.append(None); continue
            vals = row * sign
            j = int(np.nanargmax(vals))
            v = float(vals[j])
            if v >= MIN_MAGNITUDE:
                ext_lon.append(float(sub_lons[j])); ext_val.append(v)
            else:
                ext_lon.append(None); ext_val.append(None)
        k = 0
        while k < len(times):
            if ext_lon[k] is None:
                k += 1; continue
            j = k
            while j + 1 < len(times) and ext_lon[j + 1] is not None and abs(ext_lon[j + 1] - ext_lon[j]) <= MAX_STEP_DEG:
                j += 1
            hours = hours_between(times[k], times[j])
            tt = np.array([hours_between(times[k], times[i]) for i in range(k, j + 1)]) / 24.0   # real elapsed days, not an assumed quarter day per sample
            ll = np.array(ext_lon[k:j + 1])
            slope = float(np.polyfit(tt, ll, 1)[0]) if len(tt) >= 2 else 0.0
            amp = float(max(ext_val[k:j + 1]))
            out.append({"sign": sign, "i_start": k, "i_end": j, "start": times[k], "end": times[j], "hours": hours, "samples": j - k + 1,
                        "lon_start": ext_lon[k], "lon_end": ext_lon[j], "slope_deg_per_day": slope, "amplitude": amp,
                        "qualifies": hours >= MIN_HOURS and slope <= -MIN_WEST_RATE})
            k = j + 1
    out.sort(key=lambda s: s["i_start"])
    return out


def select_streak(record):
    qual = [s for s in streaks(record) if s["qualifies"]]
    if not qual:
        return None
    return sorted(qual, key=lambda s: (-s["amplitude"], s["i_start"]))[0]


def window_for(streak, record):
    """The window as indices into the record's times, so every bound is a record time:
    one day of padding either side, and when that exceeds the cap, a whole number of
    steps centered on the streak's middle sample, clipped to the record."""
    times = record["times"]; cad = cadence_hours(record); n = len(times)
    pad = int(round(PAD_HOURS / cad)); cap = int(round(MAX_WINDOW_HOURS / cad))
    i0, i1 = streak["i_start"] - pad, streak["i_end"] + pad
    if i1 - i0 > cap:
        mid = (streak["i_start"] + streak["i_end"]) // 2
        i0, i1 = mid - cap // 2, mid + cap // 2
    i0, i1 = max(0, i0), min(n - 1, i1)
    return {"start": times[i0], "end": times[i1], "i_start": i0, "i_end": i1, "hours": hours_between(times[i0], times[i1]), "steps": i1 - i0 + 1}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("records", nargs="+"); ap.add_argument("--exclude", nargs="*", type=int, default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSED: {a.out} exists and a record is never overwritten")
    seasons = []
    for p in a.records:
        r = json.load(open(p))
        if int(r["year"]) in a.exclude:
            continue
        seasons.append({"year": int(r["year"]), "path": p, "sha256": sha256_of(p), "amplitude": season_amplitude(r), "record": r})
    seasons.sort(key=lambda s: (-s["amplitude"], s["year"]))
    chosen, fallbacks = None, []
    for s in seasons:
        sel = select_streak(s["record"])
        if sel is None:
            fallbacks.append({"year": s["year"], "amplitude": s["amplitude"], "reason": "no qualifying streak"})
            continue
        band = (s["record"].get("declared") or {}).get("bands", {}).get(BAND)
        chosen = {"year": s["year"], "path": s["path"], "sha256": s["sha256"], "amplitude": s["amplitude"], "streak": sel, "window": window_for(sel, s["record"]),
                  "cadence_hours": cadence_hours(s["record"]),
                  "band": {"name": BAND, "lat_lo": band.get("lat_lo") if band else None, "lat_hi": band.get("lat_hi") if band else None, "source": "the record's declared bands" if band else "not declared in the record"},
                  "all_streaks": [dict(x) for x in streaks(s["record"])]}
        break
    out = {"generated_by": "scripts/pilot_stage_d_select.py", "script_sha256": sha256_of(os.path.abspath(__file__)),
           "declared": "EASTERN_STAGE_D_EVALUATION_2026-10-17.md, section 2, as corrected for stage E in EASTERN_STAGE_E_EVALUATION_2026-10-17.md, section 2",
           "rule": {"lon_range": LON_RANGE, "min_magnitude_ms": MIN_MAGNITUDE, "max_step_deg": MAX_STEP_DEG,
           "min_hours": MIN_HOURS, "min_west_rate_deg_per_day": MIN_WEST_RATE, "pad_hours": PAD_HOURS, "max_window_hours": MAX_WINDOW_HOURS, "band": BAND,
           "cadence": "read from each record and refused unless uniform", "window": "bounds chosen among the record's time steps"},
           "excluded": a.exclude, "ranking": [{"year": s["year"], "amplitude": s["amplitude"], "sha256": s["sha256"]} for s in seasons], "fallbacks": fallbacks, "selected": chosen}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("ranking:", ", ".join(f"{s['year']} {s['amplitude']:.2f}" for s in seasons[:8]))
    if fallbacks:
        print("fallbacks:", fallbacks)
    if chosen:
        st, w = chosen["streak"], chosen["window"]
        print(f"selected {chosen['year']}: streak sign {st['sign']:+d} {st['start']}..{st['end']} {st['hours']:g} h {st['lon_start']:g}->{st['lon_end']:g} E slope {st['slope_deg_per_day']:+.2f}/day amplitude {st['amplitude']:.2f} m/s; window {w['start']}..{w['end']} ({w['hours']:g} h)")
    else:
        print("no season holds a qualifying streak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
