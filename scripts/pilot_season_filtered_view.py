#!/usr/bin/env python3
"""A bounded candidate-finding display: the seasonal band means of the meridional wind,
band-passed with the propagation study's filter, in the time and longitude plane
(`EASTERN_DIRECTION_RECOMMENDATION_7_2026-10-15.md`, sections 3 and 9).

Reused unchanged: the season view's grid, band means and time axis
(`pilot_season_wind_view.py`), which call the propagation study's `read_season` and
`band_series`; and the propagation study's Lanczos band-pass for 2.5 to 6 day periods (81
weights, 40 six-hourly steps on either side), applied by its `filter_supported` inside
contiguous supported runs and kept only where the whole window lies inside the run. Every
plotted value therefore uses samples from ten days before to ten days after its date, the
displayed range is 11 June to 20 September at best, and unavailable dates are drawn as
missing and counted. Values are signed. A temporal filter selects frequencies: it keeps any
persistent spatial structure whose amplitude varies inside the passband, and a short input
event produces a longer oscillatory response, whose span for a one-day pulse is recorded as
the filter's own footprint. Nothing of the tracker and no reference coordinate is drawn.

    python3 scripts/pilot_season_filtered_view.py --year 2006 --level 700 --out <json> --png-prefix <prefix>
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_eastern_propagation as P  # noqa: E402
import pilot_season_wind_view as S  # noqa: E402

VLIM = 4.0                      # m/s, the filtered band mean's color range
FOOTPRINT_FRACTION = 0.1        # the pulse response counts while above this fraction of its peak
MONTHS = S.MONTHS
MARK_LONS = S.MARK_LONS
MISSING_COLOR = S.MISSING_COLOR
STEPS_PER_INCH = S.STEPS_PER_INCH


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def filtered_means(means, weights):
    """Each band's series through the study's filter inside its supported runs, whole
    windows only, with the unavailable samples counted after filtering."""
    out = {}
    for b, m in means.items():
        f = P.filter_supported(m["series"], weights)
        out[b] = {"series": f, "support": m["support"], "unavailable": int(np.isnan(f).sum()), "samples": int(f.size),
                  "unavailable_before_filter": int(m["unavailable"])}
    return out


def footprint_response(weights, dt_days=P.DT_DAYS, pulse_days=1.0, fraction=FOOTPRINT_FRACTION):
    """The filter's response to a unit pulse of the given length at one place, inside a
    supported run long enough to hold it: the apparent span in days over which the
    response stays above the fraction of its peak magnitude, the number of sign changes,
    and the response to a steady unit input (the sum of the weights)."""
    n = (len(weights) - 1) // 2
    steps = max(1, int(round(pulse_days / dt_days)))
    series = np.zeros(6 * n + steps)
    series[3 * n:3 * n + steps] = 1.0
    r = P.filter_supported(series, weights)
    ok = np.isfinite(r)
    rr = r[ok]
    peak = float(np.max(np.abs(rr)))
    above = np.flatnonzero(np.abs(rr) >= fraction * peak)
    span_days = float((above[-1] - above[0] + 1) * dt_days)
    signs = np.sign(rr[np.abs(rr) >= fraction * peak])
    changes = int(np.sum(signs[1:] * signs[:-1] < 0))
    return {"pulse_days": pulse_days, "fraction_of_peak": fraction, "apparent_span_days": span_days, "sign_changes": changes,
            "steady_input_response": float(np.sum(weights)), "peak_response": peak}


def draw(year, level, lons, times, filt, png_prefix, label):
    """The full-season figure and one per calendar month, both bands side by side, time
    downward, longitude across, missing samples in gray, the marked meridians dashed, the
    filter's footprint in every title. Returns one record per figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(MISSING_COLOR)
    windows = [("season", times[0], times[-1])] + [(name, dt.datetime(year, m, 1), (dt.datetime(year, m + 1, 1) if m < 12 else dt.datetime(year + 1, 1, 1)) - dt.timedelta(hours=6)) for m, name in MONTHS]
    tarr = np.array(times); records = []
    for name, t0, t1 in windows:
        sel = np.flatnonzero((tarr >= t0) & (tarr <= t1))
        if sel.size == 0:
            continue
        height = max(4.0, sel.size / STEPS_PER_INCH)
        fig, axes = plt.subplots(1, len(filt), figsize=(6.5 * len(filt) + 1.5, height + 1.6), sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (b, m) in zip(axes, filt.items()):
            arr = np.ma.masked_invalid(m["series"][sel])
            lo, hi = P.BANDS[b]
            mesh = ax.pcolormesh(lons, tarr[sel], arr, cmap=cmap, vmin=-VLIM, vmax=VLIM, shading="nearest")
            for x in MARK_LONS:
                ax.axvline(x, color="k", lw=0.8, ls="--")
            ax.set_title(f"{b} band, {lo:g} to {hi:g} N", fontsize=12)
            ax.set_xlabel("longitude (degrees E)")
            ax.yaxis.set_major_locator(mdates.DayLocator(interval=5 if name == "season" else 2))
            ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.invert_yaxis() if not ax.yaxis_inverted() else None
            ax.grid(True, axis="y", color="0.5", lw=0.3, ls=":")
        fig.colorbar(mesh, ax=axes.tolist(), shrink=0.4, label=f"{level} hPa meridional wind, 2.5 to 6 day band-pass of the band mean (m/s)")
        fig.suptitle(f"{label}, {name}: ERA5 {level} hPa meridional wind, band mean over terrain-valid rows (80 percent support), "
                     f"2.5 to 6 day Lanczos band-pass. Footprint: each plotted value uses 81 six-hourly samples, ten days before to ten days after "
                     f"its date; gray where the whole window is not supported. A filtered streak's span is not a lifetime. "
                     f"Nothing of the tracker drawn. Dashed: 40 E and the detector's 60 E edge.", fontsize=10, wrap=True)
        path = f"{png_prefix}_{name}.png"
        fig.savefig(path, dpi=100); plt.close(fig)
        records.append({"name": name, "path": path, "sha256": sha256_of(path), "first": times[sel[0]].strftime("%Y-%m-%dT%H"), "last": times[sel[-1]].strftime("%Y-%m-%dT%H"),
                        "steps": int(sel.size), "bands": list(filt), "unavailable_in_window": {b: int(np.isnan(m["series"][sel]).sum()) for b, m in filt.items()},
                        "footprint_in_title": True})
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--out", "--png-prefix"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--level", type=int, default=P.PRIMARY_LEVEL)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    lat, lon, rows, cols = S.grid(args.year, args.level)
    all_rows = np.concatenate(list(rows.values()))
    block = P.read_season(args.year, args.level, all_rows, cols)
    lons = lon[cols]
    weights = P.lanczos_bandpass()
    means = S.band_means(block, rows, args.level)
    filt = filtered_means(means, weights)
    times = S.times_of(block["time"])
    label = f"{args.year}, {args.level} hPa"
    figures = draw(args.year, args.level, lons, times, filt, args.png_prefix, label)
    files = {n: os.path.join(P.DATA_DIR, f"era5_{n}_{args.year}_6h_region.nc") for n in (f"v{args.level}", "sp")}
    here = os.path.dirname(os.path.abspath(__file__))
    supported = {b: [i for i in range(len(times)) if np.isfinite(m["series"][i]).any()] for b, m in filt.items()}
    out = {"generated_by": "scripts/pilot_season_filtered_view.py", "script_sha256": sha256_of(os.path.abspath(__file__)),
           "reuses": {"grid, band_means, times_of": "scripts/pilot_season_wind_view.py, unchanged", "season_view_script_sha256": sha256_of(os.path.join(here, "pilot_season_wind_view.py")),
                      "read_season, band_series, lanczos_bandpass, filter_supported": "scripts/pilot_eastern_propagation.py, unchanged",
                      "propagation_script_sha256": sha256_of(os.path.join(here, "pilot_eastern_propagation.py"))},
           "year": args.year, "level_hpa": args.level, "primary_level_hpa": P.PRIMARY_LEVEL, "role": "primary" if args.level == P.PRIMARY_LEVEL else "sensitivity",
           "sources": {n: {"path": p, "sha256": sha256_of(p)} for n, p in files.items()},
           "declared": {"bands": {b: {"lat_lo": lo, "lat_hi": hi, "rows": int(rows[b].size), "latitudes": lat[rows[b]].tolist()} for b, (lo, hi) in P.BANDS.items()},
                        "weights": "equal over the valid rows of the band", "valid_cell": f"wind finite and surface pressure above {args.level} hPa strictly, at that time",
                        "min_valid_fraction": P.MIN_VALID_FRACTION, "lon_range": list(S.LON_RANGE), "lon_step": float(np.median(np.diff(lons))), "columns": int(cols.size),
                        "time": {"first": times[0].strftime("%Y-%m-%dT%H"), "last": times[-1].strftime("%Y-%m-%dT%H"), "steps": len(times), "hours": 6},
                        "filter": {"kind": "Lanczos band-pass, the propagation study's", "periods_days": list(P.PERIODS_DAYS), "weights": int(len(weights)),
                                   "half_steps": int(P.LANCZOS_HALF), "half_days": float(P.LANCZOS_HALF * P.DT_DAYS),
                                   "applied": "filter_supported, inside contiguous supported runs, kept only where the whole window lies inside the run",
                                   "footprint": "each plotted value uses samples from ten days before to ten days after its date"},
                        "values": "signed band mean after the band-pass", "color_range_ms": VLIM, "marked_lons": list(MARK_LONS),
                        "drawn": "filtered band-mean meridional wind only; no axis, candidate, history, track or reference coordinate"},
           "footprint": {"above_0.1_of_peak": footprint_response(weights), "above_0.5_of_peak": footprint_response(weights, fraction=0.5)},
           "lons": lons.tolist(), "times": [t.strftime("%Y-%m-%dT%H") for t in times],
           "bands": {b: {"unavailable": m["unavailable"], "samples": m["samples"], "unavailable_fraction": m["unavailable"] / m["samples"],
                         "unavailable_before_filter": m["unavailable_before_filter"],
                         "first_supported": times[supported[b][0]].strftime("%Y-%m-%dT%H") if supported[b] else None,
                         "last_supported": times[supported[b][-1]].strftime("%Y-%m-%dT%H") if supported[b] else None,
                         "series": [[None if not np.isfinite(x) else round(float(x), 3) for x in row] for row in m["series"]]}
                     for b, m in filt.items()},
           "figures": figures,
           "label": "a candidate-finding display, not a census, not physical validation and not a test of the tracker; a filtered streak is an "
                    "observation in this band mean at these times, its span is not a lifetime, and it establishes no identity"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    for b, m in filt.items():
        print(f"{b}: {m['unavailable']} of {m['samples']} samples unavailable after filtering ({m['unavailable'] / m['samples']:.3%}), "
              f"{m['unavailable_before_filter']} before; supported {out['bands'][b]['first_supported']} to {out['bands'][b]['last_supported']}")
    for fp in out["footprint"].values():
        print(f"footprint: a {fp['pulse_days']:g} day pulse answers over {fp['apparent_span_days']:g} days above {fp['fraction_of_peak']:g} of its peak "
              f"with {fp['sign_changes']} sign changes; steady input response {fp['steady_input_response']:.2e}")
    print(f"{len(figures)} figures. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
