#!/usr/bin/env python3
"""Fixed-window wind and cloud context for one stored track, or for one declared point,
for the bounded case reading of the 60 E candidate evaluation.

WIND. The raw ERA5 700 hPa meridional wind from the retained 75 E retrieval file,
unsmoothed and uncontoured, averaged over a latitude band centered on the first
detection, as a Hovmoller diagram (longitude by time) over a window fixed by rule, with
the stored track overlaid. It reads circulation continuity independently of the
tracker's accepted contours, while ERA5 remains the shared input of both.

CLOUD. GridSat-B1 brightness temperature from the retained July to September imagery,
read through the sequence tool's file lookup, as the fraction of pixels colder than
240 K in the same band per degree of longitude and three-hourly time. A timestep without
a retained file is COVERAGE MISSING and is kept apart from cold cloud absent. Infrared
cloud context, not a wave-presence oracle. Neither absent cold cloud nor a continuing
cloud feature alone decides anything.

THE RULES, declared in EASTERN_60E_EVALUATION_2026-10-07.md. Window longitude from the
first-detection longitude plus 15 degrees to the westernmost stored longitude minus 25
degrees, time from 3 days before the first detection to 4 days after the last
observation, clipped to the data. For a point, 15 degrees east to 40 degrees west, 3 days
before to 7 days after. Band, the first-detection latitude plus and minus 5 degrees,
clipped to 5 S to 35 N. The record carries the window, the band, the arrays, every file
digest and the coverage summary. The reading categories are the reader's and live in
the document, never in this record.

    python3 scripts/pilot_case_wind_context.py --replay <replay json.gz> --track-index 798 --v-file <era5 v nc> \\
        --imagery data/gridsat_jas --out <fresh json> --png <fresh png>
    python3 scripts/pilot_case_wind_context.py --point 10.5 29.0 38960.0 --year 2006 --v-file <era5 v nc> --imagery data/gridsat_jas --out <json> --png <png>
"""
import argparse
import datetime as dt
import textwrap
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import gridsat_case_sequence as G  # noqa: E402
import pilot_margin_compare as MC  # noqa: E402

EPOCH = dt.datetime(1900, 1, 1)
DAYS_1900_TO_1970 = 25567.0
LON_BOUNDS = (-155.0, 75.0)


def _clip_time(t0, t1, time_bounds):
    if time_bounds is None:
        return [t0, t1]
    return [max(t0, float(time_bounds[0])), min(t1, float(time_bounds[1]))]


def window_for(track, east_ext=15.0, west_ext=25.0, days_before=3.0, days_after=4.0, lon_bounds=LON_BOUNDS, time_bounds=None):
    lon0, lon_min = float(track["lon"][0]), float(min(float(x) for x in track["lon"]))
    return {"lon": [max(lon_min - west_ext, lon_bounds[0]), min(lon0 + east_ext, lon_bounds[1])],
            "time": _clip_time(float(track["time"][0]) - days_before, float(track["time"][-1]) + days_after, time_bounds)}


def window_for_point(lon, day, east_ext=15.0, west_ext=40.0, days_before=3.0, days_after=7.0, lon_bounds=LON_BOUNDS, time_bounds=None):
    return {"lon": [max(float(lon) - west_ext, lon_bounds[0]), min(float(lon) + east_ext, lon_bounds[1])], "time": _clip_time(float(day) - days_before, float(day) + days_after, time_bounds)}


def wind_time_bounds(path):
    """The first and last retrieval times of the wind file, in days since 1900, so that a
    window never reaches beyond the data in time on any panel."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        secs = np.asarray(ds.variables["valid_time"][:], float)
    times = secs / 86400.0 + DAYS_1900_TO_1970
    return float(times.min()), float(times.max())


def band_for(track, half=5.0, clip=(-5.0, 35.0)):
    lat0 = float(track["lat"][0])
    return [max(lat0 - half, clip[0]), min(lat0 + half, clip[1])]


def v_hovmoller(path, band, window):
    """The band mean of the raw meridional wind at every retrieval time inside the window."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        secs = np.asarray(ds.variables["valid_time"][:], float)
        times = secs / 86400.0 + DAYS_1900_TO_1970
        lats, lons = np.asarray(ds.variables["latitude"][:], float), np.asarray(ds.variables["longitude"][:], float)
        tsel = np.where((times >= window["time"][0]) & (times <= window["time"][1]))[0]
        rows = np.where((lats >= band[0]) & (lats <= band[1]))[0]
        cols = np.where((lons >= window["lon"][0]) & (lons <= window["lon"][1]))[0]
        if not (tsel.size and rows.size and cols.size):
            raise SystemExit("REFUSED: the window or the band selects nothing in the wind file")
        var = ds.variables["v"]
        index = []
        for dim in var.dimensions:                                     # the retrieval carries a singleton pressure level, the fixtures may not
            if dim == "valid_time":
                index.append(slice(tsel[0], tsel[-1] + 1))
            elif dim == "latitude":
                index.append(slice(rows[0], rows[-1] + 1))
            elif dim == "longitude":
                index.append(slice(cols[0], cols[-1] + 1))
            else:
                index.append(0)
        v = np.asarray(var[tuple(index)], float)
        if v.shape != (tsel.size, rows.size, cols.size):
            raise SystemExit(f"REFUSED: the wind variable's dimensions {var.dimensions} did not reduce to (time, latitude, longitude)")
    return {"times": times[tsel], "lons": lons[cols], "lats": lats[rows], "v": v.mean(axis=1), "file": path, "file_sha256": X.digest(path)}


def cloud_hovmoller(imagery, band, window, cold_k=G.COLD_K, step_hours=3):
    """The fraction of pixels colder than cold_k in the band, per degree of longitude and
    three-hourly time, with a timestep lacking a retained file recorded as missing."""
    start = EPOCH + dt.timedelta(days=window["time"][0])
    first = start.replace(minute=0, second=0, microsecond=0)
    if first < start:
        first += dt.timedelta(hours=1)
    while first.hour % step_hours:
        first += dt.timedelta(hours=1)
    end = EPOCH + dt.timedelta(days=window["time"][1])
    bins = np.arange(np.floor(window["lon"][0]), np.floor(window["lon"][1]) + 1.0, 1.0)
    times, rows_out, coverage, files = [], [], [], []
    when = first
    while when <= end:
        times.append((when - EPOCH).total_seconds() / 86400.0)
        path = G.retained_file(imagery, when)
        if path is None:
            rows_out.append(np.full(bins.size, np.nan)); coverage.append("missing")
        else:
            import xarray as xr                                        # the retained readers' path, which decodes the scale and offset to kelvin
            with xr.open_dataset(path) as ds:
                stamped = np.asarray(ds["time"].values).ravel()[0].astype("datetime64[s]").astype(dt.datetime)
                if stamped != when:                                        # a file named for this timestep but stamped for another yields no reading
                    rows_out.append(np.full(bins.size, np.nan)); coverage.append("timestamp_mismatch")
                    when += dt.timedelta(hours=step_hours)
                    continue
                da = ds["irwin_cdr"].isel(time=0)
                glat, glon = np.asarray(da["lat"].values, float), np.asarray(da["lon"].values, float)
                r = np.where((glat >= band[0]) & (glat <= band[1]))[0]
                c = np.where((glon >= bins[0] - 0.5) & (glon < bins[-1] + 0.5))[0]
                tb = np.asarray(da.isel(lat=slice(r[0], r[-1] + 1), lon=slice(c[0], c[-1] + 1)).values, float)
            if np.nanmax(tb) > 400.0 if np.isfinite(tb).any() else False:
                raise SystemExit(f"REFUSED: {path} read as {np.nanmax(tb):g}, not a brightness temperature in kelvin")
            cold = np.where(np.isfinite(tb), (tb < cold_k).astype(float), np.nan)
            col_bins = np.floor(glon[c[0]:c[-1] + 1] + 0.5)
            row = np.array([np.nanmean(cold[:, col_bins == b]) if np.isfinite(cold[:, col_bins == b]).any() else np.nan for b in bins])
            rows_out.append(row); coverage.append("retained"); files.append({"path": path, "sha256": X.digest(path)})
        when += dt.timedelta(hours=step_hours)
    return {"times": np.array(times), "lons": bins, "cold_fraction": np.array(rows_out), "coverage": coverage, "files": files,
            "summary": {"timesteps": len(times), "retained": coverage.count("retained"), "missing": coverage.count("missing"), "timestamp_mismatch": coverage.count("timestamp_mismatch")}, "cold_k": cold_k}


COVERAGE_SHADE = {"missing": "0.85", "timestamp_mismatch": "#f1b6b6"}      # gray rows for no retained file, pink rows for a file stamped for another time
FIGURE_DPI = 110                                                           # the figure is built, measured and saved at one resolution
CAPTION_WRAP = 56                                                          # characters per caption line, so the title stays inside the figure at any count


def coverage_caption(summary):
    """The cloud panel's title, naming every non-retained coverage class with its count,
    wrapped so that it fits the figure whatever the counts are."""
    n = summary["timesteps"]
    body = (f"(gray rows, coverage missing: {summary['missing']} of {n}. "
            f"Pink rows, file stamped for another time: {summary.get('timestamp_mismatch', 0)} of {n})")
    return "GridSat-B1 cold cloud, band mean\n" + "\n".join(textwrap.wrap(body, width=CAPTION_WRAP))


def render(title, band, window, hov_v, hov_c, track, point, out_png):
    """Draw the two panels and return what was marked, the caption and the shaded rows, so a
    test can check the marking without reading pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, dpi=FIGURE_DPI)
    tv, lv, vv = hov_v["times"], hov_v["lons"], hov_v["v"]
    m = axes[0].pcolormesh(lv, tv, vv, cmap="RdBu_r", vmin=-8, vmax=8, shading="nearest")
    fig.colorbar(m, ax=axes[0], label="700 hPa meridional wind, band mean (m/s)")
    axes[0].set_title(f"ERA5 v700, band {band[0]:g} to {band[1]:g} N")
    tc, lc, cc = hov_c["times"], hov_c["lons"], hov_c["cold_fraction"]
    shaded = []
    if tc.size:
        m2 = axes[1].pcolormesh(lc, tc, cc, cmap="Blues", vmin=0, vmax=1, shading="nearest")
        fig.colorbar(m2, ax=axes[1], label=f"fraction colder than {hov_c['cold_k']:g} K")
        for t, cov in zip(tc, hov_c["coverage"]):
            if cov in COVERAGE_SHADE:
                axes[1].axhspan(t - 0.0625, t + 0.0625, color=COVERAGE_SHADE[cov], lw=0)
                shaded.append({"time": float(t), "coverage": cov, "color": COVERAGE_SHADE[cov]})
    caption = coverage_caption(hov_c["summary"])
    axes[1].set_title(caption)
    for ax in axes:
        if track is not None:
            ax.plot(track["lon"], track["time"], "k.-", lw=1.2, ms=4, label="stored track")
            ax.plot(track["lon"][0], track["time"][0], "k^", ms=8)
        if point is not None:
            ax.plot(point[1], point[2], "r*", ms=12, label="documented event point")
        ax.set_xlim(window["lon"][0], window["lon"][1]); ax.set_ylim(window["time"][0], window["time"][1])
        ax.set_xlabel("longitude (degrees east)")
        ticks = np.arange(np.ceil(window["time"][0]), np.floor(window["time"][1]) + 1)
        ax.set_yticks(ticks); ax.set_yticklabels([(EPOCH + dt.timedelta(days=float(t))).strftime("%m-%d") for t in ticks])
    axes[0].set_ylabel("date (month-day)")
    axes[0].legend(loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.canvas.draw()                                                   # the drawn title's extent against the figure, so a clipped caption is detected, not assumed
    bb = axes[1].title.get_window_extent(fig.canvas.get_renderer())
    fw, fh = fig.canvas.get_width_height()
    caption_inside = bool(bb.x0 >= 0 and bb.x1 <= fw and bb.y0 >= 0 and bb.y1 <= fh)
    if not caption_inside:                                              # a figure with a clipped caption is not published
        plt.close(fig)
        raise SystemExit(f"REFUSED: the cloud caption does not fit the figure (title spans {bb.x0:.0f} to {bb.x1:.0f} of {fw} pixels)")
    fig.savefig(out_png, dpi=FIGURE_DPI)
    plt.close(fig)
    return {"caption": caption, "shaded": shaded, "caption_inside_figure": caption_inside}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay"); ap.add_argument("--track-index", type=int)
    ap.add_argument("--point", nargs=3, type=float, metavar=("LAT", "LON", "DAYS_SINCE_1900")); ap.add_argument("--year", type=int)
    for name in ("--v-file", "--out", "--png"):
        ap.add_argument(name, required=True)
    ap.add_argument("--imagery", default="data/gridsat_jas")
    args = ap.parse_args(argv)
    for p in (args.out, args.png):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    if (args.replay is None) == (args.point is None):
        raise SystemExit("REFUSED: give a replay and a track index, or a point, not both and not neither")
    track, point, replay_id = None, None, None
    bounds = wind_time_bounds(args.v_file)
    if args.replay is not None:
        rec, sha = MC.load_gz(args.replay)
        t = rec["finished"][args.track_index]
        track = {"time": [float(x) for x in t["time"]], "lat": [float(x) for x in t["lat"]], "lon": [float(x) for x in t["lon"]]}
        replay_id = {"path": args.replay, "sha256": sha, "year": rec["year"], "configuration": rec.get("configuration"), "thresholds": rec["thresholds"], "track_index": args.track_index, "first_detection": t["birth"], "observations": len(t["time"])}
        window, band, year = window_for(track, time_bounds=bounds), band_for(track), rec["year"]
        title = f"{year} track {args.track_index}, first detection {track['lat'][0]:g} N {track['lon'][0]:g} E"
    else:
        lat, lon, day = args.point
        point = (lat, lon, day)
        window, band, year = window_for_point(lon, day, time_bounds=bounds), band_for({"lat": [lat]}), args.year
        title = f"{year} point {lat:g} N {lon:g} E at {(EPOCH + dt.timedelta(days=day)).strftime('%Y-%m-%d %HZ')}"
    hov_v = v_hovmoller(args.v_file, band, window)
    hov_c = cloud_hovmoller(args.imagery, band, window)
    render(title, band, window, hov_v, hov_c, track, point, args.png)
    out = {"generated_by": "scripts/pilot_case_wind_context.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": year,
           "replay": replay_id, "point": None if point is None else {"lat": point[0], "lon": point[1], "days_since_1900": point[2]}, "track": track,
           "rules": {"window_lon": "first-detection longitude plus 15 to westernmost stored longitude minus 25, clipped to the data", "window_time": "3 days before first detection to 4 days after the last observation, clipped to the wind file's times", "wind_file_time_bounds": list(bounds),
                     "point_window": "15 east to 40 west, 3 days before to 7 after", "band": "first-detection latitude plus and minus 5, clipped to 5 S to 35 N", "cold_k": hov_c["cold_k"]},
           "window": window, "band": band,
           "wind": {"file": hov_v["file"], "file_sha256": hov_v["file_sha256"], "times": hov_v["times"].tolist(), "lons": hov_v["lons"].tolist(), "lats": hov_v["lats"].tolist(), "v_band_mean": np.round(hov_v["v"], 3).tolist()},
           "cloud": {"imagery": args.imagery, "times": hov_c["times"].tolist(), "lons": hov_c["lons"].tolist(), "coverage": hov_c["coverage"], "summary": hov_c["summary"], "files": hov_c["files"],
                     "cold_fraction": [[None if not np.isfinite(x) else round(float(x), 3) for x in row] for row in hov_c["cold_fraction"]]},
           "figure": {"path": args.png, "sha256": X.digest(args.png)},
           "reading": "the window, the band, the arrays and the coverage. The reader's categories are recorded in the evaluation document, not here."}
    X.publish_json(args.out, out, exclusive=True)
    print(f"{title}. Window lon {window['lon']} time {window['time']}, band {band}. Wind {len(hov_v['times'])} times x {len(hov_v['lons'])} lons. "
          f"Cloud {hov_c['summary']}. Figure {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
