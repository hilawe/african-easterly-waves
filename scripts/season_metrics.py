#!/usr/bin/env python3
"""Season-level summary quantities of two trackers' finished tracks, with the published
record's interannual spread as the scale, DESCRIPTIVE and predeclared.

WHAT IS COMPARED. Version 1's and the port's finished tracks over one season on identical
input, as each program wrote them, before any deduplication. A track belongs to the season
by its FIRST observation's calendar month (June to September). Per side: tracks, distinct
waves, starts per month, lifetime in observations, genesis and lysis longitude and
latitude, and the share of duplicated observations. Every quantity is also reported per
calendar month and per ten-degree band of genesis longitude.

THE SCALE. For every count, the interannual standard deviation of the same count in the
published record over the declared years, computed with one degree of freedom removed and
with a year that has no track in a group counted as ZERO for that group, never dropped.
Distributions are compared by the two-sample Kolmogorov-Smirnov statistic and by the
differences at the 10th, 50th and 90th percentiles, never by a median alone.

THE SPARSE RULE, fixed before any result is seen. A group with fewer than MIN_GROUP tracks
on either side, or with zero interannual variance in the published record, is reported as
counts only, is listed by name, and receives no ratio and no distribution statistic.

DISTINCT WAVES are the connected components of the near-copy relation between tracks,
two tracks being near copies when they share at least WAVE_SHARED_STEPS timesteps at a
mean separation of at most WAVE_SEPARATION_DEG degrees under the comparator's
cosine-weighted separation. That is this script's own definition, stated here.

NO TOLERANCE IS APPLIED. The artifact carries differences and scales, and says that it is
descriptive. Any pass line is applied later by a reader who set it before reading this.

THE HARNESS CONTROL, optional: the port's tracks from a fresh export of the retained
standalone 1990 window must equal the retained port tracks exactly and carry the same case
identity, or the harness is not the one the record was built with.

Inputs are read once as bytes and digested. The artifact is published exclusively.
"""

import argparse
import collections
import re
import datetime as dt
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
from compare_tracker_oracle import overlap_separation, read_tracks  # noqa: E402
from aew.v1port import validate as V  # noqa: E402

SEASON_MONTHS = (6, 7, 8, 9)
# THE RECORD'S MEMBERSHIP RULE, reconstructed from version 1's own generate_ew_stats_f.m
# and verified to reproduce the archive's 1990 Africa file 508 of 508. A track's FIRST
# observation, each coordinate rounded to the nearest quarter degree, is tested against
# seven region polygons in the archive's priority order, and the first polygon containing
# it names the source region. The Africa file holds the tracks whose region is AFR. A box
# of 20 W to 40 E used before this rule was read counted 352 version 1 tracks where the
# rule counts 225, and both artifacts are retained. Whole-domain totals are reported
# beside the Africa-origin comparison and enter no ratio.
REGION_FILES = (("NEP", "northeast_pacific"), ("SEP", "southeast_pacific"), ("CAM", "central_america"),
                ("SAM", "south_america"), ("NAL", "north_atlantic"), ("SAL", "south_atlantic"),
                ("AFR", "africa"))
RECORD_REGION = "AFR"
ROUND_DEG = 0.25
MIN_GROUP = 5
WAVE_SHARED_STEPS = 3
WAVE_SEPARATION_DEG = 1.0
BAND_EDGES = list(range(-60, 61, 10))      # ten-degree bands of genesis longitude, [lo, hi)
PERCENTILES = (10, 50, 90)
RECORD_YEARS = "1983-2007"                 # the published record's years, the scale's default
DISTRIBUTIONS = ("lifetime", "genesis_lon", "genesis_lat", "lysis_lon", "lysis_lat")


def date_of(days):
    return dt.datetime(1900, 1, 1) + dt.timedelta(days=float(days))


def _from_bytes(raw_bytes, suffix, reader):
    """Parse the HASHED bytes, never the path again. A review showed a file hashed and
    then reopened, so a file changed in between would bind a result to the wrong digest."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(raw_bytes)
        path = fh.name
    try:
        return reader(path)
    finally:
        os.unlink(path)


def read_mat_tracks(raw_bytes):
    """Tracks in the (n, lat<i>, lon<i>, time<i>) layout, from bytes, through the
    comparator's VALIDATED reader (a whole nonnegative count, a case id present). The case
    id must be nonempty, since two absent identities would otherwise compare equal."""
    from scipy.io import loadmat
    declared = np.asarray(loadmat(io.BytesIO(raw_bytes), variable_names=["n"]).get("n"))
    if declared.size != 1:
        # THE COUNT MUST BE ONE NUMBER. The reused reader takes the first element of
        # whatever it finds, so a count written as [0, 1] read as zero tracks and passed.
        raise SystemExit(f"REFUSED: a track file declares a count of shape {declared.shape}, not one number")
    tracks, case = _from_bytes(raw_bytes, ".mat", read_tracks)
    if not case:
        raise SystemExit("REFUSED: a track file carries an empty case id")
    return tracks, case


def read_record_bytes(raw_bytes):
    """The published record's tracks from hashed bytes, in this script's layout."""
    return record_tracks(_from_bytes(raw_bytes, ".nc", V.read_tracks))


def record_tracks(track_dicts):
    """The published record's tracks (validate's layout) in this script's layout."""
    return [{"time": np.asarray(t["time"], float), "lat": np.asarray(t["meanlat"], float),
             "lon": np.asarray(t["meanlon"], float)} for t in track_dicts]


def in_season(track, year=None):
    d = date_of(track["time"][0])
    return d.month in SEASON_MONTHS and (year is None or d.year == year)


def load_regions(directory):
    """The archive's region polygons in its priority order, from the archived .mat files,
    with their digests. Returns (list of (code, Path), digests)."""
    from matplotlib.path import Path
    from scipy.io import loadmat
    regions, digests = [], {}
    for code, name in REGION_FILES:
        path = os.path.join(directory, f"{name}.mat")
        with open(path, "rb") as fh:
            blob = fh.read()
        digests[name] = hashlib.sha256(blob).hexdigest()
        arr = loadmat(io.BytesIO(blob))[name]
        regions.append((code, Path(np.column_stack([arr[0], arr[1]]))))
    return regions, digests


def matlab_round(x):
    """MATLAB's round: ties go AWAY from zero. Python's round sends ties to even, and a
    review showed the two disagree on 3,402 of 21,294 boundary probes, moving a first
    observation at 17.125 W, 14.5 N from the North Atlantic to Africa."""
    x = float(x)
    a = abs(x)
    whole = np.floor(a)
    # THE TIE TEST IS ON THE EXACT FRACTION, not on floor(a + 0.5): adding 0.5 to a value
    # just below a tie can round up in floating point (0.49999999999999994 + 0.5 is 1.0),
    # and a confirmation round found that moving a source region. a - floor(a) is exact.
    return float(np.sign(x) * (whole + (1.0 if a - whole >= 0.5 else 0.0)))


def source_region(track, regions):
    """The archive's rule: first observation, each coordinate rounded to the quarter degree
    as MATLAB rounds, first polygon in priority order that contains it, else OTH."""
    lon = matlab_round(float(track["lon"][0]) / ROUND_DEG) * ROUND_DEG
    lat = matlab_round(float(track["lat"][0]) / ROUND_DEG) * ROUND_DEG
    for code, path in regions:
        # inpolygon counts the boundary as inside, so test with the boundary both ways
        if path.contains_point((lon, lat), radius=1e-9) or path.contains_point((lon, lat), radius=-1e-9):
            return code
    return "OTH"


def in_record_domain(track, regions):
    return source_region(track, regions) == RECORD_REGION


def spacing(tracks):
    """How the observations are spaced in time, so an observation count is not read as a
    duration: the fraction of consecutive intervals equal to one timestep, the largest
    interval, and the share of tracks with any gap. Duration is last minus first time."""
    # a population of single-observation tracks has no intervals, and concatenating an
    # empty list raises, so the empty array is the starting point
    intervals = np.concatenate([np.array([])] + [np.diff(t["time"]) for t in tracks if t["time"].size > 1])
    with_gap = sum(1 for t in tracks if t["time"].size > 1 and not np.allclose(np.diff(t["time"]), 0.25))
    return {"timestep_days": 0.25, "intervals": int(intervals.size),
            "fraction_of_intervals_one_timestep": float(np.mean(np.isclose(intervals, 0.25))) if intervals.size else None,
            "largest_interval_days": float(intervals.max()) if intervals.size else None,
            "tracks_with_a_gap": with_gap, "tracks": len(tracks),
            "duration_days_percentiles": {str(p): float(v) for p, v in zip(
                PERCENTILES, np.percentile([t["time"][-1] - t["time"][0] for t in tracks], PERCENTILES))} if tracks else None,
            "endpoint_convention": "duration is the last observation's time minus the first's, "
                                   "so a track of n contiguous observations spans (n - 1) timesteps"}


def band_of(lon):
    for lo in BAND_EDGES[:-1]:
        if lo <= lon < lo + 10:
            return f"{lo:+d}..{lo + 10:+d}"
    return "outside"


def features(track, index=None):
    return {"i": index, "month": date_of(track["time"][0]).month, "band": band_of(float(track["lon"][0])),
            "lifetime": int(track["time"].size),      # OBSERVATIONS RECORDED, not days; see spacing()
            "genesis_lon": float(track["lon"][0]), "genesis_lat": float(track["lat"][0]),
            "lysis_lon": float(track["lon"][-1]), "lysis_lat": float(track["lat"][-1])}


def distinct_waves(tracks):
    parent = list(range(len(tracks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            n, sep = overlap_separation(tracks[i], tracks[j])
            if n >= WAVE_SHARED_STEPS and sep <= WAVE_SEPARATION_DEG:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    return len({find(i) for i in range(len(tracks))})


def duplication(tracks):
    as_record = [{"time": t["time"], "meanlat": t["lat"], "meanlon": t["lon"]} for t in tracks]
    return V.duplication_rate(as_record, months=SEASON_MONTHS)


def groups_of(feats):
    """Group keys: the whole season, each month, each band."""
    keys = ["season"] + [f"month {m}" for m in SEASON_MONTHS] + \
           [f"band {lo:+d}..{lo + 10:+d}" for lo in BAND_EDGES[:-1]]
    out = {k: [] for k in keys}
    for f in feats:
        out["season"].append(f)
        out[f"month {f['month']}"].append(f)
        key = f"band {f['band']}"
        if key in out:
            out[key].append(f)
    return out


def group_values(tracks, feats):
    """Per group: the tracks, their count, distinct waves and duplication fraction."""
    g = groups_of(feats)
    out = {}
    for k, fs in g.items():
        members = [tracks[f["i"]] for f in fs]
        dup = duplication(members) if members else {"fraction": float("nan"), "observations": 0, "duplicated": 0}
        out[k] = {"feats": fs, "count": len(fs), "distinct_waves": distinct_waves(members),
                  "duplication_fraction": dup["fraction"], "duplication": dup}
    return out


def _sd(values):
    arr = np.asarray(values, float)
    finite = arr[np.isfinite(arr)]
    return float(finite.std(ddof=1)) if finite.size > 1 else None


def published_spread(record_by_year):
    """Per group, the interannual mean and standard deviation (ddof 1) of the season count,
    of the distinct-wave count and of the duplication fraction, a year with no track in a
    group counted as ZERO for the counts and as undefined for the fraction."""
    years = sorted(record_by_year)
    per_group = collections.defaultdict(lambda: {"count": [], "distinct_waves": [], "duplication_fraction": []})
    pooled = collections.defaultdict(lambda: {name: [] for name in DISTRIBUTIONS})
    for y in years:
        season = [t for t in record_by_year[y] if in_season(t, y)]
        values = group_values(season, [features(t, i) for i, t in enumerate(season)])
        for k in groups_of([]):
            for field in ("count", "distinct_waves", "duplication_fraction"):
                per_group[k][field].append(values[k][field])
            for f in values[k]["feats"]:
                for name in DISTRIBUTIONS:
                    pooled[k][name].append(f[name])
    out = {}
    for k, series in per_group.items():
        counts = np.asarray(series["count"], float)
        out[k] = {"years": len(years), "mean": float(counts.mean()), "sd": _sd(counts),
                  "counts_by_year": dict(zip([str(y) for y in years], [int(c) for c in counts])),
                  "distinct_waves_mean": float(np.mean(series["distinct_waves"])),
                  "distinct_waves_sd": _sd(series["distinct_waves"]),
                  "duplication_fraction_mean": float(np.nanmean(series["duplication_fraction"]))
                  if np.isfinite(series["duplication_fraction"]).any() else None,
                  "duplication_fraction_sd": _sd(series["duplication_fraction"]),
                  # THE RECORD'S OWN DISTRIBUTIONS, pooled over the years, at the same
                  # percentiles the comparison reports, so the page's context column is read
                  # from this artifact and not recomputed elsewhere
                  "percentiles": {name: ({str(p): float(v) for p, v in zip(PERCENTILES, np.percentile(pooled[k][name], PERCENTILES))}
                                         if pooled[k][name] else None) for name in DISTRIBUTIONS},
                  "pooled_tracks": len(pooled[k]["lifetime"])}
    return out


def distribution_comparison(a, b):
    from scipy.stats import ks_2samp
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, pb = np.percentile(a, PERCENTILES), np.percentile(b, PERCENTILES)
    return {"ks_statistic": float(ks_2samp(a, b).statistic),
            "percentiles": {str(p): {"v1": float(x), "port": float(y), "port_minus_v1": float(y - x)}
                            for p, x, y in zip(PERCENTILES, pa, pb)}}


def _ratio(diff, sd):
    return None if sd is None or sd == 0.0 else float(diff) / sd


def compare(v1_tracks, port_tracks, spread):
    """Every group: counts, distinct waves and duplication with their own published scales,
    and distributions where the sparse rule allows. Each ratio uses THAT GROUP'S published
    standard deviation of THAT QUANTITY, never the season's."""
    gv = group_values(v1_tracks, [features(t, i) for i, t in enumerate(v1_tracks)])
    gp = group_values(port_tracks, [features(t, i) for i, t in enumerate(port_tracks)])
    out, counts_only = {}, []
    for k in gv:
        nv, npo = gv[k]["count"], gp[k]["count"]
        pub = spread.get(k, {})
        sd = pub.get("sd")
        entry = {"v1": nv, "port": npo, "port_minus_v1": npo - nv,
                 "published_mean": pub.get("mean"), "published_sd": sd,
                 "distinct_waves": {"v1": gv[k]["distinct_waves"], "port": gp[k]["distinct_waves"],
                                    "port_minus_v1": gp[k]["distinct_waves"] - gv[k]["distinct_waves"],
                                    "published_mean": pub.get("distinct_waves_mean"),
                                    "published_sd": pub.get("distinct_waves_sd")},
                 "duplication": {"v1_fraction": gv[k]["duplication_fraction"],
                                 "port_fraction": gp[k]["duplication_fraction"],
                                 "port_minus_v1_points": 100.0 * (gp[k]["duplication_fraction"]
                                                                  - gv[k]["duplication_fraction"]),
                                 "published_mean_fraction": pub.get("duplication_fraction_mean"),
                                 "published_sd_fraction": pub.get("duplication_fraction_sd")}}
        sparse = nv < MIN_GROUP or npo < MIN_GROUP
        zero_var = sd is None or sd == 0.0
        if sparse or zero_var:
            entry["reported_as"] = "counts only"
            entry["why"] = ("fewer than %d tracks on a side" % MIN_GROUP if sparse else
                            "zero interannual variance in the published record")
            counts_only.append(k)
        else:
            entry["difference_over_published_sd"] = (npo - nv) / sd
            entry["distinct_waves"]["difference_over_published_sd"] = _ratio(
                entry["distinct_waves"]["port_minus_v1"], pub.get("distinct_waves_sd"))
            entry["duplication"]["difference_over_published_sd"] = _ratio(
                gp[k]["duplication_fraction"] - gv[k]["duplication_fraction"], pub.get("duplication_fraction_sd"))
            entry["distributions"] = {
                name: distribution_comparison([f[name] for f in gv[k]["feats"]], [f[name] for f in gp[k]["feats"]])
                for name in DISTRIBUTIONS}
        out[k] = entry
    return out, counts_only


def producer_of(raw_bytes):
    """The producer record a tracker_port.mat carries, or None when it has none."""
    from scipy.io import loadmat
    raw = loadmat(io.BytesIO(raw_bytes), variable_names=["producer_json"])
    if "producer_json" not in raw:
        return None
    return json.loads(str(np.asarray(raw["producer_json"]).ravel()[0]))


PROTOCOL_KEYS = ("years", "level_hpa", "hours", "grid", "decimation", "domain", "smoothing",
                 "sign_convention", "wind_mask", "tracker_flags", "initialization",
                 "calibration", "membership", "reporting")


def producer_problems(record, manifest, manifest_sha256, year, side):
    """Why a producer record does not establish a protocol run of `year` under `manifest`.
    An empty list means it does. A review passed two empty settings blocks as equal, so
    the settings are checked against the retained manifest, key by key, and the stage,
    the year and the dataset identity are required."""
    problems = []
    if record is None:
        return [f"{side}: no producer record"]
    ps = record.get("protocol_settings") or {}
    for k in PROTOCOL_KEYS:
        if ps.get(k) != manifest.get(k):
            problems.append(f"{side}: protocol setting {k} is not the manifest's")
    if ps.get("manifest_sha256") != manifest_sha256:
        problems.append(f"{side}: manifest digest {ps.get('manifest_sha256')} is not the retained manifest's")
    if record.get("stage") != "tracking":
        problems.append(f"{side}: stage {record.get('stage')!r} is not tracking")
    ds = record.get("dataset_specific") or {}
    if ds.get("year") != year:
        problems.append(f"{side}: the record is for {ds.get('year')}, not {year}")
    # THE DATASET IDENTITY IS RESOLVED THROUGH THE MANIFEST: the name must be one the
    # manifest declares, and the label and prefix must be that entry's
    entry = (manifest.get("datasets") or {}).get(ds.get("dataset"))
    if entry is None:
        problems.append(f"{side}: dataset {ds.get('dataset')!r} is not in the manifest")
    elif ds.get("label") != entry.get("label") or ds.get("prefix") != entry.get("prefix"):
        problems.append(f"{side}: label or prefix is not the manifest's for {ds.get('dataset')}")
    # THE SOURCE INVENTORY MUST BE COMPLETE: every tracking source in this tree, each
    # with a full digest, so two records can only agree on the whole inventory
    sources = record.get("source_sha256") or {}
    if set(sources) != expected_tracking_sources() or not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in sources.values()):
        problems.append(f"{side}: the source inventory is not the complete tracking inventory with full digests")
    return problems


def expected_tracking_sources():
    """The files whose digests a tracking record must carry: every module of the port and
    the entry point, named relative to the repository."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = {"scripts/export_protocol_case.py"}
    for root, _dirs, files in os.walk(os.path.join(repo, "src", "aew", "v1port")):
        for name in files:
            if name.endswith(".py"):
                names.add(os.path.relpath(os.path.join(root, name), repo))
    return names


def settings_difference(a, b):
    """The keys on which two producer records' protocol settings or tracking sources
    differ. None when either lacks a settings block."""
    if a is None or b is None or "protocol_settings" not in a or "protocol_settings" not in b:
        return None
    pa, pb = a["protocol_settings"], b["protocol_settings"]
    out = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
    sa, sb = a.get("source_sha256") or {}, b.get("source_sha256") or {}
    out += sorted(f"source {k}" for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    return out


def manifest_agrees_with_this_instrument(manifest):
    """The manifest's membership and reporting blocks must be what this script computes,
    since it takes them from its own constants. Returns the disagreements."""
    p = []
    mem, rep = manifest.get("membership") or {}, manifest.get("reporting") or {}
    if list(mem.get("priority", [])) != [c for c, _ in REGION_FILES]:
        p.append("membership priority")
    if mem.get("rule") != ("the archive's source-region rule: the first observation, each coordinate rounded to "
                           "the nearest quarter degree with ties away from zero, tested against the region "
                           "polygons in priority order, the first containing polygon naming the region, else OTH"):
        p.append("membership rule")
    if mem.get("product_region") != RECORD_REGION:
        p.append("product region")
    if list(rep.get("season_months", [])) != list(SEASON_MONTHS):
        p.append("season months")
    return p


def _day(year, month, day):
    return float((dt.date(year, month, day) - dt.date(1900, 1, 1)).days)


def boundary_lines(tracks, year):
    """The protocol's boundary predicates on retained observations, for one calendar year.

    CROSSING IN: first observation before June 1 00Z and last at or after it. CROSSING OUT:
    first at or after June 1 00Z and before October 1 00Z, last at or after October 1 00Z.
    YEAR END: last observation on December 31, counted as potentially censored since nothing
    retained says whether the association would have continued it. A track whose
    observations straddle a boundary with a gap across it is classified by these predicates
    on its first and last observation and by nothing else."""
    june1, oct1, dec31 = _day(year, 6, 1), _day(year, 10, 1), _day(year, 12, 31)
    crossing_in = [t for t in tracks if t["time"][0] < june1 <= t["time"][-1]]
    crossing_out = [t for t in tracks if june1 <= t["time"][0] < oct1 <= t["time"][-1]]
    next_jan1 = _day(year + 1, 1, 1)
    year_end = [t for t in tracks if dec31 <= t["time"][-1] < next_jan1]
    return {"crossing_in": {"tracks": len(crossing_in),
                            "observations_at_or_after_june_1": int(sum(int(np.sum(t["time"] >= june1)) for t in crossing_in)),
                            "predicate": "first observation before June 1 00Z and last at or after it"},
            "crossing_out": {"tracks": len(crossing_out),
                             "predicate": "first observation at or after June 1 00Z and before October 1 00Z, last at or after October 1 00Z"},
            "year_end_potentially_censored": {"tracks": len(year_end),
                                              "predicate": "last observation on December 31, not a demonstrated truncation"}}


def harness_control(new_bytes, retained_bytes):
    """The fresh window export's port tracks against the retained ones, exactly."""
    new, case_new = read_mat_tracks(new_bytes)
    old, case_old = read_mat_tracks(retained_bytes)

    def canon(tracks):
        return X.canonical([{"time": t["time"], "meanlat": t["lat"], "meanlon": t["lon"]} for t in tracks])
    return {"case_id_new": case_new, "case_id_retained": case_old,
            "tracks_new": len(new), "tracks_retained": len(old),
            "passed": bool(case_new and case_old and case_new == case_old and canon(new) == canon(old))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("implementation", "reanalysis"), default="implementation",
                    help="implementation: two programs on one export, whose case identities must be "
                         "equal. reanalysis: one program on two datasets, whose case identities must "
                         "differ and whose producer records must carry equal protocol settings; the "
                         "artifact keys v1 and port then denote side A and side B, labeled by dataset")
    ap.add_argument("--manifest", default=None, help="the retained protocol manifest, required in reanalysis mode")
    ap.add_argument("--v1", required=True, help="side A's tracks (.mat): version 1's, or dataset A's")
    ap.add_argument("--port", required=True, help="side B's tracks (.mat): the port's, or dataset B's")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--record-dir", default="data/aewc")
    ap.add_argument("--record-years", default=RECORD_YEARS)
    ap.add_argument("--published-year-file", default=None,
                    help="the published record's file for the season year, as a context column")
    ap.add_argument("--window-new", default=None, help="fresh export's tracker_port.mat for the harness control")
    ap.add_argument("--window-retained", default=None, help="retained tracker_port.mat it must equal")
    ap.add_argument("--regions-dir", default="data/aewc_v2_pilot/v1_src",
                    help="the archived source directory holding the region polygon .mat files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = time.perf_counter()
    launched = {"script_sha256": X.digest(__file__),
                "git_head_at_launch": X.repository_head(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    digests, raw = {}, {}
    for name, path in (("v1", args.v1), ("port", args.port), ("window_new", args.window_new),
                       ("window_retained", args.window_retained), ("published_year_file", args.published_year_file)):
        if path:
            with open(path, "rb") as fh:
                raw[name] = fh.read()
            digests[name] = {"path": path, "sha256": hashlib.sha256(raw[name]).hexdigest()}
    regions, region_digests = load_regions(args.regions_dir)
    y0, y1 = (int(x) for x in args.record_years.split("-"))
    record_by_year, record_digests = {}, {}
    for y in range(y0, y1 + 1):
        path = os.path.join(args.record_dir, f"ERA-Int_ew_700hPa_{y}_AFR.nc")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            blob = fh.read()
        record_digests[str(y)] = hashlib.sha256(blob).hexdigest()
        record_by_year[y] = read_record_bytes(blob)
    if not record_by_year:
        raise SystemExit("REFUSED: no published record files found for the scale")
    spread = published_spread(record_by_year)

    v1_all, case_v1 = read_mat_tracks(raw["v1"])
    port_all, case_port = read_mat_tracks(raw["port"])
    sides_record = None
    if args.mode == "implementation":
        if not case_v1 or case_v1 != case_port:
            raise SystemExit(f"REFUSED: the two sides are not one case ({case_v1} against {case_port})")
    else:
        # ONE PROGRAM, TWO DATASETS: the identities digest the exported fields and must
        # differ, both are retained, and the producer records' protocol settings must be
        # equal, checked key by key, so the two runs share every rule and differ in data.
        if not args.manifest:
            raise SystemExit("REFUSED: reanalysis mode needs --manifest, the retained protocol manifest")
        with open(args.manifest, "rb") as fh:
            manifest_blob = fh.read()
        manifest = json.loads(manifest_blob.decode())
        manifest_sha256 = hashlib.sha256(manifest_blob).hexdigest()
        digests["manifest"] = {"path": args.manifest, "sha256": manifest_sha256}
        disagreements = manifest_agrees_with_this_instrument(manifest)
        if disagreements:
            raise SystemExit(f"REFUSED: the manifest's {disagreements} are not what this instrument computes")
        if args.regions_dir != (manifest.get("membership") or {}).get("regions_dir"):
            raise SystemExit("REFUSED: the regions directory is not the manifest's")
        prod_a, prod_b = producer_of(raw["v1"]), producer_of(raw["port"])
        problems = producer_problems(prod_a, manifest, manifest_sha256, args.year, "side A") \
            + producer_problems(prod_b, manifest, manifest_sha256, args.year, "side B")
        if problems:
            raise SystemExit("REFUSED: " + "; ".join(problems))
        differing = settings_difference(prod_a, prod_b)
        if differing:
            raise SystemExit(f"REFUSED: the two runs differ in {differing}")
        if prod_a["dataset_specific"]["dataset"] == prod_b["dataset_specific"]["dataset"]:
            raise SystemExit("REFUSED: both sides name the same dataset")
        if not case_v1 or not case_port or case_v1 == case_port:
            raise SystemExit(f"REFUSED: two datasets must carry different case identities ({case_v1} against {case_port})")
        # THE CALENDAR-YEAR CONTRACT: every retained observation lies in the requested year
        y0, y1 = _day(args.year, 1, 1), _day(args.year + 1, 1, 1)
        for side, tracks in (("side A", v1_all), ("side B", port_all)):
            if any(np.any(t["time"] < y0) or np.any(t["time"] >= y1) for t in tracks):
                raise SystemExit(f"REFUSED: {side} holds observations outside {args.year}")
        sides_record = {"note": "in reanalysis mode the keys v1 and port denote side A and side B",
                        "v1": {"label": prod_a["dataset_specific"].get("label"), "dataset": prod_a["dataset_specific"].get("dataset"),
                               "case_id": case_v1, "manifest_sha256": prod_a["protocol_settings"].get("manifest_sha256")},
                        "port": {"label": prod_b["dataset_specific"].get("label"), "dataset": prod_b["dataset_specific"].get("dataset"),
                                 "case_id": case_port, "manifest_sha256": prod_b["protocol_settings"].get("manifest_sha256")},
                        "protocol_settings_equal": True, "tracking_sources_equal": True}
    sides_all = {"v1": v1_all, "port": port_all}
    whole = {"v1": [t for t in v1_all if in_season(t, args.year)],
             "port": [t for t in port_all if in_season(t, args.year)]}
    v1 = [t for t in whole["v1"] if in_record_domain(t, regions)]
    port = [t for t in whole["port"] if in_record_domain(t, regions)]
    sides = {"v1": v1, "port": port}
    columns = {}
    for name, tracks in sides.items():
        columns[name] = {"tracks_all": len(v1_all) if name == "v1" else len(port_all),
                         "tracks_in_season_whole_domain": len(whole[name]),
                         "tracks_in_season_record_domain": len(tracks),
                         "source_regions_in_season": dict(collections.Counter(source_region(t, regions) for t in whole[name])),
                         "distinct_waves": distinct_waves(tracks),
                         "duplication": duplication(tracks), "spacing": spacing(tracks),
                         "boundaries": boundary_lines(sides_all[name], args.year)}
    comparison, counts_only = compare(v1, port, spread)
    payload = {
        "generated_by": "scripts/season_metrics.py", "descriptive": True,
        "tolerances_applied": None,
        "definitions": {"season_membership": "first observation in June to September of the year",
                        "domain": ("the archive's own source-region rule: the first observation, "
                                   "rounded to the quarter degree, tested against the region polygons "
                                   "in the archive's priority order, and the comparison is over the "
                                   f"tracks assigned {RECORD_REGION}; whole-domain totals are reported "
                                   "beside it and enter no ratio"),
                        "lifetime": "the number of observations recorded, not a duration; see each "
                                    "column's spacing block for intervals, gaps and duration in days",
                        "genesis_and_lysis": "the first and the last recorded observation, operational "
                                             "definitions and not physical onset or decay",
                        "boundaries": "each column's boundaries block: crossing in, crossing out and the "
                                      "potentially censored year end, by predicates on the first and last "
                                      "retained observation",
                        "distinct_waves": f"connected components sharing at least {WAVE_SHARED_STEPS} "
                                          f"timesteps at mean separation at most {WAVE_SEPARATION_DEG} deg",
                        "sparse_rule": f"fewer than {MIN_GROUP} tracks on a side, or zero published "
                                       f"variance, means counts only",
                        "scale": "interannual standard deviation, ddof 1, empty years counted as zero",
                        "distributions": f"two-sample KS statistic and percentiles {PERCENTILES}",
                        "bands": "ten-degree bands of genesis longitude, [lo, hi)"},
        **launched, "year": args.year, "mode": args.mode,
        "case_id": case_v1 if args.mode == "implementation" else None, "sides": sides_record,
        "inputs_sha256": digests, "published_record_sha256": record_digests,
        "region_polygons_sha256": region_digests,
        "columns": columns,
        "duplication_port_minus_v1_points": 100.0 * (columns["port"]["duplication"]["fraction"]
                                                     - columns["v1"]["duplication"]["fraction"]),
        "comparison": comparison, "groups_reported_as_counts_only": counts_only,
        "published_spread": spread,
    }
    if args.published_year_file:
        pub = [t for t in read_record_bytes(raw["published_year_file"]) if in_season(t, args.year)]
        payload["published_context_column"] = {
            "tracks_in_season": len(pub), "distinct_waves": distinct_waves(pub),
            "duplication": duplication(pub), "spacing": spacing(pub),
            "source_regions_by_the_rule": dict(collections.Counter(source_region(t, regions) for t in pub)),
            "starts_per_month": {str(m): sum(1 for t in pub if features(t)["month"] == m) for m in SEASON_MONTHS},
            "note": "the archived run over the full year, filtered by the same membership rule as "
                    "the compared sides. This artifact does not record the archived run's "
                    "initialization or inputs, and each compared side's initialization is in its own "
                    "producer record, so no difference against the archive is attributed here"}
    if args.window_new and args.window_retained:
        payload["harness_control"] = harness_control(raw["window_new"], raw["window_retained"])
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    s = comparison["season"]
    print(f"season {args.year}: v1 {s['v1']} tracks, port {s['port']}, difference {s['port_minus_v1']} "
          f"({s.get('difference_over_published_sd', float('nan')):+.2f} published sd); "
          f"{len(counts_only)} groups counts only; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
