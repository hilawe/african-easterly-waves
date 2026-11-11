#!/usr/bin/env python3
"""The tracking stage of developing the merge-placement change, declared in section 13 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before it was run.

One development season is replayed through the production association (`associate_step`,
`prune_stale_tracks` and `finalize_tracks`, called as the all-edge replay calls them) under the
released configuration and the two variants of the coarse merge, VIN and VCLIP. The released run
must reproduce the stored campaign tracks in content and order, the artifact's coast-crossing rows and
the census's own totals, or the stage refuses, which binds the loop and the scoring to production. For each configuration it scores supported correspondence with QTrack (the
census classes of the coast-reaching QTrack tracks starting between 10 and 30 E and the share of
their observations accompanied, beside the coast-crossing outcome of the AEWC v2 cohort tracks
starting there) and four counts in the African and Atlantic sector (tracks accompanied by two or more
distinct QTrack systems at different times, tracks of three days or more whose ends lie within 500 km, duplicate groups
under the instrument's whole-record rule, and all tracks). `combine` applies the provisional rule.

    python3 scripts/continuity_variant_replay.py season --year 1990 --out-dir DIR
    python3 scripts/continuity_variant_replay.py combine --out-dir DIR
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_census as C  # noqa: E402
import continuity_merge_variants as V  # noqa: E402
import continuity_stage_sample as S  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

CONFIGS = {"released": {}, **V.VARIANTS}
STATIONARY_DAYS = 3.0
CENSUS = "docs/aewc_v2/artifacts/continuity_census/continuity_census.json"
SHARE_RISE = 0.05
FALSE_JOIN_RISE = 0.25
TOL = 1e-9                                           # a threshold met exactly in counts is met
FALSE_JOIN_COUNTS = ("multi_system_tracks", "stationary_tracks", "duplicate_groups")


def season_loop(case, grids, ct, ft, flags, options):
    """The production loop over every step with the coarse merge's `options`."""
    import pilot_alledge_replay as A
    from aew.v1port import pipeline as PL
    from aew.v1port.association import associate_step, finalize_tracks, prune_stale_tracks
    exclusive, absorb = bool(flags["exclusive"]), bool(flags["absorb"])
    allc = (np.ones(case["lat_c"].size, bool), np.ones(case["lon_c"].size, bool))
    allf = (np.ones(case["latgrid"].shape[0], bool), np.ones(case["longrid"].shape[1], bool))
    times = np.asarray(case["time"], float).ravel()
    tracks, states = [], []
    for step in range(times.size):
        t = float(times[step])
        fields = A.prepared_fields(case, step, *allc, *allf)
        waves = A.detect_from_prepared(fields, t, ct, ft, grids, absorb=absorb, **options)
        um, vm = PL._median_over(fields["fine_zonal_wind_smoothed"]), PL._median_over(fields["fine_meridional_wind_smoothed"])
        tracks, states = associate_step(tracks, states, waves, step, um, vm, exclusive=exclusive)
        tracks, states = prune_stale_tracks(tracks, states, step, t, waves)
    finished = finalize_tracks(tracks, total_steps=times.size)
    return [{"time": np.asarray(f["time"], float), "lat": np.asarray(f["meanlat"], float), "lon": np.asarray(f["meanlon"], float)} for f in finished]


def as_tracks(finished):
    return [{"id": k, "time": f["time"], "lat": f["lat"], "lon": f["lon"], "n_valid": int(f["time"].size)} for k, f in enumerate(finished)]


def in_sector_tracks(tracks, sector=V.SECTOR):
    return [t for t in tracks if np.any((t["lon"] >= sector[0]) & (t["lon"] <= sector[1]) & (t["lat"] >= sector[2]) & (t["lat"] <= sector[3]))]


def multi_system(track, q_index):
    """True when two distinct QTrack systems accompany the track within 500 km at different
    times. Given two or more systems among its companions, such a pair exists exactly when the
    accompanied observations span two or more times."""
    ids, times = set(), set()
    for la, lo, d in zip(track["lat"], track["lon"], track["time"]):
        found = [tid for tid, _ in C.companions(la, lo, d, q_index)]
        if found:
            ids.update(found); times.add(C.time_key(d))
    return len(ids) >= 2 and len(times) >= 2


def stationary(track, days=STATIONARY_DAYS):
    """A track lasting three days or more whose start and end lie within 500 km."""
    return float(track["time"][-1] - track["time"][0]) >= days - 1e-9 and \
        float(C.km(track["lat"][0], track["lon"][0], track["lat"][-1], track["lon"][-1])) < C.RADIUS_KM


def side_effects(tracks, q_index, membership):
    sec = in_sector_tracks(tracks)
    ids = {t["id"] for t in sec}
    groups = {g for tid, g in membership.items() if tid in ids}
    return {"tracks_in_sector": len(sec), "multi_system_tracks": sum(1 for t in sec if multi_system(t, q_index)),
            "stationary_tracks": sum(1 for t in sec if stationary(t)), "duplicate_groups": len(groups)}


def correspondence(ours, ours_rows, q, q_rows):
    """The census of section 3 on one configuration's tracks."""
    idx = C.index_by_time(ours)
    classes = {c: 0 for c in C.Q_CLASSES}; hits = steps = 0
    for r in q_rows:
        if not (C.START_LONS[0] <= r["first"]["lon"] < C.START_LONS[1]):
            continue
        t = q[r["id"]]
        upto = r["first_atlantic_side"]["index"] if r["outcome"] == "atlantic_side" else r["last_used"]["index"]
        h, n = C.accompanied_share(t, idx, upto); hits += h; steps += n
        if r["outcome"] == "atlantic_side":
            s = C.start_set(t, idx, upto)
            ev = {tid for tid, _ in C.companions(t["lat"][upto], t["lon"][upto], t["time"][upto], idx)}
            classes[C.classify_q(s, ev)] += 1
    band = [r for r in ours_rows if C.START_LONS[0] <= r["first"]["lon"] < C.START_LONS[1]]
    return {"qtrack_classes": classes, "accompanied_steps": hits, "qtrack_steps": steps,
            "aewc_cohort": len(band), "aewc_atlantic_side": sum(1 for r in band if r["outcome"] == "atlantic_side")}


def census_row(census, year):
    """The census's own totals for one season, which the released replay must reproduce."""
    se = next(s for s in census["seasons"] if s["year"] == year)
    q = se["qtrack"]
    return {"qtrack_classes": {c: sum(1 for r in q if r.get("class") == c) for c in C.Q_CLASSES},
            "accompanied_steps": sum(r["accompanied_steps"] for r in q), "qtrack_steps": sum(r["steps"] for r in q),
            "aewc_cohort": len(se["this_record"]), "aewc_atlantic_side": sum(1 for r in se["this_record"] if r["outcome"] == "atlantic_side")}


def run_season(year, campaign=S.CAMPAIGN, evidence=S.EVIDENCE, artifact_dir=C.ARTIFACT_DIR, census_path=CENSUS, loop=season_loop):
    if year not in S.YEARS:
        raise SystemExit(f"REFUSED: {year} is not a development season of section 8")
    import coast_crossing_measurement as CC
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    import pilot_track_crosswalk as P
    import qtrack_pairing_pilot as M
    import season_metrics as SM
    census = json.load(open(census_path, encoding="utf-8"))
    if census["generated_by"]["sha256"] != C.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the census was not written by the current census script")
    art_path = os.path.join(artifact_dir, C.ARTIFACT.format(year=year))
    art = json.load(open(art_path, encoding="utf-8"))
    for key in ("this_record", "qtrack"):
        if C.sha256_of(art["inputs"][key]["path"]) != art["inputs"][key]["sha256"]:
            raise SystemExit(f"REFUSED: {art['inputs'][key]['path']} is not the file the {year} artifact names")
    run_dir = os.path.join(campaign, f"era5_{year}")
    run = P.load_run(run_dir, year=year)
    ev = json.load(open(os.path.join(evidence, f"era5_{year}", f"tracking_era5_{year}.json"), encoding="utf-8"))
    if run["tracks_sha256"] != ev["dataset_specific"]["tracks_sha256"] or run["tracks_sha256"] != art["inputs"]["this_record"]["sha256"]:
        raise SystemExit(f"REFUSED: the {year} campaign run, evidence and artifact do not name one set of tracks")
    case = SD.load_case(os.path.join(run_dir, "tracker_case.mat"), ev)
    ds = ev["dataset_specific"]; ct, ft = float(ds["coarse_threshold"]), float(ds["fine_threshold"])
    flags = ds["tracker_flags"]; grids = MR.control_grids(case)
    regions, _ = SM.load_regions(art["inputs"]["region_polygons"]["dir"])
    q_tracks = M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]
    q = {t["id"]: t for t in q_tracks}; q_index = C.index_by_time(q_tracks); q_rows = art["sides"]["qtrack"]["tracks"]
    out = {}
    for name, options in CONFIGS.items():
        finished = loop(case, grids, ct, ft, flags, options)
        if name == "released":
            gate = MR.archive_gate(finished, run)
            if not gate["passed"]:
                raise SystemExit(f"REFUSED: the {year} released loop does not reproduce the stored campaign tracks")
        tracks = as_tracks(finished)
        membership = CC.groups_ours(tracks)
        side = CC.measure(tracks, year, regions, M._day(year, 1, 1), membership)
        if name == "released":
            mine = sorted((r["id"], r["outcome"]) for r in side["tracks"])
            theirs = sorted((r["id"], r["outcome"]) for r in art["sides"]["this_record"]["tracks"])
            if mine != theirs:
                raise SystemExit(f"REFUSED: the {year} released loop's coast-crossing rows differ from the artifact's")
        corr = correspondence(tracks, side["tracks"], q, q_rows)
        if name == "released" and corr != census_row(census, year):
            raise SystemExit(f"REFUSED: the {year} released loop's correspondence differs from the census's")
        out[name] = {"tracks": len(tracks), **corr, **side_effects(tracks, q_index, membership)}
    return {"year": year, "case_sha256": case["sha256"], "artifact": {"path": art_path, "sha256": C.sha256_of(art_path)},
            "census": {"path": census_path, "sha256": C.sha256_of(census_path)}, "configs": out}


def decide(seasons):
    """Section 13's provisional rule over the seasons together, per variant."""
    pooled = {}
    for name in CONFIGS:
        rows = [s["configs"][name] for s in seasons]
        pooled[name] = {"share": sum(r["accompanied_steps"] for r in rows) / max(1, sum(r["qtrack_steps"] for r in rows)),
                        "qtrack_steps": sum(r["qtrack_steps"] for r in rows), "carried": sum(r["qtrack_classes"]["carried"] for r in rows),
                        **{k: sum(r[k] for r in rows) for k in FALSE_JOIN_COUNTS + ("tracks_in_sector", "aewc_cohort", "aewc_atlantic_side")}}
    base = pooled["released"]; verdicts = {}
    if len({pooled[n]["qtrack_steps"] for n in CONFIGS}) != 1:
        raise SystemExit("REFUSED: the configurations were scored over different QTrack observations")
    for name in V.VARIANTS:
        p = pooled[name]
        rises = {k: (p[k] - base[k]) / base[k] if base[k] else (float("inf") if p[k] else 0.0) for k in FALSE_JOIN_COUNTS}
        verdicts[name] = {"share_rise": round(p["share"] - base["share"], 4), "carried_rise": p["carried"] - base["carried"],
                          "false_join_rises": {k: round(v, 4) for k, v in rises.items()},
                          "carried_to_freezing": bool(p["share"] - base["share"] >= SHARE_RISE - TOL and p["carried"] > base["carried"]
                                                      and all(v <= FALSE_JOIN_RISE + TOL for v in rises.values()))}
    return {"pooled": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in pooled.items()}, "verdicts": verdicts}


def tables_text(rec):
    P_ = rec["decision"]["pooled"]
    L = ["# Tracking stage of the merge-placement change (declared section 13)", "",
         "1990 and 2006 pooled. Share is the fraction of coast-reaching QTrack observations (10 to 30 E starts) accompanied within 500 km.", "",
         "| configuration | share accompanied | carried | AEWC v2 cohort, Atlantic side | tracks in sector | multi-system | stationary | duplicate groups |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, p in P_.items():
        L.append(f"| {n} | {p['share']} | {p['carried']} | {p['aewc_atlantic_side']} of {p['aewc_cohort']} | {p['tracks_in_sector']} | "
                 f"{p['multi_system_tracks']} | {p['stationary_tracks']} | {p['duplicate_groups']} |")
    L += ["", "Per season (share accompanied, carried, multi-system, stationary, duplicate groups, tracks in sector):", ""]
    for y, cfgs in rec["seasons"].items():
        for n, r in cfgs.items():
            L.append(f"- {y} {n}: {r['accompanied_steps']} of {r['qtrack_steps']}, {r['qtrack_classes']['carried']}, {r['multi_system_tracks']}, "
                     f"{r['stationary_tracks']}, {r['duplicate_groups']}, {r['tracks_in_sector']}")
    L += ["", "| variant | share rise | carried rise | false-join rises | carried to freezing |", "| --- | --- | --- | --- | --- |"]
    for n, v in rec["decision"]["verdicts"].items():
        L.append(f"| {n} | {v['share_rise']} | {v['carried_rise']} | {v['false_join_rises']} | {'yes' if v['carried_to_freezing'] else 'no'} |")
    L += ["", f"Provisional rule: share rise of at least {SHARE_RISE:g}, more carried tracks, and no false-join count rising by more than {FALSE_JOIN_RISE:.0%}.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("season", "combine"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int)
    a = ap.parse_args(argv)
    if a.mode == "season":
        p = os.path.join(a.out_dir, f"continuity_variant_replay_{a.year}.json")
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
        rec = {"generated_by": {"script": "scripts/continuity_variant_replay.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                                "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 13", "configs": CONFIGS}, **run_season(a.year)}
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        print(json.dumps(rec["configs"], indent=1))
        return 0
    pj, pm = (os.path.join(a.out_dir, f"continuity_variant_replay.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    seasons = []
    for y in S.YEARS:
        r = json.load(open(os.path.join(a.out_dir, f"continuity_variant_replay_{y}.json"), encoding="utf-8"))
        if r["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
            raise SystemExit(f"REFUSED: the {y} season record was not written by the current script")
        seasons.append(r)
    rec = {"generated_by": {"script": "scripts/continuity_variant_replay.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                            "seasons": {str(s["year"]): C.sha256_of(os.path.join(a.out_dir, f"continuity_variant_replay_{s['year']}.json")) for s in seasons}},
           "seasons": {str(s["year"]): s["configs"] for s in seasons}, "decision": decide(seasons)}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    text = tables_text(rec)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
