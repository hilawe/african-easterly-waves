#!/usr/bin/env python3
"""The false-join test of the clip, declared in section 15 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before it was run.

One development season is replayed again through the production loop under the released
configuration and VCLIP, with the tracks retained. The released loop must reproduce the stored
campaign tracks, the coast-crossing rows and the census's totals, and both configurations must
reproduce section 14's season records exactly. Each multi-system track in the African and Atlantic
sector is then classed by its transitions. At an accompanied observation whose companion set holds
a QTrack system absent from the previous accompanied set, a transition that leaves a system behind
is a SWITCH when a missing system still has a position at that time, a RELAY when every missing
system has ended, and UNDETERMINED when a missing system is alive without a position, and one that
leaves none behind is an ADDITION. `combine` applies section 15's rule.

    python3 scripts/continuity_join_test.py season --year 1990 --out-dir DIR
    python3 scripts/continuity_join_test.py combine --out-dir DIR
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_census as C  # noqa: E402
import continuity_stage_sample as S  # noqa: E402
import continuity_variant_replay as R  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

CONFIGS = {"released": {}, "VCLIP": R.CONFIGS["VCLIP"]}
STAGE_DIR = "docs/aewc_v2/artifacts/continuity_census"
RATE_RISE = 0.25
TOL = 1e-9
CLASS_ORDER = ("switch", "relay", "undetermined", "addition_only", "no_transition")


def spans(q_tracks):
    """Each QTrack system's last time and its positions by step."""
    out = {}
    for t in q_tracks:
        pos = {}
        for la, lo, d in zip(t["lat"], t["lon"], t["time"]):
            if math.isfinite(la) and math.isfinite(lo) and math.isfinite(d):
                pos[C.time_key(d)] = (float(la), float(lo))
        out[t["id"]] = {"last": float(max(t["time"])), "positions": pos}
    return out


def transitions(track, q_index, q_spans):
    """The transitions of one track, in order, as section 15 defines them."""
    out, prev = [], None
    for la, lo, d in zip(track["lat"], track["lon"], track["time"]):
        now = {tid for tid, _ in C.companions(la, lo, d, q_index)}
        if not now:
            continue
        if prev is not None and now - prev:
            lost = prev - now
            if not lost:
                kind = "addition"
            else:
                alive = [a for a in lost if q_spans[a]["last"] >= float(d) - TOL]
                with_position = [a for a in alive if C.time_key(d) in q_spans[a]["positions"]]
                kind = "switch" if with_position else ("undetermined" if alive else "relay")
            out.append({"time": float(d), "kind": kind, "new": sorted(now - prev), "lost": sorted(lost)})
        prev = now
    return out


def track_class(events):
    kinds = {e["kind"] for e in events}
    for k in ("switch", "relay", "undetermined"):
        if k in kinds:
            return k
    return "addition_only" if kinds else "no_transition"


def classify(tracks, q_index, q_spans):
    rows = []
    for t in R.in_sector_tracks(tracks):
        if R.multi_system(t, q_index):
            ev = transitions(t, q_index, q_spans)
            rows.append({"id": t["id"], "class": track_class(ev), "events": ev})
    return rows


def run_season(year, campaign=S.CAMPAIGN, evidence=S.EVIDENCE, artifact_dir=C.ARTIFACT_DIR, census_path=R.CENSUS, stage_dir=STAGE_DIR, loop=R.season_loop):
    if year not in S.YEARS:
        raise SystemExit(f"REFUSED: {year} is not a development season of section 8")
    import coast_crossing_measurement as CC
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    import pilot_track_crosswalk as P
    import qtrack_pairing_pilot as M
    import season_metrics as SM
    stage_path = os.path.join(stage_dir, f"continuity_variant_replay_{year}.json")
    stage = json.load(open(stage_path, encoding="utf-8"))
    if stage["generated_by"]["sha256"] != R.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit(f"REFUSED: the {year} tracking-stage record was not written by the current variant-replay script")
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
    if case["sha256"] != stage["case_sha256"]:
        raise SystemExit(f"REFUSED: the {year} case is not the one the tracking stage read")
    ds = ev["dataset_specific"]; ct, ft = float(ds["coarse_threshold"]), float(ds["fine_threshold"])
    flags = ds["tracker_flags"]; grids = MR.control_grids(case)
    regions, _ = SM.load_regions(art["inputs"]["region_polygons"]["dir"])
    q_tracks = M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]
    q = {t["id"]: t for t in q_tracks}; q_index = C.index_by_time(q_tracks); q_rows = art["sides"]["qtrack"]["tracks"]
    q_spans = spans(q_tracks)
    out = {}
    for name, options in CONFIGS.items():
        finished = loop(case, grids, ct, ft, flags, options)
        if name == "released" and not MR.archive_gate(finished, run)["passed"]:
            raise SystemExit(f"REFUSED: the {year} released loop does not reproduce the stored campaign tracks")
        tracks = R.as_tracks(finished)
        membership = CC.groups_ours(tracks)
        side = CC.measure(tracks, year, regions, M._day(year, 1, 1), membership)
        if name == "released":
            mine = sorted((r["id"], r["outcome"]) for r in side["tracks"])
            theirs = sorted((r["id"], r["outcome"]) for r in art["sides"]["this_record"]["tracks"])
            if mine != theirs:
                raise SystemExit(f"REFUSED: the {year} released loop's coast-crossing rows differ from the artifact's")
        corr = R.correspondence(tracks, side["tracks"], q, q_rows)
        if name == "released" and corr != R.census_row(census, year):
            raise SystemExit(f"REFUSED: the {year} released loop's correspondence differs from the census's")
        counts = {"tracks": len(tracks), **corr, **R.side_effects(tracks, q_index, membership)}
        if counts != stage["configs"][name]:
            raise SystemExit(f"REFUSED: the {year} {name} loop differs from the tracking stage's record")
        rows = classify(tracks, q_index, q_spans)
        classes = {k: sum(1 for r in rows if r["class"] == k) for k in CLASS_ORDER}
        if sum(classes.values()) != counts["multi_system_tracks"]:
            raise SystemExit(f"REFUSED: the {year} {name} classes do not sum to the multi-system count")
        out[name] = {"counts": counts, "classes": classes,
                     "events": {k: sum(1 for r in rows for e in r["events"] if e["kind"] == k) for k in ("switch", "relay", "undetermined", "addition")},
                     "multi_system": rows,
                     "tracks": [{"id": t["id"], "time": t["time"].tolist(), "lat": t["lat"].tolist(), "lon": t["lon"].tolist()} for t in tracks]}
    return {"year": year, "case_sha256": case["sha256"], "stage": {"path": stage_path, "sha256": C.sha256_of(stage_path)},
            "artifact": {"path": art_path, "sha256": C.sha256_of(art_path)}, "census": {"path": census_path, "sha256": C.sha256_of(census_path)},
            "configs": out}


def decide(seasons):
    """Section 15's rule over the seasons together."""
    pooled = {}
    for name in CONFIGS:
        rows = [s["configs"][name] for s in seasons]
        n = sum(r["counts"]["tracks_in_sector"] for r in rows)
        flagged = sum(r["classes"]["switch"] + r["classes"]["undetermined"] for r in rows)
        pooled[name] = {"tracks": sum(r["counts"]["tracks"] for r in rows), "tracks_in_sector": n,
                        "accompanied_steps": sum(r["counts"]["accompanied_steps"] for r in rows), "qtrack_steps": sum(r["counts"]["qtrack_steps"] for r in rows),
                        "carried": sum(r["counts"]["qtrack_classes"]["carried"] for r in rows),
                        "multi_system_tracks": sum(r["counts"]["multi_system_tracks"] for r in rows),
                        **{k: sum(r["classes"][k] for r in rows) for k in CLASS_ORDER},
                        "stationary_tracks": sum(r["counts"]["stationary_tracks"] for r in rows), "duplicate_groups": sum(r["counts"]["duplicate_groups"] for r in rows),
                        "flagged_rate": flagged / n if n else 0.0}
        p = pooled[name]
        p["share"] = p["accompanied_steps"] / p["qtrack_steps"] if p["qtrack_steps"] else 0.0
        p["stationary_rate"] = p["stationary_tracks"] / n if n else 0.0
        p["duplicate_rate"] = p["duplicate_groups"] / n if n else 0.0
    base, v = pooled["released"], pooled["VCLIP"]
    if base["qtrack_steps"] != v["qtrack_steps"]:
        raise SystemExit("REFUSED: the configurations were scored over different QTrack observations")

    def within(key):
        return v[key] <= (1.0 + RATE_RISE) * base[key] + TOL

    checks = {"share_rise": v["share"] - base["share"] >= R.SHARE_RISE - TOL, "carried_rise": v["carried"] > base["carried"],
              "flagged_rate": within("flagged_rate"), "stationary_rate": within("stationary_rate"), "duplicate_rate": within("duplicate_rate")}
    return {"pooled": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in p.items()} for k, p in pooled.items()},
            "checks": checks, "carried_to_freezing": all(checks.values())}


def tables_text(rec):
    P_, D = rec["decision"]["pooled"], rec["decision"]
    L = ["# False-join test of the clip (declared section 15)", "",
         "1990 and 2006 pooled. Rates are per track in the African and Atlantic sector.", "",
         "| configuration | tracks | in sector | multi-system | switch | relay | undetermined | addition only | no transition | switch and undetermined per sector track | stationary per sector track | duplicate groups per sector track |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, p in P_.items():
        L.append(f"| {n} | {p['tracks']} | {p['tracks_in_sector']} | {p['multi_system_tracks']} | {p['switch']} | {p['relay']} | {p['undetermined']} | "
                 f"{p['addition_only']} | {p['no_transition']} | {p['flagged_rate']} | {p['stationary_rate']} | {p['duplicate_rate']} |")
    L += ["", "Per season (switch, relay, undetermined, addition only, no transition; transition events switch, relay, undetermined, addition):", ""]
    for y, cfgs in rec["seasons"].items():
        for n, r in cfgs.items():
            c, e = r["classes"], r["events"]
            L.append(f"- {y} {n}: {c['switch']}, {c['relay']}, {c['undetermined']}, {c['addition_only']}, {c['no_transition']}; "
                     f"events {e['switch']}, {e['relay']}, {e['undetermined']}, {e['addition']}")
    L += ["", "Checks: " + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in D["checks"].items()) + ".",
          f"Carried to freezing: {'yes' if D['carried_to_freezing'] else 'no'}.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("season", "combine"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int)
    a = ap.parse_args(argv)
    if a.mode == "season":
        p = os.path.join(a.out_dir, f"continuity_join_test_{a.year}.json.gz")
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
        rec = {"generated_by": {"script": "scripts/continuity_join_test.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                                "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 15", "configs": CONFIGS}, **run_season(a.year)}
        with gzip.open(p, "xt", encoding="utf-8") as fh:
            json.dump(rec, fh)
        print(json.dumps({n: {"classes": c["classes"], "events": c["events"]} for n, c in rec["configs"].items()}, indent=1))
        return 0
    pj, pm = (os.path.join(a.out_dir, f"continuity_join_test.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    seasons, digests = [], {}
    for y in S.YEARS:
        path = os.path.join(a.out_dir, f"continuity_join_test_{y}.json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            r = json.load(fh)
        if r["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
            raise SystemExit(f"REFUSED: the {y} season record was not written by the current script")
        seasons.append(r); digests[str(y)] = C.sha256_of(path)
    rec = {"generated_by": {"script": "scripts/continuity_join_test.py", "sha256": SCRIPT_SHA256_AT_LOAD, "seasons": digests},
           "seasons": {str(s["year"]): {n: {"classes": c["classes"], "events": c["events"], "counts": c["counts"]} for n, c in s["configs"].items()} for s in seasons},
           "decision": decide(seasons)}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    text = tables_text(rec)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
