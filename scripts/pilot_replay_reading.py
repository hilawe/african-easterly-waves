#!/usr/bin/env python3
"""Read the two runs' association replays side by side for one season's families: at
each window step, every final candidate inside the family's box, what the association did
with it (which live track claimed it in which pass, or which new track it seeded), the
fate of every track involved, and the nearest candidate of the other run at the same step
with its own handling. Nothing is recomputed. Every statement is a lookup in the replay
records, which carry their gates.

    python3 scripts/pilot_replay_reading.py --replay-control <B json> --replay-treatment <C json> --families <json> --out <fresh json>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

MATCH_DEG = 1.0


def load_replay_bytes(path):
    """A replay record parsed from the bytes that are hashed, so a consumer binds to the
    complete content it read and not to the metadata inside it."""
    import hashlib
    with open(path, "rb") as fh:
        blob = fh.read()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def in_box(c, box):
    return box["lat"][0] <= c["lat_mean"] <= box["lat"][1] and box["lon"][0] <= c["lon_mean"] <= box["lon"][1]


def handling(entry, index):
    """What the association did with candidate `index` at this step."""
    out = []
    for c in entry.get("claims", []):
        if c["candidate"] == index:
            out.append({"claimed_by": c["birth"], "pass": c["pass"], "n_obs_after": c["n_obs"]})
    for s in entry.get("seeds", []):
        if s["candidate"] == index:
            out.append({"seeded": s["birth"]})
    return out


def fate_of(replay, birth):
    f = dict(replay["fates"][birth])
    h = replay["histories"].get(birth)
    if h:
        f["steps"] = h["steps"]
        f["lon_claimed"] = [round(x, 3) for x in h["lon_claimed"]]
        f["lat_claimed"] = [round(x, 3) for x in h["lat_claimed"]]
    return f


def read_family(fam, rb, rc):
    steps_b = {e["step"]: e for e in rb["steps"]}
    steps_c = {e["step"]: e for e in rc["steps"]}
    box = fam["box"]
    rows, births = [], {"B": set(), "C": set()}
    for k in sorted(set(steps_b) & set(steps_c)):
        eb, ec = steps_b[k], steps_c[k]
        if k not in rb["family_steps"][fam["name"]]:
            continue
        row = {"step": k, "date": eb["date"], "B": [], "C": [],
               "due_in_box": {"B": [t for t in eb["live_before"] if t["due"] and box["lat"][0] <= t["last_lat"] <= box["lat"][1] and box["lon"][0] <= t["last_lon"] <= box["lon"][1]],
                              "C": [t for t in ec["live_before"] if t["due"] and box["lat"][0] <= t["last_lat"] <= box["lat"][1] and box["lon"][0] <= t["last_lon"] <= box["lon"][1]]}}
        for mine, other, em, eo, run in (("B", "C", eb, ec, "B"), ("C", "B", ec, eb, "C")):
            for c in em["candidates"]:
                if not in_box(c, box):
                    continue
                d = [float(np.hypot(c["lat_mean"] - o["lat_mean"], c["lon_mean"] - o["lon_mean"])) for o in eo["candidates"]]
                j = int(np.argmin(d)) if d else None
                h = handling(em, c["index"])
                for x in h:
                    births[run].add(x.get("claimed_by") or x.get("seeded"))
                nearest = None
                if j is not None:
                    o = eo["candidates"][j]
                    nearest = {"index": j, "lat_mean": round(o["lat_mean"], 3), "lon_mean": round(o["lon_mean"], 3), "distance_deg": round(d[j], 3),
                               "same_region_digest": o["region_sha256"] == c["region_sha256"], "handling_in_other_run": handling(eo, j)}
                    for x in nearest["handling_in_other_run"]:
                        births[other].add(x.get("claimed_by") or x.get("seeded"))
                row[mine].append({"index": c["index"], "lat_mean": round(c["lat_mean"], 3), "lon_mean": round(c["lon_mean"], 3), "n_points": c["n_points"],
                                  "u_median": None if c["u_median"] is None else round(c["u_median"], 2), "v_median": None if c["v_median"] is None else round(c["v_median"], 2),
                                  "region_sha256": c["region_sha256"][:12], "handling": h, "nearest_other_run": nearest})
        # the matching reads of due tracks in the box: what was inside their polygons
        for run, e in (("B", eb), ("C", ec)):
            due = {t["birth"] for t in row["due_in_box"][run]}
            row["matching_in_box_" + run] = [m for m in e["matching"] if m["birth"] in due]
            for t in row["due_in_box"][run]:
                births[run].add(t["birth"])
        row["due_in_box"] = {r: [{"birth": t["birth"], "n_obs": t["n_obs"], "last": [round(t["last_lat"], 3), round(t["last_lon"], 3)], "due": t["due"],
                                  "est": t["est_" + t["due"]]} for t in v] for r, v in row["due_in_box"].items()}
        rows.append(row)
    fates = {"B": {b: fate_of(rb, b) for b in sorted(births["B"])}, "C": {b: fate_of(rc, b) for b in sorted(births["C"])}}
    named = {}
    for run, r in (("B", rb), ("C", rc)):
        for name, info in r["named_tracks"].items():
            if info["family"] == fam["name"]:
                named[name] = {"birth": info["birth"], "fate": fate_of(r, info["birth"])}
    return {"name": fam["name"], "window": fam["window"], "box": box, "steps": rows, "fates_of_tracks_involved": fates, "named_tracks": named}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--replay-control", "--replay-treatment", "--families", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    (rb, sha_b), (rc, sha_c) = load_replay_bytes(args.replay_control), load_replay_bytes(args.replay_treatment)
    for r in (rb, rc):
        if not r["gate"]["passed"]:
            raise SystemExit(f"REFUSED: the {r['label']} {r['year']} replay did not pass its gate")
    if rb["year"] != rc["year"] or rb["label"] != "B" or rc["label"] != "C":
        raise SystemExit("REFUSED: the replays are not a B and a C of one year")
    fams = json.load(open(args.families))["families"]
    out = {"generated_by": "scripts/pilot_replay_reading.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": rb["year"],
           "families_file": args.families, "families_sha256": X.digest(args.families),
           "replays": {r["label"]: {"path": p, "replay_sha256": sha, "script_sha256": r["script_sha256"], "gate": r["gate"], "case_sha256": r["case_sha256"], "tracks_sha256": r["tracks_sha256"]}
                       for r, p, sha in ((rb, args.replay_control, sha_b), (rc, args.replay_treatment, sha_c))},
           "families": [read_family(f, rb, rc) for f in fams]}
    X.publish_json(args.out, out, exclusive=True)
    for fam in out["families"]:
        print(f"== {fam['name']}")
        for row in fam["steps"]:
            for run in "BC":
                for c in row[run]:
                    near = c["nearest_other_run"]
                    print(f"{row['date']} {run} cand {c['index']} ({c['lat_mean']}, {c['lon_mean']}) {c['handling']} | nearest other {near['distance_deg'] if near else None} deg {near['handling_in_other_run'] if near else ''}")
        for run in "BC":
            for b, f in fam["fates_of_tracks_involved"][run].items():
                print(f"  {run} {b}: {f['fate']} born {f['born_step']} n_obs {f.get('n_obs')} {'finished ' + str(f.get('finished_index')) if f['fate'] == 'finished' else 'fate step ' + str(f.get('fate_step'))}")
        for name, info in fam["named_tracks"].items():
            print(f"  named {name}: birth {info['birth']} {info['fate']['fate']} n_obs {info['fate'].get('n_obs')} steps {info['fate'].get('steps')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
