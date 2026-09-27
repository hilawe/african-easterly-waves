#!/usr/bin/env python3
"""Threshold sensitivities of the protocol campaign: a dataset's retained cases
replayed through the entry point's own tracking replay with one named alternative
threshold pair in place of the dataset's own Rule A thresholds, then compared year by
year through the reviewed instrument, and summarized with the collector's row builder
and multi-year block. Two pairs exist, each an experiment of its own: the archived
constants of the 2013 tracker on ERA-Interim, and ERA-Interim's Rule A values
transferred numerically to ERA5 (a common numerical threshold-pair comparison, each
dataset keeping its own climatology and anomaly fields).

EXECUTION. A replay runs FROM A FROZEN CODE SNAPSHOT (a checkout of one commit, given by
--code-snapshot and --snapshot-commit), refuses to run from anywhere else, binds every
record to that commit, and requires the snapshot's entry point to be the one that made
the case. One launch at a time per output root: the launch takes a lock naming its
process, registers every worker's process id, and terminates its workers when it is
stopped, so a stopped launch leaves no worker writing and a second launch cannot start
while a first is alive. A first attempt of the ERA-Interim experiment lost eight minutes
to workers that outlived a stopped parent under a driver file edited meanwhile, which
is what these three measures answer.

WHAT IS HELD FIXED, BY CONSTRUCTION. A retained case holds every field the tracker
reads after preparation and the producer record of the run that made it, so replaying
it changes nothing upstream of detection: inputs, climatology, anomalies, decimation,
grids, initialization and run length are the campaign's. Each case is read only after
its digest matches the collected record's. Only the two detection thresholds change.

WHAT THIS IS NOT. Not a calibration, not a tuning, not a second production rule. The
thresholds are the archived constants of the 2013 tracker for ERA-Interim at 700 hPa,
taken from the port's own profile table, and they are written into every record beside
the Rule A values they replace, under the label "sensitivity".

    .venv/bin/python3 <snapshot>/scripts/run_threshold_sensitivity.py replay --dataset era5 --pair eraint-rule-a \\
        --code-snapshot <snapshot> --snapshot-commit <sha> --campaign data/protocol_runs/campaign \\
        --evidence docs/aewc_v2/evidence/validation/protocol_campaign --manifest <json> --out-dir <root> --workers 4
    .venv/bin/python3 <snapshot>/scripts/run_threshold_sensitivity.py control --dataset era5 --year 1990 ...
    .venv/bin/python3 scripts/run_threshold_sensitivity.py compare --dataset era5 --against own ...
    .venv/bin/python3 scripts/run_threshold_sensitivity.py compare --dataset era5 --against eraint-baseline ...
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import exact_tracks as X  # noqa: E402

DATASET = "eraint"                                   # the default dataset; the command names its own
LABEL = "the archived constants of version 1 for ERA-Interim at 700 hPa"
SOURCE = "src/aew/v1port/profiles.py, V1_ERAINT_700"
PAIRS = {"archived": {"datasets": ("eraint",), "label": LABEL},
         "eraint-rule-a": {"datasets": ("era5",), "label": "ERA-Interim's Rule A thresholds transferred numerically to ERA5"}}
DEFAULT_PAIR_SOURCE = "docs/aewc_v2/artifacts/thresholds_protocol_eraint_1979_2010.json"


def archived_constants():
    from aew.v1port import profiles
    coarse, fine = profiles.V1_ERAINT_700
    return float(coarse), float(fine)


def threshold_pair(pair, dataset, pair_source=None):
    """The named pair's coarse and fine values, its label and its source. `archived` is the
    port's own constant table; `eraint-rule-a` is read from the retained ERA-Interim
    calibration artifact and bound to its digest. A pair is refused on a dataset it was
    not defined for, so the archived constants cannot be run on ERA5 by accident."""
    if pair not in PAIRS or dataset not in PAIRS[pair]["datasets"]:
        raise SystemExit(f"REFUSED: the pair {pair!r} is not defined for {dataset}")
    if pair == "archived":
        coarse, fine = archived_constants()
        return coarse, fine, PAIRS[pair]["label"], {"kind": "port constant table", "where": SOURCE}
    path = pair_source or DEFAULT_PAIR_SOURCE
    art = json.load(open(path))
    if art.get("prefix") != "eraint" or art.get("schema") != "thresholds-v2":
        raise SystemExit(f"REFUSED: {path} is not the ERA-Interim thresholds-v2 calibration artifact")
    coarse, fine = float(art["coarse"]["threshold"]), float(art["fine"]["threshold"])
    return coarse, fine, PAIRS[pair]["label"], {"kind": "calibration artifact", "path": path, "sha256": _sha256(path), "case_id": art.get("case_id")}


def snapshot_problems(snapshot, commit, embedded):
    """Why this process is NOT running from the frozen snapshot of `commit` whose entry
    point made the case: the driver file must lie under the snapshot, the snapshot must
    hold the entry point with the digest the case's record names, and the commit must be
    named. Empty when bound."""
    problems = []
    if not snapshot or not commit:
        problems.append("no code snapshot or commit was named")
        return problems
    here, root = os.path.realpath(__file__), os.path.realpath(snapshot)
    if not here.startswith(root + os.sep):
        problems.append(f"this driver runs from {here}, not from the snapshot {root}")
    entry = os.path.join(root, "scripts", "export_protocol_case.py")
    named = (embedded.get("source_sha256") or {}).get("scripts/export_protocol_case.py")
    if not os.path.exists(entry) or _sha256(entry) != named:
        problems.append("the snapshot's entry point is not the one that made the case")
    if not re.fullmatch(r"[0-9a-f]{7,40}", str(commit)):
        problems.append(f"{commit!r} is not a commit")
    return problems


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _record_of(evidence, year, dataset=DATASET):
    path = os.path.join(evidence, f"{dataset}_{year}", f"tracking_{dataset}_{year}.json")
    with open(path) as fh:
        return json.load(fh)


def replay_year(campaign, evidence, manifest_path, out_dir, year, coarse, fine, dataset=DATASET, pair=None,
                snapshot=None, commit=None, control=False):
    """One year: verify the case against the collected record, replay tracking with the
    named pair (or, as a control, the case's own thresholds), publish the tracks and a
    record exclusively. Returns the record. Refuses (SystemExit) rather than guessing at
    any mismatch. A control replays with the collected record's own thresholds and
    requires its tracks to equal the campaign's exactly, canonical set against set."""
    import export_protocol_case as E
    from scipy.io import loadmat
    manifest, manifest_sha256 = E.load_manifest(manifest_path)
    started = time.perf_counter()
    collected = _record_of(evidence, year, dataset)["dataset_specific"]
    case_path = os.path.join(campaign, f"{dataset}_{year}", "tracker_case.mat")
    if not os.path.exists(case_path):
        raise SystemExit(f"REFUSED: {case_path} is absent")
    with open(case_path, "rb") as fh:
        blob = fh.read()
    case_sha256 = hashlib.sha256(blob).hexdigest()
    if case_sha256 != collected.get("case_sha256"):
        raise SystemExit(f"REFUSED: {case_path} does not have the digest the collected record names")
    raw = loadmat(io.BytesIO(blob))
    del blob
    embedded = json.loads(str(raw["producer_json"][0]))
    import season_metrics as S
    problems = S.producer_problems(embedded, manifest, manifest_sha256, year, "v1")   # the comparison gate's own check
    if problems or (embedded.get("dataset_specific") or {}).get("dataset") != dataset:
        raise SystemExit(f"REFUSED: the case at {case_path} is not an {dataset} {year} tracking case under this manifest: "
                         + "; ".join(problems or ["dataset differs"]))
    bound = snapshot_problems(snapshot, commit, embedded)
    if bound:
        raise SystemExit("REFUSED: not bound to a frozen snapshot: " + "; ".join(bound))
    rule_a = {"coarse": collected.get("coarse_threshold"), "fine": collected.get("fine_threshold"),
              "calibration_sha256": collected.get("calibration_sha256")}
    if rule_a["coarse"] is None or rule_a["fine"] is None:
        raise SystemExit("REFUSED: the collected record carries no Rule A thresholds to replace")
    if control:
        coarse, fine = rule_a["coarse"], rule_a["fine"]
    payload = {k: raw[k] for k in ("lat_c", "lon_c", "latgrid", "longrid", "time", "u_c", "v_c", "currv_anom_c",
                                   "advcurrv_anom_c", "u", "v", "currv_anom")}
    payload["case_id"] = str(raw["case_id"][0])
    if payload["case_id"] != collected.get("case_id"):
        raise SystemExit("REFUSED: the case id inside the case is not the collected record's")
    flags = manifest["tracker_flags"]
    t0 = time.perf_counter()
    tracks = E.track_case(payload, coarse, fine, exclusive=bool(flags["exclusive"]), absorb=bool(flags["absorb"]))
    tracking_seconds = round(time.perf_counter() - t0, 1)
    if control:
        campaign_tracks = os.path.join(evidence, f"{dataset}_{year}", "tracker_port.mat")
        with open(campaign_tracks, "rb") as fh:
            retained, _ = S.read_mat_tracks(fh.read())
        mine = [{"time": np.asarray(t["time"], float), "meanlat": np.asarray(t["meanlat"], float), "meanlon": np.asarray(t["meanlon"], float)} for t in tracks]
        theirs = [{"time": t["time"], "meanlat": t["lat"], "meanlon": t["lon"]} for t in retained]
        if X.canonical(mine) != X.canonical(theirs):
            raise SystemExit(f"CONTROL FAILED: the replay of {dataset} {year} under its own thresholds does not reproduce the campaign's tracks "
                             f"({len(mine)} against {len(theirs)})")
    pair_info = pair or {"name": "archived", "label": LABEL, "source": {"kind": "port constant table", "where": SOURCE}}
    record = json.loads(json.dumps(embedded))
    record["stage"] = "tracking"
    record["producer"] = "scripts/run_threshold_sensitivity.py"
    record["produced_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["source_sha256"] = {**(embedded.get("source_sha256") or {}), "scripts/run_threshold_sensitivity.py": X.digest(__file__)}
    record["sensitivity"] = {"what_changed": "the two detection thresholds only" if not control else "nothing: a control under the case's own thresholds",
                             "control": bool(control), "pair": pair_info["name"], "label": pair_info["label"] if not control else "control, the dataset's own Rule A thresholds",
                             "source": pair_info["source"], "dataset": dataset,
                             "coarse_threshold": coarse, "fine_threshold": fine, "replaces_rule_a": rule_a,
                             "replayed_from_case": case_path, "case_sha256": case_sha256,
                             "campaign_record_tracks_sha256": collected.get("tracks_sha256"),
                             "code_snapshot": {"commit": commit, "path": os.path.realpath(snapshot)}}
    ds = record["dataset_specific"]
    ds.update({"coarse_threshold": coarse, "fine_threshold": fine, "case_sha256": case_sha256, "case_id": payload["case_id"],
               "calibration_path": None, "calibration_sha256": None,
               "phase_seconds": {"tracking_seconds": tracking_seconds}})
    run_dir = os.path.join(out_dir, f"{dataset}_{year}")
    os.makedirs(run_dir, exist_ok=True)
    port_path = os.path.join(run_dir, "tracker_port.mat")
    port = {"n": float(len(tracks)), "case_id": payload["case_id"], "exclusive": float(bool(flags["exclusive"])),
            "absorb": float(bool(flags["absorb"])), "producer_json": json.dumps(record, sort_keys=True)}
    for i, t in enumerate(tracks):
        port[f"lat{i}"] = np.asarray(t["meanlat"], float)
        port[f"lon{i}"] = np.asarray(t["meanlon"], float)
        port[f"time{i}"] = np.asarray(t["time"], float)
    E.publish_mat(port_path, port)
    ds["tracks_sha256"] = _sha256(port_path)
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    E.publish_record(os.path.join(run_dir, f"tracking_{dataset}_{year}.json"), record)
    return record, len(tracks)


def sensitivity_record_problems(record_path, evidence, year, coarse, fine, manifest_sha256, dataset=DATASET, pair_name=None, commit=None):
    """Why a record at a sensitivity path is NOT this experiment's record of that year: a
    review copied a genuine Rule A campaign record and its tracks to the path and the
    driver skipped the year as done. Beyond the driver's own identity predicate, the
    record must carry the sensitivity block with these thresholds, the manifest's digest,
    the collected record's case id, and a tracks file beside it."""
    import campaign_record_ok as R
    problems = list(R.problems(record_path))
    try:
        record = json.load(open(record_path))
    except (OSError, ValueError) as exc:
        return problems + [f"unreadable: {exc}"]
    sens = record.get("sensitivity") or {}
    ds = record.get("dataset_specific") or {}
    if sens.get("coarse_threshold") != coarse or sens.get("fine_threshold") != fine \
            or ds.get("coarse_threshold") != coarse or ds.get("fine_threshold") != fine:
        problems.append("the record is not a run under the archived constants" if pair_name in (None, "archived")
                        else f"the record is not a run under the pair {pair_name}")
    if pair_name is not None and sens.get("pair") != pair_name:
        problems.append(f"the record names the pair {sens.get('pair')!r}, not {pair_name!r}")
    if commit is not None and (sens.get("code_snapshot") or {}).get("commit") != commit:
        problems.append("the record was not made from this code snapshot")
    if sens.get("control"):
        problems.append("the record is a control, not an experiment run")
    if (record.get("protocol_settings") or {}).get("manifest_sha256") != manifest_sha256:
        problems.append("the record's manifest digest is not this manifest's")
    try:
        collected = _record_of(evidence, year, dataset)["dataset_specific"]
    except (OSError, ValueError, KeyError):
        collected = {}
    if ds.get("case_id") != collected.get("case_id") or sens.get("case_sha256") != collected.get("case_sha256"):
        problems.append("the record's case is not the collected campaign record's case")
    tracks = os.path.join(os.path.dirname(record_path), "tracker_port.mat")
    if not os.path.exists(tracks):
        problems.append("no tracks file beside the record")
        return problems
    # the record beside the tracks can be relabeled, the one inside the tracks file was
    # written by the run itself: the two must agree on the thresholds, the case and the manifest
    try:
        from scipy.io import loadmat
        inside = json.loads(str(loadmat(tracks, variable_names=["producer_json"])["producer_json"][0]))
    except Exception as exc:                                              # a reporting boundary: a problem, never a pass
        return problems + [f"the tracks file carries no readable producer record: {type(exc).__name__}: {exc}"]
    ids = inside.get("dataset_specific") or {}
    isens = inside.get("sensitivity") or {}
    if (ids.get("coarse_threshold"), ids.get("fine_threshold"), isens.get("coarse_threshold"), isens.get("fine_threshold")) != (coarse, fine, coarse, fine):
        problems.append("the producer record inside the tracks file was not written under the archived constants" if pair_name in (None, "archived")
                        else f"the producer record inside the tracks file was not written under the pair {pair_name}")
    if ids.get("case_id") != ds.get("case_id") or (inside.get("protocol_settings") or {}).get("manifest_sha256") != manifest_sha256:
        problems.append("the producer record inside the tracks file does not agree with the record beside it on the case or the manifest")
    # the sidecar can be relabeled in every field; the record inside the tracks was written by
    # the run, so pair, commit, control and dataset must agree between the two
    for key, inside_value, beside_value in (("pair", isens.get("pair"), sens.get("pair")),
                                            ("commit", (isens.get("code_snapshot") or {}).get("commit"), (sens.get("code_snapshot") or {}).get("commit")),
                                            ("control", bool(isens.get("control")), bool(sens.get("control"))),
                                            ("dataset", ids.get("dataset"), ds.get("dataset"))):
        if inside_value != beside_value:
            problems.append(f"the producer record inside the tracks file does not agree with the record beside it on the {key} ({inside_value!r} against {beside_value!r})")
    return problems


_LAUNCH = {"dir": None}


def _register_worker(launch_dir):
    """Each worker writes its process id under its launch's registry, so the launch's stop
    handler, or a person, can terminate every worker of this launch and no other. A
    worker keeps the default signal disposition, so a stop ends it at once: under a fork
    context it would otherwise inherit the launch's own stop handler. A worker that starts
    after its launch was stopped finds the launch's STOP flag and does nothing."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _LAUNCH["dir"] = launch_dir
    # a launch that is gone, or stopped, is never recreated by a late worker: the worker
    # creates NO directory, it only opens its own record inside the registry the launch
    # made, and exits when that registry is gone or the flag is up
    if not _launch_is_live(launch_dir):
        os._exit(0)
    try:
        with open(os.path.join(launch_dir, "workers", str(os.getpid())), "x") as fh:
            fh.write(str(os.getpid()))
    except OSError:
        os._exit(0)
    if not _launch_is_live(launch_dir):                                    # stopped between the check and the record
        os._exit(0)


def _launch_is_live(launch_dir=None):
    """A launch is live while its LIVE marker exists and its STOP flag does not. The
    marker is made at acquisition and removed FIRST by a stop, and nothing ever recreates
    it, so no ordering of a worker's checks against the stop's removals can find a live
    launch after the stop began: a directory half removed has no marker either."""
    d = launch_dir or _LAUNCH["dir"]
    return bool(d) and os.path.exists(os.path.join(d, "LIVE")) and not os.path.exists(os.path.join(d, "STOP"))


def _worker(args):
    campaign, evidence, manifest_path, out_dir, year, coarse, fine, dataset, pair, snapshot, commit = args
    if _LAUNCH["dir"] and not _launch_is_live():
        return year, False, "the launch was stopped or is gone, so this year did not start"
    try:
        record, n = replay_year(campaign, evidence, manifest_path, out_dir, year, coarse, fine, dataset=dataset, pair=pair,
                                snapshot=snapshot, commit=commit)
        return year, True, f"{n} tracks in {record['elapsed_seconds']} s"
    except SystemExit as exc:
        return year, False, str(exc)
    except Exception as exc:                       # a worker's failure is a reported failure, never a lost year
        return year, False, f"{type(exc).__name__}: {exc}"


class LaunchLock:
    """One launch at a time per output root. The lock is an operating-system lock on
    `<root>/.launch-lock` (flock, exclusive, non-blocking), which the kernel releases
    when the holding process ends, so there is no stale-lock recovery to race: a second
    launch is refused while the first holds it. Each launch keeps its own directory under
    `<root>/.launches/<pid>/` with a STOP flag and a registry of its workers' ids, and a
    second launch is refused while any earlier launch's registered workers are alive.
    Stopping the launch raises its STOP flag first, so a worker that has not registered
    yet does nothing when it starts, then terminates every registered worker and waits
    for them before the process ends and the kernel releases the lock."""

    def __init__(self, root):
        self.root = root
        self.path = os.path.join(root, ".launch-lock")
        self.launches = os.path.join(root, ".launches")
        self.launch_dir = os.path.join(self.launches, str(os.getpid()))
        self.registry = os.path.join(self.launch_dir, "workers")
        self._fh = None

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    @staticmethod
    def _registered_under(launch_dir):
        try:
            return [int(n) for n in os.listdir(os.path.join(launch_dir, "workers")) if n.isdigit()]
        except OSError:
            return []

    def acquire(self):
        import fcntl
        os.makedirs(self.root, exist_ok=True)
        self._fh = open(self.path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.seek(0)
            owner = self._fh.read().strip() or "unknown"
            self._fh.close()
            self._fh = None
            raise SystemExit(f"REFUSED: a launch (process {owner}) holds {self.path}; stop it first, its workers stop with it")
        for name in (os.listdir(self.launches) if os.path.isdir(self.launches) else []):
            live = [p for p in self._registered_under(os.path.join(self.launches, name)) if self._alive(p)]
            if live:
                self._fh.close()
                self._fh = None
                raise SystemExit(f"REFUSED: launch {name} is gone but its workers {live} are still alive; stop them first")
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        os.makedirs(self.registry, exist_ok=True)
        with open(os.path.join(self.launch_dir, "LIVE"), "w") as fh:
            fh.write(str(os.getpid()))

    def registered(self):
        return self._registered_under(self.launch_dir)

    def stop_flag(self):
        """The LIVE marker goes first, then the STOP flag is written for the record."""
        try:
            os.unlink(os.path.join(self.launch_dir, "LIVE"))
        except FileNotFoundError:
            pass
        os.makedirs(self.launch_dir, exist_ok=True)
        with open(os.path.join(self.launch_dir, "STOP"), "w") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def record(self, pids):
        """Every child the launch knows of goes into the registry, registered or not, so a
        later launch refuses while any of them is alive."""
        os.makedirs(self.registry, exist_ok=True)
        for pid in pids:
            with open(os.path.join(self.registry, str(int(pid))), "w") as fh:
                fh.write(str(int(pid)))

    def release(self):
        """The kernel lock goes, and the launch directory goes ONLY when no child of this
        launch is alive: while one survives, its STOP flag and registry stay, so a late
        worker exits on the flag and a later launch refuses the survivor by name."""
        if self._fh is not None:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        if not any(self._alive(p) for p in self.registered()):
            shutil.rmtree(self.launch_dir, ignore_errors=True)


def stop_launch(lock, children, bound=10.0):
    """THE ONE SHUTDOWN RULE. Raise STOP first, so a worker that has not registered yet
    does nothing when it starts. Record every child known to the process table beside
    every registered worker, terminate them all, and wait up to `bound` seconds for all
    of them to be gone. Release the launch state only when none survives; otherwise leave
    the STOP flag and the registry in place and report the survivors, which a later
    launch refuses by name. Returns the survivors."""
    lock.stop_flag()
    pids = {p.pid for p in children if getattr(p, "pid", None)} | set(lock.registered())
    lock.record(pids)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + bound
    while time.time() < deadline:
        pids = set(lock.registered())                                # late registrations join the set
        if not any(lock._alive(p) for p in pids):
            break
        time.sleep(0.1)
    survivors = sorted(p for p in lock.registered() if lock._alive(p))
    lock.release()
    return survivors


def replay(args, context="spawn"):
    """The experiment's command has no threshold option: it runs one named pair, so no
    other experiment can run under that pair's label. Every record is bound to the frozen
    snapshot the command runs from."""
    import export_protocol_case as E
    dataset = getattr(args, "dataset", DATASET)
    pair_name = getattr(args, "pair", "archived")
    coarse, fine, label, source = threshold_pair(pair_name, dataset, getattr(args, "pair_source", None))
    pair = {"name": pair_name, "label": label, "source": source}
    snapshot, commit = getattr(args, "code_snapshot", None), getattr(args, "snapshot_commit", None)
    _, manifest_sha256 = E.load_manifest(args.manifest)
    y0, y1 = (int(x) for x in args.years.split("-"))
    if y1 < y0:
        raise SystemExit(f"REFUSED: the year range {args.years} is reversed")
    years = list(range(y0, y1 + 1))

    def record_path(year):
        return os.path.join(args.out_dir, f"{dataset}_{year}", f"tracking_{dataset}_{year}.json")

    def problems_of(year):
        return sensitivity_record_problems(record_path(year), args.evidence, year, coarse, fine, manifest_sha256, dataset, pair_name, commit)
    lock = LaunchLock(args.out_dir)
    lock.acquire()
    stopped = {"by": None}

    def on_stop(signum, frame):
        import multiprocessing
        stopped["by"] = signum
        left = stop_launch(lock, multiprocessing.active_children())
        print(f"STOPPED by signal {signum}: workers terminated{'' if not left else ', still alive and left registered: ' + str(left)}", flush=True)
        os._exit(130)
    previous = {s: signal.signal(s, on_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        todo = []
        for year in years:
            if os.path.exists(record_path(year)):
                problems = problems_of(year)
                if problems:
                    raise SystemExit(f"REFUSED: {record_path(year)} exists and is not this experiment's record of {year} ({'; '.join(problems)})")
                print(f"skip {year}, record exists")
                continue
            todo.append((args.campaign, args.evidence, args.manifest, args.out_dir, year, coarse, fine, dataset, pair, snapshot, commit))
        print(f"sensitivity replay of {dataset}: coarse {coarse:.3e}, fine {fine:.3e} ({label}), {len(todo)} years to run, "
              f"{args.workers} at a time, snapshot {commit} at {snapshot}, lock {lock.path}", flush=True)
        failures = []
        if todo:
            if args.workers > 1:
                # an executor reports a worker that dies without returning (a signal, an abrupt
                # exit) as a broken pool, so the command reaches its reconciliation instead of
                # waiting forever on a result that will never come
                from concurrent.futures import ProcessPoolExecutor
                from concurrent.futures.process import BrokenProcessPool
                from multiprocessing import get_context
                pending = {item[4] for item in todo}
                try:
                    with ProcessPoolExecutor(args.workers, mp_context=get_context(context),
                                             initializer=_register_worker, initargs=(lock.launch_dir,)) as pool:
                        for year, ok, msg in pool.map(_worker, todo):
                            pending.discard(year)
                            print(f"{'done' if ok else 'FAILED'} {year}: {msg}", flush=True)
                            if not ok:
                                failures.append(year)
                except BrokenProcessPool as exc:
                    print(f"FAILED: a worker died without reporting ({exc}), years still pending: {sorted(pending)}", flush=True)
                    failures.extend(sorted(pending))
            else:
                for item in todo:
                    year, ok, msg = _worker(item)
                    print(f"{'done' if ok else 'FAILED'} {year}: {msg}", flush=True)
                    if not ok:
                        failures.append(year)
        missing = [y for y in years if not os.path.exists(record_path(y)) or problems_of(y)]
        print(f"replay ended: {len(years) - len(missing)} of {len(years)} records, {len(failures)} failed")
        if missing or failures:
            print(f"REPLAY INCOMPLETE: missing records for {missing}")
            return 1
        return 0
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
        lock.release()


def control(args):
    """One year replayed under the case's own thresholds from the snapshot, required to
    reproduce the campaign's tracks exactly, published under `<out-dir>/control/`."""
    out = os.path.join(args.out_dir, "control")
    record, n = replay_year(args.campaign, args.evidence, args.manifest, out, args.year, None, None, dataset=args.dataset,
                            pair={"name": "control", "label": "control", "source": {"kind": "the collected record's own thresholds"}},
                            snapshot=args.code_snapshot, commit=args.snapshot_commit, control=True)
    print(f"CONTROL PASSED: {args.dataset} {args.year} replayed under its own thresholds reproduces the campaign's {n} tracks exactly, "
          f"record under {out}")
    return 0


def existing_comparison_problems(path, year, rule_a_tracks, sensitivity_tracks, collected_case_id, mode="implementation"):
    """Why an artifact already at the comparison path is NOT this year's comparison of
    these two track files: a review planted the retained ERA-Interim against ERA5
    artifact at the path and the driver summarized it under this experiment's label."""
    try:
        art = json.load(open(path))
    except (OSError, ValueError) as exc:
        return [f"unreadable: {exc}"]
    if not isinstance(art, dict):
        return ["not an artifact object"]
    problems = []
    if art.get("mode") != mode:
        problems.append(f"mode {art.get('mode')!r} is not {mode}")
    if art.get("year") != year:
        problems.append(f"year {art.get('year')} is not {year}")
    if mode == "implementation" and art.get("case_id") != collected_case_id:
        problems.append("case id is not the collected record's")
    if mode == "reanalysis" and not art.get("threshold_transfer"):
        problems.append("the artifact records no declared threshold transfer")
    digests = art.get("inputs_sha256") or {}
    for side, tracks in (("v1", rule_a_tracks), ("port", sensitivity_tracks)):
        if not os.path.exists(tracks) or (digests.get(side) or {}).get("sha256") != _sha256(tracks):
            problems.append(f"side {side} input digest is not the {side} tracks file's")
    return problems


def collect_and_compare(args):
    """Copy each year's tracks and record into evidence (never over different bytes), run the
    instrument, and summarize with the collector's own row builder and multi-year block.
    `--against own`: implementation mode, the dataset's campaign tracks (its own Rule A) as
    side A and the replayed tracks as side B, one case identity. `--against
    eraint-baseline`: reanalysis mode with the transfer declared, the campaign's
    ERA-Interim Rule A tracks as side A and the replayed ERA5 tracks as side B, a common
    numerical threshold pair on two datasets that keep their own climatologies."""
    import collect_protocol_campaign as C
    import season_metrics as S
    import export_protocol_case as E
    dataset = getattr(args, "dataset", DATASET)
    against = getattr(args, "against", "own")
    pair_name = getattr(args, "pair", "archived")
    if against == "eraint-baseline" and dataset == "eraint":
        raise SystemExit("REFUSED: the ERA-Interim baseline comparison is for a replay of another dataset")
    coarse, fine, _, _ = threshold_pair(pair_name, dataset, getattr(args, "pair_source", None))
    _, manifest_sha256 = E.load_manifest(args.manifest)
    commit = getattr(args, "snapshot_commit", None)
    y0, y1 = (int(x) for x in args.years.split("-"))
    per_year, statuses = {}, {}
    for year in range(y0, y1 + 1):
        src = os.path.join(args.runs, f"{dataset}_{year}")
        dst = os.path.join(args.evidence, f"{dataset}_{year}")
        record_name = f"tracking_{dataset}_{year}.json"
        if not os.path.exists(os.path.join(src, record_name)):
            per_year[year] = (None, "a run is missing")
            continue
        # a run is compared only as THIS experiment's run: the same validation the replay
        # applies, so a baseline or another experiment's record can never be summarized here
        problems = sensitivity_record_problems(os.path.join(src, record_name), args.campaign_evidence, year, coarse, fine,
                                               manifest_sha256, dataset, pair_name, commit)
        if problems:
            per_year[year] = (None, "refused: the run is not this experiment's (" + "; ".join(problems) + ")")
            continue
        os.makedirs(dst, exist_ok=True)
        for name in ("tracker_port.mat", record_name):
            target = os.path.join(dst, name)
            if os.path.exists(target):
                if _sha256(target) != _sha256(os.path.join(src, name)):
                    raise SystemExit(f"REFUSED: {target} exists with different bytes and is never overwritten")
            else:
                shutil.copyfile(os.path.join(src, name), target)
        if against == "own":
            side_a = os.path.join(args.campaign_evidence, f"{dataset}_{year}", "tracker_port.mat")
            out = os.path.join(args.artifacts, f"sensitivity_{year}_{dataset}_{'archived' if getattr(args, 'pair', 'archived') == 'archived' else 'transferred'}_vs_rule_a.json")
            expect_case = _record_of(args.campaign_evidence, year, dataset)["dataset_specific"].get("case_id")
            mode_argv = ["--mode", "implementation"]
        else:
            side_a = os.path.join(args.campaign_evidence, f"eraint_{year}", "tracker_port.mat")
            out = os.path.join(args.artifacts, f"sensitivity_{year}_{dataset}_transferred_vs_eraint_baseline.json")
            expect_case = None                                                    # reanalysis mode records no single case
            mode_argv = ["--mode", "reanalysis", "--manifest", args.manifest, "--declared-threshold-transfer"]
        if os.path.exists(out):
            problems = existing_comparison_problems(out, year, side_a, os.path.join(dst, "tracker_port.mat"), expect_case,
                                                    mode="implementation" if against == "own" else "reanalysis")
            per_year[year] = (out, "exists") if not problems else \
                (None, "refused: an artifact exists at the path and is not this comparison (" + "; ".join(problems) + "), never overwritten")
            continue
        argv = [*mode_argv, "--v1", side_a, "--port", os.path.join(dst, "tracker_port.mat"),
                "--year", str(year), "--regions-dir", args.regions_dir, "--record-dir", args.record_dir, "--out", out]
        published = os.path.join(args.published_dir, f"ERA-Int_ew_700hPa_{year}_AFR.nc")
        if os.path.exists(published):
            argv += ["--published-year-file", published]
        try:
            S.main(argv)
        except SystemExit as exc:
            per_year[year] = (None, f"refused: {exc}")
            continue
        per_year[year] = (out, "published")
    summary = C.summarize(per_year)
    summary["distributions_mean_over_years"] = distribution_means(summary["years"], args.artifacts)
    pair_label = PAIRS.get(getattr(args, "pair", "archived"), {}).get("label", LABEL)
    summary.update({"generated_by": "scripts/run_threshold_sensitivity.py", "script_sha256": X.digest(__file__),
                    "dataset": dataset, "against": against,
                    "sides": ({"v1": f"the campaign's Rule A tracks on {dataset}", "port": pair_label} if against == "own"
                              else {"v1": "the campaign's Rule A tracks on ERA-Interim", "port": f"{pair_label}, on {dataset}"}),
                    "instrument_mode": ("implementation, one case identity, two threshold pairs" if against == "own"
                                        else "reanalysis with a declared threshold transfer, two datasets on one numerical pair")})
    try:
        X.publish_json(args.summary, summary, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.summary} exists and artifacts are never overwritten")
    print(f"compared {len(summary['years'])} years, {len(summary['years_without_a_comparison'])} without; wrote {args.summary}")
    return 0


def distribution_means(rows, artifacts):
    """Means over the compared years of the per-year distribution lines the instrument
    records for the season's Africa-origin tracks: the percentiles of lifetime and of
    genesis and lysis position on each side, and the KS statistic. Read from the per-year
    artifacts, nothing recomputed from tracks."""
    acc = {}
    for y, row in sorted(rows.items()):
        art = json.load(open(os.path.join(artifacts, row["artifact"])))
        dists = art["comparison"]["season"].get("distributions") or {}
        for name, d in dists.items():
            a = acc.setdefault(name, {"ks": [], "percentiles": {}})
            if d.get("ks_statistic") is not None:
                a["ks"].append(float(d["ks_statistic"]))
            for p, v in (d.get("percentiles") or {}).items():
                a["percentiles"].setdefault(p, {"v1": [], "port": []})
                a["percentiles"][p]["v1"].append(float(v["v1"]))
                a["percentiles"][p]["port"].append(float(v["port"]))
    out = {}
    for name, a in acc.items():
        out[name] = {"years": len(a["ks"]) if a["ks"] else max((len(v["v1"]) for v in a["percentiles"].values()), default=0),
                     "ks_statistic_mean": float(np.mean(a["ks"])) if a["ks"] else None,
                     "percentiles_mean": {p: {"v1": float(np.mean(v["v1"])), "port": float(np.mean(v["port"])),
                                              "port_minus_v1": float(np.mean(v["port"]) - np.mean(v["v1"]))}
                                          for p, v in a["percentiles"].items()}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("replay", "control"):
        r = sub.add_parser(name)
        r.add_argument("--dataset", choices=("eraint", "era5"), default="eraint")
        r.add_argument("--campaign", required=True)
        r.add_argument("--evidence", required=True, help="the campaign's collected evidence, whose records name the cases")
        r.add_argument("--manifest", required=True)
        r.add_argument("--out-dir", required=True)
        r.add_argument("--code-snapshot", required=True, help="the frozen checkout this driver runs from")
        r.add_argument("--snapshot-commit", required=True)
        if name == "replay":
            r.add_argument("--pair", choices=tuple(PAIRS), default="archived")
            r.add_argument("--pair-source", default=None, help="the retained ERA-Interim calibration artifact for eraint-rule-a")
            r.add_argument("--years", default="1979-2010")
            r.add_argument("--workers", type=int, default=4)
        else:
            r.add_argument("--year", type=int, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--dataset", choices=("eraint", "era5"), default="eraint")
    c.add_argument("--pair", choices=tuple(PAIRS), default="archived")
    c.add_argument("--against", choices=("own", "eraint-baseline"), default="own")
    c.add_argument("--manifest", default="docs/aewc_v2/protocol/manifest_2026-09-25.json")
    c.add_argument("--pair-source", default=None)
    c.add_argument("--snapshot-commit", default=None, help="when given, every run must be bound to this commit")
    c.add_argument("--campaign-evidence", required=True)
    c.add_argument("--runs", required=True)
    c.add_argument("--evidence", required=True)
    c.add_argument("--artifacts", required=True)
    c.add_argument("--summary", required=True)
    c.add_argument("--years", default="1979-2010")
    c.add_argument("--regions-dir", default="data/aewc_v2_pilot/v1_src")
    c.add_argument("--record-dir", default="data/aewc")
    c.add_argument("--published-dir", default="data/aewc")
    args = ap.parse_args(argv)
    return {"replay": replay, "control": control, "compare": collect_and_compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
