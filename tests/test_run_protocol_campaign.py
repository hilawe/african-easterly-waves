"""The campaign driver: a failed run makes the command fail, a run whose record exists is
skipped on a restart and never rerun, and a clean campaign exits 0. The entry point is
replaced by a small fake through the ENTRY override, so no year is tracked here."""
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(ROOT, "scripts", "run_protocol_campaign.sh")

FAKE = textwrap.dedent("""
    import hashlib, json, os, sys
    args = sys.argv[1:]
    manifest_sha256 = hashlib.sha256(open(args[args.index("--manifest") + 1], "rb").read()).hexdigest()
    dataset, year, out = args[args.index("--dataset") + 1], args[args.index("--year") + 1], args[args.index("--out-dir") + 1]
    run = os.path.join(out, f"{dataset}_{year}")
    case = os.path.join(run, "tracker_case.mat")
    if os.path.exists(case):                                   # the real entry point publishes exclusively
        print("REFUSED: the retained case already exists"); sys.exit(1)
    open(case, "wb").write(b"case")                            # the case is published BEFORE tracking
    import time; time.sleep(float(os.environ.get("ENTRY_SLEEP", "0")))
    if f"{dataset} {year}" in os.environ.get("LATE_RUNS", ""):
        # a publisher that outlives its parent: after this process has failed and its parent moved on,
        # a detached child publishes a record at the pathnames this attempt was given
        import subprocess, time
        subprocess.Popen([sys.executable, "-c", "import time, json, sys; time.sleep(1.2); "
                          f"json.dump({{'dataset_specific': {{'late': True}}}}, open({os.path.join(run, f'tracking_{dataset}_{year}.json')!r}, 'w'))"],
                         start_new_session=True)
        print("failed, with a child still to publish"); sys.exit(1)
    if f"{dataset} {year}" in os.environ.get("FAIL_RUNS", ""):
        print("the fake entry point fails after publishing the case"); sys.exit(1)
    if f"{dataset} {year}" in os.environ.get("SILENT_RUNS", ""):
        sys.exit(0)                                            # a worker that ends without a record or a report
    with open(os.path.join(run, f"tracking_{dataset}_{year}.json"), "w") as fh:
        json.dump({"dataset_specific": {"dataset": dataset, "year": int(year), "fake": True, "attempt": out},
                   "protocol_settings": {"manifest_sha256": manifest_sha256}}, fh)
    print("done")
""")


def _manifest(tmp_path, text='{"protocol": "test"}'):
    """A manifest file the driver can digest, and that digest."""
    import hashlib
    path = tmp_path / f"m_{hashlib.sha256(text.encode()).hexdigest()[:8]}.json"
    path.write_text(text)
    return str(path), hashlib.sha256(text.encode()).hexdigest()


def _campaign(tmp_path, fail="", silent="", late="", years="2001-2001", manifest_text='{"protocol": "test"}'):
    fake = tmp_path / "fake_entry.py"
    fake.write_text(FAKE)
    out = tmp_path / "campaign"
    env = dict(os.environ, OUT=str(out), YEARS=years, ENTRY=f"{sys.executable} {fake}", PYTHON=sys.executable,
               FAIL_RUNS=fail, SILENT_RUNS=silent, LATE_RUNS=late,
               MANIFEST=_manifest(tmp_path, manifest_text)[0], CAL_eraint="c", CAL_era5="c", CACHE_eraint="k", CACHE_era5="k")
    r = subprocess.run(["bash", SCRIPT], env=env, cwd=ROOT, capture_output=True, text=True)
    return r, out


def test_a_failed_run_makes_the_campaign_command_fail_and_is_counted(tmp_path):
    r, out = _campaign(tmp_path, fail="era5 2001")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAILED era5 2001" in r.stdout and "1 failed" in r.stdout and "CAMPAIGN INCOMPLETE" in r.stdout
    assert (out / "eraint_2001" / "tracking_eraint_2001.json").exists()
    assert not (out / "era5_2001" / "tracking_era5_2001.json").exists()
    assert not list(out.glob(".failures.*"))                              # the counter is removed
    assert not (out / "era5_2001").exists()                              # nothing partial at the canonical path
    partial = list(out.glob("era5_2001.attempt-*"))
    assert len(partial) == 1 and (partial[0] / "era5_2001" / "tracker_case.mat").exists()
    assert not list(out.glob("eraint_2001.attempt-*"))                  # a completed attempt is promoted and its shell removed
    # a restart leaves the partial attempt where it is, runs a fresh attempt, promotes it, and leaves the record alone
    before = (out / "eraint_2001" / "tracking_eraint_2001.json").stat().st_mtime_ns
    r2, _ = _campaign(tmp_path)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "skip eraint 2001, record exists" in r2.stdout and "done era5 2001" in r2.stdout and "2 of 2 records, 0 failed" in r2.stdout
    assert (out / "eraint_2001" / "tracking_eraint_2001.json").stat().st_mtime_ns == before
    assert (out / "era5_2001" / "tracking_era5_2001.json").exists() and (out / "era5_2001" / "run.log").exists()
    assert len(list(out.glob("era5_2001.attempt-*"))) == 1 and (partial[0] / "era5_2001" / "tracker_case.mat").exists()


def test_a_publisher_surviving_its_attempt_can_never_land_a_record_in_the_promoted_run(tmp_path):
    import json
    import time
    r, out = _campaign(tmp_path, late="era5 2001")
    assert r.returncode == 1 and "FAILED era5 2001" in r.stdout
    old = list(out.glob("era5_2001.attempt-*"))
    assert len(old) == 1
    r2, _ = _campaign(tmp_path)                                          # restarts before the late child publishes
    assert r2.returncode == 0 and "done era5 2001" in r2.stdout
    time.sleep(2.0)                                                      # the late child has published by now
    late = old[0] / "era5_2001" / "tracking_era5_2001.json"
    assert late.exists() and json.load(open(late)) == {"dataset_specific": {"late": True}}
    promoted = json.load(open(out / "era5_2001" / "tracking_era5_2001.json"))
    assert promoted["dataset_specific"].get("late") is None and promoted["dataset_specific"]["attempt"] != str(old[0])
    r3, _ = _campaign(tmp_path)
    assert r3.returncode == 0 and "skip era5 2001, record exists" in r3.stdout


def test_two_campaign_commands_on_one_root_cannot_both_promote_the_same_run(tmp_path):
    """The promotion lock: when two commands race on one run, exactly one promotes, the
    other fails its run and says the lock or the destination was taken, and the promoted
    run holds no nested run directory."""
    import time
    fake = tmp_path / "fake_entry.py"
    fake.write_text(FAKE)
    out = tmp_path / "campaign"
    env = dict(os.environ, OUT=str(out), YEARS="2001-2001", ENTRY=f"{sys.executable} {fake}", PYTHON=sys.executable,
               FAIL_RUNS="", SILENT_RUNS="", LATE_RUNS="", ENTRY_SLEEP="1.0",
               MANIFEST=_manifest(tmp_path)[0], CAL_eraint="c", CAL_era5="c", CACHE_eraint="k", CACHE_era5="k")
    procs = [subprocess.Popen(["bash", SCRIPT], env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outs = [p.communicate()[0] for p in procs]
    codes = [p.returncode for p in procs]
    # each run is promoted by exactly one command; a command that lost a promotion reports it and exits 1
    assert 1 in codes and set(codes) <= {0, 1}, outs
    assert all("promotion of" in o and "refused" in o for o, c in zip(outs, codes) if c == 1)
    for ds in ("eraint", "era5"):
        final = out / f"{ds}_2001"
        assert (final / f"tracking_{ds}_2001.json").exists()
        assert not any(p.is_dir() for p in final.iterdir()), list(final.iterdir())     # nothing nested inside
        assert not (out / f"{ds}_2001.promoting").exists()                           # the lock is released


PROMOTE = os.path.join(ROOT, "scripts", "campaign_promote.sh")


def _promote(src, dst):
    """The driver's promote function alone, sourced from its own file, so the lock's
    semantics are tested without timing: a held lock refuses, a free one promotes."""
    return subprocess.run(["bash", "-c", f'. "{PROMOTE}"; promote "$1" "$2"', "_", str(src), str(dst)], capture_output=True, text=True)


def test_promotion_refuses_while_the_lock_is_held_and_leaves_everything_in_place(tmp_path):
    src = tmp_path / "attempt" / "era5_2001"
    src.mkdir(parents=True)
    (src / "tracking_era5_2001.json").write_text("{}")
    dst = tmp_path / "era5_2001"
    lock = tmp_path / "era5_2001.promoting"
    lock.mkdir()                                                          # another promoter holds it
    r = _promote(src, dst)
    assert r.returncode == 1 and "refused" in r.stdout and "is held" in r.stdout
    assert src.exists() and not dst.exists() and lock.exists()             # nothing moved, the lock is not ours to remove
    lock.rmdir()
    r = _promote(src, dst)
    assert r.returncode == 0 and (dst / "tracking_era5_2001.json").exists() and not src.exists() and not lock.exists()
    (tmp_path / "attempt2" / "era5_2001").mkdir(parents=True)
    r = _promote(tmp_path / "attempt2" / "era5_2001", dst)               # the destination now exists
    assert r.returncode == 1 and "it exists" in r.stdout and not lock.exists() and (tmp_path / "attempt2" / "era5_2001").exists()


def test_a_file_at_the_record_path_that_is_not_this_runs_record_is_neither_skipped_nor_counted(tmp_path):
    import json
    out = tmp_path / "campaign"
    settings = {"manifest_sha256": _manifest(tmp_path)[1]}
    (out / "era5_2001").mkdir(parents=True)
    (out / "era5_2001" / "tracking_era5_2001.json").write_bytes(b"")                     # an empty file
    r, _ = _campaign(tmp_path)
    assert r.returncode == 1 and "is not a record of this run" in r.stdout and "missing records for: era5/2001" in r.stdout
    assert (out / "era5_2001" / "tracking_era5_2001.json").read_bytes() == b""             # untouched
    (out / "era5_2001" / "tracking_era5_2001.json").write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 1990}, "protocol_settings": settings}))
    r, _ = _campaign(tmp_path)
    assert r.returncode == 1 and "record names era5 1990, the path says era5 2001" in r.stdout
    (out / "era5_2001" / "tracking_era5_2001.json").write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": "2001"}, "protocol_settings": settings}))
    r, _ = _campaign(tmp_path)
    assert r.returncode == 1 and "is not a record of this run" in r.stdout          # a string year is not the path's integer year
    (out / "era5_2001" / "tracking_era5_2001.json").write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 2001}, "protocol_settings": settings}))
    (out / "era5_2001" / "tracker_port.mat").write_bytes(b"tracks the record does not name")
    r, _ = _campaign(tmp_path)
    assert r.returncode == 1 and "does not have the digest the record names" in r.stdout


def test_a_canonical_directory_without_a_record_is_never_reused(tmp_path):
    out = tmp_path / "campaign"
    (out / "era5_2001").mkdir(parents=True)
    (out / "era5_2001" / "tracker_case.mat").write_bytes(b"case")
    r, _ = _campaign(tmp_path)
    assert r.returncode == 1 and "exists without a completion record" in r.stdout
    assert (out / "era5_2001" / "tracker_case.mat").exists() and not (out / "era5_2001" / "tracking_era5_2001.json").exists()


def test_a_worker_that_ends_without_a_record_or_a_report_still_fails_the_campaign(tmp_path):
    r, out = _campaign(tmp_path, silent="eraint 2001")
    assert r.returncode == 1 and "1 of 2 records, 1 failed" in r.stdout and "missing records for: eraint/2001" in r.stdout


def test_a_malformed_or_reversed_year_range_is_refused_before_anything_runs(tmp_path):
    for years in ("bogus", "2001-2000", "01-2001"):
        r, out = _campaign(tmp_path, years=years)
        assert r.returncode == 1 and "REFUSED" in r.stdout and not out.exists(), years


def test_a_clean_campaign_exits_zero(tmp_path):
    r, out = _campaign(tmp_path)
    assert r.returncode == 0 and "2 of 2 records, 0 failed" in r.stdout


def test_an_era5_only_campaign_runs_and_expects_only_era5_records(tmp_path):
    fake = tmp_path / "fake_entry.py"
    fake.write_text(FAKE)
    out = tmp_path / "campaign"
    env = dict(os.environ, OUT=str(out), YEARS="2011-2012", ENTRY=f"{sys.executable} {fake}", PYTHON=sys.executable,
               DATASETS="era5", PARALLEL="2", MANIFEST=_manifest(tmp_path)[0], CAL_eraint="c", CAL_era5="c", CACHE_eraint="k", CACHE_era5="k")
    r = subprocess.run(["bash", SCRIPT], env=env, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "2 of 2 records, 0 failed" in r.stdout, r.stdout + r.stderr
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["era5_2011", "era5_2012"]
    # a failed run is reported as missing under the datasets actually run, so the rerun hint names no ERA-Interim year
    out2 = tmp_path / "campaign_failed"
    r = subprocess.run(["bash", SCRIPT], env=dict(env, OUT=str(out2), FAIL_RUNS="era5 2012"), cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "1 of 2 records, 1 failed" in r.stdout, r.stdout + r.stderr
    assert "missing records for: era5/2012. Rerun" in r.stdout, r.stdout


def test_an_unknown_dataset_or_a_bad_parallel_count_is_refused_before_anything_runs(tmp_path):
    for env_extra, needle in (({"DATASETS": "era5 merra2"}, "unknown dataset"), ({"DATASETS": " "}, "names no dataset"),
                              ({"DATASETS": ""}, "names no dataset"), ({"DATASETS": "era5 era5"}, "names era5 twice"),
                              ({"DATASETS": "era5 eraint era5"}, "names era5 twice"),
                              ({"PARALLEL": "0"}, "PARALLEL"), ({"PARALLEL": "four"}, "PARALLEL"),
                              ({"PARALLEL": "00"}, "without a leading zero"), ({"PARALLEL": "07"}, "without a leading zero"),
                              ({"MANIFEST": "/nonexistent/m.json"}, "cannot read the manifest")):
        out = tmp_path / f"c{abs(hash(needle + str(env_extra)))}"
        env = dict(os.environ, OUT=str(out), YEARS="2011-2011", ENTRY="false", PYTHON=sys.executable,
                   MANIFEST=_manifest(tmp_path)[0], CAL_eraint="c", CAL_era5="c", CACHE_eraint="k", CACHE_era5="k")
        env.update(env_extra)
        r = subprocess.run(["bash", SCRIPT], env=env, cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 1 and needle in r.stdout, (env_extra, r.stdout)
        assert not out.exists()


def test_a_record_made_under_another_manifest_is_neither_skipped_nor_counted(tmp_path):
    """A review seeded records made under a different manifest, and the campaign skipped
    them and reported itself complete. Under a second manifest the first campaign's
    record is refused by name, left in place, and counted missing."""
    import json
    r, out = _campaign(tmp_path, manifest_text='{"protocol": "first"}')
    assert r.returncode == 0 and "2 of 2 records, 0 failed" in r.stdout, r.stdout + r.stderr
    record = out / "era5_2001" / "tracking_era5_2001.json"
    before = record.read_bytes()
    r, _ = _campaign(tmp_path, manifest_text='{"protocol": "second"}')
    assert r.returncode == 1 and "0 of 2 records, 2 failed" in r.stdout, r.stdout
    assert "FAILED era5 2001: the file at" in r.stdout and "the record was made under manifest" in r.stdout
    assert "missing records for: eraint/2001 era5/2001" in r.stdout and "skip " not in r.stdout
    assert record.read_bytes() == before                                   # never rewritten
    r, _ = _campaign(tmp_path, manifest_text='{"protocol": "first"}')      # under its own manifest it is a record again
    assert r.returncode == 0 and "skip era5 2001, record exists" in r.stdout and "2 of 2 records" in r.stdout


def test_the_record_predicate_requires_the_manifest_from_the_command_line(tmp_path):
    import json
    predicate = os.path.join(ROOT, "scripts", "campaign_record_ok.py")
    run = tmp_path / "era5_2001"
    run.mkdir()
    rec = run / "tracking_era5_2001.json"
    digest = "a" * 64
    rec.write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 2001}, "protocol_settings": {"manifest_sha256": digest}}))
    def ok(*argv):
        return subprocess.run([sys.executable, predicate, *argv], capture_output=True, text=True)
    assert ok("--manifest-sha256", digest, str(rec)).returncode == 0
    assert ok("--manifest-sha256", "b" * 64, str(rec)).returncode == 1
    for argv in ((str(rec),), ("--manifest-sha256", "", str(rec)), ("--manifest-sha256", "A" * 64, str(rec))):
        r = ok(*argv)
        assert r.returncode == 2 and "usage" in r.stdout, argv               # no unchecked path from the command line
    rec.write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 2001}, "protocol_settings": "not a mapping"}))
    assert ok("--manifest-sha256", digest, str(rec)).returncode == 1
