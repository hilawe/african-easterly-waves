#!/usr/bin/env python3
"""Run the percentile mutation catalog with a per-mutation timeout.

WHY A SEPARATE RUNNER. The repository's mutation checker has no timeout, and this catalog
deliberately contains a mutation, `progress_guard_disabled`, whose only observable effect
is an infinite loop, since the guard it disables exists to convert a hang into a refusal.
Running the catalog through the plain checker hangs on that entry, so a claim of "all
caught" was reproducible only from an ad hoc harness in the session logs. This commits the
harness. A hang past the timeout is reported as CAUGHT, because for this class of defect a
hang IS the failure the test detects.

    .venv/bin/python tests/run_percentile_mutations.py

Exits 0 only when every mutation is applied, observed changing the file on disk, and
caught. A surviving mutation, a no-op entry, or a missing anchor is a failure.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys

TIMEOUT_SECONDS = 90
ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "aew" / "v1port" / "percentile.py"
CATALOG = ROOT / "tests" / "mutations_percentile.py"
TESTS = ROOT / "tests" / "test_percentile.py"


def main():
    spec = importlib.util.spec_from_file_location("catalog", CATALOG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = TARGET.read_text()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    # the canary: the tests must pass on the unmutated code, or every mutation below
    # would be "caught" by a suite that was already red
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "-q", "--tb=no"],
        env=env, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    if baseline.returncode != 0:
        print("REFUSING TO RUN: the oracle fails on the unmutated code")
        return 2

    failures = 0
    try:
        for name, mutate in module.MUTATIONS.items():
            mutated = mutate(original)
            if mutated == original:
                print(f"  {name:36} *** NO-OP, anchor missing ***")
                failures += 1
                continue
            TARGET.write_text(mutated)
            if TARGET.read_text() == original:
                print(f"  {name:36} *** NOT APPLIED ON DISK ***")
                failures += 1
                continue
            try:
                run = subprocess.run(
                    [sys.executable, "-m", "pytest", str(TESTS), "-q", "--tb=no"],
                    env=env, capture_output=True, text=True,
                    timeout=TIMEOUT_SECONDS)
                verdict = ("CAUGHT" if run.returncode != 0
                           else "*** SURVIVED ***")
            except subprocess.TimeoutExpired:
                verdict = "CAUGHT (hung; a hang is the failure this class produces)"
            if verdict.startswith("***"):
                failures += 1
            print(f"  {name:36} {verdict}")
            TARGET.write_text(original)
    finally:
        TARGET.write_text(original)
    total = len(module.MUTATIONS)
    print(f"\n{total - failures} of {total} caught, source restored: "
          f"{TARGET.read_text() == original}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
