#!/usr/bin/env python3
"""Measure exact_percentile's memory on a production-sized year, reproducibly.

WHY THIS IS COMMITTED. The module's first docstring claimed a peak of one chunk plus an
8 MiB histogram plus a bounded candidate list, and a review measured hundreds of MiB of
growth above a 143 MiB annual chunk, because the finite copy, the boolean mask and the
int64 bin indices are each chunk-sized and coexist. A memory claim nobody can rerun is
the provenance failure this project keeps relearning, so the procedure lives here and the
docstring cites it instead of asserting numbers.

WHAT IS MEASURED, precisely, because a first version blurred it. Two numbers per run:
ABSOLUTE PEAK, the process's ru_maxrss high-water mark after the call, and GROWTH, that
peak minus the high-water mark before the call. Growth is the routine's contribution on
top of whatever the process had already touched, and neither is "total" in any other
sense.
ru_maxrss only ever rises, so every configuration runs in a FRESH SUBPROCESS, and the
figures VARY BETWEEN RUNS (a review's rerun differed from the first run by over 100 MiB
on the annual case), so each configuration runs THREE times and the range is reported.

THE POPULATION IS THE PRODUCTION SHAPE, one year of the one-degree fine tracking domain:
12,851 cells at 1,460 six-hourly steps, 18,762,460 float64 values. A first version of the
monthly case computed `1460 // 124` chunks and silently dropped the final 96 timesteps,
which a review caught through the finite-count mismatch between configurations. The
remainder chunk is now included and the counts must agree.

    .venv/bin/python scripts/bench_percentile_memory.py
"""
import json
import platform
import subprocess
import sys

ANNUAL_STEPS = 1460
FINE_CELLS = 12_851
REPEATS = 3

WORKER = r"""
import json, resource, sys
import numpy as np
sys.path.insert(0, "src")
from aew.v1port.percentile import exact_percentile

chunk_steps = int(sys.argv[1])
cells, total_steps = 12_851, 1460

def chunks():
    # The population is a pure function of the GLOBAL index, so every chunking yields the
    # identical values and the cross-configuration count check below can mean something.
    # An RNG stream consumed per chunk would give each chunking a different population,
    # and the identity check would fail for a reason that is not a defect.
    done = 0
    while done < total_steps:
        steps = min(chunk_steps, total_steps - done)
        idx = np.arange(done * cells, (done + steps) * cells, dtype=np.float64)
        block = np.sin(idx * 0.7371) * 3e-6
        block[idx.astype(np.int64) % 500 == 0] = np.nan
        yield block
        done += steps

before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
value, n = exact_percentile(chunks, 55.0)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"n": n, "value": value, "peak_mib": after / 2**20,
                  "growth_mib": (after - before) / 2**20}))
"""


def run_config(chunk_steps):
    results = []
    for _ in range(REPEATS):
        run = subprocess.run([sys.executable, "-c", WORKER, str(chunk_steps)],
                             capture_output=True, text=True, check=True)
        results.append(json.loads(run.stdout))
    return results


def main():
    import numpy as np
    print(f"environment: {platform.platform()}, python {platform.python_version()}, "
          f"numpy {np.__version__}")
    print(f"population: one synthetic year, {ANNUAL_STEPS * FINE_CELLS:,} values, "
          f"{REPEATS} repetitions per configuration\n")
    counts, values = set(), set()
    for steps, label in ((ANNUAL_STEPS, "annual chunk (143.1 MiB held at once)"),
                         (124, "monthly chunks (12.2 MiB each, 96-step remainder "
                               "included)")):
        rs = run_config(steps)
        counts.update(r["n"] for r in rs)
        values.update(r["value"] for r in rs)
        growth = [r["growth_mib"] for r in rs]
        peak = [r["peak_mib"] for r in rs]
        print(f"  {label}")
        print(f"    growth above pre-call high-water mark: "
              f"{min(growth):7.1f} .. {max(growth):7.1f} MiB over {REPEATS} runs")
        print(f"    absolute peak resident set:            "
              f"{min(peak):7.1f} .. {max(peak):7.1f} MiB")
    # EQUAL COUNTS DO NOT PROVE IDENTICAL POPULATIONS, which a first version claimed on
    # counts alone. The VALUE is the sharper invariant, since every repetition of every
    # configuration must return the bit-identical percentile, or the chunking changed
    # either the population or the arithmetic, and both are defects here.
    if len(counts) != 1 or len(values) != 1:
        print(f"\nERROR: the configurations must agree bit for bit and do not "
              f"(counts {sorted(counts)}, values {sorted(values)!r}); a chunking that "
              f"changes the population or the result is measuring the wrong thing")
        return 1
    print(f"\nevery repetition of every configuration returned the identical result, "
          f"n = {counts.pop():,}, value = {values.pop()!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
