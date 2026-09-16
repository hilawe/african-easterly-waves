#!/usr/bin/env python3
"""Add a depth probe to a copy of version 1's isolate_region_f, leaving the archive alone.

ONE INSERTED LINE, at the top of the function, recording the deepest `numel(dbstack)` the
function reaches. A counter that increments on entry would need a matching decrement at
each of this function's three exits (the out-of-bounds return, the return inside the empty
catch, and the implicit end), and getting one of those wrong is the usual way such a patch
lies. Reading the interpreter's own stack depth needs no bookkeeping at all.

The probe cannot change the walk: it writes two globals and reads nothing the function uses.
"""
import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(REPO, "data", "aewc_v2_pilot", "v1_src")
SIGNATURE = "function [output,Zn] = isolate_region_f(X,Y,Z,pos)"
PROBE = ("\nglobal AEWMAXD AEWBASE\n"
         "AEWMAXD = max(AEWMAXD, numel(dbstack));")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out_dir = args.out or os.path.join(os.environ["AEW_ORACLE_DIR"], "v1_depth")
    if os.path.abspath(out_dir).startswith(os.path.abspath(REPO) + os.sep):
        raise SystemExit(f"{out_dir} is inside the repository; write the copy outside it.")

    with open(os.path.join(ARCHIVE, "isolate_region_f.m")) as fh:
        source = fh.read()
    if SIGNATURE not in source:
        raise SystemExit(f"the signature has moved; expected {SIGNATURE!r}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(ARCHIVE, out_dir)
    with open(os.path.join(out_dir, "isolate_region_f.m"), "w") as fh:
        fh.write(source.replace(SIGNATURE, SIGNATURE + PROBE, 1))
    with open(os.path.join(ARCHIVE, "isolate_region_f.m")) as fh:
        assert fh.read() == source, "the archived source was modified"
    print(f"wrote {os.path.join(out_dir, 'isolate_region_f.m')}")
    print(f"  the archived source at {ARCHIVE} is unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
