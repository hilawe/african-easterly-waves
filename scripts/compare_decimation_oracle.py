#!/usr/bin/env python3
"""Export a real field so version 1's own decimate_f can be run against the port's.

WHY THIS IS A SCRIPT. It found the largest defect in the port, an off-by-one in the
convolution window that shifted every decimated field one cell, and the first version of it
was an ad-hoc command. Review has flagged that shape twice: a central result produced by
something nothing committed can regenerate.

THE COMPARISON IT SETS UP. `decimate_f.m` convolves with a Gaussian and subsamples. For an
EVEN-width kernel the "same" window is ambiguous by one cell and MATLAB and scipy resolve it
in opposite corners, so the two agree on odd widths and differ on even ones. Version 1's own
one-degree input gives width 2; the 0.75 degree tree this project used for years gives 3,
which is why the defect was unreachable until the regridding.

NO CLIMATOLOGY IS INVOLVED, deliberately. Decimation is a pure array operation, so this
exports a RAW wind field. That keeps the comparison independent of the anomaly chain and
lets it run without a climatology cache.

    export AEW_REPO=/path/to/aew-phd
    export AEW_ORACLE_DIR=/path/for/exchange/files
    .venv/bin/python scripts/compare_decimation_oracle.py
    octave-cli --quiet scripts/octave/decimation_check.m

The Octave side needs the image package for `fspecial`, which version 1 uses and which is
not installed by default: `octave-cli --eval "pkg install -forge image"`, once.
"""
import argparse
import os
import sys

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--coarse-resolution", type=float, default=2.5)
    ap.add_argument("--directory", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "eraint", "v1port_buffered"))
    args = ap.parse_args(argv)
    out_dir = os.environ["AEW_ORACLE_DIR"]

    times, latgrid, longrid, u, _ = L.load_year(args.year, args.directory, "eraint")
    window = slice(args.start, args.start + args.steps)
    field = np.ascontiguousarray(u[window], dtype=float)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    width, stride = clim.decimation_shape(native, args.coarse_resolution)
    port = clim.gaussian_decimate(field, native, args.coarse_resolution)

    path = os.path.join(out_dir, "decim_case.mat")
    savemat(path, {"xi": longrid[0, :], "yi": latgrid[:, 0],
                   "ti": np.asarray(times[window], dtype=float), "dati": field,
                   "resi": float(native), "reso": float(args.coarse_resolution),
                   "port": port}, do_compression=True)
    print(f"wrote {path}")
    print(f"  {field.shape} at {native} deg -> {port.shape} at "
          f"{args.coarse_resolution}")
    print(f"  kernel width {width}, stride {stride}   "
          f"{'EVEN, so the window convention matters here' if width % 2 == 0 else
             'odd, so both conventions agree and this case cannot detect the defect'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
