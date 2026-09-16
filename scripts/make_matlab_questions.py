#!/usr/bin/env python3
"""Generate the self-contained MATLAB script for the questions Octave cannot answer.

WHY THIS IS A SCRIPT AND NOT A HEREDOC. Two reviews found central results in this project
produced by ad-hoc commands that nothing committed could regenerate. The MATLAB answers
decide which of two sets of oracle numbers is real, so the thing that asks the question has
to be reproducible even though the answer has to be pasted in by hand.

THE THREE QUESTIONS, and what each decides:

  Q1  Does `convhull(x,y,'simplify',true)` succeed? find_ews_f.m builds every association
      polygon with that call inside an empty catch. Octave refuses it, so under Octave the
      hull is never computed and a much larger fallback is used, which makes the oracle's
      association far more permissive than the archived code describes. If MATLAB accepts
      it, the port matches version 1 better than the oracle showed (115 of 134 tracks with
      90 identical, rather than 96 of 117 with 60). If MATLAB refuses it too, the original
      oracle numbers stand.
  Q2  Where does convhull's index vector start and which way does it wind? merge_contours_f
      feeds the hull straight into an inflation centred on the mean, and Octave's convhull
      is not MATLAB's. Open in this project since the port began. Uses a REAL 200-point
      wave footprint rather than a toy set.
  Q3  Does `inpolygon` accept six arguments? The merge finding concludes it fails in MATLAB
      too, making the absorption defect real in the published record rather than an Octave
      artifact. That has always been inferred from the documented signature and never
      executed.

The generated script needs no toolbox, no data file and no setup, so MATLAB Online Basic
is enough.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def wrap(values, width=76):
    out, line = [], "  "
    for x in values:
        token = f"{x:g}"
        if len(line) + len(token) + 1 > width:
            out.append(line + " ...")
            line = "  "
        line += token + " "
    out.append(line)
    return "\n".join(out)


def build(lon, lat):
    return f"""% AEW port: the questions that need genuine MATLAB
%
% Paste this whole file into MATLAB Online (or any MATLAB) and run it. It needs no
% toolbox, no data file and no setup, and it changes nothing. It prints about fifteen
% lines; copy all of them back.

fprintf('MATLAB %s on %s\\n\\n', version, computer);

%% Q1  Does convhull accept the 'simplify' option?
x = [0 1 2 2 1 0 0.5 1.5]';
y = [0 0 1 2 2 1 1   1  ]';
try
  k = convhull(x, y, 'simplify', true);
  fprintf('Q1 convhull(x,y,''simplify'',true): OK, %d vertices, k = %s\\n', ...
          numel(k), mat2str(k(:)'));
catch err
  fprintf('Q1 convhull(x,y,''simplify'',true): FAILED -- %s\\n', err.message);
end
try
  k2 = convhull(x, y);
  fprintf('Q1b convhull(x,y) plain:            OK, %d vertices, k = %s\\n', ...
          numel(k2), mat2str(k2(:)'));
catch err
  fprintf('Q1b convhull(x,y) plain:            FAILED -- %s\\n', err.message);
end

%% Q2  Where does the hull start, and which way does it wind?
lon = [
{wrap(lon)}
];
lat = [
{wrap(lat)}
];
lon = lon(:); lat = lat(:);
fprintf('\\nQ2 using a real 200-point wave footprint\\n');
kk = convhull(lon, lat);
fprintf('Q2 convhull returns %d indices\\n', numel(kk));
fprintf('Q2 full index vector k = %s\\n', mat2str(kk(:)'));
fprintf('Q2 first vertex is point %d at (lon %g, lat %g)\\n', kk(1), lon(kk(1)), lat(kk(1)));
fprintf('Q2 closed (first==last)? %d\\n', kk(1) == kk(end));
a = 0;
for i = 1:numel(kk)-1
  a = a + (lon(kk(i))*lat(kk(i+1)) - lon(kk(i+1))*lat(kk(i)));
end
fprintf('Q2 signed area = %g  (positive = counterclockwise)\\n', a/2);

%% Q3  Does inpolygon accept six arguments?
try
  tf = inpolygon(0.5, 0.5, [0 1 1 0], [0 0 1 1], 'simplify', true);
  fprintf('\\nQ3 inpolygon with six arguments: OK, returned %d\\n', tf);
catch err
  fprintf('\\nQ3 inpolygon with six arguments: FAILED -- %s\\n', err.message);
end
fprintf('\\ndone. Please copy every line above.\\n');
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.expanduser(
        "~/Downloads/aew_matlab_questions.m"))
    ap.add_argument("--points", default="/tmp/wave_lon.npy",
                    help="the real wave footprint, as saved by the extraction step")
    args = ap.parse_args(argv)
    lon = np.load(args.points)
    lat = np.load(args.points.replace("lon", "lat"))
    if lon.size != lat.size:
        raise SystemExit("the longitude and latitude arrays differ in length")
    with open(args.out, "w") as fh:
        fh.write(build(lon.ravel(), lat.ravel()))
    print(f"wrote {args.out} using {lon.size} real wave points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
