#!/usr/bin/env python3
"""Build an instrumented copy of version 1's tracker that dumps its intermediate state.

WHY A COPY. `find_ews_f.m` returns finished tracks and nothing else, so no oracle so far
has been able to see its trough axes, its merged wave centers, or its live track set. Those
are exactly what is needed to say whether the port's association fragments a feature because
its axes differ or because its association is less tolerant.

THE ARCHIVED SOURCE IS NEVER MODIFIED. `data/aewc_v2_pilot/v1_src` is the record and stays
byte-identical; this writes a separate tree and refuses to write into the repository. The
copy is not committed, for the same reason the patched merge copy was not: a modified copy
of an archived record is not something to keep under version control where it could be
mistaken for the record.

WHAT IS ADDED, and it is deliberately the least that answers the question: one global and
four printf blocks, each guarded by a time match against that global. With the global empty
every guard is false, so the instrumented copy computes exactly what the original does. The
caller checks that by comparing its finished tracks against the uninstrumented run.
"""
import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(REPO, "data", "aewc_v2_pilot", "v1_src")

SIGNATURE = ("function ews = find_ews_f(u_c,v_c,currv_anom_c,advcurrv_anom_c,lat_c,"
             "lon_c,time,latgrid,longrid,u,v,currv_anom,rean,level);")

COARSE_MERGE = ("  pot_wv2 = merge_contours_f(pot_wv,longrid_c,latgrid_c,crvt,"
                "curv_thr_c); %Merge wave contours using coarse fields")

AFTER_COARSE = "  pot_wv = pot_wv2;\n  clear pot_wv2"

CLEAR_LINE = "  clear id ch ctid"

PRUNE_END = "    clear keep_id\n  end  \nend"


def patch(source):
    """Return the instrumented text, or raise if any anchor has moved."""
    out = source
    anchors = {"signature": SIGNATURE, "coarse merge": COARSE_MERGE,
               "after coarse": AFTER_COARSE, "clear line": CLEAR_LINE,
               "prune end": PRUNE_END}
    missing = [name for name, text in anchors.items() if text not in out]
    if missing:
        raise SystemExit(
            f"these anchors are not in find_ews_f.m: {', '.join(missing)}. The archived "
            f"source has changed, or this script was written against a different copy. "
            f"Fix the anchors rather than loosening the match, because a patch that lands "
            f"in the wrong place would change the answer silently.")

    out = out.replace(SIGNATURE, SIGNATURE + "\n"
                      "global AEWDBG_TIMES\n"
                      "if isempty(AEWDBG_TIMES); AEWDBG_TIMES = []; end")

    out = out.replace(COARSE_MERGE, """  if any(abs(time(t) - AEWDBG_TIMES) < 1e-6);
    for zz = 1:length(id);
      if floor(ch(2,id(zz))) == ch(2,id(zz)) & ch(2,id(zz)) > 1;
        zla = ch(2,id(zz)+1:id(zz)+ch(2,id(zz)));
        zlo = ch(1,id(zz)+1:id(zz)+ch(2,id(zz)));
        printf('AXIS %.4f %d %.4f %.4f %.4f %.4f\\n', time(t), numel(zla), ...
               mean(zla), mean(zlo), max(zla)-min(zla), max(zlo)-min(zlo));
        % every vertex of the axis, because the summary above cannot show WHICH
        % points a contour tracer joined into one line, and the third worked case
        % turned on exactly that
        printf('AXISPTS %.4f %d', time(t), numel(zla));
        printf(' %.4f %.4f', [zla(:)'; zlo(:)']);
        printf('\\n');
      end
    end
    fflush(stdout);
  end
""" + COARSE_MERGE)

    out = out.replace(AFTER_COARSE, AFTER_COARSE + """
  if any(abs(time(t) - AEWDBG_TIMES) < 1e-6);
    for zz = 1:size(pot_wv,2);
      printf('COARSE %.4f %.4f %.4f\\n', time(t), pot_wv(zz).lat_mean, ...
             pot_wv(zz).lon_mean);
    end
    fflush(stdout);
  end""")

    out = out.replace(CLEAR_LINE, """  if any(abs(time(t) - AEWDBG_TIMES) < 1e-6);
    for zz = 1:size(pot_wv,2);
      printf('FINE %.4f %.4f %.4f\\n', time(t), pot_wv(zz).lat_mean, ...
             pot_wv(zz).lon_mean);
    end
    fflush(stdout);
  end
""" + CLEAR_LINE, 1)

    out = out.replace(PRUNE_END, """    clear keep_id
  end
  if any(abs(time(t) - AEWDBG_TIMES) < 1e-6) & exist('ew_tracks') == 1;
    for zz = 1:length(ew_tracks);
      printf('TRACK %.4f %d %d %.4f %.4f %.4f\\n', time(t), zz, ...
             length(ew_tracks(zz).time), ew_tracks(zz).meanlat(end), ...
             ew_tracks(zz).meanlon(end), ew_tracks(zz).time(end));
    end
    fflush(stdout);
  end
end""")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repair-convhull", action="store_true",
                    help="also replace find_ews_f's convhull(...,'simplify',true) with "
                         "the plain two-argument form, which Octave accepts")
    ap.add_argument("--out", default=None,
                    help="where to write the copy (default $AEW_ORACLE_DIR/v1_instrumented)")
    args = ap.parse_args(argv)
    out_dir = args.out or os.path.join(os.environ["AEW_ORACLE_DIR"], "v1_instrumented")

    # REFUSE TO WRITE INSIDE THE REPOSITORY. The archived record and a modified copy of it
    # must not be able to end up in the same tree by a mistyped path.
    if os.path.abspath(out_dir).startswith(os.path.abspath(REPO) + os.sep):
        raise SystemExit(
            f"{out_dir} is inside the repository. The instrumented copy must live outside "
            f"it so it cannot be confused with the archived record.")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(ARCHIVE, out_dir)
    target = os.path.join(out_dir, "find_ews_f.m")
    with open(os.path.join(ARCHIVE, "find_ews_f.m")) as fh:
        source = fh.read()
    patched = patch(source)
    if args.repair_convhull:
        # WHY THIS SWITCH EXISTS. find_ews_f.m builds every association polygon with
        #     ply_id = convhull(x, y, 'simplify', true);
        # inside a try/catch whose catch is empty, at three sites. OCTAVE REFUSES THAT
        # CALL ("OPTIONS must be a string or cell array of strings"), so under Octave
        # ply_id is never set, the hull branch never runs, and every polygon is the
        # fallback hexagon built from the track's own extent, which is much larger.
        # MATLAB DOCUMENTS A 'Simplify' OPTION, so the same call probably succeeds there
        # and version 1 as published probably uses the hull. That makes the difference an
        # OCTAVE ARTIFACT rather than a defect in version 1, and it contaminates any
        # comparison of association behavior run under Octave.
        #
        # 'Simplify' only drops collinear vertices; it does not change the polygon's
        # shape. So the plain two-argument form is a faithful stand-in for what MATLAB
        # computes, for every purpose the polygon is put to here (an inpolygon test).
        broken = ("convhull(longrid(pot_wv(id).wave_points),"
                  "latgrid(pot_wv(id).wave_points),'simplify',true)")
        fixed = ("convhull(longrid(pot_wv(id).wave_points),"
                 "latgrid(pot_wv(id).wave_points))")
        n = patched.count(broken)
        if n != 3:
            raise SystemExit(f"expected 3 convhull sites, found {n}")
        patched = patched.replace(broken, fixed)
        print(f"  repaired {n} convhull call sites")
    with open(target, "w") as fh:
        fh.write(patched)

    with open(os.path.join(ARCHIVE, "find_ews_f.m")) as fh:
        assert fh.read() == source, "the archived source was modified"
    print(f"wrote {target}")
    print(f"  {len(patched) - len(source)} characters added at 4 points, "
          f"all inert while AEWDBG_TIMES is empty")
    print(f"  the archived source at {ARCHIVE} is unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
