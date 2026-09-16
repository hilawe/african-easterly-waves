"""Tests for the grid-route comparator's coverage check.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_compare_grid_routes.py.

    G1  a direct grid missing rows or columns is compared on the overlap and reported as
        agreeing
    G2  a strided grid missing rows or columns is likewise passed
    G3  the timestamp check compares the shared count against min(), so a STRICT SUBSET on
        either side passes
    G4  the timestamp check compares against one side only, so extra timestamps on the
        other side pass
    G5  a mismatch is reported but does not change the verdict, so the script still exits 0

WHY THIS FILE EXISTS AT ALL. The script had no tests, and two successive reviews found two
successive false-positive paths in it, both by constructing a direct tree that agreed
everywhere it overlapped but did not cover the same ground. G1 was the missing row and
column. G3 was the missing timestep, which survived the repair for G1 because
`shared != min(a, b)` is satisfied by a strict subset. G4 is the mistake the obvious repair
of G3 would introduce next.

The check is a pure function so these cases are cheap and exhaustive, rather than each one
needing a NetCDF pair on disk. The end-to-end seam is covered separately by
`test_the_verdict_follows_the_coverage_check`.
"""
import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "scripts", "compare_grid_routes.py")

_spec = importlib.util.spec_from_file_location("compare_grid_routes", _SCRIPT)
cgr = importlib.util.module_from_spec(_spec)
sys.modules["compare_grid_routes"] = cgr
_spec.loader.exec_module(cgr)


def mismatches(shape_s=(6, 7), shape_d=(6, 7), shared_rows=6, shared_cols=7,
               n_times_s=4, n_times_d=4, n_shared_times=4):
    """The check, with everything agreeing by default so each test varies one thing."""
    return cgr.coverage_mismatches(shape_s, shape_d, shared_rows, shared_cols,
                                   n_times_s, n_times_d, n_shared_times)


def test_two_identical_retrievals_raise_nothing():
    """The baseline. Without this the others could pass by always reporting a mismatch."""
    assert mismatches() == []


@pytest.mark.parametrize("shape_d,shared_rows,shared_cols", [
    ((5, 7), 5, 7),      # a row short
    ((6, 6), 6, 6),      # a column short
    ((5, 6), 5, 6),      # both, the case a review actually built
])
def test_a_direct_grid_that_does_not_cover_the_same_ground_is_refused(
        shape_d, shared_rows, shared_cols):
    """G1. Agreeing on the overlap is not the question when the extents differ.

    A production input whose grid stops one row earlier moves the tracker's domain edge,
    so it is not a substitute however well the shared cells agree.

    THE MESSAGE NAMES THE STRIDED SIDE HERE, which is right and worth stating because the
    first version of this test expected the other one. When the direct grid is smaller and
    wholly contained, every shared coordinate is one it holds, so the side carrying
    something the other lacks is the strided one. The refusal is what matters, and which side
    is named follows from which one has the surplus.
    """
    out = mismatches(shape_d=shape_d, shared_rows=shared_rows, shared_cols=shared_cols)
    assert out, "a direct grid smaller than the strided one must be refused"
    assert any("grid" in m for m in out), "the refusal must be about coverage"
    assert not any("timestamp" in m for m in out), \
        "the timestamps agree in this case, so only the grid may be at fault"


def test_a_strided_grid_holding_coordinates_the_direct_one_lacks_is_refused():
    """G2. The asymmetric case, which a check written from one side only would miss."""
    out = mismatches(shape_s=(7, 8), shared_rows=6, shared_cols=7)
    assert out, "extra coordinates on the strided side must be refused too"
    assert any("strided grid" in m for m in out)


def test_a_direct_grid_holding_coordinates_the_strided_one_lacks_is_refused():
    """G1, isolated. The mirror of the test above, and it was missing.

    WHY IT IS SEPARATE. The parametrized cases above shrink the DIRECT grid, and in every
    one of them the surplus therefore sits on the strided side, so the strided clause fires
    and the direct clause is never the only thing standing between the comparator and a
    false pass. Deleting the direct clause left the whole file green, which mutation testing
    caught and reading did not. Here the direct grid is the LARGER one, so only its own
    clause can refuse it.
    """
    out = mismatches(shape_d=(7, 8), shared_rows=6, shared_cols=7)
    assert out, "extra coordinates on the direct side must be refused"
    assert any("direct grid" in m for m in out), \
        "only the direct-side clause can catch this, so it must be the one reporting"
    assert not any("strided grid" in m for m in out), \
        "the strided grid is fully shared here, so it must not be blamed"


@pytest.mark.parametrize("n_times_s,n_times_d,n_shared", [
    (4, 3, 3),   # G3: the direct tree is a strict subset, the case a review built
    (3, 4, 3),   # G4: the direct tree has one the strided lacks
    (4, 4, 3),   # neither is a subset, they merely overlap
])
def test_timestamps_must_match_on_both_sides(n_times_s, n_times_d, n_shared):
    """G3 and G4. Equality on both sides, not a comparison against the smaller count.

    `shared != min(a, b)` passes a strict subset, which is how a direct tree missing its
    last timestep was reported as agreeing. Comparing against one side alone would then
    pass the mirror image. Both clauses are needed and each case here isolates one.
    """
    out = mismatches(n_times_s=n_times_s, n_times_d=n_times_d, n_shared_times=n_shared)
    assert out, (f"{n_times_s} and {n_times_d} timestamps sharing {n_shared} must be "
                 f"refused; a substitute must carry the same ones")
    assert any("timestamp" in m for m in out)


def test_the_verdict_follows_the_coverage_check(tmp_path, monkeypatch, capsys):
    """G5. A reported mismatch must change the EXIT CODE, not just print a line.

    The first repair of G1 printed a note and carried on to report agreement, so the check
    existed and the verdict ignored it. This drives `main` end to end on a real NetCDF pair
    so the seam between the two is covered rather than assumed.
    """
    nc = pytest.importorskip("netCDF4")
    import datetime

    import numpy as np

    def write(directory, n_times, n_lat):
        os.makedirs(directory, exist_ok=True)
        lat = 20.0 - 1.0 * np.arange(n_lat)
        lon = -10.0 + 1.0 * np.arange(9)
        start = (datetime.date(1981, 1, 1) - datetime.date(1970, 1, 1)).days * 86400
        seconds = [start + 21600 * i for i in range(n_times)]
        LO, LA = np.meshgrid(lon, lat)
        for var, name in (("u700", "u"), ("v700", "v")):
            base = np.sin(np.deg2rad(120 * LO)) * np.cos(np.deg2rad(150 * LA))
            values = np.stack([(-8.0 if name == "u" else 3.0) * base
                               for _ in range(n_times)])
            with nc.Dataset(os.path.join(
                    directory, f"era5_{var}_1981_6h_region.nc"), "w") as ds:
                ds.createDimension("valid_time", n_times)
                ds.createDimension("latitude", lat.size)
                ds.createDimension("longitude", lon.size)
                t = ds.createVariable("valid_time", "i8", ("valid_time",))
                t.units = "seconds since 1970-01-01"
                t[:] = np.asarray(seconds)
                ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
                ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
                ds.createVariable(name, "f4",
                                  ("valid_time", "latitude", "longitude"))[:] = values

    strided = str(tmp_path / "strided")
    write(strided, n_times=4, n_lat=8)

    same = str(tmp_path / "same")
    write(same, n_times=4, n_lat=8)
    assert cgr.main(["--strided-dir", strided, "--direct-dir", same,
                     "--year", "1981", "--subsample", "1", "--steps", "4"]) == 0, \
        "two identical retrievals must pass, or the refusals below prove nothing"

    subset = str(tmp_path / "subset")
    write(subset, n_times=3, n_lat=8)
    assert cgr.main(["--strided-dir", strided, "--direct-dir", subset,
                     "--year", "1981", "--subsample", "1", "--steps", "4"]) == 1, \
        "a strict timestamp subset must fail, and it must fail through the exit code"

    short = str(tmp_path / "short")
    write(short, n_times=4, n_lat=7)
    assert cgr.main(["--strided-dir", strided, "--direct-dir", short,
                     "--year", "1981", "--subsample", "1", "--steps", "4"]) == 1, \
        "a grid missing a row must fail through the exit code"
    capsys.readouterr()


# --- the value comparison, on the actual production route -------------------------------

@pytest.fixture
def half_degree_pair(tmp_path):
    """A 0.5 degree tree and a 1.0 degree tree derived from it, plus knobs to break them.

    WHY THIS EXISTS SEPARATELY from the coverage tests above. Those bind whether the two
    retrievals cover the same ground, and a review pointed out that they leave the VALUE
    half of the script unbound: the end-to-end case ran `--subsample 1` on two identical
    one-degree trees, which is not the production route. The route in production is
    half-degree winds strided to whole degrees, and that is what these build.
    """
    nc = pytest.importorskip("netCDF4")
    import datetime

    import numpy as np

    def write(directory, stride=1, perturb=0.0, phase=0.0, n_times=4):
        os.makedirs(directory, exist_ok=True)
        lat = 20.0 - 0.5 * np.arange(24)
        lon = -10.0 + 0.5 * np.arange(28)
        if stride > 1:
            lat, lon = lat[::stride], lon[::stride]
        lat = lat + phase
        lon = lon + phase
        LO, LA = np.meshgrid(lon, lat)
        start = (datetime.date(1981, 1, 1) - datetime.date(1970, 1, 1)).days * 86400
        seconds = [start + 21600 * i for i in range(n_times)]
        for var, name in (("u700", "u"), ("v700", "v")):
            # grid-scale structure, so striding is not a degenerate operation
            base = (np.sin(np.deg2rad(120 * LO)) * np.cos(np.deg2rad(150 * LA))
                    if name == "u" else
                    np.cos(np.deg2rad(140 * LO)) * np.sin(np.deg2rad(110 * LA)))
            scale = -8.0 if name == "u" else 3.0
            values = np.stack([scale * base for _ in range(n_times)])
            if perturb:
                values = values + perturb * np.sin(np.arange(values.shape[-1]))
            with nc.Dataset(os.path.join(
                    directory, f"era5_{var}_1981_6h_region.nc"), "w") as ds:
                ds.createDimension("valid_time", n_times)
                ds.createDimension("latitude", lat.size)
                ds.createDimension("longitude", lon.size)
                t = ds.createVariable("valid_time", "i8", ("valid_time",))
                t.units = "seconds since 1970-01-01"
                t[:] = np.asarray(seconds)
                ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
                ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
                ds.createVariable(name, "f4",
                                  ("valid_time", "latitude", "longitude"))[:] = values
        return directory
    return write


def run(strided, direct, subsample=2):
    return cgr.main(["--strided-dir", strided, "--direct-dir", direct,
                     "--year", "1981", "--subsample", str(subsample), "--steps", "4"])


def test_a_one_degree_tree_built_by_striding_is_reported_as_agreeing(
        half_degree_pair, tmp_path, capsys):
    """The baseline for the value half. Without it, every refusal below proves nothing.

    A one-degree tree built by taking every second point of the half-degree one IS the
    strided route by construction, so the comparator must pass it. If this ever fails the
    others are worthless, because a script that refuses everything refuses correctly by
    accident.
    """
    half = half_degree_pair(str(tmp_path / "half"))
    one = half_degree_pair(str(tmp_path / "one"), stride=2)
    assert run(half, one) == 0, "the strided route compared against itself must agree"
    capsys.readouterr()


def test_a_thousandth_of_a_metre_per_second_is_reported_as_differing(
        half_degree_pair, tmp_path, capsys):
    """The value half must actually discriminate, at a level far below a real difference.

    0.001 m/s is orders of magnitude smaller than any difference two retrieval routes
    could plausibly show, so passing this leaves margin. It is the case that fails if the
    agreement tolerance is ever widened or the verdict is hardwired.
    """
    half = half_degree_pair(str(tmp_path / "half"))
    off = half_degree_pair(str(tmp_path / "off"), stride=2, perturb=1e-3)
    assert run(half, off) == 1, "a perturbed direct tree must be reported as differing"
    capsys.readouterr()


def test_a_grid_shifted_off_phase_is_refused(half_degree_pair, tmp_path, capsys):
    """A half-degree phase shift puts the direct grid on the coordinates the stride skips.

    This is the failure a stride is meant to avoid and the one a comparison on shared
    coordinates alone would miss entirely, because two grids half a degree out of phase
    share NOTHING, and a script that quietly compared their overlap would be comparing an
    empty set.
    """
    half = half_degree_pair(str(tmp_path / "half"))
    shifted = half_degree_pair(str(tmp_path / "shifted"), stride=2, phase=0.5)
    with pytest.raises(SystemExit):
        run(half, shifted)
    capsys.readouterr()


def test_a_direct_tree_at_the_wrong_resolution_is_refused(half_degree_pair, tmp_path,
                                                          capsys):
    """Striding by two must be compared against ONE degree, not against another half.

    Comparing a half-degree tree strided by two against an unstrided half-degree tree is
    comparing two different target grids, and the answer would be meaningless rather than
    merely wrong.
    """
    half = half_degree_pair(str(tmp_path / "half"))
    also_half = half_degree_pair(str(tmp_path / "also_half"))
    with pytest.raises(SystemExit, match="not"):
        run(half, also_half)
    capsys.readouterr()
