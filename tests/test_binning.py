import numpy as np

from aew.binning import bin_sum


def test_interior_points_count_identically_both_variants():
    glon = np.array([0.0, 4.0, 8.0])  # bin edges at -2,2,6,10
    glat = np.array([0.0, 1.0, 2.0])  # bin edges at -0.5,0.5,1.5,2.5
    # points clearly inside cells
    x = np.array([0.1, 4.0, 7.9, 0.0])
    y = np.array([0.0, 1.0, 2.0, 2.0])
    for variant in ("fixed", "buggy"):
        gbin, gcnt = bin_sum(glon, glat, x, y, variant=variant)
        assert gcnt.shape == (3, 3)
        assert gcnt.sum() == 4
    # default z=ones => gbin == gcnt
    gbin, gcnt = bin_sum(glon, glat, x, y)
    np.testing.assert_array_equal(gbin, gcnt.astype(float))


def test_z_values_summed_into_correct_cell():
    glon = np.array([0.0, 10.0])
    glat = np.array([0.0, 10.0])
    x = np.array([0.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0])
    z = np.array([2.0, 3.0, 5.0])
    gbin, gcnt = bin_sum(glon, glat, x, y, z)
    assert gbin[0, 0] == 5.0  # 2+3 into (lat0, lon0)
    assert gcnt[0, 0] == 2
    assert gbin[1, 1] == 5.0  # the single 5.0 into (lat1, lon1)
    assert gcnt[1, 1] == 1


def test_fixed_skips_out_of_domain_buggy_folds():
    glon = np.array([0.0, 4.0, 8.0])  # lon domain [-2, 10]
    glat = np.array([0.0, 1.0, 2.0])
    x = np.array([-3.0])  # outside the lower lon bound
    y = np.array([1.0])  # inside lat
    gbin_fixed, gcnt_fixed = bin_sum(glon, glat, x, y, variant="fixed")
    gbin_buggy, gcnt_buggy = bin_sum(glon, glat, x, y, variant="buggy")
    assert gcnt_fixed.sum() == 0  # fixed rejects out-of-range point
    assert gcnt_buggy.sum() == 1  # buggy folds it onto the edge bin
    assert gcnt_buggy[1, 0] == 1


def test_descending_lat_axis_handled():
    glon = np.array([0.0, 4.0])
    glat = np.array([2.0, 1.0, 0.0])  # descending
    x = np.array([0.0, 0.0])
    y = np.array([2.0, 0.0])
    gbin, gcnt = bin_sum(glon, glat, x, y, variant="fixed")
    assert gcnt.sum() == 2
    assert gcnt[0, 0] == 1  # y=2 -> first lat bin
    assert gcnt[2, 0] == 1  # y=0 -> last lat bin


def test_fill_value_points_skipped():
    glon = np.array([0.0, 4.0])
    glat = np.array([0.0, 4.0])
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 0.0])
    z = np.array([1.0, -999.0])
    gbin, gcnt = bin_sum(glon, glat, x, y, z, fill_value=-999.0)
    assert gcnt.sum() == 1
    assert gbin[0, 0] == 1.0
