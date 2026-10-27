""" Tests for wrangler.utils """

import numpy as np
import pytest

from wrangler import utils


def test_calc_grad2_extremum_not_cancelled():
    """An isolated peak must not drop to zero (the central-difference sparkle)."""
    f = np.zeros((5, 5))
    f[2, 2] = 1.
    g2 = utils.calc_grad2(f, dx=1.)
    # x: slopes +1 and -1 -> squares 1, 1 -> mean 1; same along y
    assert g2[2, 2] == pytest.approx(2.)
    # np.gradient (the pre-2026-10-27 implementation) gives exactly 0 there
    old = np.gradient(f, axis=1)**2 + np.gradient(f, axis=0)**2
    assert old[2, 2] == 0.
    # Neighbours of the peak see one slope of 1 and one of 0 along that axis
    assert g2[2, 1] == pytest.approx(0.5)
    assert g2[1, 2] == pytest.approx(0.5)


def test_calc_grad2_linear_field_is_exact():
    """A plane has a uniform gradient, edges included."""
    jj, ii = np.mgrid[0:6, 0:7]
    dx = 2.
    f = 3. * ii * dx + 4. * jj * dx
    g2 = utils.calc_grad2(f, dx=dx)
    assert g2.shape == f.shape
    np.testing.assert_allclose(g2, 25.)


def test_calc_grad2_matches_explicit_stencil():
    """Interior and edge values against a direct loop."""
    rng = np.random.default_rng(42)
    f = rng.normal(size=(6, 8))
    dx = 0.5
    g2 = utils.calc_grad2(f, dx=dx)
    ny, nx = f.shape
    for j in range(ny):
        for i in range(nx):
            sx = [((f[j, k + 1] - f[j, k]) / dx)**2 for k in (i - 1, i) if 0 <= k < nx - 1]
            sy = [((f[k + 1, i] - f[k, i]) / dx)**2 for k in (j - 1, j) if 0 <= k < ny - 1]
            assert g2[j, i] == pytest.approx(np.mean(sx) + np.mean(sy))


def test_calc_grad2_nan_propagates():
    """A cell next to land (NaN) is NaN; cells further away are not."""
    f = np.arange(25, dtype=float).reshape(5, 5)
    f[2, 2] = np.nan
    g2 = utils.calc_grad2(f, dx=1.)
    assert np.isnan(g2[2, 2]) and np.isnan(g2[2, 1]) and np.isnan(g2[1, 2])
    assert np.isfinite(g2[0, 0]) and np.isfinite(g2[4, 4])


def test_calc_grad2_too_small():
    with pytest.raises(ValueError):
        utils.calc_grad2(np.zeros((1, 5)), dx=1.)
