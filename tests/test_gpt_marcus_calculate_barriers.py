"""只测试 calculate_marcus_barriers：对称Marcus公式及自由能差。"""

TARGET = "src.physics.marcus_barriers.calculate_marcus_barriers"
import numpy as np
import pytest
from src.physics.marcus_barriers import calculate_marcus_barriers


@pytest.mark.parametrize("lam",[.2,.81,2.])
def test_equal_curvature_marcus_barriers(lam):
    dg=np.array([-.1,0.,.1])
    forward,reverse=calculate_marcus_barriers(dg,lam,lam)
    np.testing.assert_allclose(forward,(lam+dg)**2/(4*lam),rtol=1e-13)
    np.testing.assert_allclose(reverse,(lam-dg)**2/(4*lam),rtol=1e-13)


def test_forward_reverse_barriers_differ_by_reaction_free_energy():
    dg=np.array([-.1,0.,.1]);forward,reverse=calculate_marcus_barriers(dg,.81,.61)
    np.testing.assert_allclose(forward-reverse,dg,atol=2e-15,rtol=1e-13)
    np.testing.assert_equal(np.all(forward>=0.) and np.all(reverse>=0.),True)


def test_activationless_and_inverted_regions():
    forward,_=calculate_marcus_barriers(np.array([-1.6,-.8,0.]),.8,.8)
    np.testing.assert_allclose(forward,[.2,0.,.2],atol=1e-14,rtol=0.)


def test_near_tangent_real_crossing_retains_free_energy_difference():
    dg=-2.+1e-6
    forward,reverse=calculate_marcus_barriers(dg,1.,2.)
    np.testing.assert_allclose(forward-reverse,dg,atol=2e-14,rtol=0.)


@pytest.mark.parametrize("reservoir,dg,l12,l21,expected", [
    ("cbm", -3., 1., 2., 0.), ("vbm", 3., 2., 1., 3.),
    ("cbm", -2., 1., 1., 0.), ("vbm", 2., 1., 1., 2.),
    ("metal", -3., 1., 2., 0.), ("metal", 3., 2., 1., 3.),
])
def test_fd_table_includes_real_weak_and_no_real_crossings(reservoir,dg,l12,l21,expected):
    forward,reverse=calculate_marcus_barriers(dg,l12,l21,reservoir=reservoir)
    np.testing.assert_allclose(forward,expected,atol=1e-14)
    np.testing.assert_allclose(reverse,expected-dg,atol=1e-14)


@pytest.mark.parametrize("reservoir,dg,l12,l21", [
    ("vbm", -3., 1., 2.), ("cbm", 3., 2., 1.),
    ("band_edge", -3., 1., 2.),
])
def test_no_allowed_fd_branch_has_infinite_barriers(reservoir,dg,l12,l21):
    np.testing.assert_array_equal(
        np.isposinf(calculate_marcus_barriers(dg,l12,l21,reservoir=reservoir)),[True,True])


@pytest.mark.parametrize("reservoir,l12,l21,boundary", [
    ("cbm",1.,2.,-1.),("cbm",1.,2.,-2.),
    ("vbm",2.,1.,1.),("vbm",2.,1.,2.),
])
def test_fd_barrier_is_continuous_at_weak_and_no_crossing_boundaries(reservoir,l12,l21,boundary):
    dg=boundary+np.array([-1e-8,0.,1e-8])
    forward,reverse=calculate_marcus_barriers(dg,l12,l21,reservoir=reservoir)
    np.testing.assert_allclose(forward,forward[1],atol=2e-8,rtol=0.)
    np.testing.assert_allclose(forward-reverse,dg,atol=1e-14)


def test_strong_coupling_is_unchanged_for_all_reservoirs():
    expected=calculate_marcus_barriers(.2,1.,2.)
    for reservoir in ("vbm","cbm","metal"):
        np.testing.assert_allclose(calculate_marcus_barriers(.2,1.,2.,reservoir=reservoir),expected)
