"""只测试 solve_crossing：抛物线交点方程，而不是复制当前分支规则。"""

TARGET = "src.physics.marcus_barriers.solve_crossing"
import numpy as np
import pytest
from src.physics.marcus_barriers import solve_crossing


@pytest.mark.parametrize("dg,l12,l21",[(0.,.8,.8),(.2,.81,.61),(-.2,.5,1.5),(.5,1.5,.5)])
def test_real_solution_satisfies_quadratic_and_selects_nearest_crossing(dg,l12,l21):
    q=solve_crossing(dg,l12,l21)
    residual=(l12-l21)*q*q-2*l12*q+dg+l12
    np.testing.assert_allclose(residual,0.,atol=1e-13,rtol=0.)
    roots=np.roots([l12-l21,-2*l12,dg+l12]) if l12!=l21 else np.array([(dg+l12)/(2*l12)])
    expected=roots[np.argmin(abs(roots))]
    np.testing.assert_allclose(q,expected,rtol=1e-13,atol=1e-14)


def test_equal_curvature_broadcasts_linear_crossing():
    dg=np.array([-.3,0.,.3])[:,None];lam=np.array([.5,1.])
    np.testing.assert_allclose(solve_crossing(dg,lam,lam),.5+dg/(2*lam),rtol=1e-14)


@pytest.mark.parametrize("epsilon",[1e-3,1e-6,1e-10,1e-14])
def test_small_positive_discriminant_is_not_a_double_root(epsilon):
    dg=-2.+epsilon;q=solve_crossing(dg,1.,2.)
    np.testing.assert_allclose(-q*q-2*q+dg+1.,0.,atol=1e-12,rtol=0.,
                               err_msg="正判别式被人为钳为零，返回值不满足交点方程")
    # 使用实际可表示的dg，避免把输入舍入误差误认为求根误差。
    np.testing.assert_allclose(q,-1.+np.sqrt(dg+2.),atol=2e-15,rtol=0.)
    np.testing.assert_array_less(0.,abs(q+1.))


def test_exact_zero_discriminant_is_a_double_root():
    np.testing.assert_allclose(solve_crossing(-2.,1.,2.),-1.,atol=0.,rtol=0.)


@pytest.mark.parametrize("scale",[1.,1e-3,1e-6])
def test_crossing_is_invariant_under_common_energy_scaling(scale):
    roots=np.roots([.81-.61,-2*.81,.1+.81])
    expected=roots[np.argmin(abs(roots))]
    np.testing.assert_allclose(solve_crossing(.1*scale,.81*scale,.61*scale),expected,rtol=1e-13)


def test_no_real_crossing_is_reported_instead_of_fabricated():
    # -q^2-2q-2=0 的判别式为-4，没有实交点。
    try:q=solve_crossing(-3.,1.,2.)
    except ValueError:return
    np.testing.assert_equal(np.all(~np.isfinite(q)),True,
                             err_msg="无实交点时不应伪造有限的实交点坐标")


def test_mixed_real_and_nonreal_crossings_are_independent():
    q=solve_crossing(np.array([0.,-3.]),1.,2.)
    np.testing.assert_equal(np.isfinite(q),[True,False])
    np.testing.assert_allclose(-q[0]**2-2*q[0]+1.,0.,atol=1e-14)


def test_nearly_equal_curvatures_keep_small_root_accuracy():
    q=solve_crossing(.2,1.,1.+1e-12)
    np.testing.assert_allclose((1.-(1.+1e-12))*q*q-2*q+1.2,0.,atol=1e-14)
