"""只测试 advance_occupation：主方程解析解、概率守恒及时间组合。"""

TARGET = "src.physics.solver.step_defects.advance_occupation"
import numpy as np
from src.physics.solver.step_defects import advance_occupation


def test_analytic_master_equation_and_disconnected_state():
    f=np.array([.2,.8,.37]);a=np.array([2.,1.,0.]);b=np.array([1.,4.,0.]);before=f.copy()
    expected=[2/3+(.2-2/3)*np.exp(-.3),.2+(.8-.2)*np.exp(-.5),.37]
    np.testing.assert_allclose(advance_occupation(f,a,b,.1),expected,rtol=1e-14)
    np.testing.assert_array_equal(f,before)


def test_equilibrium_limits_and_semigroup_property():
    f=np.array([0.,1.]);a=np.array([2.,2.]);b=np.array([1.,1.])
    np.testing.assert_allclose(advance_occupation(f,a,b,100.),[2/3,2/3],rtol=1e-14)
    once=advance_occupation(f,a,b,.3);twice=advance_occupation(advance_occupation(f,a,b,.1),a,b,.2)
    np.testing.assert_allclose(once,twice,rtol=1e-14)
    np.testing.assert_equal(np.all((once>=0)&(once<=1)),True)
    np.testing.assert_array_equal(advance_occupation(f,a,b,0.),f)
