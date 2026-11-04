"""只测试独立 solve_psi_s：解析根、端点根和无包根区间。"""

TARGET = "src.physics.solver.solve_psi_s.solve_psi_s"
import numpy as np
import pytest
from src.physics.solver.solve_psi_s import solve_psi_s


@pytest.mark.parametrize("root",[-1.,0.,.4,1.])
def test_known_linear_root(root):
    np.testing.assert_allclose(solve_psi_s(lambda x:3*(x-root),(-1.,1.)),root,atol=2e-12,rtol=0.)


def test_nonlinear_root_residual():
    root=solve_psi_s(lambda x:x*x-2,(0.,2.),xtol=1e-14)
    np.testing.assert_allclose(root,np.sqrt(2.),atol=2e-14,rtol=0.)
    np.testing.assert_allclose(root*root-2,0.,atol=1e-13,rtol=0.)


def test_non_bracketed_interval_raises():
    with np.testing.assert_raises(ValueError):solve_psi_s(lambda x:x*x+1,(-1.,1.))
