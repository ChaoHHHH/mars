"""只测试 solver.get_potential：边界参考、电极场和电荷面贡献。"""

TARGET = "src.physics.solver.get_potential.get_potential"
import numpy as np
from src.physics.solver.get_potential import get_potential

EPS=8.8541878188e-12


def test_no_charge_potential_is_piecewise_linear():
    psi=.2;Qs=-.001;eps1=3.9*EPS;eps2=20*EPS
    layers=[(1e-9,eps1),(2e-9,eps2)]
    expected=psi-Qs*np.array([0.,1e-9/eps1,1e-9/eps1+2e-9/eps2])
    np.testing.assert_allclose(get_potential(np.array([0.,1e-9,3e-9]),psi,Qs,[],layers),expected,rtol=1e-14)


def test_potential_includes_sheet_charge_with_correct_sign():
    eps=4*EPS;sigma=.002;psi=.2;Qs=-.001
    sheet=[(np.array([1e-9]),np.array([sigma]))]
    np.testing.assert_allclose(get_potential(np.array([0.,1e-9,2e-9]),psi,Qs,sheet,[(3e-9,eps)]),
        [psi,psi-Qs*1e-9/eps,psi-Qs*2e-9/eps-sigma*1e-9/eps],rtol=1e-14)
