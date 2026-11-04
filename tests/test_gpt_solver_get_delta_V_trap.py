"""只测试 solver.get_delta_V_trap：分层Poisson核、因果位置和场跳变。"""

TARGET = "src.physics.solver.get_delta_V_trap.get_delta_V_trap"
import numpy as np
from src.physics.solver.get_delta_V_trap import get_delta_V_trap

EPS=8.8541878188e-12


def test_piecewise_dielectric_poisson_integral_and_shapes():
    layers=[(1e-9,2*EPS),(2e-9,4*EPS)]
    sheets=[(np.array([.5e-9,1.5e-9,2.5e-9]),np.array([.001,.002,-.001]))]
    coefficients=np.array([[0.,0.,0.],[.5/(2*EPS),0.,0.],
        [.5/(2*EPS)+1/(4*EPS),.5/(4*EPS),0.],
        [.5/(2*EPS)+2/(4*EPS),1.5/(4*EPS),.5/(4*EPS)]])*1e-9
    expected=-coefficients@sheets[0][1]
    x=np.array([[0.,1e-9],[2e-9,3e-9]])
    np.testing.assert_allclose(get_delta_V_trap(x,sheets,layers),expected.reshape(2,2),rtol=1e-14,atol=1e-16)
    np.testing.assert_allclose(get_delta_V_trap(2e-9,sheets,layers),expected[2],rtol=1e-14)
    np.testing.assert_equal(isinstance(get_delta_V_trap(0.,sheets,layers),float),True)


def test_sheet_contributes_only_beyond_its_position_and_jumps_field():
    eps=4*EPS;sigma=.001;sheet=[(np.array([1e-9]),np.array([sigma]))]
    x=np.array([0.,.5e-9,1e-9,1.5e-9,2e-9])
    v=get_delta_V_trap(x,sheet,[(3e-9,eps)])
    np.testing.assert_allclose(v,[0.,0.,0.,-.5e-9*sigma/eps,-1e-9*sigma/eps],atol=1e-15)
    np.testing.assert_allclose((v[3]-v[2])/.5e-9,-sigma/eps,rtol=1e-14)


def test_empty_sheet_and_linear_superposition():
    x=np.linspace(0,3e-9,7);layers=[(3e-9,4*EPS)]
    a=[(np.array([.5e-9]),np.array([.001]))];b=[(np.array([1.5e-9]),np.array([-.002]))]
    np.testing.assert_array_equal(get_delta_V_trap(x,[],layers),np.zeros_like(x))
    np.testing.assert_allclose(get_delta_V_trap(x,a+b,layers),get_delta_V_trap(x,a,layers)+get_delta_V_trap(x,b,layers),atol=1e-15)
