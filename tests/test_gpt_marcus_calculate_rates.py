"""只测试 calculate_marcus_rates：FGR单位、势垒、细致平衡及热激活。"""

TARGET = "src.physics.marcus_rates.calculate_marcus_rates"
import numpy as np
import pytest
from src.physics.marcus_rates import calculate_marcus_rates

KB=8.6173332621e-5;HBAR=1.054571817e-34/1.602176634e-19


def test_symmetric_rate_matches_independent_si_prefactor():
    T=300.;lam=.8;vc=.01;dg=.2
    out,incoming=calculate_marcus_rates(0.,dg,vc,lam,lam,T)
    prefactor=2*np.pi/HBAR*vc*vc/np.sqrt(4*np.pi*lam*KB*T)
    np.testing.assert_allclose(out,prefactor*np.exp(-(lam+dg)**2/(4*lam*KB*T)),rtol=1e-8)
    np.testing.assert_allclose(incoming,prefactor*np.exp(-(lam-dg)**2/(4*lam*KB*T)),rtol=1e-8)


@pytest.mark.parametrize("T",[100.,300.,500.])
def test_asymmetric_rates_obey_band_reservoir_detailed_balance(T):
    edges=np.array([-.2,0.,.2])
    a,b=calculate_marcus_rates(0.,edges,.01,.81,.61,T,return_log=True)
    np.testing.assert_allclose(b-a,edges/(KB*T)+.5*np.log(.81/.61),atol=2e-14,rtol=1e-13)


def test_coupling_squared_zero_coupling_and_sign_invariance():
    args=(0.,.1)
    a=calculate_marcus_rates(*args,.01,.8,.8,300.)
    b=calculate_marcus_rates(*args,-.02,.8,.8,300.)
    np.testing.assert_allclose(b,np.array(a)*4,rtol=1e-13)
    np.testing.assert_array_equal(calculate_marcus_rates(*args,0.,.8,.8,300.),[0.,0.])


def test_temperature_dependence_encodes_activation_energy():
    k=np.array([calculate_marcus_rates(0.,0.,.01,1.,1.,t)[0] for t in [200.,400.]])
    np.testing.assert_allclose(np.log(k[1]*np.sqrt(400.)/(k[0]*np.sqrt(200.))),.25/KB*(1/200.-1/400.),rtol=1e-13)


def test_reverse_rate_survives_forward_underflow():
    a,b=calculate_marcus_rates(0.,.5,1e-167,.5,.5,300.)
    np.testing.assert_equal(a,0.)
    np.testing.assert_array_less(0.,b)


def test_fd_rates_remain_finite_and_obey_detailed_balance_without_crossing():
    for reservoir,dg,l12,l21 in [("cbm",-3.,1.,2.),("vbm",3.,2.,1.)]:
        logs=calculate_marcus_rates(0.,dg,.01,l12,l21,300.,return_log=True,reservoir=reservoir)
        np.testing.assert_equal(np.all(np.isfinite(logs)),True)
        np.testing.assert_allclose(logs[1]-logs[0],dg/(KB*300.)+.5*np.log(l12/l21),rtol=1e-13)


def test_blocked_band_edge_channel_returns_zero_instead_of_nan():
    rates=calculate_marcus_rates(0.,-3.,.01,1.,2.,300.,reservoir="vbm")
    np.testing.assert_array_equal(rates,[0.,0.])
