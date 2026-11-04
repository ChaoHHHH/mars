"""只测试 Channel.cal_Qs：Poisson–Boltzmann电荷、平带和耗尽近似。"""

TARGET = "src.models.device_channel.Channel.cal_Qs"
import numpy as np
import pytest

Q=1.602176634e-19;KB=8.6173332621e-5


@pytest.mark.parametrize("psi",[-.2,-1e-12,0.,1e-12,.2])
def test_intrinsic_channel_has_analytic_odd_charge(make_channel,monkeypatch,psi):
    c=make_channel();monkeypatch.setattr(c,"get_carrier_density",lambda T:(1e16,1e16))
    expected=-np.sqrt(8*c.eps*KB*300.*Q*1e16)*np.sinh(psi/(2*KB*300.))
    np.testing.assert_allclose(c.cal_Qs(psi,300.),expected,rtol=2e-13,atol=1e-30)


def test_charge_is_monotone_and_opposes_surface_potential(make_channel):
    c=make_channel();psi=np.array([-.4,-.1,0.,.1,.4,.8])
    charge=np.array([c.cal_Qs(v,300.) for v in psi])
    np.testing.assert_array_less(np.diff(charge),0.)
    np.testing.assert_array_less(charge[psi!=0]*psi[psi!=0],0.)
    np.testing.assert_allclose(charge[2],0.,atol=0.,rtol=0.)


def test_p_type_depletion_charge_matches_asymptotic_poisson_solution(make_channel,monkeypatch):
    c=make_channel();monkeypatch.setattr(c,"get_carrier_density",lambda T:(1e7,1e23))
    psi=.4;expected=-np.sqrt(2*c.eps*Q*1e23*(psi-KB*300.))
    np.testing.assert_allclose(c.cal_Qs(psi,300.),expected,rtol=1e-6)
