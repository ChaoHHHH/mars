"""只测试 Channel.get_Ei：本征载流子平衡及能量参考。"""

TARGET = "src.models.device_channel.Channel.get_Ei"
import numpy as np
import pytest

KB=8.6173332621e-5


@pytest.mark.parametrize("ratio",[.25,1.,4.])
@pytest.mark.parametrize("T",[200.,300.,500.])
def test_intrinsic_level_equalizes_electron_and_hole_densities(make_channel,monkeypatch,ratio,T):
    c=make_channel(Eg0=1.2,Eg1=0.)
    monkeypatch.setattr(c,"get_Nv_Nc",lambda T:(ratio*1e25,1e25))
    ei=c.get_Ei(T)
    n=1e25*np.exp((ei-1.2)/(KB*T));p=ratio*1e25*np.exp(-ei/(KB*T))
    np.testing.assert_allclose(n,p,rtol=1e-13)
    np.testing.assert_allclose(ei,.6+KB*T/2*np.log(ratio),rtol=1e-14)


def test_intrinsic_level_is_relative_to_vbm_and_not_doping(make_channel):
    a=make_channel();b=make_channel(Nd=1e24,Na=0.,Evbm=-6.)
    np.testing.assert_equal(a.get_Ei(300.),b.get_Ei(300.))
