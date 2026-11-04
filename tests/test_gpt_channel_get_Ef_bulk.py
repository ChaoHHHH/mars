"""只测试 Channel.get_Ef_bulk：电中性及n/p型费米能级位置。"""

TARGET = "src.models.device_channel.Channel.get_Ef_bulk"
import numpy as np
import pytest

KB=8.6173332621e-5


@pytest.mark.parametrize("Nd,Na",[(1e23,0.),(0.,1e23),(1e23,1e23)])
@pytest.mark.parametrize("T",[100.,300.,500.])
def test_fermi_level_satisfies_charge_neutrality(make_channel,monkeypatch,Nd,Na,T):
    c=make_channel(Nd=Nd,Na=Na,Eg0=1.2,Eg1=0.)
    monkeypatch.setattr(c,"get_Nv_Nc",lambda T:(1e25,1e25))
    ef=c.get_Ef_bulk(T);n=1e25*np.exp((ef-1.2)/(KB*T));p=1e25*np.exp(-ef/(KB*T))
    np.testing.assert_allclose(n-p,Nd-Na,rtol=2e-13,atol=1e-5)
    if Nd==Na:np.testing.assert_allclose(ef,.6,rtol=1e-14)
    elif Nd>Na:np.testing.assert_array_less(.6,ef)
    else:np.testing.assert_array_less(ef,.6)
