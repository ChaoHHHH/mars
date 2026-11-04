"""只测试 Channel.get_carrier_density：Boltzmann分布和质量作用定律。"""

TARGET = "src.models.device_channel.Channel.get_carrier_density"
import numpy as np

KB=8.6173332621e-5


def test_fermi_shift_changes_carriers_reciprocally(make_channel,monkeypatch):
    c=make_channel(Eg0=1.2,Eg1=0.)
    monkeypatch.setattr(c,"get_Nv_Nc",lambda T:(1e25,1e25))
    monkeypatch.setattr(c,"get_Ef_bulk",lambda T:.6)
    n,p=c.get_carrier_density(300.)
    np.testing.assert_allclose(n,p,rtol=1e-14)
    monkeypatch.setattr(c,"get_Ef_bulk",lambda T:.6+KB*T*np.log(10.))
    nn,pp=c.get_carrier_density(300.)
    np.testing.assert_allclose([nn/n,pp/p],[10.,.1],rtol=2e-14)
    np.testing.assert_allclose(nn*pp,n*p,rtol=2e-14)


def test_actual_doped_channel_is_neutral_in_bulk(make_channel):
    for Nd,Na in [(1e23,0.),(0.,1e23),(1e23,1e23)]:
        c=make_channel(Nd=Nd,Na=Na);n,p=c.get_carrier_density(300.)
        np.testing.assert_allclose(n-p,Nd-Na,rtol=2e-13,atol=1e6)
