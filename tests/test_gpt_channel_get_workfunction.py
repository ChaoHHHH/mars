"""只测试 Channel.get_workfunction：真空参考与掺杂趋势。"""

TARGET = "src.models.device_channel.Channel.get_workfunction"
import numpy as np


def test_energy_reference_conversion(make_channel,monkeypatch):
    c=make_channel(Evbm=-5.)
    monkeypatch.setattr(c,"get_Ef_bulk",lambda T:.4)
    np.testing.assert_allclose(c.get_workfunction(300.),4.6,rtol=1e-14)
    c.Evbm=-5.2
    np.testing.assert_allclose(c.get_workfunction(300.),4.8,rtol=1e-14)


def test_n_type_has_lower_workfunction_than_p_type(make_channel):
    n=make_channel(Nd=1e23,Na=0.);p=make_channel(Nd=0.,Na=1e23)
    np.testing.assert_array_less(n.get_workfunction(300.),p.get_workfunction(300.))
