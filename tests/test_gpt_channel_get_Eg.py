"""只测试 Channel.get_Eg：本项目线性带隙温度模型。"""

TARGET = "src.models.device_channel.Channel.get_Eg"
import numpy as np


def test_silicon_bandgap_temperature_values(make_channel):
    c=make_channel()
    np.testing.assert_allclose([c.get_Eg(t) for t in [100.,200.,300.,500.]], [1.1787,1.1514,1.1241,1.0695],rtol=1e-14)


def test_bandgap_is_independent_of_doping_and_vacuum_offset(make_channel):
    a=make_channel();b=make_channel(Nd=1e24,Na=0.,Evbm=-6.)
    np.testing.assert_equal(a.get_Eg(300.),b.get_Eg(300.))
