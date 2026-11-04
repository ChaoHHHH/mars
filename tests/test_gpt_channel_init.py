"""只测试 Channel.__init__ 的SI介电常数和材料参数保存。"""

TARGET = "src.models.device_channel.Channel.__init__"
import numpy as np


def test_relative_to_absolute_permittivity(make_channel):
    c=make_channel(eps_r=11.7,Evbm=-5.17,Nd=1e23,Na=2e22)
    np.testing.assert_allclose(c.eps,11.7*8.8541878188e-12,rtol=1e-15)
    np.testing.assert_equal([c.Evbm,c.Nd,c.Na],[-5.17,1e23,2e22])
