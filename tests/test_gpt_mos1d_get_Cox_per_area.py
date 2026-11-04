"""只测试 MOS1D.get_Cox_per_area：多层串联电容及层拆分不变性。"""

TARGET = "src.models.device_mis1d.MOS1D.get_Cox_per_area"
import numpy as np
from src.models.device_insulatinglayer import InsulatingLayer

EPS=8.8541878188e-12


def test_series_capacitance_uses_absolute_permittivity(make_mos):
    m=make_mos()
    np.testing.assert_allclose(m.get_Cox_per_area(),EPS/(1e-9/3.9+2e-9/20.),rtol=1e-14)


def test_splitting_identical_dielectric_does_not_change_capacitance(make_mos):
    a=make_mos(layers=[InsulatingLayer(3.9,3e-9)])
    b=make_mos(layers=[InsulatingLayer(3.9,1e-9),InsulatingLayer(3.9,2e-9)])
    np.testing.assert_allclose(a.get_Cox_per_area(),b.get_Cox_per_area(),rtol=1e-14)
