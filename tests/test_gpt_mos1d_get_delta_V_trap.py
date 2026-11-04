"""只测试 MOS1D.get_delta_V_trap：多种缺陷的电荷面汇总。"""

TARGET = "src.models.device_mis1d.MOS1D.get_delta_V_trap"
from types import SimpleNamespace
import numpy as np


def test_multiple_populations_are_summed_without_losing_positions(make_mos):
    m=make_mos()
    def population(x,sigma):
        return SimpleNamespace(data={'x':np.array(x).reshape(-1,1,1,1)},get_charge_sheets=lambda:np.array(sigma))
    m.defects=[population([.5e-9],[.001]),population([1.5e-9,2.5e-9],[.002,-.001])]
    e1=3.9*8.8541878188e-12;e2=20*8.8541878188e-12
    expected=-(.001*(.5e-9/e1+2e-9/e2)+.002*1.5e-9/e2-.001*.5e-9/e2)
    np.testing.assert_allclose(m.get_delta_V_trap(3e-9),expected,rtol=1e-14)
    np.testing.assert_array_equal(m.get_delta_V_trap(np.zeros((2,2))),np.zeros((2,2)))
