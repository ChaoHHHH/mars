"""只测试 MOS1D.__init__：初始器件状态。"""

TARGET = "src.models.device_mis1d.MOS1D.__init__"
import numpy as np


def test_initial_bias_and_empty_defect_state(make_mos):
    d=make_mos()
    np.testing.assert_equal(d.Vg,0.)
    np.testing.assert_equal(len(d.defects),0)
    np.testing.assert_equal(d.electrostatic_state,None)
