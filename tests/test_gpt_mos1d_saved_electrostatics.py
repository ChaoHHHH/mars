"""只测试 MOS1D._saved_electrostatics：禁止错温度、错偏压和缺失快照。"""

TARGET = "src.models.device_mis1d.MOS1D._saved_electrostatics"
import numpy as np


def test_snapshot_requires_matching_voltage_and_temperature(make_mos):
    m=make_mos()
    with np.testing.assert_raises(RuntimeError):m._saved_electrostatics(300.)
    m.update_electrostatics(300.)
    np.testing.assert_equal(m._saved_electrostatics(300.) is m.electrostatic_state,True)
    with np.testing.assert_raises(RuntimeError):m._saved_electrostatics(310.)
    m.Vg=.1
    with np.testing.assert_raises(RuntimeError):m._saved_electrostatics(300.)
