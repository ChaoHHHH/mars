"""只测试 MOS1D.add_defects：多缺陷保留及电荷变化后的状态失效。"""

TARGET = "src.models.device_mis1d.MOS1D.add_defects"
import numpy as np


def test_new_population_invalidates_previous_electrostatics(make_mos,make_defect):
    m=make_mos();a=make_defect();b=make_defect(type='acceptor')
    m.add_defects(a);m.update_electrostatics(300.);m.add_defects(b)
    np.testing.assert_equal(len(m.defects),2)
    np.testing.assert_equal(m.defects[0] is a,True)
    np.testing.assert_equal(m.defects[1] is b,True)
    np.testing.assert_equal(m.electrostatic_state,None)
