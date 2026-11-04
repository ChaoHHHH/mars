"""只测试 MOS1D.set_Vg：偏压改变后旧静电快照失效。"""

TARGET = "src.models.device_mis1d.MOS1D.set_Vg"
import numpy as np


def test_bias_change_invalidates_snapshot_without_touching_occupation(make_mos,make_defect):
    m=make_mos();d=make_defect();m.add_defects(d);m.update_electrostatics(300.)
    f=d.data['f'].copy();m.set_Vg(-.5)
    np.testing.assert_equal(m.Vg,-.5)
    np.testing.assert_equal(m.electrostatic_state,None)
    np.testing.assert_array_equal(d.data['f'],f)
