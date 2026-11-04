"""只测试 MOS1D.get_potential：电极边界及显式冻结快照。"""

TARGET = "src.models.device_mis1d.MOS1D.get_potential"
import numpy as np


def test_channel_and_gate_boundaries_with_charge(make_mos,make_defect):
    m=make_mos();m.add_defects(make_defect());m.set_Vg(.3)
    x=np.array([[0.,3e-9]])
    values=m.get_potential(x,300.)
    resistance=sum(l.tox/l.eps for l in m.layers)
    psi=values[0,0];phi_ms=m.gate.workfunction+m.channel.Evbm+m.channel.get_Ef_bulk(300.)
    np.testing.assert_allclose(values[0,-1],m.Vg-phi_ms,atol=2e-10,rtol=0.)
    np.testing.assert_array_equal(values.shape,x.shape)
    np.testing.assert_equal(m.electrostatic_state,None)


def test_saved_potential_is_explicitly_frozen(make_mos,make_defect):
    m=make_mos();d=make_defect();m.add_defects(d);m.update_electrostatics(300.)
    before=m.get_potential(1e-9,300.,use_saved_state=True);state=m.electrostatic_state
    d.data['f'].fill(.9)
    np.testing.assert_equal(m.get_potential(1e-9,300.,use_saved_state=True),before)
    np.testing.assert_array_less(1e-8,abs(m.get_potential(1e-9,300.)-before))
    np.testing.assert_equal(m.electrostatic_state is state,True)
