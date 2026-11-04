"""只测试 MOS1D.update_electrostatics：统一静电快照与转变能级。"""

TARGET = "src.models.device_mis1d.MOS1D.update_electrostatics"
import numpy as np


def test_one_surface_solve_updates_all_local_energies(make_mos,make_defect,monkeypatch):
    m=make_mos();a=make_defect(Nt=0.,Et=-4.5)
    b=make_defect(Nt=0.,Et=-4.5,Er12=1.7,Er21=.3)
    m.add_defects(a);m.add_defects(b);m.Vg=m.get_psi_ms(300.)
    original=m.solve_psi_s;calls=[]
    def counted(T):calls.append(T);return original(T)
    monkeypatch.setattr(m,'solve_psi_s',counted)
    m.update_electrostatics(300.)
    np.testing.assert_equal(len(calls),1)
    np.testing.assert_allclose(m.electrostatic_state['psi_s'],0.,atol=1e-14)
    np.testing.assert_allclose(m.VBM_local,m.channel.Evbm,atol=1e-14)
    np.testing.assert_allclose(m.CBM_local-m.VBM_local,m.channel.get_Eg(300.),rtol=1e-14)
    np.testing.assert_allclose(a.data['transition_level_local'],b.data['transition_level_local'],atol=1e-14)
    np.testing.assert_array_equal(a.data['Et_local'],a.data['Et'])
    np.testing.assert_array_equal(a.data['f'],np.zeros_like(a.data['f']))


def test_local_transition_level_is_et_minus_potential(make_mos,make_defect):
    m=make_mos();d=make_defect(Et_sigma=.1,Er12_sigma=.2,Er21_sigma=.1)
    m.add_defects(d);m.set_Vg(.4);m.update_electrostatics(300.)
    x=d.data['x'][:,0,0,0]
    potential=m.get_potential(x,300.)[:,None,None,None]
    np.testing.assert_allclose(d.data['transition_level_local'],d.data['Et']-potential,atol=1e-14)
    np.testing.assert_array_equal(d.data['transition_level_local'],d.data['Et_local'])
