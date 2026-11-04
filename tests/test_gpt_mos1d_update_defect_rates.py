"""只测试 MOS1D.update_defect_rates：局部能级转换和裸Marcus速率。"""

TARGET = "src.models.device_mis1d.MOS1D.update_defect_rates"
import numpy as np

KB=8.6173332621e-5;HBAR=4.135667696e-15/(2*np.pi)


def test_flatband_zero_driving_force_matches_symmetric_marcus_rate(make_mos,make_defect):
    m=make_mos();d=make_defect(Nt=0.,Et=m.channel.Evbm,Er12=.8,Er21=.8,Vc_vbm=.01,Vc_cbm=0.,decay_vbm=0.);m.add_defects(d)
    m.Vg=m.get_psi_ms(300.);m.update_defect_rates(300.)
    expected=2*np.pi/HBAR*.01**2/np.sqrt(4*np.pi*.8*KB*300.)*np.exp(-.8/(4*KB*300.))
    np.testing.assert_allclose(d.data['k12_vbm'],expected,rtol=1e-13)
    np.testing.assert_allclose(d.data['k21_vbm'],expected,rtol=1e-13)
    np.testing.assert_array_equal(d.data['k12_cbm'],np.zeros_like(d.data['x']))


def test_bare_rates_obey_local_band_detailed_balance(make_mos,make_defect):
    m=make_mos();d=make_defect(Nt=0.);m.add_defects(d);m.Vg=-.2;m.update_defect_rates(300.)
    for band,edge in [('vbm',m.VBM_local),('cbm',m.CBM_local)]:
        np.testing.assert_allclose(np.log(d.data[f'k21_{band}'])-d.data[f'log_k12_{band}'],(edge-d.data['transition_level_local'])/(KB*300.)+.5*np.log(d.data['Er12']/d.data['Er21']),atol=1e-12)


def test_device_selects_cbm_fd_and_leaves_vbm_blocked(make_mos,make_defect):
    m=make_mos()
    d=make_defect(Nt=0.,Et=m.channel.Evbm+m.channel.get_Eg(300.)+3.,
                  Er12=1.,Er21=2.,decay_vbm=0.,decay_cbm=0.)
    m.add_defects(d);m.Vg=m.get_psi_ms(300.);m.update_defect_rates(300.)
    np.testing.assert_equal(np.all(d.data["k12_cbm"]>0.),True)
    np.testing.assert_array_equal(d.data["k12_vbm"],np.zeros_like(d.data["k12_vbm"]))
    np.testing.assert_equal(np.all(np.isfinite(d.data["log_k12_cbm"])),True)


def test_gate_fermi_level_matches_workfunction_and_electrostatic_boundary(make_mos,make_defect):
    m=make_mos();d=make_defect(Vc_gate=.01,decay_gate=1e9)
    m.add_defects(d)
    for vg in (-.3,0.,.4):
        m.set_Vg(vg);m.update_defect_rates(300.)
        thickness=sum(layer.tox for layer in m.layers)
        expected=-m.gate.workfunction-m.get_potential(thickness,300.)
        np.testing.assert_allclose(m.EF_gate,expected,atol=1e-13)
        np.testing.assert_allclose(np.log(d.data['k21_gate'])-d.data['log_k12_gate'],
            (expected-d.data['transition_level_local'])/(KB*300.)+.5*np.log(d.data['Er12']/d.data['Er21']),atol=1e-12)


def test_gate_rates_use_metal_barriers_and_coupling_squared(make_mos,make_defect):
    m=make_mos();m.set_Vg(m.get_psi_ms(300.))
    d=make_defect(Nt=0.,Et=-m.gate.workfunction-3.,Vc_gate=.01,Er12=1.,Er21=2.)
    m.add_defects(d);m.update_defect_rates(300.)
    # 平带下交点在两个极小值之外，metal分支B12=EF_gate-Et=3 eV。
    expected=2*np.pi/HBAR*.01**2/np.sqrt(4*np.pi*KB*300.)*np.exp(-3./(KB*300.))
    np.testing.assert_allclose(d.data['k12_gate'],expected,rtol=1e-12)
    before=d.data['k12_gate'].copy();d.Vc_gate*=2
    m.update_defect_rates(300.)
    np.testing.assert_allclose(d.data['k12_gate'],4*before,rtol=1e-12)
