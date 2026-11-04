"""只测试 MOS1D.update_defect_total_rates：独立归一化和共同Fermi目标。"""

TARGET = "src.models.device_mis1d.MOS1D.update_defect_total_rates"
import numpy as np

KB=8.6173332621e-5


def test_species_keep_their_own_dft_normalizations(make_mos,make_defect):
    m=make_mos();a=make_defect(n_DFT_m3=1e27)
    b=make_defect(n_DFT_m3=2e27)
    for d in [a,b]:d.data['f'].fill(1.);m.add_defects(d)
    m.Vg=m.get_psi_ms(300.);m.update_defect_total_rates(300.)
    for key in ['k12_vbm_effective','k21_vbm_effective','k12_cbm_effective','k21_cbm_effective','rate_in','rate_out']:
        np.testing.assert_allclose(b.data[key],a.data[key]/2,rtol=2e-13)
    ef=m.channel.Evbm+m.channel.get_Ef_bulk(300.)
    expected=1/(1+np.exp((a.data['transition_level_local']-ef)/(KB*300.))*np.sqrt(a.data['Er21']/a.data['Er12']))
    np.testing.assert_allclose(a.data['rate_in']/(a.data['rate_in']+a.data['rate_out']),expected,rtol=1e-13)


def test_explicit_saved_state_reuse_does_not_recompute_surface_potential(make_mos,make_defect,monkeypatch):
    m=make_mos();m.add_defects(make_defect());m.update_electrostatics(300.)
    def forbidden(T):np.testing.assert_equal(True,False,err_msg='复用静电快照时不应重新求表面势')
    monkeypatch.setattr(m,'solve_psi_s',forbidden)
    m.update_defect_total_rates(300.,refresh_electrostatics=False)
    np.testing.assert_equal(np.all(np.isfinite(m.defects[0].data['rate_in'])),True)


def test_density_changes_charge_but_not_rates_in_saved_environment(make_mos,make_defect):
    m=make_mos();d=make_defect();m.add_defects(d)
    m.update_defect_total_rates(300.)
    keys=('rate_in','rate_out','k12_vbm_effective','k21_vbm_effective',
          'k12_cbm_effective','k21_cbm_effective')
    before={key:d.data[key].copy() for key in keys}
    charge=d.get_charge_sheets().copy()
    for factor in (2.,0.):
        d.Nt=1e24*factor
        m.update_defect_total_rates(300.,refresh_electrostatics=False)
        for key in keys:
            np.testing.assert_array_equal(d.data[key],before[key])
        np.testing.assert_allclose(d.get_charge_sheets(),charge*factor)


def test_gate_only_device_occupancy_follows_analytic_master_equation(make_mos,make_defect):
    from src.physics.solver.advance import advance
    m=make_mos();m.set_Vg(.2)
    # Nt=0 removes electrostatic feedback while keeping the single-defect kinetics.
    d=make_defect(Nt=0.,Vc_vbm=0.,Vc_cbm=0.,Vc_gate=1e-6)
    m.add_defects(d);m.update_defect_total_rates(300.)
    incoming=d.data['rate_in'].copy();outgoing=d.data['rate_out'].copy()
    total=incoming+outgoing;target=incoming/total
    expected=-target*np.expm1(-total*.1)
    advance(m,.1,300.,dt_max_s=.01)
    np.testing.assert_allclose(d.data['f'],expected,rtol=1e-12,atol=1e-15)
    ef_gate=m.channel.Evbm+m.channel.get_Ef_bulk(300.)-.2
    np.testing.assert_allclose(target,1/(1+np.exp((d.data['transition_level_local']-ef_gate)/(KB*300.))*np.sqrt(d.data['Er21']/d.data['Er12'])),rtol=1e-13)
