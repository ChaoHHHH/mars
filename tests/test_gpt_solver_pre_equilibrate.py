"""只测试 pre_equilibrate：静态固定点、回溯、极慢点及真实Fermi占据。"""

TARGET = "src.physics.solver.pre_equilibrate.pre_equilibrate"
from types import SimpleNamespace
import numpy as np
import pytest
from src.physics.solver.pre_equilibrate import pre_equilibrate


class EquilibriumProblem:
    def __init__(self,initial=(.1,),mode='feedback'):
        self.Vg=0.;self.mode=mode
        self.defects=[SimpleNamespace(data={'f':np.array(initial,dtype=float)})]
    def set_Vg(self,v):self.Vg=v
    def get_Vth0(self,T):return 1.
    def get_delta_Vth(self):return .1*float(self.defects[0].data['f'].sum())
    def update_defect_total_rates(self,T):
        d=self.defects[0].data
        target=1-d['f'] if self.mode=='feedback' else np.full_like(d['f'],.8)
        with np.errstate(divide='ignore'):
            d['log_rate_in']=np.log(target);d['log_rate_out']=np.log1p(-target)
        if self.mode=='slow':
            d['log_rate_in']-=1000.;d['log_rate_out']-=1000.
        if self.mode=='disconnected':
            d['log_rate_in'].fill(-np.inf);d['log_rate_out'].fill(-np.inf)
        d['rate_in']=np.exp(d['log_rate_in']);d['rate_out']=np.exp(d['log_rate_out'])
        self.VBM_local=-.2*float(d['f'].sum());d['Et_local']=-.3*d['f']


def test_backtracking_converges_oscillatory_fixed_point():
    d=EquilibriumProblem();r=pre_equilibrate(d,-.4,300.,mixing=1.)
    np.testing.assert_allclose(d.defects[0].data['f'],.5,atol=1e-8)
    np.testing.assert_equal(r['occupancy_residual']<=1e-8,True)
    np.testing.assert_equal(r['potential_change_V']<=1e-8,True)
    np.testing.assert_equal(d.Vg,-.4)
    np.testing.assert_equal('equilibration_time_s' in r,False)


def test_constant_total_charge_does_not_hide_local_imbalance():
    d=EquilibriumProblem(initial=(.1,.9));initial=d.get_delta_Vth()
    pre_equilibrate(d,0.,300.)
    np.testing.assert_allclose(d.defects[0].data['f'],[.5,.5],atol=1e-8)
    np.testing.assert_allclose(d.get_delta_Vth(),initial,atol=1e-14)


def test_underflow_is_not_misidentified_as_disconnection():
    d=EquilibriumProblem(mode='slow');pre_equilibrate(d,0.,300.)
    np.testing.assert_allclose(d.defects[0].data['f'],.8,atol=1e-8)
    np.testing.assert_array_equal(d.defects[0].data['rate_in'],[0.])


def test_truly_disconnected_states_preserve_initial_occupation():
    d=EquilibriumProblem(initial=(.37,),mode='disconnected');r=pre_equilibrate(d,0.,300.)
    np.testing.assert_array_equal(d.defects[0].data['f'],[.37])
    np.testing.assert_equal(r['occupancy_residual'],0.)


def test_iteration_limit_does_not_report_false_equilibrium():
    with np.testing.assert_raises(RuntimeError):
        pre_equilibrate(EquilibriumProblem(),0.,300.,min_number_steps=1,max_number_steps=1)


@pytest.mark.parametrize('initial',[0.,1.])
def test_real_device_reaches_fermi_electron_occupation(make_mos,make_defect,initial):
    m=make_mos();ef=m.channel.Evbm+m.channel.get_Ef_bulk(300.)
    d=make_defect(Nt=0.,Et=ef+.1);d.data['f'].fill(initial);m.add_defects(d)
    pre_equilibrate(m,m.get_psi_ms(300.),300.)
    expected=1/(1+np.exp(.1/(8.6173332621e-5*300.))*np.sqrt(d.data['Er21']/d.data['Er12']))
    np.testing.assert_allclose(d.data['f'],expected,atol=1e-8,rtol=0.)
