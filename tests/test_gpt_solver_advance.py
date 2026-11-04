"""只测试 advance：用可解析主方程验证步长、反馈收敛及阶段状态。"""

TARGET = "src.physics.solver.advance.advance"
from types import SimpleNamespace
import numpy as np
from src.physics.solver.advance import advance


class RateProblem:
    """控制器的解析参考问题，不代表具体材料。"""
    def __init__(self,initial=(.2,),incoming=(2.,),outgoing=(1.,),numbers=None,feedback=False):
        self.Vg=0.;self.incoming=incoming;self.outgoing=outgoing;self.feedback=feedback
        self.layers=[SimpleNamespace(tox=1.,eps=1.)]
        self.defects=[SimpleNamespace(Nt=n,x_range=(0.,1.),data={
            'x':np.full((1,1,1,1),.5),'weights':np.ones((1,1,1,1)),
            'f':np.full((1,1,1,1),f),
        }) for f,n in zip(initial,numbers or [1.]*len(initial))]
        self.refreshes=0;self.electrostatic_state=None
    def get_Cox_per_area(self):return 1.
    def get_delta_Vth(self):return -.02*sum(d.Nt*float((1-d.data['f']).sum()) for d in self.defects)
    def update_defect_total_rates(self,T):
        self.refreshes+=1
        self.electrostatic_state={'psi_s':-.1*sum(float(d.data['f'].sum()) for d in self.defects)}
        for i,d in enumerate(self.defects):
            d.data['rate_in']=np.full_like(d.data['f'],self.incoming[i])+(2*d.data['f'] if self.feedback else 0.)
            d.data['rate_out']=np.full_like(d.data['f'],self.outgoing[i])


def test_constant_rates_match_analytic_solution_and_one_clock():
    d=RateProblem(initial=(.1,.8),incoming=(2.,1.),outgoing=(1.,4.),numbers=[2.,7.])
    r=advance(d,1.,300.,dt_initial_s=.01,dt_max_s=.2,electrostatics_recalc_threshold_V=0.)
    expected=np.array([2/3+(.1-2/3)*np.exp(-3.),.2+(.8-.2)*np.exp(-5.)])
    np.testing.assert_allclose([p.data['f'].item() for p in d.defects],expected,rtol=1e-14)
    h=r['history'];np.testing.assert_equal(h[-1,0],1.)
    np.testing.assert_allclose(h[1:,1].sum(),1.,atol=1e-15)
    np.testing.assert_array_less(0.,h[1:,1]);np.testing.assert_equal(np.all(h[1:,1]<=.2),True)
    np.testing.assert_allclose(h[-1,3],(2*expected[0]+7*expected[1])/9,rtol=1e-14)
    np.testing.assert_equal(r['environment_refreshes'],d.refreshes)
    np.testing.assert_equal(h[-1,2],d.electrostatic_state['psi_s'])


def test_feedback_converges_to_independent_riccati_solution():
    # df/dt=2-f-2f²，两个定点为(-1±sqrt17)/4。
    hi=(-1+np.sqrt(17.))/4;lo=(-1-np.sqrt(17.))/4
    z=(.2-hi)/(.2-lo)*np.exp(-np.sqrt(17.))
    exact=(hi-z*lo)/(1-z);errors=[]
    for cap in [.2,.05,.001]:
        d=RateProblem(feedback=True)
        advance(d,1.,300.,dt_initial_s=cap,dt_max_s=cap,electrostatics_recalc_threshold_V=0.)
        errors.append(abs(d.defects[0].data['f'].item()-exact))
    np.testing.assert_array_less(errors[1:],errors[:-1])
    np.testing.assert_allclose(errors[-1],0.,atol=1e-4,rtol=0.)


def test_saturated_fast_rates_do_not_lock_the_time_step():
    d=RateProblem(initial=(1.,),incoming=(1e30,),outgoing=(1e10,))
    r=advance(d,1000.,300.,dt_max_s=500.)
    np.testing.assert_array_less(r['steps'],10)
    np.testing.assert_equal(r['history'][-1,0],1000.)
    np.testing.assert_array_equal(d.defects[0].data['f'],np.ones((1,1,1,1)))


def test_zero_duration_and_changed_next_phase():
    d=RateProblem();before=d.defects[0].data['f'].copy()
    r=advance(d,0.,300.)
    np.testing.assert_equal(r['steps'],0);np.testing.assert_array_equal(d.defects[0].data['f'],before)
    advance(d,.1,300.,dt_initial_s=.01,dt_max_s=.1)
    f=d.defects[0].data['f'].copy();d.Vg=-.4;d.incoming=(0.,);d.outgoing=(2.,)
    r=advance(d,.1,300.,dt_initial_s=.01,dt_max_s=.1)
    np.testing.assert_allclose(d.defects[0].data['f'],f*np.exp(-.2),rtol=1e-14)
    np.testing.assert_equal(r['history'][0,0],0.)


def test_real_mos_endpoint_matches_current_electrostatics(make_mos,make_defect):
    m=make_mos();d=make_defect();m.add_defects(d)
    r=advance(m,.05,300.,dt_initial_s=.001,dt_max_s=.01)
    np.testing.assert_equal(r['history'][-1,0],.05)
    np.testing.assert_allclose(r['history'][-1,4],m.get_delta_Vth(),atol=1e-14)
    np.testing.assert_array_equal(d.data['Et_local'],d.data['Et']-m.get_potential(d.data['x'],300.))


def test_voltage_precheck_limits_accepted_states_and_hides_rejected_trials():
    d=RateProblem(initial=(.1,.8),incoming=(20.,1.),outgoing=(1.,4.),numbers=[2.,7.])
    limit=1e-4
    accepted=[];refresh_states=[]
    update=d.update_defect_total_rates
    def refresh(T):
        refresh_states.append(tuple(float(p.data['f'].item()) for p in d.defects))
        update(T)
        d.anchor=d.get_delta_Vth()
    d.update_defect_total_rates=refresh
    def observe(device,t):
        assert abs(device.get_delta_Vth()-device.anchor)<=limit
        accepted.append((t,tuple(float(p.data['f'].item()) for p in device.defects)))
    r=advance(d,1.,300.,dt_initial_s=1.,dt_max_s=1.,
              electrostatics_recalc_threshold_V=limit,on_step=observe)
    assert r['rejected_trials']>0
    assert len(accepted)==r['steps']+1
    np.testing.assert_array_equal([x[0] for x in accepted],r['history'][:,0])
    assert np.all(np.diff(r['history'][:,0])>0)
    assert r['max_environment_shift_V']<=limit
    # 环境更新只能看到初态或已接受的状态，不能看到被拒绝的候选。
    assert all(state in [x[1] for x in accepted] for state in refresh_states)
    expected=np.array([20/21+(.1-20/21)*np.exp(-21.),.2+(.8-.2)*np.exp(-5.)])
    np.testing.assert_allclose([p.data['f'].item() for p in d.defects],expected,atol=2e-14)


def test_voltage_tolerance_refinement_converges_with_feedback():
    hi=(-1+np.sqrt(17.))/4;lo=(-1-np.sqrt(17.))/4
    z=(.2-hi)/(.2-lo)*np.exp(-np.sqrt(17.))
    exact=(hi-z*lo)/(1-z);errors=[]
    for limit in [1e-3,1e-4,1e-5]:
        d=RateProblem(feedback=True)
        advance(d,1.,300.,dt_initial_s=1.,dt_max_s=1.,
                electrostatics_recalc_threshold_V=limit)
        errors.append(abs(d.defects[0].data['f'].item()-exact))
    np.testing.assert_array_less(errors[1:],errors[:-1])
    np.testing.assert_allclose(errors[-1],0.,atol=1e-4,rtol=0.)


def test_subresolution_step_does_not_commit_occupancy_at_unchanged_time():
    d=RateProblem(incoming=(2.,),outgoing=(1.,))
    update=d.update_defect_total_rates
    def refresh(T):
        update(T)
        if d.defects[0].data['f'].item()>.3:
            d.defects[0].data['rate_in'].fill(1e30)
    d.update_defect_total_rates=refresh
    accepted=[]
    def observe(device,t):
        accepted.append((t,device.defects[0].data['f'].copy()))
    with np.testing.assert_raises_regex(RuntimeError,'time step cannot advance'):
        advance(d,1.,300.,dt_initial_s=.1,dt_max_s=.1,
                electrostatics_recalc_threshold_V=.001,on_step=observe)
    np.testing.assert_equal(np.all(np.diff([x[0] for x in accepted])>0),True)
    np.testing.assert_array_equal(d.defects[0].data['f'],accepted[-1][1])
