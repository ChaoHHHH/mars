"""指定真实时间的占据观测与分段历史拼接。"""
TARGET = 'src.io.advance_with_snapshots.advance_with_snapshots'
import importlib
from types import SimpleNamespace
import numpy as np
from src.io.advance_with_snapshots import advance_with_snapshots


def test_exact_observation_times_and_history_preserve_current_state(monkeypatch):
    module=importlib.import_module('src.io.advance_with_snapshots')
    device=SimpleNamespace(time=0.,f=0.,refreshes=0)
    def refresh(T):
        device.refreshes+=1
    device.update_defect_total_rates=refresh
    calls=[]
    def advance(device,duration,T,**options):
        calls.append((duration,options))
        initial=device.f
        device.time+=duration
        device.f=1.-np.exp(-device.time)
        return dict(history=np.array([[0.,0.,0.,initial,0.],
                                      [duration,duration,0.,device.f,0.]]),
                    columns=('time_s','dt_s','psi_s_environment_V','mean_f','dVth_V'),
                    steps=1,environment_refreshes=2)
    monkeypatch.setattr(module,'advance',advance)
    observed=[]
    result=advance_with_snapshots(device,2.,300.,snapshot_times_s=[2.,.3,0.,.3,1.],
        on_snapshot=lambda d,t: observed.append((t,d.time,d.f)),dt_max_s=.1)
    np.testing.assert_allclose(np.array(observed)[:,:2],[[0.,0.],[.3,.3],[1.,1.],[2.,2.]])
    np.testing.assert_allclose(np.array(observed)[:,2],1.-np.exp(-np.array([0.,.3,1.,2.])))
    np.testing.assert_allclose(result['history'][:,0],[0.,.3,1.,2.])
    np.testing.assert_allclose([c[0] for c in calls],[.3,.7,1.])
    np.testing.assert_equal(all(c[1]=={'dt_max_s':.1} for c in calls),True)
    np.testing.assert_equal(result['steps'],3)
    np.testing.assert_equal(result['environment_refreshes'],7)


def test_per_step_observer_uses_phase_clock_without_duplicate_split_boundaries(monkeypatch):
    module=importlib.import_module('src.io.advance_with_snapshots')
    device=SimpleNamespace(update_defect_total_rates=lambda T: None)
    def advance(current,duration,T,on_step=None,**options):
        on_step(current,0.);on_step(current,duration)
        return dict(history=np.array([[0.,0.,0.,0.,0.],[duration,duration,0.,0.,0.]]),
                    columns=(),steps=1,environment_refreshes=2)
    monkeypatch.setattr(module,'advance',advance)
    times=[]
    result=advance_with_snapshots(device,2.,300.,snapshot_times_s=[.3,1.],
          on_snapshot=lambda d,t: None,on_step=lambda d,t: times.append(t))
    np.testing.assert_allclose(times,[0.,.3,1.,2.])
    np.testing.assert_allclose(result['history'][:,0],times)
