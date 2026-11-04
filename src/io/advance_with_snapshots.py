"""在指定物理时间准确停步并观测，复用原自适应推进器。"""
import numpy as np
from src.physics.solver.advance import advance


def advance_with_snapshots(device, duration, T, *, snapshot_times_s=(),
                           on_snapshot, on_step=None, **solver_options):
    """固定偏压阶段推进，时间列表相对此次调用起点，单位s，范围[0,duration]。

    回调on_snapshot(device, phase_time_s)读取此时状态，不能修改器件参数或f。
    重复时间去重。每个观测点为真实推进终点，刷新静电及速率后调用，
    不做占据插值；切分会增加环境刷新，结果允许在求解容差内略变。
    返回与advance相同的history/columns/steps/environment_refreshes。
    """
    times=sorted(set(snapshot_times_s))
    requested=set(times)
    histories=[];steps=refreshes=0;elapsed=0.
    if 0. in requested:
        device.update_defect_total_rates(T)
        refreshes+=1
        on_snapshot(device,0.)
    endpoints=[t for t in times if t>0.]
    if not endpoints or endpoints[-1]!=duration:
        endpoints.append(duration)
    for end in endpoints:
        options=dict(solver_options)
        if on_step is not None:
            def observe(current, time):
                if time > 0. or not histories:
                    on_step(current, elapsed + time)
            options['on_step']=observe
        result=advance(device,end-elapsed,T,**options)
        h=result['history'].copy()
        h[:,0]+=elapsed
        if histories:
            h=h[1:]  # 分段共同边界只保留一次
        histories.append(h)
        steps+=result['steps'];refreshes+=result['environment_refreshes']
        elapsed=end
        if end in requested and end > 0.:
            on_snapshot(device,end)
    return dict(history=np.concatenate(histories),columns=result['columns'],
                steps=steps,environment_refreshes=refreshes)
