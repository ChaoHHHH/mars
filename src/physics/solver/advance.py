"""固定偏压阶段的自适应占据推进与静电环境刷新。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.device_mis1d import MOS1D

from src.physics.solver.step_defects import advance_occupation


def advance(
    device: MOS1D, duration: float, T: float, *,
    dt_initial_s: float = 1E-13, dt_max_s: float = 10.0,
    electrostatics_recalc_threshold_V: float = 0.001,
    on_step=None,
) -> dict:
    """在当前固定 Vg 下自适应推进 duration 秒，所有缺陷同步更新

    本函数原地更新 device 的缺陷占据、静电状态和速率

    试算占据对应的电压变化，超过阈值时刷新已有环境或缩步重试；
    on_step(device, elapsed_s) 在起点和每个接受步后观测，不得修改状态。
    """
    number_weights = [
        d.Nt * (d.x_range[1] - d.x_range[0]) * d.data["weights"] for d in device.defects
        ]
    number_sum = sum(float(w.sum()) for w in number_weights)
    trial_evaluations = 0
    rejected_trials = 0
    max_environment_shift = 0.0
    history = []
    elapsed = 0.0
    dt_cap = min(dt_initial_s, dt_max_s)
    needs_refresh = True
    steps = 0
    refreshes = 0
    while True:
        if needs_refresh:
            device.update_defect_total_rates(T)
            anchor_shift = device.get_delta_Vth()
            assert device.electrostatic_state is not None
            environment_psi = device.electrostatic_state["psi_s"]
            refreshes += 1
            needs_refresh = False
        if not history:
            mean_f = (sum(float(np.sum(w * d.data["f"]))
                          for w, d in zip(number_weights, device.defects))
                      / number_sum if number_sum > 0 else 0.0)
            history.append([0.0, 0.0, environment_psi, mean_f, anchor_shift])
            if on_step is not None:
                on_step(device, 0.0)
        if elapsed == duration:
            break

        remaining = duration - elapsed
        dt = min(dt_cap, remaining)
        old_f = [d.data["f"] for d in device.defects]
        current_shift = device.get_delta_Vth()
        retry_environment = False
        while True:
            # 缩步可能低于当前累计时间的浮点分辨率，不能在同一时刻提交新占据。
            if dt <= 0 or elapsed + dt == elapsed:
                raise RuntimeError("Adaptive time step cannot advance")
            next_occupancies = [
                advance_occupation(f, d.data["rate_in"], d.data["rate_out"], dt) for d, f in zip(device.defects, old_f)
                ]
            # 仅试算候选电荷的电压贡献；检查通过前不推进时间或写历史。
            # 不求解新环境，且始终恢复已接受的占据。
            try:
                for d, f in zip(device.defects, next_occupancies):
                    d.data["f"] = f
                trial_shift = device.get_delta_Vth()
            finally:
                for d, f in zip(device.defects, old_f):
                    d.data["f"] = f
            trial_evaluations += 1
            change = abs(trial_shift - anchor_shift)
            limit = electrostatics_recalc_threshold_V
            if limit == 0 or change <= limit:
                break
            rejected_trials += 1
            # 已有电压变化时，先在当前已接受状态刷新，避免剩余容限耗尽。
            if current_shift != anchor_shift:
                needs_refresh = retry_environment = True
                break
            dt *= min(0.8, 0.9 * limit / change)
        if retry_environment:
            continue
        max_environment_shift = max(max_environment_shift, change)
        for defects, f_new in zip(device.defects, next_occupancies):
            defects.data["f"] = f_new
        elapsed = duration if dt == remaining else elapsed + dt
        shift = device.get_delta_Vth()
        mean_f = (sum(float(np.sum(w * d.data["f"]))
                      for w, d in zip(number_weights, device.defects))
                  / number_sum if number_sum > 0 else 0.0)
        history.append([elapsed, dt, environment_psi, mean_f, shift])
        if on_step is not None:
            on_step(device, elapsed)
        steps += 1
        dt_cap = dt_max_s
        needs_refresh = (electrostatics_recalc_threshold_V == 0
                         or change >= 0.9 * electrostatics_recalc_threshold_V)
        if elapsed == duration:
            break

    # 阶段终点一定与最终占据一致；下次调用无条件从新 Vg/T 刷新。
    device.update_defect_total_rates(T)
    refreshes += 1
    assert device.electrostatic_state is not None
    history[-1][2] = device.electrostatic_state["psi_s"]
    return {
        "history": np.asarray(history, dtype=float),
        "columns": ("time_s", "dt_s", "psi_s_environment_V", "mean_f", "dVth_V"),
        "steps": steps, "environment_refreshes": refreshes,
        "trial_evaluations": trial_evaluations, "rejected_trials": rejected_trials,
        "max_environment_shift_V": max_environment_shift,
    }
