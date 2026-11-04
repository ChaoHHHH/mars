"""器件静态预平衡：阻尼迭代与收敛判定，不代表物理经过时间。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.device_mis1d import MOS1D


def pre_equilibrate(
    device: MOS1D, Vg: float, T: float, *,
    min_number_steps: int = 10, max_number_steps: int = 200,
    occupancy_tolerance: float = 1E-8, potential_tolerance_V: float = 1E-8,
    mixing: float = 0.5,
) -> dict:
    """求指定栅压下的自洽稳态占据情况

    栅极与沟道同时耦合且费米能级不同时，目标是多热库稳态而非热平衡。
    本函数原地设置 device 的栅压，并更新占据、静电状态和速率
    """
    device.set_Vg(Vg)
    Vth0 = device.get_Vth0(T)
    device.update_defect_total_rates(T)

    # 统一评估初始状态和每次回溯试探状态的平衡目标。
    def equilibrium_targets() -> list[np.ndarray]:
        """从对数速率计算平衡占据，保留极慢点并识别真正断开的点。"""
        targets = []
        for defects in device.defects:
            data = defects.data
            li, lo = data["log_rate_in"], data["log_rate_out"]
            total = np.logaddexp(li, lo)
            connected = np.isfinite(total)
            log_target = np.subtract(
                li, total, out=np.zeros_like(data["f"]), where=connected,
            )
            target = np.where(connected, np.exp(log_target), data["f"])
            targets.append(target)
        return targets

    targets = equilibrium_targets()
    residual = max(
        (float(np.max(np.abs(d.data["f"] - target)))
         for d, target in zip(device.defects, targets)),
        default=0.0,
    )
    damping = mixing
    potential_change = 0.0
    for step in range(1, max_number_steps + 1):
        old_f = [d.data["f"].copy() for d in device.defects]
        old_levels = [
            d.data["Et_local"].copy()
            for d in device.defects
        ]
        old_vbm = float(device.VBM_local)
        old_shift = device.get_delta_Vth()
        trial_mixing = damping
        while True:
            for d, f, target in zip(device.defects, old_f, targets):
                d.data["f"] = f + trial_mixing * (target - f)
            device.update_defect_total_rates(T)
            new_targets = equilibrium_targets()
            new_residual = max(
                (float(np.max(np.abs(d.data["f"] - target)))
                 for d, target in zip(device.defects, new_targets)),
                default=0.0,
            )
            if new_residual <= occupancy_tolerance or new_residual < residual:
                break
            trial_mixing *= 0.5
            if trial_mixing < 1E-10:
                for d, f in zip(device.defects, old_f):
                    d.data["f"] = f
                device.update_defect_total_rates(T)
                raise RuntimeError(
                    f"Pre-equilibration stalled; occupancy residual={residual:.3e}"
                )
        potential_change = max(
            abs(device.get_delta_Vth() - old_shift),
            abs(float(device.VBM_local) - old_vbm),
            max((float(np.max(np.abs(d.data["Et_local"] - old)))
                 for d, old in zip(device.defects, old_levels)), default=0.0),
        )
        occupancy_change = max((float(np.max(np.abs(d.data["f"] - f)))
                                for d, f in zip(device.defects, old_f)), default=0.0)
        residual, targets = new_residual, new_targets
        damping = min(mixing, 2.0 * trial_mixing)
        if (step >= min_number_steps and residual <= occupancy_tolerance
                and occupancy_change <= occupancy_tolerance
                and potential_change <= potential_tolerance_V):
            shift = device.get_delta_Vth()
            return {
                "steps": step,
                "occupancy_residual": residual,
                "occupancy_change": occupancy_change,
                "potential_change_V": potential_change,
                "mixing": trial_mixing,
                "Vth0": Vth0,
                "dVth": shift,
                "Vth": Vth0 + shift,
            }
    raise RuntimeError(
        f"Pre-equilibration did not converge after {max_number_steps} iterations; "
        f"occupancy residual={residual:.3e}, potential change={potential_change:.3e} V"
    )
