"""冻结速率下的单步占据更新，不求电势、不修改器件状态。"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


def advance_occupation(
    f: ndarray, rate_in: ndarray, rate_out: ndarray, dt: float,
) -> ndarray:
    """解析推进 df/dt=(1-f)*rate_in-f*rate_out，返回新占据数组。

    dt 单位 s，速率单位 s^-1；输入数组保持不变，零总速率保留原占据。
    """
    if dt == 0:
        return f.copy()
    total_rate = rate_in + rate_out
    f_eq = np.divide(
        rate_in, total_rate, out=np.zeros_like(f), where=total_rate > 0,
    )
    # expm1 避免小步长下 1-exp(-rate*dt) 的有效数字损失。
    with np.errstate(over="ignore"):
        moved = -np.expm1(-total_rate * dt)
    return f + (f_eq - f) * moved
