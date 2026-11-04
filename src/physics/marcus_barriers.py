"""Marcus 交点与势垒，能量单位 eV；12 为缺陷向热库发射电子。"""
from __future__ import annotations

import numpy as np

__all__ = ["calculate_marcus_barriers", "solve_crossing"]


def solve_crossing(delta_g_12, lambda12, lambda21):
    """求解 (λ12−λ21)q²−2λ12q+(dG12+λ12)=0，取绝对值较小的根。

    正判别式不截断，负判别式返回 NaN；此处不应用物理势垒近似。
    使用稳定的二次求根形式，避免曲率接近时相减消去有效数字。
    """
    dg, l12, l21 = np.broadcast_arrays(delta_g_12, lambda12, lambda21)
    a = l12 - l21
    b = -2.0 * l12
    c = dg + l12
    discriminant = b**2 - 4.0 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_d = np.sqrt(discriminant)
        numerator = -0.5 * (b + np.copysign(sqrt_d, b))
        x1 = numerator / a
        x2 = c / numerator
        x = np.where(np.abs(x2) < np.abs(x1), x2, x1)
        x = np.where(discriminant == 0.0, -b / (2.0 * a), x)
        return np.where(a == 0.0, -c / b, x)


def calculate_marcus_barriers(delta_g_12, lambda12, lambda21, *,
                               reservoir="band_edge"):
    """返回 (B12, B21)，eV；B12−B21=dG12。

    band_edge 使用经典交点势垒，无实交点时两势垒均为 inf。
    vbm/cbm/metal 在弱耦合区按 Comphy 表 4.4 应用 FD 近似：
    交点不在两个极小值 q=0、1 之间（含无实交点）时，
    VBM 且 dG12>0 取 B12=dG12；CBM 且 dG12<0 取 B12=0；
    metal 取 B12=max(dG12,0)。其余情况保持经典势垒。
    近似反向势垒由 B21=B12−dG12 构造，保持细致平衡。
    """
    dg, l12, l21 = np.broadcast_arrays(delta_g_12, lambda12, lambda21)
    q = solve_crossing(dg, l12, l21)
    real = np.isfinite(q)
    barrier12 = np.where(real, l21 * q**2, np.inf)
    barrier21 = np.where(real, l12 * (q - 1.0)**2, np.inf)
    weak = ~real | (q < 0.0) | (q > 1.0)
    if reservoir == "vbm":
        approximate = weak & (dg > 0.0)
        forward = dg
    elif reservoir == "cbm":
        approximate = weak & (dg < 0.0)
        forward = np.zeros_like(dg)
    elif reservoir == "metal":
        approximate = weak
        forward = np.maximum(dg, 0.0)
    else:
        return barrier12, barrier21
    return (np.where(approximate, forward, barrier12),
            np.where(approximate, forward - dg, barrier21))
