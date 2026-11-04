"""给定表面势及电荷快照，计算分层介质中的电势分布。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ndarray

from src.physics.solver.get_delta_V_trap import get_delta_V_trap


def get_potential(
    x: float | ndarray,
    psi_s: float,
    Qs: float,
    charge_sheets: Sequence[tuple[ndarray, ndarray]],
    layers: Sequence[tuple[float, float]],
) -> float | ndarray:
    """电势单位 V；x 使用 m，Qs/面电荷使用 C/m²，介电常数使用 F/m。

    layers 为沿沟道到栅极排列的 (厚度, 绝对介电常数)；本函数不求根。
    """
    x = np.asarray(x, dtype=float)
    potential = np.full_like(x, psi_s, dtype=float)
    layer_start = 0.0
    for thickness, permittivity in layers:
        dx = np.clip(x - layer_start, 0.0, thickness)
        potential -= Qs / permittivity * dx
        layer_start += thickness
    potential += get_delta_V_trap(x, charge_sheets, layers)
    return float(potential) if x.ndim == 0 else potential
