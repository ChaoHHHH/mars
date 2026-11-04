"""分层介质中电荷面的电势贡献，不依赖器件或缺陷对象。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ndarray


def get_delta_V_trap(
    x: float | ndarray,
    charge_sheets: Sequence[tuple[ndarray, ndarray]],
    layers: Sequence[tuple[float, float]],
) -> float | ndarray:
    """计算以沟道界面 x=0 为参考的电荷面电势贡献，单位 V。

    charge_sheets 中每项为 (位置数组, 面电荷密度数组)，单位 m、C/m²；
    layers 按沟道到栅极排列，每项为 (厚度, 绝对介电常数)，单位 m、F/m。
    标量 x 返回 float；数组 x 返回相同形状的数组。
    """
    x = np.asarray(x, dtype=float)
    delta_V = np.zeros_like(x, dtype=float)
    for xj, sigma in charge_sheets:
        layer_start = 0.0
        for thickness, permittivity in layers:
            layer_end = layer_start + thickness
            # 电荷面到观察点的路径与当前介质层的重叠长度。
            length = np.maximum(
                np.minimum(x[..., None], layer_end)
                - np.maximum(xj, layer_start),
                0.0,
            )
            delta_V -= np.sum(length * sigma / permittivity, axis=-1)
            layer_start = layer_end
    return float(delta_V) if x.ndim == 0 else delta_V
