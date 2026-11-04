"""Marcus 裸电子转移速率，能量 eV，时间 s；不含载流子或占据权重。"""
from __future__ import annotations

import numpy as np

from src.physics.marcus_barriers import calculate_marcus_barriers

KB_EV = 8.6173332621e-5
HBAR_EV_S = 4.135667696e-15 / (2.0 * np.pi)


def calculate_marcus_rates(transition_level, band_edge, coupling,
                           Er12, Er21, T, *, return_log=False,
                           reservoir="band_edge"):
    """返回 (k12, k21)，单位 s^-1，支持 NumPy 广播。

    12：缺陷→能带；21：能带→缺陷。
    仅由正向势垒计算 k12，反向由 log(k21/k12)=(band_edge-transition_level)/kT + 1/2*log(Er12/Er21) 构造。
    return_log=True 返回两速率的自然对数
    """
    et, eb, vc, er12, er21, temperature = np.broadcast_arrays(transition_level, band_edge, coupling, Er12, Er21, T)
    barrier12, _ = calculate_marcus_barriers(eb - et, er12, er21, reservoir=reservoir)
    thermal = KB_EV * temperature
    with np.errstate(divide="ignore"):
        log_k12 = (np.log(2.0 * np.pi / HBAR_EV_S) + 2.0 * np.log(np.abs(vc))
                   - 0.5 * np.log(4.0 * np.pi * er12 * thermal) - barrier12 / thermal)
    log_k21 = log_k12 + (eb - et) / thermal + 0.5 * (np.log(er12) - np.log(er21))
    if return_log:
        return log_k12, log_k21
    return np.exp(log_k12), np.exp(log_k21)
