"""缺陷有效速率的归一化与细致平衡计算，不依赖器件或缺陷对象。"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from src.physics.marcus_rates import KB_EV


def calculate_total_rates(
    *,
    log_k12_vbm: ndarray,
    log_k12_cbm: ndarray,
    log_k12_gate: ndarray,
    transition_level_local: ndarray,
    Er12: ndarray,
    Er21: ndarray,
    T: float,
    psi_s: float,
    cbm_local: float,
    ef: float,
    n0: float,
    p0: float,
    n_DFT_m3: float,
    ef_gate: float,
) -> dict[str, ndarray]:
    """返回沟道VBM/CBM及栅极热库的有效速率、总速率及其自然对数。

    能量使用相对真空的 eV，psi_s 使用 V，T 使用 K；n0、p0 和
    n_DFT_m3 使用 m^-3。VBM 正向速率乘 p/n_DFT_m3；
    CBM 乘 n/n_DFT_m3 及导带空态因子
    反向有效速率通过， 
    log(k21/k12)=(EF-transition_level_local)/kT + 0.5*log(Er12/Er21) 
    构造
    栅极使用自身费米能级 ef_gate
    三个通道均显式传入速率；耦合系数为0时，对应 log_k12 为 -inf，通道自然关闭
    速率单位 s^-1
    """
    thermal = KB_EV * T
    with np.errstate(divide="ignore"):
        log_n = np.log(n0) + psi_s / thermal
        log_p = np.log(p0) - psi_s / thermal
    log_cbm_empty = -np.logaddexp(0.0, (ef - cbm_local) / thermal)
    log_density_factor = -np.log(n_DFT_m3)
    log_entropy_ratio = 0.5 * (np.log(Er12) - np.log(Er21))
    log_ratio = (ef - transition_level_local) / thermal + log_entropy_ratio
    result = {}
    log_in_channels, log_out_channels = [], []
    for band, log_k12, log_factor in (
        ("vbm", log_k12_vbm, log_p + log_density_factor),
        ("cbm", log_k12_cbm, log_n + log_density_factor + log_cbm_empty),
    ):
        log_out = log_k12 + log_factor
        result[f"k12_{band}_effective"] = np.exp(log_out)
        result[f"k21_{band}_effective"] = np.exp(log_out + log_ratio)
        log_out_channels.append(log_out)
        log_in_channels.append(log_out + log_ratio)
    log_gate_out = log_k12_gate
    log_gate_in = (log_gate_out + (ef_gate - transition_level_local) / thermal
                   + log_entropy_ratio)
    result["k12_gate_effective"] = np.exp(log_gate_out)
    result["k21_gate_effective"] = np.exp(log_gate_in)
    result["log_rate_in"] = np.logaddexp(np.logaddexp(*log_in_channels), log_gate_in)
    result["log_rate_out"] = np.logaddexp(np.logaddexp(*log_out_channels), log_gate_out)
    result["rate_in"] = (result["k21_vbm_effective"] + result["k21_cbm_effective"]
                         + result["k21_gate_effective"])
    result["rate_out"] = (result["k12_vbm_effective"] + result["k12_cbm_effective"]
                          + result["k12_gate_effective"])
    return result
