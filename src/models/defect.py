from __future__ import annotations

from math import ceil
from typing import Literal

import numpy as np
from scipy.stats import norm, truncnorm

ELEMENTARY_CHARGE = 1.602176634E-19

class Defects:
    def __init__(self, x_range: tuple[float, float], Nt: float, Et: float,
                 Er12: float, Er21: float, Vc_vbm: float, Vc_cbm: float,
                 type: Literal["donor", "acceptor"], dEt: float, dEr: float,
                 dx: float, Et_sigma: float, Er12_sigma: float,
                 Er21_sigma: float, non_coverage: float = 1e-4,
                 decay_vbm: float = 0.0, decay_cbm: float = 0.0,
                 *,
                 n_DFT_m3: float,
                 Vc_gate: float = 0.0, decay_gate: float = 0.0,
                 temperature_vbm: float = 0.0, temperature_cbm: float = 0.0,
                 temperature_gate: float = 0.0) -> None:
        self.n_DFT_m3 = float(n_DFT_m3)
        self.x_range = x_range
        self.Nt = Nt
        self.Et = Et
        self.Er12 = Er12
        self.Er21 = Er21
        self.Vc_vbm = Vc_vbm
        self.Vc_cbm = Vc_cbm
        self.Vc_gate = Vc_gate
        self.decay_vbm = decay_vbm  # m^-1
        self.decay_cbm = decay_cbm  # m^-1
        self.decay_gate = decay_gate # m^-1
        self.temperature_vbm = temperature_vbm
        self.temperature_cbm = temperature_cbm
        self.temperature_gate = temperature_gate
        self.type = type # donor：被电子占据为 0 态，无电子占据为 +1 态，acceptor：被电子占据为 -1 态，无电子占据为 0 态
        self.dEt = dEt
        self.dEr = dEr
        self.dx = dx
        self.Et_sigma = Et_sigma
        self.Er12_sigma = Er12_sigma
        self.Er21_sigma = Er21_sigma
        self.non_coverage = non_coverage

        self.data: dict[str, np.ndarray] | None = None
        # 根据采样参数在init时就自动采样
        self.sampling()

    def sampling(self) -> dict[str, np.ndarray]:
        x_start, x_end = self.x_range
        sample_range = norm.ppf(1 - self.non_coverage)

        def energy_axis(mean: float, sigma: float, spacing: float, positive: bool = False) -> tuple[np.ndarray, np.ndarray]:
            if sigma == 0:
                return np.array([mean]), np.ones(1)
            if positive:
                distribution = truncnorm(-mean / sigma, np.inf, loc=mean, scale=sigma)
                lower, upper = distribution.ppf([self.non_coverage, 1.0 - self.non_coverage])
                samples = np.linspace(lower, upper, max(2, ceil((upper - lower) / spacing) + 1))
                edges = np.concatenate(([0.0], (samples[:-1] + samples[1:]) / 2, [np.inf]))
                weights = distribution.sf(edges[:-1]) - distribution.sf(edges[1:])
                weights /= weights.sum()
                return samples, weights

            lower = mean - sample_range * sigma
            upper = mean + sample_range * sigma
            samples, step = np.linspace(lower, upper, ceil((upper - lower) / spacing) + 1, retstep=True)
            weights = (norm.sf(samples - step / 2, loc=mean, scale=sigma) - norm.sf(samples + step / 2, loc=mean, scale=sigma))
            return samples, weights

        x = np.linspace(x_start, x_end, ceil((x_end - x_start) / self.dx) + 1)
        et, wt = energy_axis(self.Et, self.Et_sigma, self.dEt)
        er12, w12 = energy_axis(self.Er12, self.Er12_sigma, self.dEr, True)
        er21, w21 = energy_axis(self.Er21, self.Er21_sigma, self.dEr, True)
        xx, ee, e12, e21 = np.meshgrid(x, et, er12, er21, indexing="ij")
        weights = np.broadcast_to(wt[None, :, None, None] * w12[None, None, :, None] * w21[None, None, None, :] / x.size, xx.shape).copy()
        self.data = {"x": xx, "Et": ee, "Er12": e12, "Er21": e21, "weights": weights,
                     "f": np.zeros_like(weights, dtype=float)}

        return self.data

    def get_charge_sheets(self) -> np.ndarray:
        """sigma_j = q * Nt * thickness * SUM@x_j(weight * Q(f))。
        返回一维数组，与 self.data["x"][:, 0, 0, 0] 中的 x_j 对应
        donor: Q(f) = 1 - f；acceptor: Q(f) = -f
        """

        f = self.data["f"]
        if self.type == "donor":
            # 电子占据时中性，未占据时带 +q 电荷。
            charge_state = 1.0 - f
        else: # self.type == "acceptor"
            # 电子占据时带 -q 电荷，未占据时中性。
            charge_state = -f
        thickness = self.x_range[1] - self.x_range[0]
        return ELEMENTARY_CHARGE * self.Nt * thickness * np.sum(self.data["weights"] * charge_state, axis=(1, 2, 3))
