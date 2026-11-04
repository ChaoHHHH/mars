"""表面势方程的数值求根；器件物理方程由调用方提供。"""

from __future__ import annotations

from collections.abc import Callable

from scipy.optimize import brentq


def solve_psi_s(
    residual: Callable[[float], float],
    bracket: tuple[float, float],
    *,
    xtol: float = 2e-12,
) -> float:
    """在 bracket 内求 residual(psi_s)=0，返回表面势，单位 V。
    """
    lower, upper = bracket
    return float(brentq(residual, lower, upper, xtol=xtol))
