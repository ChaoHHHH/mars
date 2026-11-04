"""nmos_sio2 的端到端回归：预平衡、三周期漂移/恢复及时间精度细化。"""
import json
from pathlib import Path

import numpy as np
from benchmarks.run import simulate

CASE = 'nmos_sio2'
REFERENCE_MV = [[41.74801682351065, 41.75437586245841, 41.75437586245841, 41.75437586245841], [3.88216002826149, 1.9963824852552008, 0.8486552987257225, 0.2824459434640508], [41.75292427270456, 41.75437586245842, 41.75437586245842, 41.75437586245842], [3.8821600282614908, 1.9963824852552003, 0.8486552987257225, 0.2824459434640508], [41.75292427270456, 41.75437586245842, 41.75437586245842, 41.75437586245842], [3.8821600282614908, 1.9963824852552003, 0.8486552987257225, 0.2824459434640508]]


def test_nmos_sio2_stress_recovery_regression():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "benchmarks" / f"{CASE}.json").read_text())
    base, equilibrium, initial = simulate(config, config["solver"])
    fine, fine_equilibrium, _ = simulate(config, config["refined_solver"])
    for eq in (equilibrium, fine_equilibrium):
        np.testing.assert_array_less(eq["occupancy_residual"], 1e-8)
        np.testing.assert_array_less(eq["potential_change_V"], 1e-8)
    np.testing.assert_equal(np.all((initial["f"] >= 0) & (initial["f"] <= 1)), True)
    np.testing.assert_allclose(base[0, 7], 0., atol=1e-14)
    limits = config["acceptance"]
    previous = 0.
    for phase, settings in enumerate(config["phases"]):
        a = base[base[:, 1] == phase]
        b = fine[fine[:, 1] == phase]
        for h in (a, b):
            np.testing.assert_equal(np.all(np.isfinite(h)), True)
            np.testing.assert_equal(np.all((h[:, 6] >= 0) & (h[:, 6] <= 1)), True)
            np.testing.assert_allclose(h[[0, -1], 2], [0., settings["duration_s"]], atol=1e-12)
            np.testing.assert_equal(np.all(np.diff(h[:, 2]) > 0.), True)
            np.testing.assert_allclose(h[:, 4], settings["Vg"])
        np.testing.assert_allclose(a[0, 7], previous, atol=1e-12)
        previous = a[-1, 7]
        error_mV = np.abs(np.interp(b[:, 2], a[:, 2], a[:, 7]) - b[:, 7]) * 1e3
        np.testing.assert_array_less(error_mV.max(), limits["max_curve_difference_mV"])
        # 移除缺陷浓度因子后的回归基线；旧曲线已通过100倍时间缩放独立核对。
        observed = np.interp([1., 10., 100., 1000.], b[:, 2], b[:, 7]) * 1e3
        np.testing.assert_allclose(observed, REFERENCE_MV[phase], atol=.1, rtol=0.)
        if settings["kind"] == "stress":
            peak_mV = a[-1, 7] * 1e3
            np.testing.assert_array_less(limits["min_first_stress_magnitude_mV"], 1 * peak_mV)
        else:
            np.testing.assert_array_less(abs(a[-1, 7] * 1e3 / peak_mV), limits["max_first_recovery_fraction"])
