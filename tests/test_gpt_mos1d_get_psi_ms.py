"""只测试 MOS1D.get_psi_ms：金属/半导体功函数差的符号。"""

TARGET = "src.models.device_mis1d.MOS1D.get_psi_ms"
import numpy as np


def test_workfunction_difference_is_voltage(make_mos,monkeypatch):
    m=make_mos(workfunction=4.5)
    monkeypatch.setattr(m.channel,'get_workfunction',lambda T:4.3)
    np.testing.assert_allclose(m.get_psi_ms(300.),.2,atol=1e-15,rtol=0.)
    m.gate.workfunction=4.1
    np.testing.assert_allclose(m.get_psi_ms(300.),-.2,atol=1e-15,rtol=0.)
