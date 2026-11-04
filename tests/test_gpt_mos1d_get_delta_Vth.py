"""只测试 MOS1D.get_delta_Vth：电荷的栅极投影及符号。"""

TARGET = "src.models.device_mis1d.MOS1D.get_delta_Vth"
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('sigma',[.001,-.001])
def test_signed_gate_projection_of_single_sheet(make_mos,sigma):
    m=make_mos();m.defects=[SimpleNamespace(data={'x':np.array([.5e-9]).reshape(1,1,1,1)},get_charge_sheets=lambda:np.array([sigma]))]
    expected=-sigma*(.5e-9/(3.9*8.8541878188e-12)+2e-9/(20*8.8541878188e-12))
    np.testing.assert_allclose(m.get_delta_Vth(),expected,rtol=1e-14)
    np.testing.assert_array_less(m.get_delta_Vth()*sigma,0.)


def test_no_defects_have_no_threshold_shift(make_mos):
    np.testing.assert_equal(make_mos().get_delta_Vth(),0.)
