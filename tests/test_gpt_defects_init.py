"""只测试 Defects.__init__：DFT归一化参数和自动建立采样数据。"""

TARGET = "src.models.defect.Defects.__init__"
import numpy as np
import pytest


def test_normalization_is_owned_by_each_defect(make_defect):
    a=make_defect(n_DFT_m3=2e27)
    b=make_defect(n_DFT_m3=4e27)
    np.testing.assert_equal(a.n_DFT_m3,2e27)
    np.testing.assert_equal(b.n_DFT_m3,4e27)
    np.testing.assert_equal(a.data is not None,True)


@pytest.mark.parametrize("field",["n_DFT_m3"])
@pytest.mark.parametrize("value",[0.,-1.,np.nan,np.inf])
def test_dft_number_densities_must_be_finite_positive(make_defect,field,value):
    with np.testing.assert_raises(ValueError):make_defect(**{field:value})


def test_default_gate_channel_is_disabled_and_coefficients_are_zero(make_defect):
    d=make_defect()
    for name in ('Vc_gate','decay_gate','temperature_vbm','temperature_cbm','temperature_gate'):
        np.testing.assert_equal(getattr(d,name),0.)
