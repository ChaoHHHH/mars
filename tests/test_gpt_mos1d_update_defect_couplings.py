"""只测试 MOS1D.update_defect_couplings：绝对距离的指数衰减。"""

TARGET = "src.models.device_mis1d.MOS1D.update_defect_couplings"
import numpy as np


def test_distance_is_measured_from_channel_not_layer_start(make_mos,make_defect):
    m=make_mos();d=make_defect(x_range=(1e-9,2e-9),dx=.5e-9,Vc_vbm=.1,Vc_cbm=.2,decay_vbm=2e9,decay_cbm=4e9);m.add_defects(d)
    m.update_defect_couplings()
    np.testing.assert_allclose(d.data['Vc_vbm_local'][:,0,0,0],.1*np.exp(-np.array([2.,3.,4.])),rtol=1e-14)
    np.testing.assert_allclose(d.data['Vc_cbm_local'][:,0,0,0],.2*np.exp(-np.array([4.,6.,8.])),rtol=1e-14)


def test_zero_coupling_and_zero_decay_limits(make_mos,make_defect):
    m=make_mos();d=make_defect(Vc_cbm=0.,decay_vbm=0.);m.add_defects(d);m.update_defect_couplings()
    np.testing.assert_array_equal(d.data['Vc_cbm_local'],np.zeros_like(d.data['x']))
    np.testing.assert_array_equal(d.data['Vc_vbm_local'],np.full_like(d.data['x'],.01))


def test_gate_distance_uses_total_stack_thickness(make_mos,make_defect):
    m=make_mos();d=make_defect(x_range=(1e-9,3e-9),dx=1e-9,
                              Vc_gate=.3,decay_gate=2e9)
    m.add_defects(d);m.update_defect_couplings()
    np.testing.assert_allclose(d.data['Vc_gate_local'][:,0,0,0],
                               .3*np.exp(-np.array([4.,2.,0.])),rtol=1e-14)


def test_temperature_coefficients_are_stored_without_changing_couplings(make_mos,make_defect):
    m=make_mos();d=make_defect(Vc_gate=.02,temperature_vbm=.1,
                              temperature_cbm=.2,temperature_gate=.3)
    m.add_defects(d);m.update_defect_rates(250.)
    before={band:d.data[f'Vc_{band}_local'].copy() for band in ('vbm','cbm','gate')}
    m.update_defect_rates(400.)
    for band in before:
        np.testing.assert_array_equal(d.data[f'Vc_{band}_local'],before[band])
        np.testing.assert_equal(getattr(d,f'temperature_{band}'),
                                {'vbm':.1,'cbm':.2,'gate':.3}[band])
