"""只测试 MOS1D.get_Vth0：强反型阈值条件及不改变器件状态。"""

TARGET = "src.models.device_mis1d.MOS1D.get_Vth0"
import numpy as np
import pytest
from scipy.optimize import brentq

KB=8.6173332621e-5


@pytest.mark.parametrize('Nd,Na',[(1e23,0.),(0.,1e23)])
def test_threshold_reaches_twice_bulk_fermi_potential(make_mos,make_channel,Nd,Na):
    m=make_mos(channel=make_channel(Nd=Nd,Na=Na));voltage=m.get_Vth0(300.)
    c=m.channel;n,p=c.get_carrier_density(300.);target=KB*300.*np.log(p/n)
    R=sum(l.tox/l.eps for l in m.layers);phi_ms=m.gate.workfunction+c.Evbm+c.get_Ef_bulk(300.)
    psi=brentq(lambda v:v-c.cal_Qs(v,300.)*R+phi_ms-voltage,-2.,2.,xtol=1e-14)
    np.testing.assert_allclose(psi,target,atol=2e-12,rtol=0.)


def test_ideal_threshold_ignores_traps_and_preserves_bias_occupation(make_mos,make_defect):
    m=make_mos();before=m.get_Vth0(300.);d=make_defect();d.data['f'].fill(.3);m.add_defects(d);m.Vg=1.2
    f=d.data['f'].copy()
    np.testing.assert_equal(m.get_Vth0(300.),before)
    np.testing.assert_equal(m.Vg,1.2);np.testing.assert_array_equal(d.data['f'],f)
