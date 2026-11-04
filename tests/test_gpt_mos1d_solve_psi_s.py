"""只测试 MOS1D.solve_psi_s：含陷阱电荷的MOS边界条件。"""

TARGET = "src.models.device_mis1d.MOS1D.solve_psi_s"
import numpy as np
import pytest


@pytest.mark.parametrize('Nd,Na',[(1e23,0.),(0.,1e23)])
@pytest.mark.parametrize('effective_bias',[-1.5,-.3,0.,.3,1.5])
def test_surface_potential_satisfies_charge_balance(make_mos,make_channel,make_defect,Nd,Na,effective_bias):
    m=make_mos(channel=make_channel(Nd=Nd,Na=Na));d=make_defect();m.add_defects(d)
    resistance=sum(l.tox/l.eps for l in m.layers)
    # 初态donor为空；三个等权电荷面位于0、1、2nm。
    sigma=1.602176634e-19*1e24*2e-9/3
    shift=-sigma*((1e-9/(3.9*8.8541878188e-12)+2e-9/(20*8.8541878188e-12))
                  +(2e-9/(20*8.8541878188e-12))+(1e-9/(20*8.8541878188e-12)))
    psi_ms=m.gate.workfunction+m.channel.Evbm+m.channel.get_Ef_bulk(300.)
    m.Vg=psi_ms+shift+effective_bias
    psi=m.solve_psi_s(300.)
    np.testing.assert_allclose(psi-m.channel.cal_Qs(psi,300.)*resistance,effective_bias,atol=2e-10,rtol=0.)
    np.testing.assert_equal(np.isfinite(psi),True)
    np.testing.assert_equal(abs(psi)<=abs(effective_bias)+1e-12,True)
    if effective_bias!=0:np.testing.assert_array_less(0.,psi*effective_bias)
