"""只测试 Channel.get_mneff_mpeff：有效质量的数值、单位和温度趋势。"""

TARGET = "src.models.device_channel.Channel.get_mneff_mpeff"
import numpy as np
import pytest

ME=9.1093837015e-31


@pytest.mark.parametrize("T",[100.,300.,500.])
def test_analytic_constant_effective_masses(make_channel,T):
    c=make_channel(Eg0=1.,Eg1=0.,ncoeff=[2.,2.],pcoeff=[8.,0.,0.,0.,0.,0.,0.,0.,0.])
    np.testing.assert_allclose(c.get_mneff_mpeff(T),[2*ME,4*ME],rtol=1e-14)


def test_temperature_dependent_valence_mass(make_channel):
    c=make_channel(Eg0=1.,Eg1=0.,ncoeff=[1.,1.],pcoeff=[1.,.01,0.,0.,0.,0.,0.,0.,0.])
    masses=np.array([c.get_mneff_mpeff(t) for t in [100.,300.,500.]])
    np.testing.assert_allclose(masses[:,0],ME,rtol=1e-14)
    np.testing.assert_allclose(masses[:,1]/ME,np.array([2.,4.,6.])**(2/3),rtol=1e-14)
    np.testing.assert_array_less(0.,np.diff(masses[:,1]))
