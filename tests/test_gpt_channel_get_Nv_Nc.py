"""只测试 Channel.get_Nv_Nc：三维态密度、SI单位及简并度。"""

TARGET = "src.models.device_channel.Channel.get_Nv_Nc"
import numpy as np
import pytest

ME=9.1093837015e-31;KB=1.380649e-23;H=6.62607015e-34


@pytest.mark.parametrize("T",[100.,300.,500.])
def test_density_of_states_matches_si_free_particle_result(make_channel,T):
    c=make_channel(Eg0=1.,Eg1=0.,ncoeff=[1.,1.],pcoeff=[1.,0.,0.,0.,0.,0.,0.,0.,0.],Mv=1,Mc=1)
    expected=2*(2*np.pi*ME*KB*T/H**2)**1.5
    np.testing.assert_allclose(c.get_Nv_Nc(T),[expected,expected],rtol=1e-9)


def test_degeneracy_and_temperature_scaling(make_channel):
    p=dict(Eg0=1.,Eg1=0.,ncoeff=[1.,1.],pcoeff=[1.,0.,0.,0.,0.,0.,0.,0.,0.])
    a=make_channel(**p,Mv=1,Mc=1);b=make_channel(**p,Mv=2,Mc=6)
    np.testing.assert_allclose(np.array(b.get_Nv_Nc(300.))/a.get_Nv_Nc(300.),[2.,6.],rtol=1e-14)
    np.testing.assert_allclose(np.array(a.get_Nv_Nc(400.))/a.get_Nv_Nc(100.),[8.,8.],rtol=1e-14)
