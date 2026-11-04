"""只测试 Defects.get_charge_sheets：电荷态、体密度积分及离散权重。"""

TARGET = "src.models.defect.Defects.get_charge_sheets"
import numpy as np
import pytest

Q=1.602176634e-19


def prepare(defect):
    defect.Nt=2e24;defect.x_range=(0.,3e-9)
    defect.data={
        "x":np.broadcast_to(np.array([.5e-9,2.5e-9])[:,None,None,None],(2,2,1,1)).copy(),
        "weights":np.array([[.1,.2],[.3,.4]])[:,:,None,None],
        "f":np.array([[0.,.25],[.5,1.]])[:,:,None,None],
    }
    return defect


@pytest.mark.parametrize("kind,expected",[("donor",[.25,.15]),("acceptor",[-.05,-.55])])
def test_weighted_electron_charge_states(make_defect,kind,expected):
    d=prepare(make_defect(type=kind))
    np.testing.assert_allclose(d.get_charge_sheets(),Q*2e24*3e-9*np.array(expected),rtol=1e-14)


@pytest.mark.parametrize("kind,f,charge",[("donor",0.,1.),("donor",1.,0.),("acceptor",0.,0.),("acceptor",1.,-1.)])
def test_neutral_and_fully_charged_limits(make_defect,kind,f,charge):
    d=prepare(make_defect(type=kind));d.data["f"].fill(f)
    np.testing.assert_allclose(d.get_charge_sheets(),Q*2e24*3e-9*np.array([.3,.7])*charge,atol=0.,rtol=1e-14)


@pytest.mark.parametrize("kind",["donor","acceptor"])
def test_adding_electrons_always_reduces_electric_charge(make_defect,kind):
    d=prepare(make_defect(type=kind));d.data["f"].fill(.2);before=d.get_charge_sheets()
    d.data["f"].fill(.3)
    np.testing.assert_allclose(d.get_charge_sheets()-before,-Q*2e24*3e-9*.1*np.array([.3,.7]),rtol=1e-14)


def test_volume_density_and_thickness_set_areal_number(make_defect):
    d=prepare(make_defect());initial=d.get_charge_sheets()
    d.Nt*=2;np.testing.assert_allclose(d.get_charge_sheets(),2*initial,rtol=1e-14)
    d.x_range=(0.,6e-9);np.testing.assert_allclose(d.get_charge_sheets(),4*initial,rtol=1e-14)
    d.Nt=0.;np.testing.assert_array_equal(d.get_charge_sheets(),[0.,0.])


def test_splitting_quadrature_nodes_does_not_create_charge(make_defect):
    d=prepare(make_defect());initial=d.get_charge_sheets()
    d.data={key:np.repeat(value,2,axis=1)/(2 if key=="weights" else 1) for key,value in d.data.items()}
    np.testing.assert_allclose(d.get_charge_sheets(),initial,rtol=1e-14)
