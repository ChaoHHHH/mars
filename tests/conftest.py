"""数值测试使用的物理参数；只准备输入，不作为待测公式的参考实现。"""
import pytest

from src.models.defect import Defects
from src.models.device_channel import Channel
from src.models.device_gate import Gate
from src.models.device_insulatinglayer import InsulatingLayer
from src.models.device_mis1d import MOS1D


@pytest.fixture
def make_defect():
    def make(**changes):
        parameters = dict(
            x_range=(0., 2e-9), Nt=1e24, Et=-4.5, Er12=.8, Er21=1.4,
            Vc_vbm=.01, Vc_cbm=.02, type="donor", dEt=.1, dEr=.1,
            dx=1e-9, Et_sigma=0., Er12_sigma=0., Er21_sigma=0.,
            non_coverage=1e-4, decay_vbm=2e9, decay_cbm=3e9, n_DFT_m3=1e27,
        )
        parameters.update(changes)
        return Defects(**parameters)
    return make


@pytest.fixture
def make_channel():
    def make(**changes):
        parameters = dict(
            Nd=0., Na=1e23, Evbm=-5.17, Eg0=1.206, Eg1=-2.73e-4,
            ncoeff=[.1905, .9163],
            pcoeff=[.443587, .003609528, .0001173515, 1.263218e-6,
                    3.025581e-9, .004683382, .0002286895, 7.469271e-7,
                    1.727481e-9],
            Mv=1, Mc=6, eps_r=11.7,
        )
        parameters.update(changes)
        return Channel(**parameters)
    return make


@pytest.fixture
def make_mos(make_channel):
    def make(*, channel=None, layers=None, workfunction=4.5):
        return MOS1D(
            Gate(workfunction),
            channel if channel is not None else make_channel(),
            layers if layers is not None else [
                InsulatingLayer(3.9, 1e-9), InsulatingLayer(20., 2e-9),
            ],
        )
    return make
