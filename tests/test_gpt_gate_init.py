"""只测试 Gate.__init__：功函数以eV保存。"""

TARGET = "src.models.device_gate.Gate.__init__"
import numpy as np
import pytest
from src.models.device_gate import Gate


@pytest.mark.parametrize("workfunction",[3.9,4.71,5.2])
def test_workfunction_energy_units(workfunction):
    np.testing.assert_equal(Gate(workfunction).workfunction,workfunction)
