"""只测试 InsulatingLayer.__init__：介电常数和厚度的SI单位。"""

TARGET = "src.models.device_insulatinglayer.InsulatingLayer.__init__"
import numpy as np
import pytest
from src.models.device_insulatinglayer import InsulatingLayer


@pytest.mark.parametrize("eps_r,tox",[(3.9,1e-9),(20.,2.1e-9)])
def test_oxide_geometry_and_absolute_permittivity(eps_r,tox):
    layer=InsulatingLayer(eps_r,tox)
    np.testing.assert_equal(layer.tox,tox)
    np.testing.assert_allclose(layer.eps,eps_r*8.8541878188e-12,rtol=1e-15)
