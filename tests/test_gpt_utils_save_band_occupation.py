"""能带占据图与原始状态快照的一致性，不推进f。"""
TARGET = 'src.io.band_occupation.save_band_occupation'
import numpy as np
from src.io.band_occupation import save_band_occupation


def test_snapshot_preserves_occupations_and_saves_current_electrostatics(tmp_path,make_mos,make_defect):
    device=make_mos();defect=make_defect(Et_sigma=.1,dEt=.1)
    device.add_defects(defect)
    other=make_defect(Et=-4.,Nt=2e24,type='acceptor')
    device.add_defects(other)
    defect.data['f'].fill(.3);other.data['f'].fill(.7)
    before=[d.data['f'].copy() for d in device.defects]
    device.set_Vg(.4)
    bands=[dict(name='SiO2',electron_affinity_eV=.82,bandgap_eV=9.),
           dict(name='HfO2',electron_affinity_eV=2.272,bandgap_eV=5.8)]
    result=save_band_occupation(device,300.,bands,tmp_path/'state.png',time_s=12.5)
    np.testing.assert_equal(result['figure'].read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
    with np.load(result['state']) as saved:
        np.testing.assert_equal(saved['time_s'],12.5)
        np.testing.assert_equal(saved['Vg_V'],.4)
        np.testing.assert_allclose(saved['psi_s_V'],device.solve_psi_s(300.))
        for i,d in enumerate(device.defects):
            np.testing.assert_array_equal(d.data['f'],before[i])
            np.testing.assert_array_equal(saved[f'defect_{i}_f'],before[i])
            np.testing.assert_allclose(saved[f'defect_{i}_transition_level_local'],
                d.data['Et']-device.get_potential(d.data['x'][:,0,0,0],300.)[:,None,None,None])
            np.testing.assert_equal(np.all(np.isfinite(saved[f'defect_{i}_rate_in'])),True)
