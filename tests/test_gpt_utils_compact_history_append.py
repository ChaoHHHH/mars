"""紧凑记录的概率加权、静电一致性与不扰动推进。"""
TARGET='src.io.compact_history.CompactOccupationHistory.append'
from contextlib import closing
import copy
import h5py
import numpy as np
from src.io.compact_history import CompactOccupationHistory,COLUMNS
from src.physics.solver.advance import advance
from src.physics.solver.get_potential import get_potential

BANDS=[dict(name='SiO2',electron_affinity_eV=.82,bandgap_eV=9.),
       dict(name='HfO2',electron_affinity_eV=2.272,bandgap_eV=5.8)]


def test_compact_histogram_and_electrostatics_are_sufficient_without_microstates(tmp_path,make_mos,make_defect):
    m=make_mos();d=make_defect(Et_sigma=.1,Er12_sigma=.1)
    m.add_defects(d);d.data['f']=np.linspace(0.,1.,d.data['f'].size).reshape(d.data['f'].shape)
    m.set_Vg(.3);m.update_defect_total_rates(300.)
    snapshot=m.electrostatic_state
    rates=d.data['rate_in'].copy();before=d.data['f'].copy()
    path=tmp_path/'history.h5'
    with closing(CompactOccupationHistory(path,m,BANDS,bins=12)) as archive:
        archive.append(m,300.,0.)
        d.data['f'].fill(.7)
        archive.append(m,300.,1.25,phase=1)
        np.testing.assert_equal(m.electrostatic_state is snapshot,True)
        np.testing.assert_array_equal(d.data['rate_in'],rates)
    with h5py.File(path,'r') as f:
        g=f['defects/0'];edges=g['energy_edges_intrinsic_eV'][:]
        intrinsic=d.data['Et']
        for x in range(len(g['x_m'])):
            mass=np.histogram(intrinsic[x].ravel(),edges,weights=d.data['weights'][x].ravel())[0]
            filled=np.histogram(intrinsic[x].ravel(),edges,weights=(d.data['weights'][x]*before[x]).ravel())[0]
            expected=np.divide(filled,mass,out=np.zeros_like(filled),where=mass>0)
            np.testing.assert_allclose(g['bin_weights'][x],mass,atol=1e-15)
            np.testing.assert_allclose(g['f'][0,x],expected,atol=3e-8,rtol=1e-7)
        np.testing.assert_equal(g['f'].dtype,np.dtype('float32'))
        np.testing.assert_equal(f.attrs['complete_frames'],2)
        row=dict(zip(COLUMNS,f['frames'][-1]))
        x=np.linspace(0.,3e-9,19)
        actual=get_potential(x,row['psi_s_V'],row['Qs_C_m2'],
              [(g['x_m'][:],g['sigma_C_m2'][-1])],[(l.tox,l.eps) for l in m.layers])
        np.testing.assert_allclose(actual,m.get_potential(x,300.),atol=1e-13)
        mass=g['bin_weights'][:]
        np.testing.assert_allclose(np.sum(mass*g['f'][-1])/mass.sum(),.7,atol=3e-8)
        np.testing.assert_equal('Et' in g,False)


def test_streaming_archive_does_not_change_adaptive_trajectory(tmp_path,make_mos,make_defect):
    device=make_mos();device.add_defects(make_defect(Vc_vbm=1e-6,Vc_cbm=1e-6))
    device.set_Vg(.4)
    baseline=copy.deepcopy(device)
    expected=advance(baseline,1.,300.,dt_max_s=.2)
    with closing(CompactOccupationHistory(tmp_path/'steps.h5',device,BANDS,bins=10)) as archive:
        result=advance(device,1.,300.,dt_max_s=.2,
                       on_step=lambda m,t: archive.append(m,300.,t))
    np.testing.assert_array_equal(result['history'],expected['history'])
    np.testing.assert_array_equal(device.defects[0].data['f'],baseline.defects[0].data['f'])
    with h5py.File(tmp_path/'steps.h5','r') as f:
        np.testing.assert_allclose(f['frames'][:,0],result['history'][:,0])
        np.testing.assert_equal(len(f['frames']),result['steps']+1)
