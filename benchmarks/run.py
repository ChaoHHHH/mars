"""运行：python -m benchmarks.run [--case all|nmos_sio2|pmos_sio2_hfo2]。"""
from pathlib import Path
import argparse
import json
import numpy as np
from src.models.device_channel import Channel
from src.models.device_gate import Gate
from src.models.device_insulatinglayer import InsulatingLayer
from src.models.device_mis1d import MOS1D
from src.models.defect import Defects
from src.physics.solver.advance import advance
from src.physics.solver.pre_equilibrate import pre_equilibrate

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = 'time_s,phase,phase_time_s,dt_s,Vg_V,psi_s_environment_V,mean_f,delta_Vth_V'


def simulate(config, solver):
    device = MOS1D(channel=Channel(**config['channel']), gate=Gate(**config['gate']),
                   layers=[InsulatingLayer(**p) for p in config['layers']])
    defect = Defects(**config['defect'])
    device.add_defects(defect)
    equilibrium = pre_equilibrate(device, config['pre_equilibrium_Vg'], config['T_K'])
    reference = device.get_delta_Vth()
    initial = {key: defect.data[key].copy() for key in ('x','weights','f','transition_level_local')}
    records, elapsed = [], 0.
    for phase, settings in enumerate(config['phases']):
        device.set_Vg(settings['Vg'])
        result = advance(device, settings['duration_s'], config['T_K'], **solver)
        h = result['history']
        records.append(np.column_stack((h[:,0]+elapsed, np.full(len(h),phase),
                       h[:,0],h[:,1],np.full(len(h),settings['Vg']),h[:,2:4],h[:,4]-reference)))
        elapsed += settings['duration_s']
    return np.concatenate(records), equilibrium, initial


def plot(config, data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    phases=config['phases']
    fig,axes=plt.subplots(3,len(phases),figsize=(12,6),sharey='row',squeeze=False,
                          gridspec_kw={'wspace':0.,'hspace':.16})
    for i,p in enumerate(phases):
        h=data[data[:,1]==i];x=np.arange(len(h))
        old=config['pre_equilibrium_Vg'] if i==0 else phases[i-1]['Vg']
        axes[0,i].plot(np.r_[0,x],np.r_[old,h[:,4]],lw=1.6)
        axes[1,i].plot(x,h[:,7]*1e3,lw=1.6)
        axes[2,i].plot(x,h[:,6],lw=1.6)
        axes[0,i].set_title(p['kind'],fontsize=9)
        for ax in axes[:,i]:
            ax.set_xlim(0,max(1,x[-1]));ax.set_xticks([x[-1]/2],[f"{p['duration_s']:g}s"])
            ax.grid(axis='y',alpha=.25);ax.tick_params(direction='in')
    axes[0,0].set_ylabel('Vg [V]');axes[1,0].set_ylabel('Vth - Vth(pre) [mV]')
    axes[2,0].set_ylabel('Mean electron f');axes[2,0].set_ylim(-.03,1.03)
    fig.suptitle(config['name']+' | 300 K | samples equally spaced within each phase')
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def run_case(name, output):
    config=json.loads((ROOT/'benchmarks'/f'{name}.json').read_text())
    folder=output/name;folder.mkdir(parents=True,exist_ok=True)
    base,eq,initial=simulate(config,config['solver'])
    fine,_,_=simulate(config,config['refined_solver'])
    differences=[]
    for phase in range(len(config['phases'])):
        a=base[base[:,1]==phase];b=fine[fine[:,1]==phase]
        # 比较同一真实物理时刻；不用绘图点序号对齐。
        differences.extend(abs(np.interp(b[:,2],a[:,2],a[:,7])-b[:,7])*1e3)
    stress=base[base[:,1]==0][-1,7]*1e3
    recovery=base[base[:,1]==1][-1,7]*1e3
    expected_sign=1 if name.startswith('nmos') else -1
    limits=config['acceptance']
    np.testing.assert_equal(np.all(np.isfinite(base)),True)
    np.testing.assert_equal(np.all((base[:,6]>=0)&(base[:,6]<=1)),True)
    np.testing.assert_array_less(limits['min_first_stress_magnitude_mV'],stress*expected_sign)
    np.testing.assert_array_less(abs(recovery/stress),limits['max_first_recovery_fraction'])
    np.testing.assert_array_less(max(differences),limits['max_curve_difference_mV'])
    summary=dict(name=name,Vth0_V=eq['Vth0'],pre_Vth_V=eq['Vth'],
                 first_stress_delta_mV=float(stress),first_recovery_delta_mV=float(recovery),
                 recovery_fraction=float(abs(recovery/stress)),max_curve_difference_mV=float(max(differences)),
                 base_rows=len(base),refined_rows=len(fine),acceptance='passed')
    for filename,data in [('history.csv',base),('refined_history.csv',fine)]:
        np.savetxt(folder/filename,data,delimiter=',',header=COLUMNS,comments='')
    np.savez_compressed(folder/'pre_equilibrium.npz',**initial)
    (folder/'parameters.json').write_text(json.dumps(config,indent=2)+'\n')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(config,base,folder/'stress_recovery.png')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['all','nmos_sio2','pmos_sio2_hfo2'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'figs'/'benchmarks')
    args=parser.parse_args()
    for name in (['nmos_sio2','pmos_sio2_hfo2'] if args.case=='all' else [args.case]):
        run_case(name,args.output)


if __name__=='__main__':
    main()
