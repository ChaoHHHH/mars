"""保存当前时刻自洽能带和缺陷占据；仅绘图分布做平滑。"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


def save_band_occupation(device, T, band_parameters, output_path, *, time_s=None,
                         defect_labels=None, title=None, show_rate_notes=True):
    """按当前f刷新静电/速率，保存PNG及同名NPZ，不推进或预平衡占据。

    band_parameters按层给出name/electron_affinity_eV/bandgap_eV，仅用于带边绘图。
    NPZ保留原始采样点、权重、f、局部转变能级和有效速率；颜色为电子占据。
    支持多个缺陷，数据前缀defect_0_、defect_1_等；能量均相对真空，eV。
    show_rate_notes=False 时隐藏右下角占据/速率说明，保留图例。
    """
    device.update_defect_total_rates(T)
    state=device.electrostatic_state
    path=Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig=Figure(figsize=(8,5.4),dpi=110)
    FigureCanvasAgg(fig)
    ax=fig.subplots()
    ef=device.channel.Evbm+device.channel.get_Ef_bulk(T)
    saved=dict(T_K=T,Vg_V=device.Vg,psi_s_V=state['psi_s'],EF_eV=ef,
               time_s=np.nan if time_s is None else time_s)
    start=0.;vb_x=[];vb=[];cb=[]
    for layer,band in zip(device.layers,band_parameters):
        end=start+layer.tox
        x=np.linspace(start,end,160)
        potential=device.get_potential(x,T,use_saved_state=True)
        conduction=-band['electron_affinity_eV']-potential
        valence=conduction-band['bandgap_eV']
        ax.fill_between(x*1e9,valence,conduction,color='lightgray',alpha=.45)
        ax.text((start+end)*.5e9,.97,band['name'],ha='center',va='top',
                transform=ax.get_xaxis_transform())
        vb_x.extend(x*1e9);vb.extend(valence);cb.extend(conduction)
        start=end
    ax.plot([-.1,0.],[device.VBM_local]*2,'b--',lw=1.5)
    ax.plot([-.1,0.],[device.CBM_local]*2,'r--',lw=1.5)
    ax.plot(vb_x,vb,'b--',lw=1.5,label='VB')
    ax.plot(vb_x,cb,'r--',lw=1.5,label='CB')
    ax.axhline(ef,color='#303846',lw=1.1,label=r'$E_F$ channel bulk')
    saved.update(band_x_m=np.asarray(vb_x)*1e-9,VB_eV=vb,CB_eV=cb)
    cmap=LinearSegmentedColormap.from_list('electron_occupation',['#f5f6fa','#122980'])
    notes=[]
    for index,defect in enumerate(device.defects):
        data=defect.data;energy=data['transition_level_local'];weights=data['weights']
        x_nm=data['x'][:,0,0,0]*1e9
        width=.43*np.min(np.diff(x_nm)) if len(x_nm)>1 else .03
        edges=np.linspace(float(energy.min())-.02,float(energy.max())+.02,181)
        polygons=[];colors=[]
        for j,x in enumerate(x_nm):
            density=np.histogram(energy[j].ravel(),edges,weights=weights[j].ravel())[0]
            filled=np.histogram(energy[j].ravel(),edges,weights=(weights[j]*data['f'][j]).ravel())[0]
            density=gaussian_filter1d(density,1.1,mode='constant')
            filled=gaussian_filter1d(filled,1.1,mode='constant')
            if density.max()<=0:
                continue
            for k in np.flatnonzero(density>0):
                w=width*density[k]/density.max();lo,hi=edges[k:k+2]
                polygons.append([(x-w,lo),(x+w,lo),(x+w,hi),(x-w,hi)])
                colors.append(np.clip(filled[k]/density[k],0.,1.))
        ax.add_collection(PolyCollection(polygons,array=np.asarray(colors),cmap=cmap,
                                         norm=Normalize(0,1),edgecolors='none',rasterized=True))
        spatial_mass=weights.sum(axis=(1,2,3))
        center=np.sum(weights*energy,axis=(1,2,3))/spatial_mass
        label=(defect_labels[index] if defect_labels is not None else f'Defect {index+1}')
        ax.plot(x_nm,center,color=['#c77c00','#148f77','#8e44ad'][index % 3],
                lw=1.5,marker='_',markersize=9,
                label=f'{label}: mean transition level')
        mean_f=float(np.sum(weights*data['f'])/weights.sum())
        outgoing=float(np.sum(weights*data['rate_out'])/weights.sum())
        incoming=float(np.sum(weights*data['rate_in'])/weights.sum())
        notes.append(f'{label} ({defect.type}): mean f={mean_f:.3e}\n'
                     f'Mean k12={outgoing:.2e}, k21={incoming:.2e} s$^{{-1}}$')
        prefix=f'defect_{index}_'
        for key in ('x','weights','f','Et_local','transition_level_local','rate_in','rate_out'):
            saved[prefix+key]=data[key].copy()
        saved[prefix+'center_x_m']=x_nm*1e-9
        saved[prefix+'transition_center_eV']=center
    from matplotlib.cm import ScalarMappable
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),ax=ax,
                 label='Electron occupation f',pad=.025)
    if title is None:
        timing='' if time_s is None else f't={time_s:g} s, '
        title=f'{timing}Vg={device.Vg:+.3f} V, T={T:g} K'
    ax.set(xlim=(-.12,start*1e9+.08),xlabel='Position (nm)',
           ylabel='Energy relative to vacuum (eV)',title=title)
    if notes and show_rate_notes:
        ax.text(.98,.05,'\n'.join(notes),transform=ax.transAxes,ha='right',va='bottom',
                fontsize=8,bbox=dict(facecolor='white',edgecolor='.85',alpha=.9))
    ax.legend(loc='lower left',fontsize=8);ax.tick_params(direction='in')
    fig.tight_layout();fig.savefig(path,dpi=200)
    np.savez_compressed(path.with_suffix('.npz'),**saved)
    fig.clear()
    return {'figure':path,'state':path.with_suffix('.npz')}
