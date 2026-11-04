"""按能量分箱记录每步占据，HDF5流式压缩；不保存微观采样点历史。"""
from pathlib import Path
import json
import numpy as np
import h5py
from src.physics.solver.get_potential import get_potential

COLUMNS = ('time_s','phase','Vg_V','T_K','psi_s_V','Qs_C_m2','EF_eV','delta_Vth_V',
           'VBM_local_eV','CBM_local_eV')


class CompactOccupationHistory:
    """固定结构只写一次，每步写float32分箱占据和float64静电参数。

    append仅读取f，独立求当前静电；不更新器件冻结快照或速率。
    仅适用于采样网格、权重、器件材料和结构在记录期间不变的模拟。
    用contextlib.closing确保结束或异常时close；每32帧flush一次。
    """
    def __init__(self, path, device, band_parameters, *, bins=180, defect_labels=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=h5py.File(self.path,'w')
        self.file.attrs['format']='mars_compact_occupation_v1'
        self.file.attrs['band_parameters_json']=json.dumps(band_parameters)
        self.file.attrs['layers_json']=json.dumps([(d.tox,d.eps) for d in device.layers])
        self.file.attrs['frame_columns_json']=json.dumps(COLUMNS)
        self.frames=self.file.create_dataset('frames',shape=(0,len(COLUMNS)),
            maxshape=(None,len(COLUMNS)),chunks=(128,len(COLUMNS)),dtype='f8')
        self.axes=[]
        group=self.file.create_group('defects')
        for i,d in enumerate(device.defects):
            data=d.data
            energy=data['Et']
            edges=np.linspace(float(energy.min())-.02,float(energy.max())+.02,bins+1)
            x=data['x'][:,0,0,0];nx=len(x)
            indices=np.searchsorted(edges,energy.reshape(nx,-1),side='right')-1
            indices=np.clip(indices,0,bins-1)+np.arange(nx)[:,None]*bins
            weights=data['weights'].reshape(nx,-1)
            mass=np.bincount(indices.ravel(),weights=weights.ravel(),minlength=nx*bins).reshape(nx,bins)
            g=group.create_group(str(i))
            g.attrs['label']=defect_labels[i] if defect_labels is not None else f'Defect {i+1}'
            g.attrs['type']=d.type;g.attrs['Nt_m3']=d.Nt
            g.create_dataset('x_m',data=x)
            g.create_dataset('energy_edges_intrinsic_eV',data=edges)
            g.create_dataset('bin_weights',data=mass)
            center=np.sum(data['weights']*energy,axis=(1,2,3))/data['weights'].sum(axis=(1,2,3))
            g.create_dataset('center_intrinsic_eV',data=center)
            occupancy=g.create_dataset('f',shape=(0,nx,bins),maxshape=(None,nx,bins),
                chunks=(1,nx,bins),dtype='f4',compression='gzip',compression_opts=4,shuffle=True)
            sigma=g.create_dataset('sigma_C_m2',shape=(0,nx),maxshape=(None,nx),
                chunks=(128,nx),dtype='f8',compression='gzip',shuffle=True)
            self.axes.append((indices,weights,mass,occupancy,sigma))
        self.file.attrs['complete_frames']=0

    def append(self, device, T, time_s, *, phase=0):
        """保存此时占据和对应静电，不使用可能落后的冻结静电快照。"""
        psi=device.solve_psi_s(T)
        Qs=device.channel.cal_Qs(psi,T)
        ef=device.channel.Evbm+device.channel.get_Ef_bulk(T)
        frame=len(self.frames)
        for d,(indices,weights,mass,occupancy,sigma) in zip(device.defects,self.axes):
            filled=np.bincount(indices.ravel(),weights=(weights*d.data['f'].reshape(weights.shape)).ravel(),
                               minlength=mass.size).reshape(mass.shape)
            average=np.divide(filled,mass,out=np.zeros_like(filled),where=mass>0)
            occupancy.resize(frame+1,axis=0);occupancy[frame]=average
            sigma.resize(frame+1,axis=0);sigma[frame]=d.get_charge_sheets()
        self.frames.resize(frame+1,axis=0)
        self.frames[frame]=[time_s,phase,device.Vg,T,psi,Qs,ef,device.get_delta_Vth(),
                            device.channel.Evbm-psi,device.channel.Evbm+device.channel.get_Eg(T)-psi]
        self.file.attrs.modify('complete_frames',frame+1)
        if (frame+1)%32==0:
            self.file.flush()

    def close(self):
        self.file.close()


def plot_compact_history(path, output_path, *, frame=-1):
    """按帧序号重画能带/分箱占据，默认最后一帧；不需要原器件或微观f。

    同一阶段边界可有切换前后两帧（相同time、不同phase/Vg），用帧序号区分。
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.ndimage import gaussian_filter1d
    with h5py.File(path,'r') as f:
        count=int(f.attrs['complete_frames'])
        index=frame if frame>=0 else count+frame
        row=dict(zip(COLUMNS,f['frames'][index]))
        layers=json.loads(f.attrs['layers_json'])
        bands=json.loads(f.attrs['band_parameters_json'])
        defects=[]
        for g in f['defects'].values():
            defects.append(dict(x=g['x_m'][:],sigma=g['sigma_C_m2'][index],
                edges=g['energy_edges_intrinsic_eV'][:],weights=g['bin_weights'][:],
                f=g['f'][index],center=g['center_intrinsic_eV'][:],
                label=g.attrs['label'],type=g.attrs['type']))
    sheets=[(d['x'],d['sigma']) for d in defects]
    def potential(x):
        return get_potential(x,row['psi_s_V'],row['Qs_C_m2'],sheets,layers)
    fig=Figure(figsize=(8,5.4),dpi=110);FigureCanvasAgg(fig);ax=fig.subplots()
    xband=[];vb=[];cb=[];start=0.
    for (thickness,eps),band in zip(layers,bands):
        x=np.linspace(start,start+thickness,160)
        conduction=-band['electron_affinity_eV']-potential(x)
        valence=conduction-band['bandgap_eV']
        ax.fill_between(x*1e9,valence,conduction,color='lightgray',alpha=.45)
        ax.text((start+thickness/2)*1e9,.97,band['name'],ha='center',va='top',transform=ax.get_xaxis_transform())
        xband.extend(x*1e9);vb.extend(valence);cb.extend(conduction);start+=thickness
    ax.plot([-.1,0.],[row['VBM_local_eV']]*2,'b--')
    ax.plot([-.1,0.],[row['CBM_local_eV']]*2,'r--')
    ax.plot(xband,vb,'b--',label='VB');ax.plot(xband,cb,'r--',label='CB')
    ax.axhline(row['EF_eV'],color='#303846',lw=1.1,label=r'$E_F$ channel bulk')
    cmap=LinearSegmentedColormap.from_list('occupation',['#f5f6fa','#122980'])
    notes=[]
    for i,d in enumerate(defects):
        shifts=potential(d['x']);x_nm=d['x']*1e9
        width=.43*np.min(np.diff(x_nm)) if len(x_nm)>1 else .03
        polygons=[];colors=[]
        for j,x in enumerate(x_nm):
            density=gaussian_filter1d(d['weights'][j],1.1,mode='constant')
            filled=gaussian_filter1d(d['weights'][j]*d['f'][j],1.1,mode='constant')
            if density.max()<=0:continue
            for k in np.flatnonzero(density>0):
                w=width*density[k]/density.max();lo,hi=d['edges'][k:k+2]-shifts[j]
                polygons.append([(x-w,lo),(x+w,lo),(x+w,hi),(x-w,hi)])
                colors.append(np.clip(filled[k]/density[k],0.,1.))
        ax.add_collection(PolyCollection(polygons,array=np.array(colors),cmap=cmap,
            norm=Normalize(0,1),edgecolors='none',rasterized=True))
        ax.plot(x_nm,d['center']-shifts,color=['#c77c00','#148f77','#8e44ad'][i%3],
            marker='_',label=f"{d['label']}: mean transition level")
        mean=float(np.sum(d['weights']*d['f'])/d['weights'].sum())
        notes.append(f"{d['label']} ({d['type']}): mean f={mean:.3e}")
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),ax=ax,label='Electron occupation f',pad=.025)
    ax.set(xlim=(-.05,start*1e9+.08),xlabel='Position (nm)',ylabel='Energy relative to vacuum (eV)',
        title=f"t={row['time_s']:g} s, Vg={row['Vg_V']:+.3f} V, T={row['T_K']:g} K")
    ax.text(.98,.05,'\n'.join(notes),transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
    ax.legend(loc='lower left',fontsize=8);fig.tight_layout()
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=200);fig.clear()
    return output
