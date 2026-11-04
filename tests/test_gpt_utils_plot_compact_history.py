"""仅从HDF5重画指定帧。"""
TARGET='src.io.compact_history.plot_compact_history'
from contextlib import closing
import numpy as np
from src.io.compact_history import CompactOccupationHistory,plot_compact_history


def test_replot_requires_only_archive_file(tmp_path,make_mos,make_defect):
    m=make_mos();m.add_defects(make_defect());m.defects[0].data['f'].fill(.25)
    bands=[dict(name='SiO2',electron_affinity_eV=.82,bandgap_eV=9.),
           dict(name='HfO2',electron_affinity_eV=2.272,bandgap_eV=5.8)]
    with closing(CompactOccupationHistory(tmp_path/'history.h5',m,bands,bins=20)) as archive:
        archive.append(m,300.,12.)
    del m
    image=plot_compact_history(tmp_path/'history.h5',tmp_path/'replot.png')
    np.testing.assert_equal(image.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
