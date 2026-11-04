# 运行与占据历史绘图工具

模拟入口现在位于仓库根目录 `run.py`：

```bash
python run.py
```

默认读取 `config/simulation.json`，并按其中的 `defect_files` 列表加载同目录的 `defect_1.json`、`defect_2.json` 等。各文件保存一种缺陷的构造参数；按列表顺序输出独立的占据和电荷。结果中的 `parameters.json` 内嵌全部缺陷参数。

也可从其他工作目录通过绝对路径运行。结果始终写到该仓库的 `results/<年月日_时分秒>/`，不跟随工作目录。入口使用 `from src...` 导入物理模型与工具；内部导入已更新为新的 models/physics/io 路径。

```text
results/<运行时间>/
├── threshold_voltage_drift.csv
└── plotting_data/
    ├── occupation_history.h5
    └── pre_equilibrium/
        ├── band_occupation.png
        └── band_occupation.npz
```

run.py负责计算、预平衡图及数据保存，不弹出汇总绘图窗口。results按仓库规则不加入Git。

## 紧凑历史记录

在 JSON 的 `output` 对象中设置：

```json
{
  "save_compact_history": true,
  "occupation_energy_bins": 180
}
```

固定空间坐标、能量分箱权重、平均本征转变能级、介电层参数和绘图带边参数仅写一次；每个接受步保存float32分箱平均电子占据，以及float64静电/偏压/温度/时间信息与各x处的面电荷。分箱占据为sum(weight*f)/sum(weight)。分箱在未减电势的转变能级轴固定：统一使用Et（输入即转变能级）；读出时减去重建的局部电势。

HDF5采用gzip+shuffle分块压缩，每32帧flush，退出时关闭。每个阶段含起点及各接受步；阶段边界可以有相同物理时间、不同phase/Vg的两条记录。波形t=0时占据来自预平衡，偏压已经切至首阶段。

文件结构：

- `/frames`：时间、阶段、Vg、T、当前f对应的自洽表面势、沟道电荷、费米能级、缺陷阈值贡献及沟道带边。列名见frame_columns_json属性。
- `/defects/0/x_m`、`energy_edges_intrinsic_eV`、`bin_weights`、`center_intrinsic_eV`：固定信息。
- `/defects/0/f`：(帧数,x点数,能量箱数)，float32。
- `/defects/0/sigma_C_m2`：每帧的真实面电荷。更多缺陷位于`/defects/1`等。
- complete_frames属性表示完整追加的帧数。

记录器独立计算当前占据对应的静电，不刷新推进器正在使用的冻结环境或速率。记录有分箱、静电求解和写盘开销；在JSON中关闭save_compact_history可停止记录。文件足够重画能带与占据，但没有每个Er12/Er21节点的微观f，不能精确续算。采样网格、权重和结构在记录期间应保持不变。

CSV记录真实时间、阶段时间、dt、Vg、冻结环境表面势、平均占据、理想/实际阈值、绝对缺陷贡献及相对预平衡的V/mV漂移。CSV的psi_s_environment_V为推进时使用的环境，HDF5的psi_s_V为当前f的自洽值，二者不必逐项相等。

## 离线绘图

从仓库根目录执行，并显式传入本次结果路径：

```bash
python -m src.io.plot_threshold_voltage_drift results/<运行时间>/threshold_voltage_drift.csv
python -m src.io.plot_occupation_history results/<运行时间>/plotting_data/occupation_history.h5 --output-dir results/<运行时间>/occupation_frames --stride 10
```

当前绘图脚本仍有固定日期的默认输入/输出，因此推荐使用上述显式路径；占据图的--output-dir也应传入。阈值图默认与CSV同目录，也可用--output指定。每阶段横轴按采样点序号等距排列，中央标注真实持续时间；纵轴为带符号线性µV。帧间隔不等于时间间隔；占据导出默认额外补画末帧，可用--no-final关闭。

仅画最后一帧：

```python
from src.io.compact_history import plot_compact_history

plot_compact_history("results/<运行时间>/plotting_data/occupation_history.h5",
                     "results/replot.png", frame=-1)
```

选择最接近10秒的已保存帧：

```python
import h5py
import numpy as np

history = "results/<运行时间>/plotting_data/occupation_history.h5"
with h5py.File(history, "r") as f:
    times = f["frames"][:int(f.attrs["complete_frames"]), 0]
    index = int(np.argmin(abs(times - 10.0)))
    print("实际保存时间：", times[index])
plot_compact_history(history, "results/t10_nearest.png", frame=index)
```

这是最近帧，不是对占据做时间插值。离线图保留分布中心线和电子占据颜色，但不显示未保存的微观速率。

## 单次原始快照和精确选时工具

`save_band_occupation(device,T,band_parameters,path,time_s=...)` 保存调用时的PNG和同名NPZ，包含原始采样状态；刷新静电与速率，不推进占据。band_parameters按层提供name/electron_affinity_eV/bandgap_eV，仅用于绘图。

`advance_with_snapshots(...,snapshot_times_s=[...],on_snapshot=...)` 可让一个固定偏压阶段准确停到指定时刻；时间相对此次调用的起点。回调只能读取器件，不能修改参数或f。切分会增加环境刷新，可能在原求解误差内略微影响轨迹。

当前根目录run.py直接使用advance和每步HDF5记录，不配置snapshot_times_s；如必须恰好观测任意时刻，需要显式使用上述工具安排停步，而不是修改最近帧的时间标签。
