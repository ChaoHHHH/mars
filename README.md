# MARS 一维 MOS 缺陷模拟

## 目录

- `src/models/`：沟道、栅极、绝缘层、MOS1D 器件与缺陷分布。
- `src/physics/`：Marcus 势垒与裸速率；`solver/` 包含静电、有效速率、预平衡和时间推进。
- `src/io/`：JSON 配置、模拟流程、CSV/HDF5 记录与离线绘图。
- `config/simulation.json`：器件、波形、求解器和输出设置；`config/defect_1.json` 等保存各类缺陷参数。
- `src_old/`：保留原代码。

## 运行

```bash
conda activate mars
python run.py --config config/simulation.json
python run.py --config config/simulation.json --output results/custom
python -m pytest tests -q
```

省略 `--config` 时使用仓库内的 `config/simulation.json`。指定的配置路径相对当前工作目录；输出目录相对仓库根目录，绝对路径直接使用。每次运行在输出根目录下新建时间戳目录，并保存完整 `parameters.json`。

导入 `run.py` 不会构造器件或启动计算。程序调用方式：

```python
from src.io.config import load_config
from src.io.simulation import run_simulation

config = load_config("config/simulation.json")
result_dir = run_simulation(config, "results")
```

## JSON 参数

- `channel_parameters`、`gate_parameter`：器件参数，直接传入对应模型构造函数。
- `defect_files`：按顺序列出缺陷 JSON，路径相对 `simulation.json`，每个文件直接存放一组 `Defects` 构造参数。
- `insulating_layer_parameters`：介质层数组，按沟道到栅极顺序。
- `insulating_band_parameters`：绘图带边参数，顺序与介质层一致。
- `simulation`：`T`、`equilibration_Vg`、`initial_f`、`period`、`n_cycles`、`Vmin`、`Vmax`。每周期先 Vmin 后 Vmax，各持续半周期；预平衡不计入物理时间。
- `solver`：`dt_initial_s`、`dt_max_s`、`electrostatics_recalc_threshold_V`。
- `output`：`directory`、`save_compact_history`、`occupation_energy_bins`、`save_pre_equilibrium_plot`、`defect_labels`。

例如在 `simulation.json` 中设置：

```json
"defect_files": ["defect_1.json", "defect_2.json"]
```

可复制 `defect_1.json` 为 `defect_2.json`，再修改该缺陷的浓度、能级、重组能、空间分布和耦合参数。仅加载列表中指定的文件，顺序决定输出中的缺陷编号。
`simulation.initial_f` 可以是统一初始占据，或按文件顺序给出的数组（如 `[0.0, 1.0]`）；随后仍执行预平衡。`output.defect_labels` 若提供，也必须按顺序为每种缺陷提供一个标签。
保存到结果目录的 `parameters.json` 内嵌全部已加载的 `defects_parameters` 数组，无需依赖原缺陷文件即可再次加载。旧版内嵌单缺陷对象仍可读取。

长度 m，温度 K，电势 V，能量 eV，体浓度 m^-3，面浓度 m^-2，时间 s。JSON 中 `x_range` 使用数组，布尔值为 `true/false`，空值为 `null`。默认加载一种缺陷，可在 `defect_files` 中追加更多文件。DFT参考载流子体密度统一为 `n_DFT_m3`；有效速率仅使用沟道载流子浓度与该参考密度的比值，不乘缺陷浓度归一化因子。`Nt` 仍用于缺陷电荷、占据权重及静电反馈。

结果包括阈值漂移 CSV、可选预平衡 PNG/NPZ 和可选紧凑占据 HDF5。原有绘图方式改用新路径：

```bash
python -m src.io.plot_threshold_voltage_drift results/<时间戳>/threshold_voltage_drift.csv
python -m src.io.plot_occupation_history results/<时间戳>/plotting_data/occupation_history.h5 --output-dir results/<时间戳>/occupation_frames --stride 10
```

NMOS/PMOS 基准继续使用独立的 benchmark JSON 格式：`python -m benchmarks.run`。

`Et` 统一表示相对真空的转变能级，局部转变能级为 `Et - potential`，不再加 `Er12`。默认 JSON 中的 `Et` 数值保持不变，现直接按转变能级解释；旧配置中使用 `Et + Er12` 的结果将因此变化。

速率包含一维经典振动熵修正：`log(k21/k12) = (E_reservoir - Et_local)/(kB*T) + 0.5*log(Er12/Er21)`。裸速率使用相应能带边或栅极费米能级，有效速率使用各热库费米能级；两态电子简并度相同，`Et` 不含此振动熵修正。曲率相同时恢复原有关系。


栅极耦合参数也放在每个 `defect_{num}.json`：`Vc_gate`（eV）和 `decay_gate`（m^-1），默认均为0。
VBM/CBM耦合按距沟道的距离x衰减，栅极耦合为 `Vc_gate * exp(-decay_gate * (总介质厚度-x))`。
栅极采用现有 `metal` 势垒近似，费米能级为 `EF_channel - Vg`；栅极速率不乘沟道浓度比，也不使用缺陷浓度归一化。
三热库的流入/流出速率分别求和后推进占据；栅极与沟道费米能级不同时，预平衡求的是自洽稳态。
默认 `Vc_gate=0` 关闭栅极交换。

`temperature_vbm`、`temperature_cbm`、`temperature_gate` 默认均为0，当前只接收、保存并随完整输入写入 `parameters.json`，尚未指定公式和单位，不应用于耦合或速率计算。
原有Marcus热激活及沟道材料本身的温度依赖仍按温度T计算。
