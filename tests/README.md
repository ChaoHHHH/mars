# 数值与物理测试（第一轮）

每个 `test_gpt_xxx.py` 只对应一个生产方法或独立函数，目标由文件中的 `TARGET` 标明。共有 **43 个逐方法测试文件**，另有 **2 个端到端 benchmark 测试文件**，覆盖当前5个类的全部直接方法及各独立数值函数。`conftest.py` 仅准备输入夹具，不包含测试断言。`run.py` 是示例脚本，不是本轮独立数值目标；嵌套辅助函数由所属方法的测试覆盖。

断言统一使用 `np.testing`，包括 `np.testing.assert_raises`；没有裸 `assert`，没有 `xfail`、跳过或为迁就当前源码而改写物理预期。交点、FD 势垒及正重组能采样修正涉及 src，不增加输入检查。

## 运行

在仓库根目录执行：

```bash
python -m pytest tests/test_gpt_defects_sampling.py -v
python -m pytest tests/test_gpt_solver_advance.py -v
python -m pytest tests -q
```

`numpy`、`scipy`、`pytest` 已包含在项目依赖中。可通过 `--junitxml=figs/gpt_test_results.xml` 导出结果。

## 参考依据与检查次序

1. Defects：δ能量分布、独立Gaussian区间的数值积分、覆盖概率、均值/方差与细网格收敛、正重组能支持域；体浓度乘厚度得到面数量，donor/acceptor电荷态及电子增加时电荷减少。
2. Channel：线性带隙模型、有效质量的解析构造、SI三维态密度、简并度与T^(3/2)标度、n=p的本征能级、电中性、质量作用定律、真空功函数、Poisson–Boltzmann电荷和平带/耗尽极限。
3. 势垒/速率：交点应满足原二次方程，对称Marcus解析式、反应自由能差、耦合平方标度、热激活、细致平衡、零通道及下溢。
4. 静电：多层电容串联、电荷面位置投影、场跳变、沟道和栅极边界、n/p型阈值条件、电子/缺陷电荷的符号；显式静电快照及每次统一更新只求一次表面势。
5. 时间：恒定速率主方程解析解、零速率、概率区间、时间组合律、反馈Riccati方程的独立闭式解与步长收敛、多缺陷统一时钟和数量加权、预平衡回溯及极慢/断开点的区别。

独立参考包括解析结果、SI常数、Gaussian概率密度数值积分及可解析的人工反馈问题。人工反馈问题只用于检验求解流程，不是材料模型。DFT归一化测试检查的是项目声明公式及逐缺陷参数归属，**并不证明该归一化的DFT物理推导正确**。固定参数的单次采样方差不应与完整Gaussian方差强行精确相等，因此相应测试使用截断分布矩和细化误差。

## 当前运行结果

2026-10-30：**204 个用例，204 通过、0 失败、0 跳过、0 收集错误**。按用户约定，不测试域外输入是否抛异常；有效输入域内的数值与物理判据全部保留。

### 已修正：正重组能分布的概率质量

Er12、Er21 分别采用正半轴条件 Gaussian：原始 N(mu,sigma²) 在 Er>0 条件下重新归一化。mu/sigma 仍是截断前参数，不能把它们直接视为截断后的矩。

节点覆盖条件分布的 non_coverage 与 1−non_coverage 分位区间，分箱边界为相邻节点中点，首尾边界为0和正无穷；端点代表极小尾部概率。每个重组能轴权重和为1，保持相互独立，负半轴质量不进入分箱。零方差且均值非正时没有有效的正值条件分布，显式抛 ValueError。

sampling 的21个用例全部通过，包括独立概率积分、mu=0.55/sigma=0.70的原失败参数、宽分布双轴权重和、donor空态总电荷、负均值Gaussian截断后的解析矩与网格收敛。Et轴仍沿用原有限范围Gaussian权重，未改为归一化；因此Et有展宽时，总权重仍含该轴原有的微小尾部损失。

### 已修正：小正判别式

删除了D<1e-5的正值截断，D>0保留真实值，D=0才按重根计算。新增了四档正判别式、精确重根及共同能量缩放的回归测试；近切点势垒仍满足正反势垒差等于反应自由能差。原例q约−0.999，方程残差为0。D<0现在返回NaN；稳定二次求根避免曲率接近时的消去误差。

### 已修正：弱耦合 FD 近似

依据用户提供的 Comphy 第56–57页和表4.4，在交点位于两个极小值之外或无实交点时：VBM且ΔG>0取正向势垒ΔG，CBM且ΔG<0取0；metal取max(ΔG,0)。反向势垒由正反势垒差等于ΔG构造。未被上述分支覆盖的无实交点通道取无限势垒、零速率。

`calculate_marcus_barriers` 与 `calculate_marcus_rates` 的 `reservoir="band_edge"` 表示纯经典带边模型，供独立对照；器件显式传入 `vbm` 或 `cbm` 启用 FD 近似。该近似不是完整 Fermi–Dirac 能量积分，保留当前项目的速率前因子及归一化。有效反向速率继续由沟道费米能级的细致平衡构造。

新增19个用例覆盖：无交点和有实交点的弱耦合、两类边界连续性、未允许通道为零、器件热库分支接入、近等曲率求根和细致平衡，全部通过。原无实交点失败也已通过。

### 输入域约定

按用户要求，移除原来9项要求输入检查的用例：calculate_marcus_rates 的非正重组能/零温度3项，以及 advance_occupation 的负值、NaN、无限时间或速率6项。源码不增加输入检查，调用方负责提供有效输入。这些测试被移除，不使用 skip 或 xfail；有效输入下的解析解、收敛性和细致平衡测试保持不变。

## 文件与目标对应

一文件一目标的清单也保存于 [coverage_targets.json](coverage_targets.json)。公共夹具调用构造函数或依赖方法用于准备输入，不表示同一个文件在同时测试这些依赖；每个文件的断言只服务于其指定目标。

|文件|唯一目标|
|---|---|
|[test_gpt_defects_init.py](test_gpt_defects_init.py)|`src.models.defect.Defects.__init__`|
|[test_gpt_defects_sampling.py](test_gpt_defects_sampling.py)|`src.models.defect.Defects.sampling`|
|[test_gpt_defects_get_charge_sheets.py](test_gpt_defects_get_charge_sheets.py)|`src.models.defect.Defects.get_charge_sheets`|
|[test_gpt_channel_init.py](test_gpt_channel_init.py)|`src.models.device_channel.Channel.__init__`|
|[test_gpt_channel_get_Eg.py](test_gpt_channel_get_Eg.py)|`src.models.device_channel.Channel.get_Eg`|
|[test_gpt_channel_get_mneff_mpeff.py](test_gpt_channel_get_mneff_mpeff.py)|`src.models.device_channel.Channel.get_mneff_mpeff`|
|[test_gpt_channel_get_Nv_Nc.py](test_gpt_channel_get_Nv_Nc.py)|`src.models.device_channel.Channel.get_Nv_Nc`|
|[test_gpt_channel_get_Ei.py](test_gpt_channel_get_Ei.py)|`src.models.device_channel.Channel.get_Ei`|
|[test_gpt_channel_get_Ef_bulk.py](test_gpt_channel_get_Ef_bulk.py)|`src.models.device_channel.Channel.get_Ef_bulk`|
|[test_gpt_channel_get_workfunction.py](test_gpt_channel_get_workfunction.py)|`src.models.device_channel.Channel.get_workfunction`|
|[test_gpt_channel_get_carrier_density.py](test_gpt_channel_get_carrier_density.py)|`src.models.device_channel.Channel.get_carrier_density`|
|[test_gpt_channel_cal_Qs.py](test_gpt_channel_cal_Qs.py)|`src.models.device_channel.Channel.cal_Qs`|
|[test_gpt_gate_init.py](test_gpt_gate_init.py)|`src.models.device_gate.Gate.__init__`|
|[test_gpt_insulatinglayer_init.py](test_gpt_insulatinglayer_init.py)|`src.models.device_insulatinglayer.InsulatingLayer.__init__`|
|[test_gpt_mos1d_init.py](test_gpt_mos1d_init.py)|`src.models.device_mis1d.MOS1D.__init__`|
|[test_gpt_mos1d_saved_electrostatics.py](test_gpt_mos1d_saved_electrostatics.py)|`src.models.device_mis1d.MOS1D._saved_electrostatics`|
|[test_gpt_mos1d_set_Vg.py](test_gpt_mos1d_set_Vg.py)|`src.models.device_mis1d.MOS1D.set_Vg`|
|[test_gpt_mos1d_add_defects.py](test_gpt_mos1d_add_defects.py)|`src.models.device_mis1d.MOS1D.add_defects`|
|[test_gpt_mos1d_get_Cox_per_area.py](test_gpt_mos1d_get_Cox_per_area.py)|`src.models.device_mis1d.MOS1D.get_Cox_per_area`|
|[test_gpt_mos1d_get_psi_ms.py](test_gpt_mos1d_get_psi_ms.py)|`src.models.device_mis1d.MOS1D.get_psi_ms`|
|[test_gpt_mos1d_solve_psi_s.py](test_gpt_mos1d_solve_psi_s.py)|`src.models.device_mis1d.MOS1D.solve_psi_s`|
|[test_gpt_mos1d_get_delta_V_trap.py](test_gpt_mos1d_get_delta_V_trap.py)|`src.models.device_mis1d.MOS1D.get_delta_V_trap`|
|[test_gpt_mos1d_get_potential.py](test_gpt_mos1d_get_potential.py)|`src.models.device_mis1d.MOS1D.get_potential`|
|[test_gpt_mos1d_update_electrostatics.py](test_gpt_mos1d_update_electrostatics.py)|`src.models.device_mis1d.MOS1D.update_electrostatics`|
|[test_gpt_mos1d_get_Vth0.py](test_gpt_mos1d_get_Vth0.py)|`src.models.device_mis1d.MOS1D.get_Vth0`|
|[test_gpt_mos1d_get_delta_Vth.py](test_gpt_mos1d_get_delta_Vth.py)|`src.models.device_mis1d.MOS1D.get_delta_Vth`|
|[test_gpt_mos1d_update_defect_couplings.py](test_gpt_mos1d_update_defect_couplings.py)|`src.models.device_mis1d.MOS1D.update_defect_couplings`|
|[test_gpt_mos1d_update_defect_rates.py](test_gpt_mos1d_update_defect_rates.py)|`src.models.device_mis1d.MOS1D.update_defect_rates`|
|[test_gpt_mos1d_update_defect_total_rates.py](test_gpt_mos1d_update_defect_total_rates.py)|`src.models.device_mis1d.MOS1D.update_defect_total_rates`|
|[test_gpt_marcus_solve_crossing.py](test_gpt_marcus_solve_crossing.py)|`src.physics.marcus_barriers.solve_crossing`|
|[test_gpt_marcus_calculate_barriers.py](test_gpt_marcus_calculate_barriers.py)|`src.physics.marcus_barriers.calculate_marcus_barriers`|
|[test_gpt_marcus_calculate_rates.py](test_gpt_marcus_calculate_rates.py)|`src.physics.marcus_rates.calculate_marcus_rates`|
|[test_gpt_solver_advance.py](test_gpt_solver_advance.py)|`src.physics.solver.advance.advance`|
|[test_gpt_solver_calculate_total_rates.py](test_gpt_solver_calculate_total_rates.py)|`src.physics.solver.calculate_total_rates.calculate_total_rates`|
|[test_gpt_solver_get_delta_V_trap.py](test_gpt_solver_get_delta_V_trap.py)|`src.physics.solver.get_delta_V_trap.get_delta_V_trap`|
|[test_gpt_solver_get_potential.py](test_gpt_solver_get_potential.py)|`src.physics.solver.get_potential.get_potential`|
|[test_gpt_solver_pre_equilibrate.py](test_gpt_solver_pre_equilibrate.py)|`src.physics.solver.pre_equilibrate.pre_equilibrate`|
|[test_gpt_solver_solve_psi_s.py](test_gpt_solver_solve_psi_s.py)|`src.physics.solver.solve_psi_s.solve_psi_s`|
|[test_gpt_solver_advance_occupation.py](test_gpt_solver_advance_occupation.py)|`src.physics.solver.step_defects.advance_occupation`|

本轮用例不是所有物理机制或所有参数空间的证明，也不包括DFT数据的材料真实性验证；测试结论限定于所覆盖的有效输入域。未自动提交Git commit。


## 端到端 benchmark 回归

- `test_gpt_benchmark_nmos_sio2.py`：p型Si沟道、单层SiO2的NMOS。
- `test_gpt_benchmark_pmos_sio2_hfo2.py`：n型Si沟道、SiO2/HfO2的PMOS。

这两个文件属于器件全流程集成测试，不受上文逐方法目标映射限制。直接读取benchmarks中的JSON并调用simulate；不调用绘图或文件输出。每项运行基准与细化设置，覆盖预平衡残差、占据区间、真实时间推进与阶段连续性、三周期应力方向/恢复幅度、两档精度的曲线一致性。每阶段1/10/100/1000秒的带符号漂移还与固定参考值比较，绝对容差0.1mV；参考值来自当前已验证的细化模拟，是回归基线，不是独立实验真值。两档曲线差异容差由JSON给定，当前为0.5mV。

```bash
python -m pytest tests/test_gpt_benchmark_nmos_sio2.py tests/test_gpt_benchmark_pmos_sio2_hfo2.py -v
```

上述用例数为历史记录。当前 advance 已删除占据预算，新增电压预检查、候选拒绝后观测状态一致性，以及电压阈值细化下的解析反馈解收敛测试；原占据预算卡顿不再对应当前算法。


## 指定时间快照

`test_gpt_utils_advance_with_snapshots.py` 检查真实时间停步、去重、状态观测和分段历史拼接；`test_gpt_utils_save_band_occupation.py` 用两种缺陷检查PNG输出、NPZ原始占据、最新静电/局部能级一致性，并确认保存不改变f。工具说明见src/utils/README.md。


## 紧凑HDF5历史

新增record append和离线绘图的两个测试文件。检查独立histogram积分的分箱占据、float32存储、由真实面电荷重建电势、记录器不修改冻结环境或占据、开关记录时推进轨迹逐项一致、文件帧与计算步一一对应，以及脱离器件后从文件重画PNG。指定时间工具另检查全步回调在分段时的时钟和去重。HDF5测试使用Windows comphy环境的h5py；源码没有安装或修改环境。

完整run.py已额外实跑验证：6000s共622帧，紧凑HDF5为6207831字节，最终帧可离线重画；这次实跑不作为pytest固定体积断言。


## 载流子参考密度更新

参数统一为 `n_DFT_m3`，删除缺陷参考面密度和缺陷浓度归一化因子。
有效速率使用 p/n_DFT_m3、n/n_DFT_m3 及导带空态因子；Nt 继续影响电荷和静电反馈。
新增固定静电环境下改变 Nt（包括零浓度）不改变单缺陷速率、n_DFT_m3 反比缩放速率的检查。
两套 benchmark 的旧固定因子0.01删除后速率为原来的100倍；已将阶段时间和步长同时缩短100倍，与旧固定时刻基线核对，通过后更新新公式的基线，验收容差保持不变。


## 栅极耦合

新增构造参数、距栅极距离、栅极功函数/电势边界、metal势垒、耦合平方标度、栅极独立费米能级、三热库速率求和、下溢和纯栅极通道主方程解析解检查。
温度系数仅保存，测试确认在不同温度下不额外改变耦合；JSON入口使用非零栅极耦合与温度参数验证保存和全流程。
默认关闭栅极交换，原NMOS/PMOS回归基线保持不变。
