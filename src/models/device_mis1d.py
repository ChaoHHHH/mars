"""一维 MIS 器件：结构、静电状态及缺陷转移速率。"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from src.models.defect import Defects
from src.models.device_channel import Channel
from src.models.device_gate import Gate
from src.models.device_insulatinglayer import InsulatingLayer
from src.physics.marcus_rates import calculate_marcus_rates
from src.physics.solver.get_delta_V_trap import get_delta_V_trap as calculate_trap_potential
from src.physics.solver.get_potential import get_potential as calculate_potential
from src.physics.solver.solve_psi_s import solve_psi_s as solve_surface_potential
from src.physics.solver.calculate_total_rates import calculate_total_rates


class MOS1D:
    """金属–绝缘层–半导体器件

    长度使用 m，温度使用 K，电势使用 V，能量使用相对真空的 eV；
    缺陷电子占据与电势相互耦合，偏移量由当前缺陷电荷计算。
    """

    # 器件构造与外部设置

    def __init__(
        self, gate: Gate, channel: Channel, layers: list[InsulatingLayer],
    ) -> None:
        self.gate = gate
        self.channel = channel
        self.layers = layers
        self.Vg = 0.0
        self.defects: list[Defects] = []
        # 上次显式刷新的静电快照；f 原地改变时不自动更新
        self.electrostatic_state: dict | None = None

    def set_Vg(self, Vg: float) -> None:
        if Vg != self.Vg:
            self.electrostatic_state = None # TODO：应该自动更新
        self.Vg = Vg

    def add_defects(self, defects: Defects) -> None:
        self.defects.append(defects)
        self.electrostatic_state = None # TODO：应该自动更新

    # 静电势与阈值电压

    def get_Cox_per_area(self) -> float:
        """单位面积绝缘层电容 Cox，F/m²；1/Cox = sum(tox/eps)。"""
        Ci: ndarray = np.array([layer.eps / layer.tox for layer in self.layers])
        Cox = 1.0 / np.sum(1.0 / Ci)
        return float(Cox)

    def get_psi_ms(self, T: float) -> float:
        """金属与半导体的功函数差，V；输入功函数以 eV 表示。"""
        return self.gate.workfunction - self.channel.get_workfunction(T)

    def solve_psi_s(self, T: float) -> float:
        """在当前缺陷占据概率 f 下求解静电自洽的 MOS 表面势，单位 V。"""

        Cox = self.get_Cox_per_area()
        psi_ms = self.get_psi_ms(T)
        thickness = sum(layer.tox for layer in self.layers)
        delta_V_gate = self.get_delta_V_trap(thickness)

        # 目标方程
        def f(psi_s: float) -> float:
            Qs = self.channel.cal_Qs(psi_s, T)
            return float(psi_s - Qs / Cox + psi_ms + delta_V_gate - self.Vg)

        Veff = self.Vg - psi_ms - delta_V_gate
        if Veff == 0.0:
            return 0.0
        bracket = (min(0.0, Veff), max(0.0, Veff))
        return solve_surface_potential(f, bracket)

    def get_delta_V_trap(self, x: float | ndarray) -> float | ndarray:
        """缺陷面电荷引起的电势变化，V，以沟道界面 x=0 为参考。"""
        charge_sheets = [
            (defects.data["x"][:, 0, 0, 0], defects.get_charge_sheets())
            for defects in self.defects
        ]
        layers = [(layer.tox, layer.eps) for layer in self.layers]
        return calculate_trap_potential(x, charge_sheets, layers)

    def get_potential(
        self, x: float | ndarray, T: float, *, use_saved_state: bool = False,
    ) -> float | ndarray:
        """介电层电势，V；默认按当前占据计算，不隐式修改静电快照。

        use_saved_state=True 时使用上次显式刷新的电荷、电势和介质参数，
        不重新求根；该快照允许在冻结环境推进期间与当前 f 不同。
        """
        if use_saved_state:
            state = self._saved_electrostatics(T)
            return calculate_potential(
                x, state["psi_s"], state["Qs"],
                state["charge_sheets"], state["layers"],
            )

        psi_s = self.solve_psi_s(T)
        Qs = self.channel.cal_Qs(psi_s, T)
        charge_sheets = [
            (d.data["x"][:, 0, 0, 0], d.get_charge_sheets())
            for d in self.defects
        ]
        layers = [(layer.tox, layer.eps) for layer in self.layers]
        return calculate_potential(x, psi_s, Qs, charge_sheets, layers)

    def _saved_electrostatics(self, T: float) -> dict:
        """检查显式复用的快照是否与请求的偏压和温度相符。"""
        state = self.electrostatic_state
        if state is None or state["T"] != T or state["Vg"] != self.Vg:
            raise RuntimeError("Call update_electrostatics(T) before reusing this state")
        return state

    def update_electrostatics(self, T: float) -> None:
        """只求一次表面势，保存静电快照并统一更新带边和缺陷局部能级。

        快照保存 Vg、T、psi_s、Qs、电荷面、介质参数及各缺陷位置的电势。
        修改 f 或材料/几何参数后，应显式刷新；本方法不隐式更新速率。
        """
        psi_s = self.solve_psi_s(T)
        Qs = self.channel.cal_Qs(psi_s, T)
        charge_sheets = tuple(
            (d.data["x"][:, 0, 0, 0].copy(), d.get_charge_sheets().copy())
            for d in self.defects
        )
        layers = tuple((layer.tox, layer.eps) for layer in self.layers)
        potentials = tuple(
            calculate_potential(x, psi_s, Qs, charge_sheets, layers)
            for x, _ in charge_sheets
        )
        self.VBM_local = self.channel.Evbm - psi_s
        self.CBM_local = self.channel.Evbm + self.channel.get_Eg(T) - psi_s
        for defects, potential in zip(self.defects, potentials):
            data = defects.data
            data["Et_local"] = data["Et"] - potential[:, None, None, None]
            data["transition_level_local"] = data["Et_local"] # TODO：这个之前Et被认为是本征能级，现在多余了，应该删除
        self.electrostatic_state = {
            "Vg": self.Vg,
            "T": T,
            "psi_s": psi_s,
            "Qs": Qs,
            "charge_sheets": charge_sheets,
            "layers": layers,
            "defect_potentials": potentials,
        }

    def get_Vth0(self, T: float) -> float:
        """无缺陷电荷的理想阈值电压，V

        psi_th = -2*(EF-Ei)，直接反算栅压，等价于关闭缺陷耦合后求根。
        不改变当前栅压、缺陷占据或器件状态。
        """
        intrinsic_level = self.channel.get_Ei(T)
        psi_th = -2.0 * (self.channel.get_Ef_bulk(T) - intrinsic_level)
        return float(
            self.get_psi_ms(T) + psi_th
            - self.channel.cal_Qs(psi_th, T) / self.get_Cox_per_area()
        )

    def get_delta_Vth(self) -> float:
        """当前缺陷电荷相对理想器件的阈值偏移，V

        不扣除初始电荷贡献，不更新占据；Vth = get_Vth0(T) + get_delta_Vth()。
        """
        thickness = sum(layer.tox for layer in self.layers)
        return float(self.get_delta_V_trap(thickness))

    # 局部能级、耦合及转移速率

    def update_defect_couplings(self) -> None:
        """更新三个热库的耦合，单位 eV。

        VBM/CBM 距离取 x，gate 距离取总厚度减 x，均按 exp(-decay*距离) 衰减。
        temperature_vbm/cbm/gate 仅保存，暂不影响耦合。
        """
        # TODO：加入温度系数
        thickness = sum(layer.tox for layer in self.layers)
        for defects in self.defects:
            x = defects.data["x"]
            defects.data["Vc_vbm_local"] = (defects.Vc_vbm * np.exp(-defects.decay_vbm * x))
            defects.data["Vc_cbm_local"] = (defects.Vc_cbm * np.exp(-defects.decay_cbm * x))
            defects.data["Vc_gate_local"] = (defects.Vc_gate * np.exp(-defects.decay_gate * (thickness - x)))

    def update_defect_rates(
        self, T: float, *, refresh_electrostatics: bool = True,
    ) -> None:
        """更新裸速率，s^-1；默认刷新一次静电状态，也可显式复用快照。"""
        if refresh_electrostatics:
            self.update_electrostatics(T)
        else:
            self._saved_electrostatics(T)
        self.update_defect_couplings()
        # EF_gate = EF_channel - Vg，相同真空能量参考，单位 eV
        self.EF_gate = self.channel.Evbm + self.channel.get_Ef_bulk(T) - self.Vg
        for defects in self.defects:
            data = defects.data
            for band, edge, reservoir in (
                ("vbm", self.VBM_local, "vbm"),
                ("cbm", self.CBM_local, "cbm"),
                ("gate", self.EF_gate, "metal"),
            ):
                log_k12, log_k21 = calculate_marcus_rates(
                    data["transition_level_local"], edge,
                    data[f"Vc_{band}_local"], data["Er12"], data["Er21"], T,
                    return_log=True, reservoir=reservoir,
                )
                data[f"log_k12_{band}"] = log_k12
                data[f"k12_{band}"] = np.exp(log_k12)
                data[f"k21_{band}"] = np.exp(log_k21)

    def update_defect_total_rates(
        self, T: float, *,
        refresh_electrostatics: bool = True,
    ) -> None:
        """准备沟道与各缺陷参数，更新三热库的流入/流出有效速率和总速率，单位 s^-1"""
        self.update_defect_rates(T, refresh_electrostatics=refresh_electrostatics)
        psi_s = self.channel.Evbm - self.VBM_local
        n0, p0 = self.channel.get_carrier_density(T)
        ef = self.channel.Evbm + self.channel.get_Ef_bulk(T)
        for defects in self.defects:
            data = defects.data
            data.update(calculate_total_rates(
                log_k12_vbm=data["log_k12_vbm"],
                log_k12_cbm=data["log_k12_cbm"],
                transition_level_local=data["transition_level_local"],
                Er12=data["Er12"], Er21=data["Er21"],
                T=T, psi_s=psi_s, cbm_local=self.CBM_local,
                ef=ef, n0=n0, p0=p0,
                n_DFT_m3=defects.n_DFT_m3,
                log_k12_gate=data["log_k12_gate"], ef_gate=self.EF_gate,
            ))
