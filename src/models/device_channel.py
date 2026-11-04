import numpy as np

EPS0  = 8.8541878188E-12 # F/m
KB_EV = 8.6173332621E-5  # eV/K
H = 4.135667696E-15 # eV⋅s
ME = 9.1093837015E-31 # kg
EV_TO_J = 1.602176634e-19  # J/eV

class Channel:
    def __init__(self, Nd: float, Na: float, Evbm: float,
                 Eg0: float, Eg1: float,
                 ncoeff: list[float], pcoeff: list[float],
                 Mv: int, Mc: int, eps_r: float) -> None:
        """Evbm应该为相对于真空能级的值，用来标定其它能量"""
        self.Nd = Nd
        self.Na = Na
        self.Eg0 = Eg0
        self.Eg1 = Eg1
        self.Evbm = Evbm
        self.ncoeff = ncoeff
        self.pcoeff = pcoeff
        self.Mv = Mv
        self.Mc = Mc
        self.eps_r = eps_r
        self.eps = eps_r * EPS0

    def get_Eg(self, T: float) -> float:
        return self.Eg0 + self.Eg1 * T

    def get_mneff_mpeff(self, T: float) -> tuple[float, float]:

        Eg = self.get_Eg(T)
        # Conduction band
        a, b = self.ncoeff
        mn_rel = ((a * self.Eg0 / Eg)**2 * b)**(1.0 / 3.0)
        # Valence band
        A, B, C, D, E, F, G, H, I = self.pcoeff
        aa = A + T * (B + T * (C + T * (D + T * E)))
        bb = 1 + T * (F + T * (G + T * (H + T * I)))
        mp_rel = (aa / bb)**(2.0 / 3.0)

        return float(mn_rel * ME), float(mp_rel * ME)

    def get_Nv_Nc(self, T: float) -> tuple[float, float]:
        mneff, mpeff = self.get_mneff_mpeff(T)
        kT = KB_EV * T * EV_TO_J
        h = H * EV_TO_J
        Nv = self.Mv * 2 * (2 * np.pi * mpeff * kT / h**2)**1.5
        Nc = self.Mc * 2 * (2 * np.pi * mneff * kT / h**2)**1.5
        return Nv, Nc

    def get_Ei(self, T: float) -> float:
        """本征能级，相对沟道 VBM 的能量，单位 eV。

        与 get_Ef_bulk 使用相同能量零点；相对真空的值为 Evbm + Ei。
        """
        Nv, Nc = self.get_Nv_Nc(T)
        kT = KB_EV * T
        return float(self.get_Eg(T) / 2.0 + kT / 2.0 * np.log(Nv / Nc))

    def get_Ef_bulk(self, T: float) -> float:
        """体区费米能级，相对沟道 VBM 的能量，单位 eV。"""
        Eg = self.get_Eg(T)
        Nv, Nc = self.get_Nv_Nc(T)
        kT = KB_EV * T
        ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * kT))
        Ei = self.get_Ei(T)
        Ef = Ei + kT * np.arcsinh((self.Nd - self.Na) / (2 * ni))
        return float(Ef)

    def get_workfunction(self, T: float) -> float:
        """半导体体区功函数，eV"""
        return -self.Evbm - self.get_Ef_bulk(T)

    def get_carrier_density(self, T: float) -> tuple[float, float]:
        """返回体区平衡载流子浓度 n0, p0，单位 m^-3"""

        Ef = self.get_Ef_bulk(T)
        Eg = self.get_Eg(T)
        Nv, Nc = self.get_Nv_Nc(T)
        kT = KB_EV * T

        n0 = Nc * np.exp((Ef - Eg) / kT)
        p0 = Nv * np.exp(-Ef / kT)

        return float(n0), float(p0)

    def cal_Qs(self, psi_s: float, T: float) -> float:
        """半导体实际空间电荷，C/m²；正表面势对应负电荷。"""
        n0, p0 = self.get_carrier_density(T)
        kT = KB_EV * T
        u = psi_s / kT
        # 稳定计算 exp(x) - 1 - x
        def f(x: float) -> float:
            if abs(x) < 1e-4:
                return x**2 * (0.5 + x * (1/6 + x * (1/24 + x * (1/120 + x/720))))
            return float(np.expm1(x) - x)
        F = p0 * f(-u) + n0 * f(u)
        # eV 转换为 J
        kT_J = kT * EV_TO_J
        Qs = -np.sign(psi_s) * np.sqrt(2 * self.eps * kT_J * F)
        return float(Qs)