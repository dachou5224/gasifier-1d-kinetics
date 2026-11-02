from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class SourceTerm(ABC):
    """
    Abstract Base Class for generic source terms in the conservation equations.
    """
    @abstractmethod
    def get_sources(self, cell_idx: int, z: float, dz: float) -> Tuple[np.ndarray, float, float]:
        """
        Calculate source terms for a specific cell.
        
        Returns:
            gas_src (np.array): Molar source for 8 gas species [mol/s]
            solid_src (float): Mass source for solid [kg/s] (Negative for consumption)
            energy_src (float): Energy source [Watts] (Negative for sink/loss)
        """
        pass

class EvaporationSource(SourceTerm):
    """
    Handles Moisture/Slurry Evaporation:
    - Mass Source: +H2O (Gas)
    - Energy Source: Enthalpy of liquid water (sink = latent + sensible to gas at T).

    If L_evap_m is 0 or very small, all evaporation is applied in the first cell (Cell 0),
    which can overestimate the heat sink there and lead to under-predicted reaction temperature.
    Use L_evap_m > 0 (e.g. 0.5–1.5 m) to distribute evaporation over the first L_evap_m meters
    (op_conds['L_evap_m']); default remains "all in Cell 0" for backward compatibility.
    """
    # Standard enthalpy of liquid water (J/mol), NIST
    H_LIQUID_J_MOL = -285830.0

    def __init__(self, water_flow_mol_s: float, enthalpy_per_mol_J: float = None,
                 L_evap_m: float = 0.0):
        self.water_flow = water_flow_mol_s
        self.h_liq = enthalpy_per_mol_J if enthalpy_per_mol_J is not None else self.H_LIQUID_J_MOL
        self.L_evap = max(L_evap_m, 1e-6)

    def _fraction_in_cell(self, z: float, dz: float) -> float:
        """Fraction of total evaporation in this cell. z is cell center; cell spans [z-dz/2, z+dz/2].
        Linear distribution over [0, L_evap] so sum over cells = 1."""
        z_start = z - dz / 2.0
        z_end = z + dz / 2.0
        overlap_start = max(0.0, z_start)
        overlap_end = min(self.L_evap, z_end)
        if overlap_end <= overlap_start:
            return 0.0
        return (overlap_end - overlap_start) / self.L_evap

    def get_sources(self, cell_idx: int, z: float, dz: float) -> Tuple[np.ndarray, float, float]:
        gas_src = np.zeros(8)
        solid_src = 0.0
        energy_src = 0.0

        frac = self._fraction_in_cell(z, dz)
        if frac <= 0.0:
            return gas_src, solid_src, energy_src

        gas_src[7] = self.water_flow * frac
        energy_src = self.water_flow * frac * self.h_liq
        return gas_src, solid_src, energy_src


class PyrolysisSource(SourceTerm):
    """
    Handles Coal Pyrolysis (Devolatilization):
    - Gas Source: +Volatiles (CH4, CO, H2, etc.)
    - Solid Source: -Volatile Matter (kg/s)
    - Energy Source: 挥发分在热解温度下携带的焓

    P1.6-A（2026-11-02，审计 B9）**不改这里的能量源项**，只改固相焓口径（见
    ``MaterialService.get_solid_enthalpy`` 与 ``Cell._calc_energy_balance``）：
    入炉煤按 ``Hf_coal`` 计焓，热解后的焦炭碳取石墨生成焓 0、灰只计显热。
    于是热解热自动等于
        ΔH_pyro = H_char(T) + H_vol(T) - H_coal(T)
    并全部落在热解所在单元的能量残差里（不再隐含在焦炭的生成焓中，也不会被下游带走）。

    本类保留 ``coal_enthalpy_W`` / ``char_enthalpy_W`` / ``pyrolysis_heat_W`` 三个诊断量，
    用来做单元级能量闭合检查；它们不进入方程。

    修改前（legacy 口径）的等效热解热是
        ΔH_legacy = H_vol - m_vol*(cp*ΔT + Hf_coal)
    两者之差恰为 ``m_vol * Hf_coal``，这就是 B9 的量级。
    """
    def __init__(self, volatile_fluxes_mol_s: np.ndarray, solid_loss_kg_s: float, target_cell_idx: int = 0,
                 T_pyro: float = 300.0, coal_mass_flow_kg_s: float = None,
                 coal_props: dict = None, solid_enthalpy_model: str = "char_formation_zero"):
        self.vol_fluxes = volatile_fluxes_mol_s
        self.solid_loss = solid_loss_kg_s
        self.target_idx = target_cell_idx
        self.T_pyro = float(T_pyro)
        self.solid_enthalpy_model = str(solid_enthalpy_model)

        # 挥发分在热解温度下携带的焓（含各自的生成焓）
        from model.physics import get_enthalpy_molar
        from model.material import SPECIES_NAMES
        self.volatile_enthalpy_W = 0.0
        for i, sp in enumerate(SPECIES_NAMES):
            if self.vol_fluxes[i] > 0:
                h_i = get_enthalpy_molar(sp, self.T_pyro)
                self.volatile_enthalpy_W += self.vol_fluxes[i] * h_i

        # 能量源项：两种口径一致，差别只在固相焓是否带 Hf_coal
        self.energy_src_W = self.volatile_enthalpy_W

        # 诊断量（不进入方程）
        self.coal_enthalpy_W = float("nan")
        self.char_enthalpy_W = float("nan")
        if coal_mass_flow_kg_s is not None and coal_props is not None:
            cp_s = float(coal_props.get("cp_char", 1300.0))
            hf_coal = float(coal_props.get("Hf_coal", 0.0))
            m_coal = float(coal_mass_flow_kg_s)
            m_char = m_coal - float(solid_loss_kg_s)
            dT = self.T_pyro - 298.15
            self.coal_enthalpy_W = m_coal * (cp_s * dT + hf_coal)
            if self.solid_enthalpy_model == "legacy_coal_hf":
                self.char_enthalpy_W = m_char * (cp_s * dT + hf_coal)
            else:
                self.char_enthalpy_W = m_char * (cp_s * dT)

    @property
    def pyrolysis_heat_W(self) -> float:
        """H_coal - H_char - H_vol（>0 = 放热，<0 = 吸热）。仅在提供煤流时有效。"""
        return self.coal_enthalpy_W - self.char_enthalpy_W - self.volatile_enthalpy_W

    def get_sources(self, cell_idx: int, z: float, dz: float) -> Tuple[np.ndarray, float, float]:
        gas_src = np.zeros(8)
        solid_src = 0.0
        energy_src = 0.0

        if cell_idx == self.target_idx:
            # Gas Source
            gas_src = self.vol_fluxes.copy()
            # Solid Consumption (Volatiles leaving solid phase)
            solid_src = -self.solid_loss
            # Energy of volatiles injected into gas phase
            energy_src = self.energy_src_W

        return gas_src, solid_src, energy_src
