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
    - Energy Source: +/- Heat of Pyrolysis (Often neglected or lumped, set to 0 by default)
    """
    def __init__(self, volatile_fluxes_mol_s: np.ndarray, solid_loss_kg_s: float, target_cell_idx: int = 0):
        self.vol_fluxes = volatile_fluxes_mol_s
        self.solid_loss = solid_loss_kg_s
        self.target_idx = target_cell_idx
        
    def get_sources(self, cell_idx: int, z: float, dz: float) -> Tuple[np.ndarray, float, float]:
        gas_src = np.zeros(8)
        solid_src = 0.0
        energy_src = 0.0
        
        if cell_idx == self.target_idx:
            # Gas Source
            gas_src = self.vol_fluxes.copy()
            # Solid Consumption (Volatiles leaving solid phase)
            solid_src = -self.solid_loss
            
        return gas_src, solid_src, energy_src
