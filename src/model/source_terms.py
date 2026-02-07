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
    Handles Moisture Evaporation:
    - Mass Source: +H2O (Gas)
    - Solid Source: -Moisture (If tracking moisture as solid, but we simplify to Gas Source only for now, as Inlet handling varies)
    - Energy Source: -Latent Heat (Sink)
    """
    def __init__(self, water_flow_mol_s: float, latent_heat_watts: float, target_cell_idx: int = 0):
        self.water_flow = water_flow_mol_s
        self.Q_evap = latent_heat_watts # Positive magnitude
        self.target_idx = target_cell_idx
        
    def get_sources(self, cell_idx: int, z: float, dz: float) -> Tuple[np.ndarray, float, float]:
        gas_src = np.zeros(8)
        solid_src = 0.0
        energy_src = 0.0
        
        if cell_idx == self.target_idx:
            # Add H2O (Index 7)
            gas_src[7] = self.water_flow
            
            # Energy Sink: Total Enthalpy of Incoming Liquid Water
            # energy_src = m_dot * h_liquid
            # h_liquid = h_gas(T_in) - LatentHeat
            # We assume T_in = 298.15 K for the feed water basis or use constant.
            # Hf(H2O, gas) = -241.8 kJ/mol
            # Latent = 44.0 kJ/mol (at 298K)
            # h_liquid ~ -285.8 kJ/mol
            
            h_liquid_J_mol = -285830.0 # J/mol (Standard Enthalpy of Liquid Water)
            
            # If we want to be precise with T_feed, we could add Cp_liq * (T - 298).
            # But strictly, the source term must balance the Enthalpy added to Gas.
            # Balance: H_out = H_in + Source.
            # H_out (Gas) includes Hf_gas (-241 kJ/mol).
            # H_in = 0 (for water).
            # Source must bridge 0 to -241.
            # AND provide Latent heat (-44).
            # So Source = -285 kJ/mol.
            
            energy_src = self.water_flow * h_liquid_J_mol
            
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
