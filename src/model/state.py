from dataclasses import dataclass, field
import numpy as np
from typing import List
from .physics import R_CONST

@dataclass
class StateVector:
    """
    Represents the full thermodynamic and chemical state of the gasifier at a specific point.
    Designed to be immutable-ish (or at least strictly typed) to avoid index errors.
    """
    # Gas Phase (mol/s)
    # Order: [O2, CH4, CO, CO2, H2S, H2, N2, H2O]
    gas_moles: np.ndarray 
    
    # Solid Phase
    solid_mass: float      # kg/s (Total solid flow)
    carbon_fraction: float # X_c (kg_C / kg_Solid)
    
    # Thermodynamics
    T: float               # K (Temperature)
    P: float               # Pa (Pressure)
    
    # Metadata (Optional but helpful for debugging)
    z: float = 0.0         # Axial position (m)
    
    @property
    def total_gas_moles(self) -> float:
        return np.sum(self.gas_moles)
    
    @property
    def gas_fractions(self) -> np.ndarray:
        s = self.total_gas_moles
        if s < 1e-9: return np.zeros_like(self.gas_moles)
        return self.gas_moles / s

    def copy(self):
        return StateVector(
            gas_moles=self.gas_moles.copy(),
            solid_mass=self.solid_mass,
            carbon_fraction=self.carbon_fraction,
            T=self.T,
            P=self.P,
            z=self.z
        )

    def to_array(self) -> np.ndarray:
        """Serialize to Solver Array [F0-F7, Ws, Xc, T]"""
        return np.concatenate([
            self.gas_moles, 
            [self.solid_mass, self.carbon_fraction, self.T]
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, P: float, z: float = 0.0):
        """Deserialize from Solver Array"""
        return cls(
            gas_moles=arr[:8],
            solid_mass=arr[8],
            carbon_fraction=arr[9],
            T=arr[10],
            P=P,
            z=z
        )

    def get_concentration(self, species_idx: int) -> float:
        """
        Calculate molar concentration of a species (kmol/m³).
        Formula: C_i = (F_i / F_total) * (P / RT)
        """
        F_i = self.gas_moles[species_idx]
        F_total = self.total_gas_moles
        if F_total < 1e-9: F_total = 1e-9
        
        # P [Pa], T [K], R [J/mol.K] -> P/RT [mol/m3]
        # Result needed in kmol/m3 for kinetics
        c_mol_m3 = (F_i / F_total) * (self.P / (R_CONST * self.T))
        return c_mol_m3 / 1000.0
