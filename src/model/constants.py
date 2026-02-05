from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicalConstants:
    """
    Central repository for physical constants and default model parameters.
    Replaces magic numbers in the codebase.
    """
    
    # Physics & Transport
    PARTICLE_DENSITY: float = 1400.0 # kg/m^3 (Default coal density)
    HEAT_CAPACITY_SOLID: float = 1500.0 # J/kgK (Default char specific heat)
    MIN_SLIP_VELOCITY: float = 0.1   # m/s (Minimum gas-solid slip velocity)
    MIN_FLOW_SCALE: float = 0.1      # kg/s or fraction (Safe denominator)
    GRAVITY: float = 9.81            # m/s^2
    UNIVERSAL_GAS_CONSTANT: float = 8.314 # J/(mol*K)
    STEFAN_BOLTZMANN: float = 5.67e-8     # W/(m^2*K^4)
    
    # Grid Generation
    FIRST_CELL_LENGTH: float = 0.05  # m (Reduced from 0.40 for shorter tau)
    IGNITION_ZONE_DZ: float = 0.05   # m (Reduced from 0.10)
    MIN_GRID_SIZE: float = 0.05      # m (Reduced from 0.10)
    
    # Numerical / Solver
    TOLERANCE_SMALL: float = 1e-9    # Small number for division safety
    DEFAULT_TEMPERATURE: float = 298.15 # K
    
    # Wall / Reactor props
    WALL_EMISSIVITY: float = 0.7     
    PARTICLE_EMISSIVITY: float = 0.85 
    DEFAULT_HEAT_LOSS_PERCENT: float = 3.0 # % of LHV input
    DEFAULT_HHV: float = 25000.0 # kJ/kg (Reference coal)
