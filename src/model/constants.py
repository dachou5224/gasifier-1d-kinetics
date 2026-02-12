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
    DEFAULT_HEAT_LOSS_PERCENT: float = 2.0  # % of inlet coal HHV (文献: 散热损失为入炉煤HHV的2%)
    DEFAULT_HHV: float = 25000.0 # kJ/kg (Reference coal)

    # Char combustion vs gas-phase: 焦炭燃烧比气相燃烧更慢，下调 C+O2 速率
    CHAR_COMBUSTION_RATE_FACTOR: float = 0.3  # C+O2 速率缩放 (0.2–0.5 合理)

    # 颗粒瞬态传热 (Fortran Line 264-274, docs/fortran_temperature_analysis.md)
    EF_PARTICLE_TRANSIENT: float = 0.9  # ef (Fortran)
    CONDUT_COEFF: float = 7.7e-7 * 418.4  # 7.7e-7 cal/(cm·s·K) -> 3.22e-4 W/(m·K) per (Tg+Ts)^0.75
    TS_MAX_FOR_RATES: float = 1250.0  # Fortran: if tsm>1250 then tsm=1250
    TS_TRANSIENT_NC: int = 30  # 子步数 (Fortran nc=20-50)
    USE_RK_GILL_COMBUSTION: bool = False  # 燃烧区用 RK-Gill 含反应热 (Fortran 505)；开启后计算量约 4x
