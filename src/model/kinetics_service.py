import numpy as np
from typing import Dict, List, Optional
from model.state import StateVector
from model.kinetics import (
    HeterogeneousKinetics,
    calculate_wgs_equilibrium,
    calculate_cstm_equilibrium,
)
from model.constants import PhysicalConstants
from model.physics import R_CONST
from model.material import SPECIES_NAMES

class KineticsService:
    """
    Service for calculating all reaction rates (Pyrolysis, Heterogeneous, Homogeneous).
    Decouples logic from the Solver/Cell.
    """
    def __init__(self):
        # Underlying Physics Model
        self.het_model = HeterogeneousKinetics()
        
        # Homogeneous Parameters (Fortran: exp(-E_cal/(1.987*ts)) → E_J = E_cal × 4.184)
        # WGS: wgshift L1271,1281  A=2.877e5 E=27760 cal/mol
        # MSR: ch4ref L1355  A=312 E=30000 cal/mol, ts≤1000K rate=0
        CAL2J = 4.184
        self.A_homo = {
            'CO_Ox':  2.23e12,
            'H2_Ox':  1.08e13,
            'WGS':    2.877e5,   # Fortran wgshift L1281
            'RWGS':   1.0e5,
            'CH4_Ox': 1.6e10,
            'MSR':    312.0      # Fortran ch4ref L1355
        }
        self.E_homo = {
            'CO_Ox':  1.25e8 / 1000.0,
            'H2_Ox':  8.37e7 / 1000.0,
            'WGS':    27760.0 * CAL2J,   # 116.1 kJ/mol
            'RWGS':   6.27e7 / 1000.0,
            'CH4_Ox': 1.256e8 / 1000.0,
            'MSR':    30000.0 * CAL2J    # 125.5 kJ/mol
        }
        
    def calc_heterogeneous_rates(self, state: StateVector, particle_diameter: float, 
                                 surface_area: float, X_total: float = 0.0, 
                                 Re: float = 0.0, Sc: float = 1.0,
                                 T_particle: float = None,
                                 char_combustion_factor: float = None,
                                 use_fortran_diffusion: bool = False) -> Dict[str, float]:
        """
        Calculate Surface Reaction Rates (mol/s).
        T_particle: 颗粒温度 Ts_avg；None 时用 Ross 修正 T_p = T + 6.6e4*C_O2（向后兼容）
        Returns: Dict {'C+O2': rate, ...}
        """
        r = {}
        F_total = state.total_gas_moles
        if F_total < PhysicalConstants.TOLERANCE_SMALL:
             # This is expected for initial cells with no gas yet, debug level only
             # logging.debug(f"LogicWarning: F_total ~ 0 at T={state.T:.1f}")
             F_total = PhysicalConstants.TOLERANCE_SMALL
        
        # 1. Particle Temperature: Fortran 式用 Ts_avg；否则用 Ross 修正
        P, T = state.P, state.T
        if T_particle is not None:
            T_p = min(T_particle, 4000.0)
        else:
            C_O2_kmol = state.get_concentration(0)
            T_p = T + 6.6e4 * C_O2_kmol
            T_p = min(T_p, 4000.0)
        
        # 2. Mechanism Factor phi (Determines CO/CO2 ratio)
        p_fac = 2500.0 * np.exp(-5.19e4 / (R_CONST * T_p))
        phi = (2*p_fac + 2) / (p_fac + 2)
        r['phi'] = phi 
        
        # 3. Y (Shrinking Core Factor)
        # Y = Rc / Rp = (1-X)**(1/3) for sphere
        Y = (1.0 - X_total)**(1.0/3.0)
        Y = max(Y, 1e-3)
        
        mapping = {'C+O2': 'O2', 'C+H2O': 'H2O', 'C+CO2': 'CO2', 'C+H2': 'H2'}
        P_atm = 101325.0

        for rxn, sp in mapping.items():
            idx = SPECIES_NAMES.index(sp)
            F_i = max(state.gas_moles[idx], 0.0)
            P_i = (F_i / F_total) * P  # Pa
            P_eq = 0.0

            # Fortran 安全与温度阈值 (fortran_analysis.md)
            if rxn == 'C+CO2':
                if T_p <= 850.0 or P_i < 1e-6 or X_total >= 0.999:
                    r[rxn] = 0.0
                    continue
            elif rxn == 'C+H2':
                if T_p <= 1200.0 or X_total >= 0.999:
                    r[rxn] = 0.0
                    continue
            elif rxn == 'C+H2O':
                if P_i < 0.001 * P_atm or X_total >= 0.999:
                    r[rxn] = 0.0
                    continue
                P_CO = (state.gas_moles[2] / F_total) * P
                P_H2 = (state.gas_moles[5] / F_total) * P
                if P_CO < 1e-6 or P_H2 < 1e-6:
                    P_eq = 0.0  # Fortran goto 5: pexc = psteam
                else:
                    K_cstm = calculate_cstm_equilibrium(T_p)
                    cts = 17.644 - 16811.0 / T_p
                    if abs(cts) > 16.0 or K_cstm > 10000.0:
                        r[rxn] = 0.0
                        continue
                    P_eq = (P_H2 * P_CO) / (K_cstm * P_atm + 1e-12)

            # 4. Driving Force Correction (P_eq)
            # C+H2O 的 P_eq 已在 Fortran 安全检查块中计算
            if rxn == 'C+CO2':
                from model.kinetics import calculate_boudouard_equilibrium
                Kp_atm = calculate_boudouard_equilibrium(T_p)
                P_CO = (state.gas_moles[2] / F_total) * P
                P_eq = (P_CO**2 / (Kp_atm * P_atm + 1e-9))

            # 5. Stoichiometric factor nu (Mols Carbon per mol oxidant)
            nu = phi if rxn == 'C+O2' else 1.0

            # UCSM Rate Evaluation: flux [kmol/(m²·s)] 已折算到颗粒外表面积
            P_eff = max(P_i - P_eq, 0.0)
            if rxn == 'C+H2O' and P_eff < 1e-6:
                r[rxn] = 0.0
                continue

            rate_kmols = self.het_model.calculate_total_rate(
                rxn, T, P, P_eff, particle_diameter, Y, nu=nu, Re=Re, Sc=Sc, T_p=T_p,
                use_fortran_diffusion=use_fortran_diffusion, phi=r.get('phi', 1.0)
            )

            rate_flux = rate_kmols * 1000.0
            r[rxn] = rate_flux * surface_area
            # 焦炭燃烧比气相燃烧更慢，对 C+O2 施加缩放因子（可工况级覆盖）
            if rxn == 'C+O2':
                fac = char_combustion_factor if char_combustion_factor is not None else PhysicalConstants.CHAR_COMBUSTION_RATE_FACTOR
                r[rxn] *= fac
            
        return r

    def calc_homogeneous_rates(self, state: StateVector, volume: float, 
                                inlet_state: StateVector = None, gas_src: np.ndarray = None,
                                Ts_particle: float = None, wgs_rat_factor: bool = False,
                                msr_tmin_k: float = 1000.0,
                                wgs_catalytic_factor: float = None,
                                wgs_k_factor: float = None) -> Dict[str, float]:
        """
        Calculate Homogeneous Rates (mol/s) based on Table 2-5.
        
        Uses AVERAGE concentration: C_avg = (C_inlet + C_outlet) / 2
        This provides a more physically meaningful rate estimate for plug flow reactors.
        
        WGS: Fortran wgshift 逻辑 — Ts_particle<=1000K 时不计 WGS；若未提供 Ts_particle 则用气相 T。
        """
        rates = {}
        P, T = state.P, state.T
        
        # Calculate concentrations using AVERAGE of inlet and outlet
        if inlet_state is not None and gas_src is not None:
            # Inlet + Source moles (what enters the cell)
            inlet_moles = {}
            for i, sp in enumerate(SPECIES_NAMES):
                inlet_moles[sp] = max(inlet_state.gas_moles[i] + gas_src[i], 0.0)
            
            # Outlet moles (current state)
            outlet_moles = {sp: max(state.gas_moles[i], 0.0) for i, sp in enumerate(SPECIES_NAMES)}
            
            # Average moles = (inlet + outlet) / 2
            avg_moles = {sp: (inlet_moles[sp] + outlet_moles[sp]) / 2.0 for sp in SPECIES_NAMES}
            
            F_total_avg = sum(avg_moles.values())
            if F_total_avg < PhysicalConstants.TOLERANCE_SMALL:
                F_total_avg = PhysicalConstants.TOLERANCE_SMALL
            
            # Calculate concentrations from average moles
            C = {}
            for sp in SPECIES_NAMES:
                y_i = avg_moles[sp] / F_total_avg
                # C_i = y_i * P / (R*T) [mol/m³] -> [kmol/m³]
                C[sp] = (y_i * P / (R_CONST * T)) / 1000.0
            
            # Debug: Show average moles for key species (only once per solve)
            import logging
            logger = logging.getLogger(__name__)
            if inlet_moles.get('CH4', 0) > 1.0:  # Only log if there's significant CH4
                logger.info(f"[AVG CONC] CH4: inlet={inlet_moles['CH4']:.2f}, outlet={outlet_moles['CH4']:.2f}, avg={avg_moles['CH4']:.2f}")
                logger.info(f"[AVG CONC] O2:  inlet={inlet_moles['O2']:.2f}, outlet={outlet_moles['O2']:.2f}, avg={avg_moles['O2']:.2f}")


        else:
            # Fallback: Use outlet (current) concentration
            F_total = state.total_gas_moles
            if F_total < PhysicalConstants.TOLERANCE_SMALL:
                F_total = PhysicalConstants.TOLERANCE_SMALL
            
            C = {}
            for i, sp in enumerate(SPECIES_NAMES):
                C[sp] = state.get_concentration(i)


        # 1-5: Second Order (r = k Ca Cb) [kmol/m3.s]
        # (1) CO Combustion
        k = self.A_homo['CO_Ox'] * np.exp(-self.E_homo['CO_Ox'] / (R_CONST * T))
        r1 = k * C['CO'] * C['O2']
        rates['CO_Ox'] = r1 * volume * 1000.0  # mol/s
        
        # (2) H2 Combustion
        k = self.A_homo['H2_Ox'] * np.exp(-self.E_homo['H2_Ox'] / (R_CONST * T))
        r2 = k * C['H2'] * C['O2']
        rates['H2_Ox'] = r2 * volume * 1000.0

        # (3) WGS 净速：R_net = k_fwd × (C_CO C_H2O - C_CO2 C_H2 / K_eq)，保证热力学一致
        # Fortran wgshift 对齐：催化因子 f=0.2、rat=exp(-8.91+5553/ts) 压低速率
        T_wgs_check = Ts_particle if Ts_particle is not None else T
        if T_wgs_check <= 1000.0:
            rates['WGS'] = 0.0
            rates['RWGS'] = 0.0
        else:
            k_fwd = self.A_homo['WGS'] * np.exp(-self.E_homo['WGS'] / (R_CONST * T))
            # 催化因子 f，可调 (op_conds['WGS_CatalyticFactor'])，Fortran=0.2
            WGS_CATALYTIC_FACTOR = wgs_catalytic_factor if wgs_catalytic_factor is not None else 0.2
            # Fortran rat 因子：exp(-8.91+5553/ts)。启用时高温下压低约2个数量级
            # 物理预估：抑制 WGS → 少放热、少 CO→CO2，CO 可能略升；温度略降
            WGS_RAT_FACTOR = np.exp(-8.91 + 5553.0 / T_wgs_check) if wgs_rat_factor else 1.0
            k_eff = k_fwd * WGS_CATALYTIC_FACTOR * WGS_RAT_FACTOR
            K_eq = calculate_wgs_equilibrium(T)
            # WGS_K_Factor: <1 使有效 K 变小，强化逆向 WGS (CO2+H2→CO+H2O)，提高 CO
            K_eff = K_eq * (wgs_k_factor if wgs_k_factor is not None else 1.0)
            r_net = k_eff * (C['CO'] * C['H2O'] - C['CO2'] * C['H2'] / (K_eff + 1e-12))
            rates['WGS'] = r_net * volume * 1000.0  # mol/s
            rates['RWGS'] = 0.0
        
        # (5) CH4 Combustion
        k = self.A_homo['CH4_Ox'] * np.exp(-self.E_homo['CH4_Ox'] / (R_CONST * T))
        r5 = k * C['CH4'] * C['O2']
        rates['CH4_Ox'] = r5 * volume * 1000.0
        
        # (6) MSR (Fortran ch4ref: ts≤1000K → rate=0)
        T_msr_check = Ts_particle if Ts_particle is not None else T
        if T_msr_check <= msr_tmin_k:
            rates['MSR'] = 0.0
        else:
            k = self.A_homo['MSR'] * np.exp(-self.E_homo['MSR'] / (R_CONST * T))
            r6 = k * C['CH4']
            rates['MSR'] = r6 * volume * 1000.0
        
        return rates

