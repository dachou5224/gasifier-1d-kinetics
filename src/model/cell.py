import numpy as np
import logging
from model.state import StateVector
from model.material import MaterialService, SPECIES_NAMES
from model.kinetics_service import KineticsService
from model.constants import PhysicalConstants

# Set up logging
logger = logging.getLogger(__name__)

class Cell:
    """
    Represents a single Control Volume (CV) in the 1D Gasifier.
    Solves for Outlet State given Inlet State and Source Terms.
    """
    MOLAR_MASSES = np.array([31.998, 16.04, 28.01, 44.01, 34.08, 2.016, 28.013, 18.015])

    def __init__(self, cell_index, z, dz, V, A, inlet_state, kinetics, pyrolysis, coal_props, op_conds, sources=None):
        self.idx = cell_index
        self.z = z
        self.dz = dz
        self.V = V
        self.A = A
        self.inlet = inlet_state
        self.kinetics = kinetics
        self.coal_props = coal_props
        self.op_conds = op_conds
        self.sources = sources if sources is not None else []
        
        # Determine Reference Scales for normalization
        self.ref_flow = max(self.inlet.total_gas_moles, 1.0)
        self.ref_solid = max(self.inlet.solid_mass, 1e-3)
        self.coal_flow_dry = getattr(self, 'coal_flow_dry', 1.0)
        self.char_Xc0 = self.inlet.carbon_fraction

    def _calc_physics_props(self, current, C_fed_total):
        """Calculates physical properties and transport coefficients."""
        # 1. Conversion relative to Original Coal
        C_local = current.solid_mass * current.carbon_fraction
        if C_fed_total < 1e-9: C_fed_total = 1e-9
        X_total = np.clip(1.0 - (C_local / C_fed_total), 0.0, 1.0)
        
        # 2. Particle Diameter (Shrinking Core vs Ash skeleton)
        d_p0 = self.op_conds.get('particle_diameter', 100e-6)
        Ash_d = self.coal_props.get('Ashd', 6.0) / 100.0
        Cd_total = self.coal_props.get('Cd', 60.0) / 100.0
        f_ash_coal = Ash_d / (Cd_total + Ash_d + 1e-9)
        d_p = d_p0 * (f_ash_coal + (1.0 - f_ash_coal) * (1.0 - X_total))**(1/3.0)
        d_p = max(d_p, 1e-6)
        
        # 3. Gas Dynamics & Transport (Based on User Formula: u_g = F_total * R * T / (P * A * epsilon))
        # Where F_total is local molar flow (mol/s)
        F_total = max(current.total_gas_moles, 1e-9)
        T_local = current.T
        P_local = current.P
        epsilon = self.op_conds.get('epsilon', 1.0) # Entrained flow voidage approx 1.0
        
        # u_g [m/s] = (mol/s * J/mol.K * K) / (Pa * m2) = m/s
        v_g = (F_total * 8.314 * T_local) / (P_local * self.A * epsilon + 1e-9)
        
        # Residence Time (Local approach: tau = dz / u_g)
        tau = self.dz / max(v_g, PhysicalConstants.MIN_SLIP_VELOCITY)
        m_hold = current.solid_mass * tau
        S_total = (6 * m_hold) / (PhysicalConstants.PARTICLE_DENSITY * d_p + 1e-9)
        
        # 4. Transport Properties
        from model.physics import calculate_gas_viscosity, calculate_diffusion_coefficient
        rho_gas = (P_local * (np.sum(current.gas_moles * self.MOLAR_MASSES)/F_total) * 1e-3) / (8.314 * T_local)
        mu_g = calculate_gas_viscosity(T_local)
        v_slip = PhysicalConstants.MIN_SLIP_VELOCITY
        Re_p = (rho_gas * v_slip * d_p) / (mu_g + 1e-9)
        D_ref = calculate_diffusion_coefficient(T_local, P_local, 'O2')
        Sc_p = mu_g / (max(rho_gas, 1e-3) * D_ref + 1e-12)
        
        return {
            'X_total': X_total, 'd_p': d_p, 'S_total': S_total, 
            'Re_p': Re_p, 'Sc_p': Sc_p, 'v_g': v_g
        }

    # Fortran 燃烧区判据: pO2 > 0.05 atm (见 docs/fortran_combustion_mechanism.md)
    P_O2_COMBUSTION_THRESHOLD_PA = 5066.0  # 0.05 * 101325

    def _calc_particle_temperature(self, Tg: float, Ts_in: float, tau: float,
                                   d_p: float, solid_mass: float,
                                   carbon_fraction: float,
                                   current: 'StateVector' = None,
                                   phys: dict = None,
                                   gas_src: np.ndarray = None,
                                   is_combustion_zone: bool = False) -> tuple:
        """
        Fortran 式颗粒瞬态传热。
        - 气化区：简单指数衰减 Ts(t+Δt) = Tg - (Tg - Ts(t)) × exp(ct)
        - 燃烧区 (有 O2)：Runge-Kutta-Gill 含反应热源 (Fortran Line 459-475)
        
        Returns:
            (Ts_avg, Ts_out): 时间步内平均温度（用于反应速率）、出口颗粒温度
        """
        nc = PhysicalConstants.TS_TRANSIENT_NC
        deltim = tau / max(nc, 1)
        
        r = d_p / 2.0  # 半径 [m]
        dens = PhysicalConstants.PARTICLE_DENSITY
        cps = self.coal_props.get('cp_char', PhysicalConstants.HEAT_CAPACITY_SOLID)
        ef = PhysicalConstants.EF_PARTICLE_TRANSIENT
        sigma = PhysicalConstants.STEFAN_BOLTZMANN  # W/(m²·K⁴)
        condut_coeff = PhysicalConstants.CONDUT_COEFF
        
        ts = Ts_in
        Ts_sum = 0.0
        
        # 燃烧区：RK-Gill 含 C+O2 放热（Ts 升高）；气化区：含 C+H2O/C+CO2 吸热（Ts 降低，可抑制 WGS）
        use_rk_gill = (PhysicalConstants.USE_RK_GILL_COMBUSTION and
                       current is not None and phys is not None and gas_src is not None)
        
        for step in range(nc):
            if use_rk_gill:
                try:
                    ts_new = self._step_particle_temp_rkgill(
                        Tg, ts, deltim, r, dens, cps, ef, sigma, condut_coeff,
                        current, phys, gas_src
                    )
                    if np.isfinite(ts_new):
                        ts = ts_new
                    else:
                        use_rk_gill = False
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    use_rk_gill = False
                if not use_rk_gill:
                    condut = condut_coeff * ((Tg + ts) ** 0.75)
                    rad_term = ef * sigma * 4.0 * (Tg ** 3)
                    ct = -(3.0 / (dens * cps * r + 1e-12)) * (condut / r + rad_term) * deltim
                    ct = np.clip(ct, -25.0, 25.0)
                    ect = np.exp(ct)
                    delts = (Tg - (Tg - ts) * ect) - ts
                    tsm = ts + delts / 2.0
                    ts = ts + delts
                else:
                    tsm = ts
            else:
                condut = condut_coeff * ((Tg + ts) ** 0.75)
                rad_term = ef * sigma * 4.0 * (Tg ** 3)
                ct = -(3.0 / (dens * cps * r + 1e-12)) * (condut / r + rad_term) * deltim
                ct = np.clip(ct, -25.0, 25.0)
                ect = np.exp(ct)
                delts = (Tg - (Tg - ts) * ect) - ts
                tsm = ts + delts / 2.0
                ts = ts + delts
            
            tsm = min(tsm, PhysicalConstants.TS_MAX_FOR_RATES)
            Ts_sum += tsm
        
        Ts_avg = Ts_sum / nc
        Ts_out = ts
        return Ts_avg, Ts_out

    def _step_particle_temp_rkgill(self, Tg: float, ts: float, deltim: float,
                                   r: float, dens: float, cps: float,
                                   ef: float, sigma: float, condut_coeff: float,
                                   current: 'StateVector', phys: dict,
                                   gas_src: np.ndarray) -> float:
        """
        Fortran Runge-Kutta-Gill 单步 (Line 459-475).
        dTs/dt = (3/(dens*cps*r)) * (conv+rad + qrh*rate - qcsm*rate2/12 - qcbco2*rate3/12)
        """
        CAL2J = 4.184

        def _dts_dt(ts_val):
            ts_val = max(min(ts_val, 3500.0), 273.0)
            r_het, _, _ = self._calc_rates(current, phys, gas_src, Ts_avg=ts_val)
            q1 = -94051.0 - 3.964*(ts_val-298.) + 3.077e-3*(ts_val**2-298.**2) - 0.874e-6*(ts_val**3-298.**3)
            q2 = -26414.0 - 0.684*(ts_val-298.) - 0.513e-3*(ts_val**2-298.**2) + 8.85e-8*(ts_val**3-298.**3)
            z = 2500.0 * np.exp(-6249.0 / ((Tg+ts_val)/2.0 + 1e-9))
            phi_val = (2.*z+2.) / (z+2.)
            qrh = (-(q1/12.)*(2./phi_val-1.) - (q2/12.)*(2.-2./phi_val)) * CAL2J
            qcsm = (31382. + 2.011*(ts_val-298.) - 0.733e-3*(ts_val**2-298.**2)/2.) * CAL2J
            qcbco2 = (41220. + 2.256*(ts_val-298.) - 7.066e-3*(ts_val**2-298.**2)/2. +
                     3.153e-6*(ts_val**3-298.**3)/3.) * CAL2J
            rate_C_O2 = r_het.get('C+O2', 0.0)
            rate_C_H2O = r_het.get('C+H2O', 0.0)
            rate_C_CO2 = r_het.get('C+CO2', 0.0)
            condut = condut_coeff * ((Tg + ts_val) ** 0.75)
            conv_rad = condut * (Tg - ts_val) / r + ef * sigma * (Tg**4 - ts_val**4)
            Q_rxn = qrh * rate_C_O2 - qcsm * rate_C_H2O - qcbco2 * rate_C_CO2
            S_total = max(phys['S_total'], 1e-12)
            Q_rxn_per_area = Q_rxn / S_total
            coef = 3.0 / (dens * cps * r + 1e-12)
            val = coef * (conv_rad + Q_rxn_per_area)
            return float(np.clip(val, -1e5, 1e5))

        tsolid = [ts]
        rk = [0.0] * 5
        for l in range(1, 5):
            ts_stage = (tsolid[0] + 0.5*deltim*rk[1] if l == 2 else
                        tsolid[0] + 0.2071*deltim*rk[1] + 0.2929*deltim*rk[2] if l == 3 else
                        tsolid[0] - 0.7071*deltim*rk[2] + 1.7071*deltim*rk[3] if l == 4 else
                        tsolid[0])
            ts_stage = max(min(ts_stage, 3500.0), 273.0)
            rk[l] = _dts_dt(ts_stage)
        
        ts_new = tsolid[0] + (deltim/6.) * (rk[1] + 0.58578*rk[2] + 3.41422*rk[3] + rk[4])
        ts_new = max(min(float(ts_new), 3500.0), 273.0)
        if not np.isfinite(ts_new):
            return tsolid[0]
        return ts_new

    def _instant_volatile_combustion(self, avail: dict) -> tuple:
        """
        Fortran 式挥发分瞬时燃烧：仅用化学计量 + O2  availability，无动力学。
        优先顺序（O2 不足时）: CH4 > CO > H2（文献 tar > CH4 > CO > H2，无 tar 时 CH4 优先）
        """
        O2 = avail['O2']
        CH4 = avail['CH4'] * 0.99
        CO = avail['CO'] * 0.99
        H2 = avail['H2'] * 0.99

        O2_demand_total = CH4 * 2.0 + CO * 0.5 + H2 * 0.5
        if O2 >= O2_demand_total:
            return CH4, CO, H2  # 完全燃烧

        # O2 不足：按优先级 CH4 > CO > H2 分配
        remaining_O2 = O2
        r_CH4 = min(CH4, remaining_O2 / 2.0)
        remaining_O2 -= r_CH4 * 2.0
        r_CO = min(CO, max(remaining_O2, 0.0) / 0.5)
        remaining_O2 -= r_CO * 0.5
        r_H2 = min(H2, max(remaining_O2, 0.0) / 0.5)
        return r_CH4, r_CO, r_H2

    def _calc_rates(self, current, phys, gas_src, Ts_avg: float = None):
        """Calculates Heterogeneous and Homogeneous reaction rates.
        Ts_avg: 颗粒平均温度，用于异相反应；None 时用 current.T（向后兼容）
        """
        T_het = Ts_avg if Ts_avg is not None else current.T
        # 1. Base Rates：异相用 Ts_avg，均相用 Tg (current.T)
        r_het = self.kinetics.calc_heterogeneous_rates(
            current, phys['d_p'], phys['S_total'],
            X_total=phys['X_total'], Re=phys['Re_p'], Sc=phys['Sc_p'],
            T_particle=T_het
        )
        phi = r_het.pop('phi', 1.0)
        r_homo = self.kinetics.calc_homogeneous_rates(
            current, self.V,
            inlet_state=self.inlet,
            gas_src=gas_src,
            Ts_particle=T_het  # Fortran wgshift: ts<=1000K 时不计 WGS
        )

        avail = {sp: max(self.inlet.gas_moles[i] + gas_src[i], 0.0) for i, sp in enumerate(SPECIES_NAMES)}
        O2_budget = avail['O2']
        raw_C_O2 = r_het['C+O2']

        # ========================================================================
        # Fortran 分区: poxyin > 0.05 atm → 燃烧区（挥发分瞬时完全燃烧）
        # 使用入口 pO2（与 Fortran Line 345 poxyin 一致）
        # ========================================================================
        F_total_in = sum(avail.values())
        F_total_in = max(F_total_in, 1e-9)
        pO2_Pa = current.P * (avail['O2'] / F_total_in)
        is_combustion_zone = pO2_Pa > self.P_O2_COMBUSTION_THRESHOLD_PA

        if is_combustion_zone:
            # 燃烧区：挥发分瞬时燃烧（无动力学），CH4 > CO > H2 分配 O2
            r_CH4, r_CO, r_H2 = self._instant_volatile_combustion(avail)
            r_homo['CH4_Ox'] = r_CH4
            r_homo['CO_Ox'] = r_CO
            r_homo['H2_Ox'] = r_H2
            O2_after_vol = O2_budget - (r_CH4 * 2.0 + r_CO * 0.5 + r_H2 * 0.5)
            # Char-O2 使用剩余 O2
            r_het['C+O2'] = min(raw_C_O2, max(O2_after_vol, 0.0) * phi)
        else:
            # 气化区：挥发分不氧化
            r_homo['CH4_Ox'] = 0.0
            r_homo['CO_Ox'] = 0.0
            r_homo['H2_Ox'] = 0.0
            r_het['C+O2'] = min(raw_C_O2, O2_budget * phi)
        
        # ========================================================================
        # NON-O2 REACTION LIMITS (GASIFICATION & MSR)
        # ========================================================================
        
        # 5. Char Gasification
        r_het['C+H2O'] = min(r_het['C+H2O'], avail['H2O'] * 0.99)
        r_het['C+CO2'] = min(r_het['C+CO2'], avail['CO2'] * 0.99)
        r_het['C+H2'] = min(r_het['C+H2'], avail['H2'] * 0.5 * 0.99) # C + 2H2 -> CH4
        
        # 6. MSR: CH4 + H2O -> CO + 3H2
        # Remaining reactants after oxidation/gasification
        rem_CH4 = max(avail['CH4'] - r_homo['CH4_Ox'], 0.0)
        rem_H2O = max(avail['H2O'] - r_het['C+H2O'], 0.0)
        r_homo['MSR'] = min(r_homo['MSR'], rem_CH4 * 0.99, rem_H2O * 0.99)
        
        # Debug: Show budgets for Cell 0
        if self.idx == 0:
            zone = "combustion" if is_combustion_zone else "gasification"
            logger.info(f"[BUDGET] zone={zone} pO2_in={pO2_Pa/101325:.3f}atm O2={avail['O2']:.2f}")
            logger.info(f"         Volatile Ox (instant): CH4={r_homo['CH4_Ox']:.2f} H2={r_homo['H2_Ox']:.2f} CO={r_homo['CO_Ox']:.2f}")
            logger.info(f"         Het: C+O2={r_het['C+O2']:.2f}, C+H2O={r_het['C+H2O']:.2f}")
        
        # WGS/RWGS are typically slower, no constraint needed

        return r_het, r_homo, phi





    def _calc_balances(self, current, r_het, r_homo, phi, gas_src, solid_src):
        """Computes component mass residuals."""
        r1, r2, r3 = r_homo['CO_Ox'], r_homo['H2_Ox'], r_homo['WGS']
        r4, r5, r6 = r_homo['RWGS'], r_homo['CH4_Ox'], r_homo['MSR']
        rCOmb, rH2Og, rCO2g, rH2g = r_het['C+O2'], r_het['C+H2O'], r_het['C+CO2'], r_het['C+H2']
        
        # Gas Balances
        res_O2 = current.gas_moles[0] - (self.inlet.gas_moles[0] + gas_src[0] - (rCOmb/phi + 0.5*r1 + 0.5*r2 + 2.0*r5))
        res_CH4 = current.gas_moles[1] - (self.inlet.gas_moles[1] + gas_src[1] + rH2g - (r5 + r6))
        res_CO = current.gas_moles[2] - (self.inlet.gas_moles[2] + gas_src[2] + (2.0-2.0/phi)*rCOmb + rH2Og + 2*rCO2g + r4 + r6 - (r1 + r3))
        res_CO2 = current.gas_moles[3] - (self.inlet.gas_moles[3] + gas_src[3] + (2.0/phi-1.0)*rCOmb + r1 + r3 + r5 - (rCO2g + r4))
        res_H2S = current.gas_moles[4] - (self.inlet.gas_moles[4] + gas_src[4])
        res_H2 = current.gas_moles[5] - (self.inlet.gas_moles[5] + gas_src[5] + rH2Og + r3 + 3*r6 - (2*rH2g + r2 + r4))
        res_N2 = current.gas_moles[6] - (self.inlet.gas_moles[6] + gas_src[6])
        res_H2O = current.gas_moles[7] - (self.inlet.gas_moles[7] + gas_src[7] + r2 + r4 + 2*r5 - (rH2Og + r3 + r6))
        
        # Solid Mass Balance (CRITICAL: include solid_src)
        W_surf_loss = (rCOmb + rH2Og + rCO2g + rH2g) * 0.012011
        res_Ws = current.solid_mass - (self.inlet.solid_mass + solid_src - W_surf_loss)
        
        # Carbon Balance for Xc
        C_loss_pyro_mols = gas_src[1] + gas_src[2] + gas_src[3] # CH4, CO, CO2 from pyro
        C_loss_pyro_kg = C_loss_pyro_mols * 0.012011
        C_in_solid = self.inlet.solid_mass * self.inlet.carbon_fraction
        C_out_solid = current.solid_mass * current.carbon_fraction
        res_Xc = C_out_solid - (C_in_solid - C_loss_pyro_kg - W_surf_loss)
        
        gas_res = [res_O2, res_CH4, res_CO, res_CO2, res_H2S, res_H2, res_N2, res_H2O]
        return gas_res, res_Ws, res_Xc

    def _calc_energy_balance(self, current, energy_src, phys=None, r_het=None, r_homo=None, phi=None, Ts_out=None, gas_src=None):
        """
        Computes enthalpy residual for the cell.
        Ts_out: 出口颗粒温度；若提供则 H_out 固相用 Ts_out，否则用 current.T
        gas_src: 保留接口（temperature_diagnosis 兼容）
        """
        H_in = MaterialService.get_total_enthalpy(self.inlet, self.coal_props)  # W (J/s)
        H_out = MaterialService.get_total_enthalpy(current, self.coal_props, T_solid_override=Ts_out)
        
        # === Detailed Energy Audit ===
        H_gas_in = MaterialService.get_gas_enthalpy(self.inlet)
        H_solid_in = MaterialService.get_solid_enthalpy(self.inlet, self.coal_props, T_solid_override=self.inlet.T_solid_or_gas)
        H_gas_out = MaterialService.get_gas_enthalpy(current)
        H_solid_out = MaterialService.get_solid_enthalpy(current, self.coal_props, T_solid_override=Ts_out)
        
        # Heat Loss: 文献为入炉煤 HHV 的 2%，按 dz 比例摊分到各格
        L_total = self.op_conds.get('L_reactor', 6.0)  # m
        loss_pct = self.op_conds.get('HeatLossPercent', 2.0)  # % of inlet coal HHV
        
        coal_flow_kg_s = self.op_conds['coal_flow']  # kg/s
        hhv_MJ_kg = self.coal_props.get('HHV_d', 30.0)  # MJ/kg (or kJ/kg, normalized below)
        
        if hhv_MJ_kg > 1000.0:
            hhv_MJ_kg = hhv_MJ_kg / 1000.0
        
        Q_total_W = coal_flow_kg_s * hhv_MJ_kg * 1e6  # total coal power (W), base for heat loss
        Q_loss_W = (loss_pct / 100.0) * Q_total_W * (self.dz / L_total)
        
        # Reaction heat from rates (for audit only)
        Q_rxn_total = 0.0
        if r_homo is not None:
            # Heats of reaction (J/mol), exothermic = positive
            Q_rxn_CO_ox = r_homo.get('CO_Ox', 0.0) * 283000.0
            Q_rxn_H2_ox = r_homo.get('H2_Ox', 0.0) * 241800.0
            Q_rxn_CH4_ox = r_homo.get('CH4_Ox', 0.0) * 802000.0
            Q_rxn_homo = Q_rxn_CO_ox + Q_rxn_H2_ox + Q_rxn_CH4_ox
            Q_rxn_total += Q_rxn_homo
        
        if r_het is not None and phi is not None:
            # C + O2 -> CO/CO2 (phi determines ratio)
            Q_rxn_C_O2 = r_het.get('C+O2', 0.0) * 393510.0 * phi
            Q_rxn_total += Q_rxn_C_O2
        
        delta_H = H_out - H_in
        
        # Audit logging (only for first few cells to avoid spam)
        if self.idx < 5:
            logger.info(f"=== Cell {self.idx} Energy Audit ===")
            logger.info(f"  H_in:  Gas={H_gas_in/1e6:.2f} MW, Solid={H_solid_in/1e6:.2f} MW, Total={H_in/1e6:.2f} MW")
            logger.info(f"  H_out: Gas={H_gas_out/1e6:.2f} MW, Solid={H_solid_out/1e6:.2f} MW, Total={H_out/1e6:.2f} MW")
            logger.info(f"  ΔH = H_out - H_in = {delta_H/1e6:.2f} MW")
            logger.info(f"  Q_reaction (from rates) = {Q_rxn_total/1e6:.2f} MW")
            # Debug breakdown
            if r_homo is not None:
                Q_CO = r_homo.get('CO_Ox', 0.0) * 283000.0 / 1e6
                Q_H2 = r_homo.get('H2_Ox', 0.0) * 241800.0 / 1e6
                Q_CH4 = r_homo.get('CH4_Ox', 0.0) * 802000.0 / 1e6
                logger.info(f"    Homo: CO_Ox={Q_CO:.2f} MW, H2_Ox={Q_H2:.2f} MW, CH4_Ox={Q_CH4:.2f} MW")
            if r_het is not None:
                Q_C_O2 = r_het.get('C+O2', 0.0) * 393510.0 * (phi if phi else 1.0) / 1e6
                logger.info(f"    Het:  C+O2={Q_C_O2:.2f} MW (r_C_O2={r_het.get('C+O2', 0.0):.2f} mol/s)")
            logger.info(f"  Ratio ΔH/Q_rxn = {delta_H/(Q_rxn_total+1e-9):.2f} (should be ~-1 for exothermic)")
            logger.info(f"  energy_src = {energy_src/1e6:.2f} MW")
            logger.info(f"  Q_loss = {Q_loss_W/1e6:.4f} MW")

        
        # Energy Balance: 0 = H_in + Source - H_out - Loss
        return H_out - (H_in + energy_src - Q_loss_W)


    def residuals(self, x_flat):
        """Standard Solver Interface."""
        current = StateVector.from_array(x_flat, P=self.inlet.P, z=self.z)
        
        # Aggregate Sources
        g_src, s_src, e_src = np.zeros(8), 0.0, 0.0
        for s in self.sources:
            g, s_m, e = s.get_sources(self.idx, self.z, self.dz)
            g_src += g
            s_src += s_m
            e_src += e
            
        C_fed = self.coal_flow_dry * (self.coal_props.get('Cd', 60.0)/100.0)
        phys = self._calc_physics_props(current, C_fed)
        
        # Fortran 式：Tg → Ts(瞬态求解) → 反应速率
        Ts_in = self.inlet.T_solid_or_gas
        tau = self.dz / max(phys['v_g'], PhysicalConstants.MIN_SLIP_VELOCITY)
        F_total_in = max(sum(self.inlet.gas_moles[i] + g_src[i] for i in range(8)), 1e-9)
        pO2_Pa = current.P * (max(self.inlet.gas_moles[0] + g_src[0], 0) / F_total_in)
        is_combustion = pO2_Pa > self.P_O2_COMBUSTION_THRESHOLD_PA
        
        Ts_avg, Ts_out = self._calc_particle_temperature(
            current.T, Ts_in, tau, phys['d_p'],
            current.solid_mass, current.carbon_fraction,
            current=current, phys=phys, gas_src=g_src,
            is_combustion_zone=is_combustion
        )
        
        r_het, r_homo, phi = self._calc_rates(current, phys, g_src, Ts_avg=Ts_avg)
        res_gas, res_Ws, res_Xc = self._calc_balances(current, r_het, r_homo, phi, g_src, s_src)
        res_E = self._calc_energy_balance(current, e_src, phys=phys, r_het=r_het, r_homo=r_homo, phi=phi, Ts_out=Ts_out, gas_src=g_src)
        
        # Scaling: most species by ref_flow; N2 (index 6) by coal-N scale so solver enforces N conservation
        # Pure O2 gasification: N in syngas comes only from coal (pyrolysis N2). Small N2 flow => weak residual if scaled by ref_flow.
        ref_N2 = max(abs(self.inlet.gas_moles[6]) + abs(g_src[6]), 1e-3)
        res_gas_sc = []
        for i, r in enumerate(res_gas):
            if i == 6:
                res_gas_sc.append(r / ref_N2)
            else:
                res_gas_sc.append(r / self.ref_flow)
        res_Ws_sc = res_Ws / self.ref_solid
        res_Xc_sc = res_Xc / max(self.char_Xc0, 0.1)
        # 能量残差相对放大，避免被质量残差主导陷入低温解（temperature_diagnosis）
        res_E_sc = res_E / 5.0e5
        
        return np.concatenate([res_gas_sc, [res_Ws_sc, res_Xc_sc, res_E_sc]])



    def diagnose_failure(self, x_flat):
        """Detailed print logging for failures."""
        current = StateVector.from_array(x_flat, P=self.inlet.P, z=self.z)
        g_src, s_src, e_src = np.zeros(8), 0.0, 0.0
        for s in self.sources:
            g, sm, e = s.get_sources(self.idx, self.z, self.dz)
            g_src += g; s_src += sm; e_src += e
            
        C_fed = self.coal_flow_dry * (self.coal_props.get('Cd', 60.0)/100.0)
        phys = self._calc_physics_props(current, C_fed)
        Ts_in = self.inlet.T_solid_or_gas
        tau = self.dz / max(phys['v_g'], PhysicalConstants.MIN_SLIP_VELOCITY)
        Ts_avg, _ = self._calc_particle_temperature(
            current.T, Ts_in, tau, phys['d_p'],
            current.solid_mass, current.carbon_fraction
        )
        r_het, r_homo, phi = self._calc_rates(current, phys, g_src, Ts_avg=Ts_avg)
        
        # Recalculate H for log
        H_in = MaterialService.get_total_enthalpy(self.inlet, self.coal_props)
        H_out = MaterialService.get_total_enthalpy(current, self.coal_props)
        
        v_g = phys['v_g']
        tau = self.dz / max(v_g, 1e-3)
        
        logger.error(f"--- Cell {self.idx} Diagnostic (T={current.T:.1f} K) ---")
        logger.error(f"  Fluid Dynamics: u_g={v_g:.2f} m/s, tau={tau:.4f} s, dz={self.dz:.3f} m")
        logger.error(f"  Enthalpy Balance: H_in={H_in/1e6:.2f}, H_out={H_out/1e6:.2f}, Src={e_src/1e6:.2f} MW")
        logger.error(f"  Residual: {(H_out - (H_in + e_src))/1e6:.2f} MW")
        logger.error(f"  Reaction Heat Potential (CH4_Ox): {r_homo['CH4_Ox']*802340.0/1e6:.2f} MW")
        for i, sp in enumerate(SPECIES_NAMES):
            if abs(current.gas_moles[i]) > 1e-3 or abs(g_src[i]) > 1e-3:
                logger.error(f"    {sp:4}: {self.inlet.gas_moles[i]:8.1f} -> {current.gas_moles[i]:8.1f} (Src: {g_src[i]:8.1f})")
        logger.error(f"  Solid: {self.inlet.solid_mass:8.3f} -> {current.solid_mass:8.3f} (Src: {s_src:8.3f})")
        
        # ✅ Detailed Mass Balance Check
        logger.error(f"=== Cell {self.idx} Mass Balance Diagnostic ===")
        
        # Available amounts
        avail = [max(self.inlet.gas_moles[i] + g_src[i], 0.0) for i in range(8)]
        
        # CH4 Balance
        r_CH4_consumed = r_homo['CH4_Ox'] + r_homo['MSR']
        r_CH4_produced = r_het['C+H2']
        CH4_in = self.inlet.gas_moles[1]
        CH4_src = g_src[1]
        CH4_out = current.gas_moles[1]
        CH4_available = avail[1]
        
        logger.error(f"CH4 Balance:")
        logger.error(f"  Inlet:     {CH4_in:.2f} mol/s")
        logger.error(f"  Source:    {CH4_src:.2f} mol/s")
        logger.error(f"  Available: {CH4_available:.2f} mol/s")
        logger.error(f"  Produced:  {r_CH4_produced:.2f} mol/s")
        logger.error(f"  Consumed:  {r_CH4_consumed:.2f} mol/s")
        logger.error(f"  Outlet:    {CH4_out:.2f} mol/s")
        logger.error(f"  Expected:  {CH4_in + CH4_src + r_CH4_produced - r_CH4_consumed:.2f} mol/s")
        logger.error(f"  Residual:  {CH4_out - (CH4_in + CH4_src + r_CH4_produced - r_CH4_consumed):.2e} mol/s")
        logger.error(f"  Consumption/Available Ratio: {r_CH4_consumed / (CH4_available + 1e-9):.2f}")
        
        # O2 Balance
        r_O2_consumed = r_het['C+O2']/phi + 0.5*r_homo['CO_Ox'] + 0.5*r_homo['H2_Ox'] + 2.0*r_homo['CH4_Ox']
        O2_in = self.inlet.gas_moles[0]
        O2_src = g_src[0]
        O2_out = current.gas_moles[0]
        O2_available = avail[0]
        
        logger.error(f"O2 Balance:")
        logger.error(f"  Inlet:     {O2_in:.2f} mol/s")
        logger.error(f"  Source:    {O2_src:.2f} mol/s")
        logger.error(f"  Available: {O2_available:.2f} mol/s")
        logger.error(f"  Consumed:  {r_O2_consumed:.2f} mol/s")
        logger.error(f"  Outlet:    {O2_out:.2f} mol/s")
        logger.error(f"  Expected:  {O2_in + O2_src - r_O2_consumed:.2f} mol/s")
        logger.error(f"  Residual:  {O2_out - (O2_in + O2_src - r_O2_consumed):.2e} mol/s")
        logger.error(f"  Consumption/Available Ratio: {r_O2_consumed / (O2_available + 1e-9):.2f}")
        
        logger.error(f"==========================================")

