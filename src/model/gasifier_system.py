import numpy as np
from scipy.optimize import least_squares
from .state import StateVector
from .cell import Cell
from .material import MaterialService
from .kinetics_service import KineticsService
from .pyrolysis_service import PyrolysisService
from .constants import PhysicalConstants
from .grid_service import AdaptiveMeshGenerator, MeshConfig
from .source_terms import EvaporationSource, PyrolysisSource
from .solver import NewtonSolver
from .jax_residual_adapter import make_jacobian_fn
from .jax_solver import (
    newton_solve_cell_numpy,
    newton_solve_multistart_numpy,
    newton_solve_cell_pure_jax_ad,
    newton_solve_multistart_cell_pure_jax_ad,
    warmup_jax,
)
import logging

# Set up logging
logger = logging.getLogger(__name__)


class GasifierSystem:
    def __init__(self, geometry, coal_props, op_conds):
        # [VALIDATION] Input integrity check
        self._validate_inputs(geometry, coal_props, op_conds)
        
        self.geometry = geometry
        self.coal_props = coal_props
        self.op_conds = op_conds.copy()
        
        # Ensure Hf_coal is consistent with HHV (Critical for Energy Balance)
        if 'Hf_coal' not in self.coal_props:
            # 1. Calculate LHV_d (Lower Heating Value Dry)
            # HHV_d is given. LHV_d = HHV_d - (Latent Heat of Water from H)
            # Latent heat of water at 25C ~ 44 kJ/mol = 2442 kJ/kg_water
            Cd = self.coal_props.get('Cd', 0.0)/100.0
            Hd = self.coal_props.get('Hd', 0.0)/100.0
            Sd = self.coal_props.get('Sd', 0.0)/100.0
            
            hhv_val = self.coal_props.get('HHV_d', 30.0)
            HHV_d_kJ = hhv_val if hhv_val > 1000.0 else hhv_val * 1000.0
            
            # LHV_d = HHV_d - 9*Hd * (Hh2o_vap / Mw_h2o)
            LHV_d_kJ = HHV_d_kJ - (9.0 * Hd * 2442.0)
            self.coal_props['LHV_d_kJ'] = LHV_d_kJ

            # 2. Hf_coal (Formation Enthalpy)
            # Basis: Q_comb (LHV) = Sum(Hf_prod) - Hf_coal
            # => Hf_coal = Sum(Hf_prod) - LHV_d
            H_prod_gas_kJ = (Cd/0.012011)*(-393.51) + \
                            (Hd/0.001008)*0.5*(-241.83) + \
                            (Sd/0.03206)*(-296.81)
            
            Hf_coal_kJ = H_prod_gas_kJ + LHV_d_kJ
            self.coal_props['Hf_coal'] = Hf_coal_kJ * 1000.0 # J/kg
            logger.info(f"Consistent Hf_coal: {self.coal_props['Hf_coal']/1e6:.2f} MJ/kg (Basis: LHV_d={LHV_d_kJ/1000.0:.2f} MJ/kg)")
        
        # Initialize Services
        self.material = MaterialService()
        self.kinetics = KineticsService()
        self.pyrolysis = PyrolysisService()
        
        self.results = []
        self.cells = []
        self.solve_stats = {"fallback_count": 0, "poor_convergence_count": 0}

    def _validate_inputs(self, geometry, coal_props, op_conds):
        """
        Validates the integrity of input dictionaries.
        Raises ValueError if required parameters are missing or invalid.
        """
        # 1. Geometry
        required_geom = ['L', 'D']
        missing_geom = [k for k in required_geom if k not in geometry]
        if missing_geom:
            raise ValueError(f"Missing geometry parameters: {missing_geom}")
        
        if geometry['L'] <= 0 or geometry['D'] <= 0:
            raise ValueError("Geometry dimensions L and D must be positive.")

        # 2. Operating Conditions
        required_op = ['coal_flow', 'o2_flow', 'P', 'T_in']
        missing_op = [k for k in required_op if k not in op_conds]
        if missing_op:
            raise ValueError(f"Missing operating conditions: {missing_op}")
            
        if op_conds['coal_flow'] < 0 or op_conds['o2_flow'] < 0:
            raise ValueError("Flow rates cannot be negative.")
        if op_conds['P'] <= 0 or op_conds['T_in'] <= 0:
            raise ValueError("Pressure and Temperature must be positive.")

        # 3. Coal Properties (Basic integrity)
        if 'Cd' not in coal_props:
             raise ValueError("Coal property 'Cd' (Carbon dry basis) is required.")

    def _initialize_inlet(self):
        """
        Prepare Cell 0 Inlet State.
        Now simplified: Raw Coal + Oxidant Only.
        Moisture and Volatiles are handled via SourceTerms in Cell 0.
        """
        # 1. Physical Feed Components (Dry Coal)
        W_wet_coal = self.op_conds['coal_flow'] 
        Mt = self.coal_props.get('Mt', 0.0)
        W_dry = W_wet_coal * (1 - Mt/100.0)
        
        # 2. Moisture Calculation (for SourceTerm later)
        W_coal_moist = W_wet_coal * (Mt/100.0)
        
        # 3. Slurry Water (liquid part only; additional to coal moisture)
        slurry_conc = self.op_conds.get('SlurryConcentration', 100.0)
        if slurry_conc < 100.0:
            W_slurry_h2o = (W_dry / (slurry_conc/100.0)) - W_dry
        else:
            W_slurry_h2o = 0.0
        
        # 4. External steam (already vapor, no latent heat)
        W_steam = self.op_conds.get('steam_flow', 0.0)
        
        # Total water for mass balance (liquid + steam)
        W_h2o_total = W_coal_moist + W_slurry_h2o + W_steam
        
        # Store for Source Term Factory
        self.tmp_W_h2o_total = W_h2o_total
        # Only liquid water (moisture + slurry) should carry latent heat sink
        self.tmp_W_liq_evap = W_coal_moist + W_slurry_h2o
        
        # 5. Oxidant + Steam (Gas)
        MW_O2 = 31.998
        F_O2 = (self.op_conds['o2_flow'] / MW_O2) * 1000.0  # mol/s
        F_N2_ox = self.op_conds.get('n2_flow', 0.0) / 28.013 * 1000.0
        MW_H2O = 18.015
        F_steam = (W_steam / MW_H2O) * 1000.0  # mol/s (pre-vaporized steam)
        
        # 6. Assemble Initial Gas State (mol/s)
        # Oxidant and any pre-vaporized steam enter as gas.
        # [P2-6] 入口 H2O 焓基准确认：gas_moles[7]=F_steam 仅含外部蒸汽（已气化），
        # 其焓由 MaterialService.get_gas_enthalpy 用 Shomate H_gas(T_inlet) 计算，正确。
        # 浆液/煤中水分(tmp_W_liq_evap)由 EvaporationSource 单独加入，使用 H_LIQUID，
        # 二者无重叠，无双重计算。
        gas_moles = np.zeros(8)
        gas_moles[0] = F_O2
        gas_moles[6] = F_N2_ox
        gas_moles[7] = F_steam
        
        # 5. Solid State (Raw Dry Coal)
        # Volatiles are still IN the solid at inlet.
        W_solid_in = W_dry
        
        # Carbon Fraction in Raw Coal
        Cd_total = self.coal_props.get('Cd', 60.0) / 100.0
        # Xc matches Cd initially
        Xc_in = Cd_total
        
        inlet = StateVector(
            gas_moles=gas_moles,
            solid_mass=W_solid_in,
            carbon_fraction=Xc_in,
            T=self.op_conds['T_in'],
            P=self.op_conds['P'],
            z=0.0
        )
        
        # Store metadata
        self.W_dry = W_dry
        self.Cd_total = Cd_total
        self.char_Xc0 = Xc_in # Initial Xc is raw coal Xc
        
        # Pre-calc Pyrolysis for Source Term
        molar_yields_per_kg, W_vol_mass_per_kg = self.pyrolysis.calc_yields(self.coal_props)

        self.tmp_F_volatiles = molar_yields_per_kg * W_dry
        self.tmp_W_vol_loss = W_dry * W_vol_mass_per_kg

        
        # Update Char Xc0 just for reference? 
        # Actually, once volatiles leave, the REMAINING solid is Char.
        # But inlet is Coal. Cell 0 output will be Char.
        # We need to know what the target Char Xc is for consistency? 
        # Let's let the mass balance handle it.
        # But we need 'self.char_Xc0' for scaling residuals.
        # Let's estimate Char Xc based on assumption that all volatiles leave.
        C_in_volatiles = np.sum(self.tmp_F_volatiles[1:4]) * 0.012011 + self.tmp_F_volatiles[2]*0.012011 # Wait, indices.
        # [O2, CH4, CO, CO2...]
        # C in CH4(1), CO(2), CO2(3)
        C_in_vol_kg = (self.tmp_F_volatiles[1] + self.tmp_F_volatiles[2] + self.tmp_F_volatiles[3]) * 0.012011
        C_total = W_dry * Cd_total
        C_char = C_total - C_in_vol_kg
        W_char = W_dry - self.tmp_W_vol_loss
        self.char_Xc0 = C_char / max(W_char, 1e-9)
        
        return inlet

    def solve(
        self,
        N_cells=100,
        solver_method='minimize',
        use_jax_jacobian=False,
        jax_warmup=True,
    ):
        """
        Sequential Solver.

        use_jax_jacobian: True 时对 least_squares 注入中心差分 Jacobian，NewtonSolver 用 centered 模式。
        solver_method='jax_newton': 使用 jax_solver 模块中的阻尼 Newton（NumPy 残差 + 中心差分 J），
            失败时回退到带 Jacobian 的 least_squares。
        jax_warmup: 首次求解前预热 JAX（摊销导入与轻量编译）。
        """
        L = self.geometry['L']
        D = self.geometry['D']
        A = np.pi * (D/2)**2
        V_total = A * L
        
        # [GRID STRATEGY] Variable Mesh via Service
        # Cell 0 尺寸：tau=dz/v_g，高流量时 v_g 大、τ 短，需更大 dz 保证挥发分燃尽
        # 见 docs/cell0_ignition_analysis.md
        dz_cell0 = self.op_conds.get('FirstCellLength')
        if dz_cell0 is None and self.op_conds.get('AdaptiveFirstCellLength', False):
            m_coal_g_s = self.op_conds['coal_flow'] * 1000.0
            m_ref = 77.0  # Texaco pilot g/s
            dz_base = PhysicalConstants.FIRST_CELL_LENGTH
            dz_cell0 = dz_base * (m_coal_g_s / m_ref) ** 0.4
            dz_cell0 = max(0.03, min(0.5, dz_cell0))
        if dz_cell0 is None:
            dz_cell0 = PhysicalConstants.FIRST_CELL_LENGTH
        dz_ignition = self.op_conds.get('IgnitionZoneDz', PhysicalConstants.IGNITION_ZONE_DZ)
        ignition_zone_total_length = self.op_conds.get('IgnitionZoneTotalLength')
        ignition_zone_fine_start_z = self.op_conds.get('IgnitionZoneFineStartZ')
        ignition_zone_stretch_ratio = self.op_conds.get('IgnitionZoneStretchRatio', 1.0)
        mesh_cfg = MeshConfig(
            total_length=L,
            n_cells=N_cells,
            ignition_zone_length=dz_cell0,
            ignition_zone_res=dz_ignition,
            ignition_zone_total_length=ignition_zone_total_length,
            ignition_zone_fine_start_z=ignition_zone_fine_start_z,
            ignition_zone_stretch_ratio=ignition_zone_stretch_ratio,
            min_grid_size=PhysicalConstants.MIN_GRID_SIZE
        )
        generator = AdaptiveMeshGenerator(mesh_cfg)
        dz_list, self.z_positions = generator.generate()
        
        # Init with Instant Pyrolysis
        current_inlet = self._initialize_inlet()
        
        # [DEBUG] Verify Volatile Flux
        _f = current_inlet.gas_moles
        # [DEBUG] Verify Volatile Flux
        _f = current_inlet.gas_moles
        logger.debug(f"Volatile Flux (mol/s) -> CH4: {_f[1]:.4f}, CO: {_f[2]:.4f}, H2: {_f[5]:.4f}, O2: {_f[0]:.4f}")
        
        self.results = []
        self.cells = []
        
        cell_ops = self.op_conds.copy()
        cell_ops['L_reactor'] = L
        # 热损归一化长度：使用当前网格总长，避免 N_cells 改变时总热损预算漂移
        # (当网格未铺满几何长度时，仍保证 ΣQ_loss = HeatLossPercent × 煤 HHV)
        cell_ops['L_heatloss_norm'] = max(float(np.sum(dz_list)), 1e-9)
        cell_ops['pyrolysis_done'] = True # Pyrolysis already accounted for in inlet
        
        logger.info(f"Starting Solver (Instant Pyrolysis Mode) with {N_cells} cells.")
        
        for i in range(N_cells):
            z_curr = self.z_positions[i]
            dz_curr = dz_list[i]
            V_curr = A * dz_curr
            
        # [SOURCE TERMS]
        # 1. Evaporation Source (distributed over first L_evap to avoid over-cooling Cell 0)
        # Only liquid water (coal moisture + slurry) consumes latent heat.
        F_H2O_liq_mol = (self.tmp_W_liq_evap / 18.015) * 1000.0
        # Evaporation zone length (m).
        L_evap_m = self.op_conds.get("L_evap_m", 0.0)
        if L_evap_m <= 0.0:
            L_evap_m = 1e-6  # effectively all in first cell
        evap_src = EvaporationSource(

            F_H2O_liq_mol,
            enthalpy_per_mol_J=EvaporationSource.H_LIQUID_J_MOL,
            L_evap_m=L_evap_m
        )
        total_evap_MW = (F_H2O_liq_mol * (-EvaporationSource.H_LIQUID_J_MOL)) / 1e6
        evap_note = "Cell 0 only (may under-predict T)" if L_evap_m < 0.01 else f"distributed over L_evap={L_evap_m:.2f} m"
        logger.info(f"Evaporation Source: Flow={F_H2O_liq_mol/1000.0:.2f} mol/s, total sink={total_evap_MW:.1f} MW, {evap_note}")
        
        # 2. Pyrolysis Source (Cell 0)
        # tmp_F_volatiles is mol/s array. tmp_W_vol_loss is kg/s.
        pyro_src = PyrolysisSource(self.tmp_F_volatiles, self.tmp_W_vol_loss, target_cell_idx=0, T_pyro=current_inlet.T)


        
        sources = [evap_src, pyro_src]
        
        logger.info(f"Starting Solver (Source Term Mode) with {N_cells} cells.")

        if jax_warmup and (use_jax_jacobian or solver_method in ('jax_newton', 'jax_pure')):
            warmup_jax()

        # jax_pure：略减每格 Newton 内层迭代上限（失败仍由 multistart / least_squares 兜底）
        jax_pure_n_iter = 45
        fallback_cost_threshold = 1e-4

        def _newton_solver():
            return NewtonSolver(
                tol=1e-8,
                max_iter=100,
                damper=0.8,
                jacobian='centered' if use_jax_jacobian else 'finite_difference',
            )

        def _needs_fallback(sol):
            return (sol is None) or (not sol.success) or (sol.cost > fallback_cost_threshold)

        def _mark_fallback():
            self.solve_stats["fallback_count"] = int(self.solve_stats.get("fallback_count", 0)) + 1

        def _least_squares_with_optional_jac(func_local, x0, bounds):
            if use_jax_jacobian:
                jac_fn = make_jacobian_fn(func_local, n_vars=11, centered=True)
                ls_max_nfev = 2000 if solver_method == "jax_newton" else 500
                ls_ftol = 1e-10 if solver_method == "jax_newton" else 1e-8
                return least_squares(
                    func_local,
                    x0,
                    jac=jac_fn,
                    bounds=bounds,
                    method='trf',
                    xtol=1e-12,
                    ftol=ls_ftol,
                    max_nfev=ls_max_nfev,
                )
            ls_max_nfev = 2000 if solver_method == "jax_newton" else 500
            ls_ftol = 1e-10 if solver_method == "jax_newton" else 1e-8
            return least_squares(
                func_local,
                x0,
                bounds=bounds,
                method='trf',
                xtol=1e-12,
                ftol=ls_ftol,
                max_nfev=ls_max_nfev,
            )

        for i in range(N_cells):
            z_curr = self.z_positions[i]
            dz_curr = dz_list[i]
            V_curr = A * dz_curr
            
            # Update ops for L reference
            cell_ops['L_reactor'] = L
            cell_ops['L_heatloss_norm'] = max(float(np.sum(dz_list)), 1e-9)
            cell_ops['HeatLossPercent'] = self.op_conds.get('HeatLossPercent', PhysicalConstants.DEFAULT_HEAT_LOSS_PERCENT)
            
            # Pass sources instead of manual Q_source_term
            cell = Cell(i, self.z_positions[i], dz_curr, V_curr, A, 
                        current_inlet, self.kinetics, self.pyrolysis, self.coal_props, cell_ops,
                        sources=sources)
            
            # Pass Coal-Basis parameters for X calculation
            cell.coal_flow_dry = self.W_dry 
            cell.Cd_total = self.Cd_total
            cell.char_Xc0 = self.char_Xc0
            self.cells.append(cell)
            
            # Bounds for least_squares (11 variables)
            # [O2, CH4, CO, CO2, H2S, H2, N2, H2O, Ws, Xc, T]
            lower = np.zeros(11)
            upper = np.ones(11) * 3000.0 # Strict upper bound for gas moles (3x theoretical max)
            
            # Ws, Xc, T specific bounds
            lower[8] = 0.0; upper[8] = self.W_dry * 1.5 
            lower[9] = 0.0; upper[9] = 1.0 # Xc
            lower[10] = 300.0; upper[10] = 4000.0 # T
            
            def func(x):
                return cell.residuals(x)
            
            if i == 0:
                # Multi-Guess ignition for first Active cell
                # [IGNITION STRATEGY] Start with HIGH temperatures (2000K) to overcome evaporation
                guesses_T = [3000.0, 2000.0, 1500.0, 1000.0, 400.0]  # 高温优先，避免低温冷态解

                def make_cell0_x0(t_start):
                    x0 = current_inlet.to_array()
                    x0[10] = t_start
                    if t_start > 900.0:
                        # [IGNITION HELPER] temperature_diagnosis.md: 必须先将挥发分加入 x0
                        x0[:8] += self.tmp_F_volatiles
                        x0[8] -= self.tmp_W_vol_loss
                        x0[9] = self.char_Xc0
                        n_CH4 = x0[1]
                        n_O2 = x0[0]
                        xi_1 = min(n_CH4 * 0.999, n_O2 * 0.999 / 0.5)
                        x0[1] -= xi_1
                        x0[0] -= 0.5 * xi_1
                        x0[2] += xi_1
                        x0[5] += 2.0 * xi_1
                        n_H2 = x0[5]
                        n_O2 = x0[0]
                        xi_2 = min(n_H2 * 0.999, n_O2 * 0.999 / 0.5)
                        x0[5] -= xi_2
                        x0[0] -= 0.5 * xi_2
                        x0[7] += xi_2
                        n_CO = x0[2]
                        n_O2 = x0[0]
                        xi_3 = min(n_CO * 0.999, n_O2 * 0.999 / 0.5)
                        x0[2] -= xi_3
                        x0[0] -= 0.5 * xi_3
                        x0[3] += xi_3
                    return x0

                if solver_method == "jax_newton":
                    x0_list = [make_cell0_x0(t) for t in guesses_T]
                    best_mult, _ = newton_solve_multistart_numpy(
                        x0_list,
                        func,
                        lower,
                        upper,
                        n_iter=40,
                        damper=0.5,
                        tol_residual=1e-8
                    )
                    if best_mult is not None and best_mult.success:
                        sol = best_mult
                    else:
                        _mark_fallback()
                        sol = _least_squares_with_optional_jac(func, x0_list[0], (lower, upper))
                elif solver_method == "jax_pure":
                    # 快路径：只用少量高温初值，减少多初值开销
                    guesses_T_fast = [3000.0, 2000.0, 1500.0]
                    x0_list_try = [make_cell0_x0(t) for t in guesses_T_fast]

                    best_mult, _ = newton_solve_multistart_cell_pure_jax_ad(
                        cell,
                        x0_list_try,
                        lower,
                        upper,
                        n_iter=jax_pure_n_iter,
                        damper=0.8,
                        tol_residual=1e-8,
                        fd_epsilon=1e-6,
                        reg=1e-8,
                        t_guesses=guesses_T_fast,
                    )
                    sol = (
                        best_mult
                        if best_mult is not None
                        else _least_squares_with_optional_jac(
                            func, x0_list_try[0], (lower, upper)
                        )
                    )
                    if best_mult is None:
                        _mark_fallback()

                    # 兜底：fast 候选失败/残差偏大时，再扩展回完整候选
                    if _needs_fallback(sol):
                        x0_list_full = [make_cell0_x0(t) for t in guesses_T]
                        best_mult, _ = newton_solve_multistart_cell_pure_jax_ad(
                            cell,
                            x0_list_full,
                            lower,
                            upper,
                            n_iter=jax_pure_n_iter,
                            damper=0.8,
                            tol_residual=1e-8,
                            fd_epsilon=1e-6,
                            reg=1e-8,
                            t_guesses=guesses_T,
                        )
                        sol = (
                            best_mult
                            if best_mult is not None
                            else _least_squares_with_optional_jac(
                                func, x0_list_full[0], (lower, upper)
                            )
                        )
                        if best_mult is None:
                            _mark_fallback()

                    # 最后仍保底：若当前解残差仍大，再跑一次 TRF
                    if _needs_fallback(sol):
                        x0_fb = x0_list_try[0] if len(x0_list_try) > 0 else make_cell0_x0(guesses_T[0])
                        _mark_fallback()
                        sol_fb = _least_squares_with_optional_jac(func, x0_fb, (lower, upper))
                        if sol_fb.success and (not sol.success or sol_fb.cost < sol.cost):
                            sol = sol_fb
                else:
                    best_sol = None
                    best_cost = 1e9

                    for t_start in guesses_T:
                        x0 = make_cell0_x0(t_start)

                        if t_start > 900.0:
                            try:
                                diag_state = StateVector.from_array(x0, P=current_inlet.P, z=0.0)
                                diag_V = self.geometry['D']**2 / 4 * np.pi * dz_list[0]
                                r_homo_diag = self.kinetics.calc_homogeneous_rates(diag_state, diag_V)
                                Q_rxn_H2 = r_homo_diag['H2_Ox'] * 241800.0
                                Q_rxn_CO = r_homo_diag['CO_Ox'] * 283000.0
                                Q_rxn_CH4 = r_homo_diag['CH4_Ox'] * 802000.0
                                Q_total_diag = Q_rxn_H2 + Q_rxn_CO + Q_rxn_CH4
                                velocity = (diag_state.total_gas_moles * 8.314 * t_start / current_inlet.P) / A
                                residence_time = dz_list[0] / max(velocity, 0.001)
                                logger.info(f"  [IGNITION DIAGNOSTIC] T_guess={t_start}K")
                                logger.info(f"    Geometry: L_cell={dz_list[0]:.3f} m, D={D:.2f} m, Vol={diag_V:.3f} m3")
                                logger.info(f"    Flow:     Vel={velocity:.2f} m/s, Tau={residence_time:.3f} s")
                                logger.info(f"    R_H2_Ox:  {r_homo_diag['H2_Ox']:.2e} mol/s -> {Q_rxn_H2/1e6:.2f} MW")
                                logger.info(f"    R_CO_Ox:  {r_homo_diag['CO_Ox']:.2e} mol/s -> {Q_rxn_CO/1e6:.2f} MW")
                                logger.info(f"    R_CH4_Ox: {r_homo_diag['CH4_Ox']:.2e} mol/s -> {Q_rxn_CH4/1e6:.2f} MW")
                                logger.info(f"    TOTAL REACT HEAT (Reacted State): {Q_total_diag/1e6:.2f} MW")
                                x_pot = current_inlet.to_array()
                                x_pot[:8] += self.tmp_F_volatiles
                                x_pot[10] = t_start
                                state_pot = StateVector.from_array(x_pot, P=current_inlet.P, z=0.0)
                                r_pot = self.kinetics.calc_homogeneous_rates(state_pot, diag_V)
                                Q_pot_H2 = r_pot['H2_Ox'] * 241800.0
                                Q_pot_CO = r_pot['CO_Ox'] * 283000.0
                                Q_pot_CH4 = r_pot['CH4_Ox'] * 802000.0
                                Q_total_pot = Q_pot_H2 + Q_pot_CO + Q_pot_CH4
                                logger.info(f"    POTENTIAL RATE (Inlet+Volatiles @ {t_start}K):")
                                logger.info(f"      R_CH4_Ox: {r_pot['CH4_Ox']:.2e} mol/s")
                                logger.info(f"      Potential Heat Release: {Q_total_pot/1e6:.2f} MW (Target: >42 MW)")
                            except Exception as e:
                                logger.warning(f"Failed to run ignition diagnostic: {e}")

                        logger.info(f"  > Attempting Solver with Initial T = {x0[10]:.1f} K (Method: {solver_method})")

                        if solver_method == 'newton':
                            sol = _newton_solver().solve(func, x0, bounds=(lower, upper))
                        else:
                            sol = _least_squares_with_optional_jac(func, x0, (lower, upper))

                        logger.info(
                            f"  [Ignition Guess] T: {t_start}K -> Res: {sol.x[10]:.1f}K, Cost: {sol.cost:.2e}, Success: {sol.success}"
                        )

                        if sol.success:
                            final_res = cell.residuals(sol.x)
                            logger.info(f"    Final Residuals (Max): {np.max(np.abs(final_res)):.4e}")
                            is_ignited = sol.x[10] > 1200.0
                            update_best = False
                            if best_sol is None:
                                update_best = True
                            else:
                                best_ignited = best_sol.x[10] > 1200.0
                                if is_ignited and not best_ignited:
                                    update_best = True
                                elif is_ignited == best_ignited:
                                    if sol.cost < best_cost:
                                        update_best = True
                                    elif abs(sol.cost - best_cost) < best_cost * 0.1 and sol.x[10] > best_sol.x[10]:
                                        update_best = True
                            if update_best:
                                best_sol = sol
                                best_cost = sol.cost
                            if is_ignited and sol.cost < 1e-6:
                                break

                    sol = best_sol if best_sol is not None else sol
            else:
                # 下游 cell：入口温度高时多初值猜测，避免陷入低温解
                # (WGS 等吸热不应造成剧烈降温，温度下降会自我限制；用户诊断：多为 cell 初值/求解问题)
                x0_base = current_inlet.to_array()
                T_in = float(x0_base[10])
                if solver_method == 'jax_newton':
                    if T_in > 1200.0:
                        T_cands = [
                            T_in,
                            min(T_in * 1.02, 3500.0),
                            min(T_in * 1.08, 3200.0),
                            min(T_in * 1.15, 3500.0),
                            max(T_in * 0.98, 1100.0),
                            max(T_in * 0.92, 1100.0),
                        ]
                        x0_list = []
                        for Tc in T_cands:
                            x0 = x0_base.copy()
                            x0[10] = Tc
                            x0_list.append(x0)
                        best_down, _ = newton_solve_multistart_numpy(
                            x0_list,
                            func,
                            lower,
                            upper,
                            t_guesses=T_cands,
                        )
                        sol = (
                            best_down
                            if best_down is not None and best_down.success
                            else newton_solve_cell_numpy(func, x0_base, lower, upper)
                        )
                        best_cost_down = sol.cost if sol.success else 1e9
                        if sol.success and T_in > 1800 and sol.x[10] < T_in * 0.8:
                            for Tc in [min(T_in * 1.2, 3500.0), min(T_in * 1.1, 3400.0)]:
                                x0 = x0_base.copy()
                                x0[10] = Tc
                                s = newton_solve_cell_numpy(func, x0, lower, upper)
                                if s.success and s.x[10] > sol.x[10] and s.cost < best_cost_down * 2.0:
                                    sol = s
                                    best_cost_down = s.cost
                                    break
                    else:
                        sol = newton_solve_cell_numpy(func, x0_base, lower, upper)
                    if _needs_fallback(sol):
                        _mark_fallback()
                        sol_fb = _least_squares_with_optional_jac(func, x0_base, (lower, upper))
                        if sol_fb.success and (not sol.success or sol_fb.cost < sol.cost):
                            sol = sol_fb
                elif solver_method == 'jax_pure':
                    # jax_pure：host 前向差分 J + 阻尼 Newton（见 jax_solver.newton_solve_cell_pure_jax_ad）
                    if T_in > 1200.0:
                        T_cands_full = [
                            T_in,
                            min(T_in * 1.02, 3500.0),
                            min(T_in * 1.08, 3200.0),
                            min(T_in * 1.15, 3500.0),
                            max(T_in * 0.98, 1100.0),
                            max(T_in * 0.92, 1100.0),
                        ]
                        # 快路径：只保留少量“高温/近温”初值，显著减少多初值开销（3 个；失败再扩到 full）
                        T_cands_fast = [
                            T_in,
                            min(T_in * 1.02, 3500.0),
                            max(T_in * 0.98, 1100.0),
                        ]

                        def _pack_x0_list(T_cands_local):
                            x0_list_local = []
                            for Tc in T_cands_local:
                                x0 = x0_base.copy()
                                x0[10] = Tc
                                x0_list_local.append(x0)
                            return x0_list_local

                        # 1) 先用 fast 候选
                        x0_list_try = _pack_x0_list(T_cands_fast)
                        best_down, _ = newton_solve_multistart_cell_pure_jax_ad(
                            cell,
                            x0_list_try,
                            lower,
                            upper,
                            n_iter=jax_pure_n_iter,
                            damper=0.8,
                            tol_residual=1e-8,
                            fd_epsilon=1e-6,
                            reg=1e-8,
                            t_guesses=T_cands_fast,
                        )
                        sol = best_down if best_down is not None else _least_squares_with_optional_jac(
                            func, x0_list_try[0], (lower, upper)
                        )
                        if best_down is None:
                            _mark_fallback()

                        # 2) 若快路径失败，再用 full 候选兜底（避免稳定性回退）
                        if _needs_fallback(sol):
                            x0_list_full = _pack_x0_list(T_cands_full)
                            best_down, _ = newton_solve_multistart_cell_pure_jax_ad(
                                cell,
                                x0_list_full,
                                lower,
                                upper,
                                n_iter=jax_pure_n_iter,
                                damper=0.8,
                                tol_residual=1e-8,
                                fd_epsilon=1e-6,
                                reg=1e-8,
                                t_guesses=T_cands_full,
                            )
                            sol = best_down if best_down is not None else _least_squares_with_optional_jac(
                                func, x0_list_full[0], (lower, upper)
                            )
                            if best_down is None:
                                _mark_fallback()
                    else:
                        sol = newton_solve_cell_pure_jax_ad(
                            cell,
                            x0_base,
                            lower,
                            upper,
                            n_iter=jax_pure_n_iter,
                            damper=0.8,
                            tol_residual=1e-8,
                        )

                    # 未收敛或残差仍大（与 diagnose_failure 阈值同量级）时回退 TRF
                    if _needs_fallback(sol):
                        _mark_fallback()
                        sol_fb = _least_squares_with_optional_jac(func, x0_base, (lower, upper))
                        if sol_fb.success and (not sol.success or sol_fb.cost < sol.cost):
                            sol = sol_fb
                elif T_in > 1200.0:
                    # 温度初值探索：先在 T_in 附近精细搜索，再探索略高/略低
                    T_cands = [
                        T_in, min(T_in * 1.02, 3500.0), min(T_in * 1.08, 3200.0),
                        min(T_in * 1.15, 3500.0), max(T_in * 0.98, 1100.0), max(T_in * 0.92, 1100.0)
                    ]
                    best_down = None
                    best_cost_down = 1e9
                    for Tc in T_cands:
                        x0 = x0_base.copy()
                        x0[10] = Tc
                        ns = _newton_solver()
                        s = ns.solve(func, x0, bounds=(lower, upper))
                        if s.success:
                            if s.cost < best_cost_down:
                                best_cost_down = s.cost
                                best_down = s
                            elif abs(s.cost - best_cost_down) < best_cost_down * 0.2 and s.x[10] > best_down.x[10]:
                                best_down = s
                                best_cost_down = s.cost
                    ns = _newton_solver()
                    sol = best_down if best_down is not None else ns.solve(func, x0_base, bounds=(lower, upper))
                    if sol.success and T_in > 1800 and sol.x[10] < T_in * 0.8:
                        for Tc in [min(T_in * 1.2, 3500.0), min(T_in * 1.1, 3400.0)]:
                            x0 = x0_base.copy()
                            x0[10] = Tc
                            ns2 = _newton_solver()
                            s = ns2.solve(func, x0, bounds=(lower, upper))
                            if s.success and s.x[10] > sol.x[10] and s.cost < best_cost_down * 2.0:
                                sol = s
                                best_cost_down = s.cost
                                break
                else:
                    ns = _newton_solver()
                    sol = ns.solve(func, x0_base, bounds=(lower, upper))
            
            if _needs_fallback(sol):
                self.solve_stats["poor_convergence_count"] = int(self.solve_stats.get("poor_convergence_count", 0)) + 1
                logger.warning(f"Cell {i} convergence poor. Cost: {sol.cost:.2e}, Success: {sol.success}, T: {sol.x[10]:.1f}K")
                # [DIAGNOSTICS]
                cell.diagnose_failure(sol.x)
            
            sol_state = StateVector.from_array(sol.x, P=current_inlet.P, z=cell.z)
            # Fortran 式：计算出口颗粒温度 Ts_out，供下一 cell 使用
            Ts_in = current_inlet.T_solid_or_gas
            phys = cell._calc_physics_props(sol_state, self.W_dry * self.Cd_total)
            tau = cell.dz / max(phys['v_g'], PhysicalConstants.MIN_SLIP_VELOCITY)
            _, Ts_out = cell._calc_particle_temperature(
                sol_state.T, Ts_in, tau, phys['d_p'],
                sol_state.solid_mass, sol_state.carbon_fraction
            )
            sol_state.T_solid = Ts_out
            self.results.append(sol_state)
            current_inlet = sol_state
            
        logger.info("Solver finished successfully.")
        return self._package_results()

    def _package_results(self):
        # Return numpy array matching old format for compatibility
        # [F0..F7, Ws, Xc, T]
        rows = [s.to_array() for s in self.results]
        return np.array(rows), self.z_positions
