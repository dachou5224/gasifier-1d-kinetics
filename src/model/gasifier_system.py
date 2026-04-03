import numpy as np
from scipy.optimize import least_squares
import warnings
from .state import StateVector
from .cell import Cell
from .material import MaterialService
from .kinetics_service import KineticsService
from .pyrolysis_service import PyrolysisService
from .constants import PhysicalConstants
from .grid_service import AdaptiveMeshGenerator, MeshConfig
from .source_terms import EvaporationSource, PyrolysisSource
from .species_bridge import mainline_gas8_to_jax9, jax9_to_mainline_gas8
from .input_contract import (
    ash_mass_fraction_dry,
    coal_flow_kg_s_for_heat_loss,
    heat_loss_norm_length_m,
    heat_loss_ref_temp_k,
    resolve_f_s_coal,
)
from .solver import NewtonSolver
from .jax_residual_adapter import make_jacobian_fn
import logging

logger = logging.getLogger(__name__)

_LEGACY_SOLVER_ALIASES = {
    "newton": "newton_fd",
    "jax_newton": "newton_fd",
    "jax_pure": "newton_fd",
}

class GasifierSystem:
    def __init__(self, geometry, coal_props, op_conds):
        self._validate_inputs(geometry, coal_props, op_conds)
        self.geometry = geometry; self.coal_props = coal_props; self.op_conds = op_conds.copy()
        if 'Hf_coal' not in self.coal_props:
            Cd = self.coal_props.get('Cd', 0.0)/100.0; Hd = self.coal_props.get('Hd', 0.0)/100.0; Sd = self.coal_props.get('Sd', 0.0)/100.0
            hhv = self.coal_props.get('HHV_d', 30.0); hhv_kj = hhv if hhv > 1000.0 else hhv * 1000.0
            lhv_kj = hhv_kj - (9.0 * Hd * 2442.0)
            hf_kj = (Cd/0.012011)*(-393.51) + (Hd/0.001008)*0.5*(-241.83) + (Sd/0.03206)*(-296.81) + lhv_kj
            self.coal_props['Hf_coal'] = hf_kj * 1000.0
        self.material = MaterialService(); self.kinetics = KineticsService(); self.pyrolysis = PyrolysisService()
        self.results = []; self.cells = []; self.solve_stats = {"fallback_count": 0, "poor_convergence_count": 0}

    def _validate_inputs(self, geometry, coal_props, op_conds):
        required_geom = ['L', 'D']
        missing_geom = [k for k in required_geom if k not in geometry]
        if missing_geom:
            raise ValueError(f"Missing geometry parameters: {missing_geom}")
        if geometry['L'] <= 0 or geometry['D'] <= 0:
            raise ValueError("Geometry dimensions L and D must be positive.")
        required_op = ['coal_flow', 'o2_flow', 'P', 'T_in']
        missing_op = [k for k in required_op if k not in op_conds]
        if missing_op:
            raise ValueError(f"Missing operating conditions: {missing_op}")
        if op_conds['coal_flow'] < 0 or op_conds['o2_flow'] < 0:
            raise ValueError("Flow rates cannot be negative.")
        if op_conds['P'] <= 0 or op_conds['T_in'] <= 0:
            raise ValueError("Pressure and Temperature must be positive.")
        if 'Cd' not in coal_props:
            raise ValueError("Coal property 'Cd' (Carbon dry basis) is required.")

    def _jax_cell0_seed_from_minimize(self, dz_list, inlet, A, F_H2O_evap, L_evap, ref_f, ref_e):
        """
        与 _solve_sequential 首格相同的 SciPy least_squares 解，转成 JAX 12 维状态，
        作为 reactor_solve_v4 第一格多起点中的第一条，避免纯启发式初值落在低温支路。
        """
        i = 0
        cell = Cell(
            i,
            self.z_positions[i],
            dz_list[i],
            A * dz_list[i],
            A,
            inlet,
            self.kinetics,
            self.pyrolysis,
            self.coal_props,
            self.op_conds,
            sources=[
                EvaporationSource(F_H2O_evap, L_evap_m=L_evap),
                PyrolysisSource(self.tmp_F_vol, self.tmp_W_vol, target_cell_idx=0, T_pyro=inlet.T),
            ],
        )
        cell.coal_flow_dry = self.W_dry
        cell.Cd_total = self.Cd_total
        cell.char_Xc0 = self.char_Xc0
        cell.ref_flow, cell.ref_energy = ref_f, ref_e
        lower = np.zeros(11)
        upper = np.ones(11) * 3000.0
        lower[8] = 0.0
        upper[8] = self.W_dry * 1.5
        lower[9] = 0.0
        upper[9] = 1.0
        lower[10] = 300.0
        upper[10] = 4000.0

        def func(x):
            return cell.residuals(x)

        best_sol, best_cost = None, 1e9
        for t in [3000, 2000, 1500, 1000, 400]:
            x0 = inlet.to_array()
            x0[10] = t
            if t > 900.0:
                x0[:8] += self.tmp_F_vol
                x0[8] -= self.tmp_W_vol
                x0[9] = self.char_Xc0
            sol = least_squares(func, x0, bounds=(lower, upper), method="trf", xtol=1e-12, ftol=1e-8, max_nfev=1000)
            if sol.success and sol.cost < best_cost:
                best_sol, best_cost = sol, sol.cost
        if best_sol is None:
            return None
        xb = best_sol.x
        g9 = mainline_gas8_to_jax9(xb[:8])
        return np.concatenate([g9, [xb[8], xb[9], xb[10]]])

    def _initialize_inlet(self):
        W_wet = self.op_conds['coal_flow']; Mt = self.coal_props.get('Mt', 0.0); W_dry = W_wet * (1 - Mt/100.0)
        slurry = self.op_conds.get('SlurryConcentration', 100.0); W_steam = self.op_conds.get('steam_flow', 0.0)
        W_coal_moist = W_wet * Mt/100.0
        W_slurry_h2o = (W_dry / (slurry/100.0)) - W_dry if slurry < 100.0 else 0.0
        self.tmp_W_h2o_total = W_coal_moist + W_slurry_h2o + W_steam
        self.tmp_W_liq_evap = W_coal_moist + W_slurry_h2o
        F_O2 = (self.op_conds['o2_flow'] / 31.998) * 1000.0; F_N2 = self.op_conds.get('n2_flow', 0.0) / 28.013 * 1000.0; F_st = W_steam / 18.015 * 1000.0
        gas = np.zeros(8); gas[0]=F_O2; gas[6]=F_N2; gas[7]=F_st
        inlet = StateVector(gas_moles=gas, solid_mass=W_dry, carbon_fraction=self.coal_props.get('Cd', 60.0)/100.0, T=self.op_conds['T_in'], P=self.op_conds['P'])
        self.W_dry = W_dry; self.Cd_total = self.coal_props.get('Cd', 60.0)/100.0
        m_y, w_y = self.pyrolysis.calc_yields(self.coal_props)
        self.tmp_F_vol = m_y * W_dry; self.tmp_W_vol = W_dry * w_y
        self.tmp_F_volatiles = self.tmp_F_vol
        self.tmp_W_vol_loss = self.tmp_W_vol
        C_char = W_dry * self.Cd_total - (self.tmp_F_vol[1]+self.tmp_F_vol[2]+self.tmp_F_vol[3])*0.012011
        self.char_Xc0 = C_char / max(W_dry - self.tmp_W_vol, 1e-9)
        return inlet

    def _normalize_solver_api(self, solver_method: str, use_jax_jacobian: bool, jacobian_mode):
        mode = "scipy" if jacobian_mode is None else str(jacobian_mode)
        if use_jax_jacobian:
            warnings.warn(
                "`use_jax_jacobian` 已弃用；请改用 `jacobian_mode='centered_fd'`。",
                DeprecationWarning,
                stacklevel=3,
            )
            mode = "centered_fd"
        if solver_method in _LEGACY_SOLVER_ALIASES:
            warnings.warn(
                f"`solver_method={solver_method}` 已弃用；请改用 `solver_method=\"{_LEGACY_SOLVER_ALIASES[solver_method]}\"`。",
                DeprecationWarning,
                stacklevel=3,
            )
            solver_method = _LEGACY_SOLVER_ALIASES[solver_method]
        valid_methods = {"minimize", "newton_fd", "jax_jit"}
        if solver_method not in valid_methods:
            raise ValueError(f"Unsupported solver_method: {solver_method}. Expected one of {sorted(valid_methods)}")
        valid_jac_modes = {"scipy", "centered_fd"}
        if mode not in valid_jac_modes:
            raise ValueError(f"Unsupported jacobian_mode: {mode}. Expected one of {sorted(valid_jac_modes)}")
        return solver_method, mode

    def solve(self, N_cells=100, solver_method='minimize', use_jax_jacobian=False, jax_warmup=True, jacobian_mode=None):
        solver_method, jacobian_mode = self._normalize_solver_api(solver_method, use_jax_jacobian, jacobian_mode)
        L, D = self.geometry['L'], self.geometry['D']; A = np.pi * (D/2)**2
        mesh_cfg = MeshConfig(total_length=L, n_cells=N_cells, ignition_zone_length=self.op_conds.get('FirstCellLength', 0.1), min_grid_size=0.001)
        dz_list, self.z_positions = AdaptiveMeshGenerator(mesh_cfg).generate()
        inlet = self._initialize_inlet()
        if jax_warmup and solver_method == "jax_jit":
            from .jax_solver import warmup_jax
            warmup_jax()
        if solver_method == "jax_jit":
            return self._solve_jax_v4(dz_list, inlet, A)
        return self._solve_sequential(dz_list, inlet, A, solver_method, jacobian_mode)

    def _solve_jax_v4(self, dz_list, inlet, A):
        # 须先加载 jax_solver，其内会 jax_enable_x64；否则此处先 import jax.numpy 会锁在 float32
        from .jax_solver import reactor_solve_v4
        import jax.numpy as jnp

        N = len(dz_list); g_src_9 = np.zeros((N, 9)); s_src = np.zeros(N); e_src = np.zeros(N)
        F_H2O_evap = (self.tmp_W_liq_evap / 18.015) * 1000.0; L_evap = self.op_conds.get("L_evap_m", 1e-6)
        for i in range(N):
            z, dz = self.z_positions[i], dz_list[i]
            z_s, z_e = z-dz/2., z+dz/2.; frac = (min(L_evap, z_e) - max(0., z_s)) / L_evap if z_e > 0 and z_s < L_evap else 0.
            g_src_9[i, 4] += F_H2O_evap * frac; e_src[i] += F_H2O_evap * frac * -285830.0
            if i == 0:
                v = self.tmp_F_vol
                # 须用 +=：与 Cell 内 EvaporationSource + PyrolysisSource 一致，勿整行覆盖以致丢失蒸发 H2O
                g_src_9[i, :] = g_src_9[i, :] + mainline_gas8_to_jax9(v)
                s_src[i] = -self.tmp_W_vol
                from model.physics import get_enthalpy_molar; from model.material import SPECIES_NAMES
                e_src[i] += sum(v[j]*get_enthalpy_molar(sp, inlet.T) for j, sp in enumerate(SPECIES_NAMES) if v[j]>0)

        f_s_coal = resolve_f_s_coal(self.coal_props, self.op_conds)
        def make_x12(t):
            v = inlet.gas_moles
            g9 = mainline_gas8_to_jax9(v)
            x = np.concatenate([g9, [inlet.solid_mass, inlet.carbon_fraction, t]])
            if t > 900.:
                x[:9] += g_src_9[0]; x[9] += s_src[0]; x[10] = self.char_Xc0
                xi1 = min(x[1]*0.99, x[0]*0.99/2.0); x[1]-=xi1; x[0]-=2.0*xi1; x[2]+=xi1; x[4]+=2.0*xi1
                xi2 = min(x[5]*0.99, x[0]*0.99/0.5); x[5]-=xi2; x[0]-=0.5*xi2; x[4]+=xi2
                xi3 = min(x[2]*0.99, x[0]*0.99/0.5); x[2]-=xi3; x[0]-=0.5*xi3; x[3]+=xi3
            return x
        hhv = self.coal_props.get('HHV_d', 30.0); hhv_mj = hhv / 1000.0 if hhv > 1000.0 else hhv
        Ash_d = ash_mass_fraction_dry(self.coal_props)
        f_ash = Ash_d / (self.Cd_total + Ash_d + 1e-9)
        ref_f = max(inlet.total_gas_moles, 1.0); ref_e = max(ref_f * 35.0 * 200.0, 5.0e5)
        guesses = np.array([make_x12(t) for t in [3000, 2000, 1500, 1000, 400]])
        if self.op_conds.get("JaxSeedCell0FromMinimize", True):
            seed12 = self._jax_cell0_seed_from_minimize(dz_list, inlet, A, F_H2O_evap, L_evap, ref_f, ref_e)
            if seed12 is not None:
                guesses = np.vstack([seed12[np.newaxis, :], guesses])
        g9_in = mainline_gas8_to_jax9(inlet.gas_moles)
        inlet_12 = jnp.asarray(
            np.concatenate([g9_in, [inlet.solid_mass, inlet.carbon_fraction, inlet.T]]),
            dtype=jnp.float64,
        )
        # 与顺序主线 _solve_sequential / Cell._calc_energy_balance 保持一致：
        # 若上层未显式给 L_heatloss_norm，则按实际 mesh 总长 sum(dz) 归一化热损，
        # 而不是几何 L。当前历史网格在 N=20 时 sum(dz) 可显著小于 L；
        # 若此处退回几何 L，会系统性低估 Q_loss，导致 jax_jit 温度单向偏高。
        op_for_heat_loss = dict(self.op_conds)
        op_for_heat_loss.setdefault("L_heatloss_norm", float(np.sum(dz_list)))
        hl_norm = heat_loss_norm_length_m(op_for_heat_loss, float(self.geometry["L"]), float(np.sum(dz_list)))
        hl_ref = heat_loss_ref_temp_k(self.op_conds)
        coal_flow_loss = coal_flow_kg_s_for_heat_loss(self.op_conds)

        all_jnp = reactor_solve_v4(
            inlet_12,
            jnp.asarray(dz_list, dtype=jnp.float64),
            jnp.asarray(g_src_9, dtype=jnp.float64),
            jnp.asarray(s_src, dtype=jnp.float64),
            jnp.asarray(e_src, dtype=jnp.float64),
            jnp.asarray(self.z_positions, dtype=jnp.float64),
            float(A),
            jnp.asarray(guesses, dtype=jnp.float64),
            float(inlet.P), float(inlet.solid_mass * inlet.carbon_fraction), float(coal_flow_loss),
            float(self.op_conds.get('particle_diameter', 100e-6)), float(self.op_conds.get('epsilon', 1.0)),
            float(self.op_conds.get('HeatLossPercent', 2.0)), float(self.geometry['L']),
            float(self.op_conds.get('CharCombustionRateFactor', 0.3)), float(self.op_conds.get('WGS_CatalyticFactor', 0.2)),
            float(self.op_conds.get('Combustion_CO2_Fraction', 1.0)), float(self.op_conds.get('P_O2_Combustion_atm', 0.05)),
            float(hl_norm), float(hl_ref),
            float(self.coal_props.get('Hf_coal', 0.0)),
            float(self.coal_props.get('cp_char', 1300.0)), float(hhv_mj), float(inlet.T), float(f_ash),
            float(ref_f), float(ref_e), float(self.char_Xc0), float(f_s_coal)
        )
        self.results = []
        for i in range(N):
            row = all_jnp[i]
            gas = np.asarray(jax9_to_mainline_gas8(row[:9]))
            self.results.append(StateVector(gas_moles=gas, solid_mass=float(row[9]), carbon_fraction=float(row[10]), T=float(row[11]), P=inlet.P, z=self.z_positions[i]))
        return self._package_results()

    def _solve_sequential(self, dz_list, inlet, A, solver_method, jacobian_mode):
        N_cells = len(dz_list); current_inlet = inlet; self.results = []; self.cells = []
        F_H2O_evap = (self.tmp_W_liq_evap / 18.015) * 1000.0; L_evap = self.op_conds.get("L_evap_m", 1e-6)
        ref_f = max(inlet.total_gas_moles, 1.0); ref_e = max(ref_f * 35.0 * 200.0, 5.0e5)
        cell_ops = self.op_conds.copy()
        cell_ops["L_reactor"] = self.geometry["L"]
        cell_ops["L_heatloss_norm"] = max(float(np.sum(dz_list)), 1e-9)
        cell_ops["pyrolysis_done"] = True

        def _newton_solver():
            return NewtonSolver(
                tol=1e-8,
                max_iter=100,
                damper=0.8,
                jacobian='centered' if jacobian_mode == "centered_fd" else 'finite_difference',
            )

        def _least_squares_with_optional_jac(func_local, x0, bounds):
            kwargs = dict(bounds=bounds, method='trf', xtol=1e-12, ftol=1e-8, max_nfev=1000)
            if jacobian_mode == "centered_fd":
                kwargs["jac"] = make_jacobian_fn(func_local, n_vars=11, centered=True)
            return least_squares(func_local, x0, **kwargs)

        def _needs_fallback(sol):
            return (sol is None) or (not sol.success) or (sol.cost > 1e-4)

        def _mark_fallback():
            self.solve_stats["fallback_count"] = int(self.solve_stats.get("fallback_count", 0)) + 1

        def _make_cell0_x0(t_start):
            x0 = current_inlet.to_array()
            x0[10] = t_start
            if t_start > 900.0:
                x0[:8] += self.tmp_F_vol
                x0[8] -= self.tmp_W_vol
                x0[9] = self.char_Xc0
            return x0

        for i in range(N_cells):
            cell = Cell(i, self.z_positions[i], dz_list[i], A*dz_list[i], A, current_inlet, self.kinetics, self.pyrolysis, self.coal_props, cell_ops, sources=[EvaporationSource(F_H2O_evap, L_evap_m=L_evap), PyrolysisSource(self.tmp_F_vol, self.tmp_W_vol, target_cell_idx=0, T_pyro=inlet.T)])
            cell.coal_flow_dry = self.W_dry; cell.Cd_total = self.Cd_total; cell.char_Xc0 = self.char_Xc0; cell.ref_flow, cell.ref_energy = ref_f, ref_e
            self.cells.append(cell)
            lower = np.zeros(11); upper = np.ones(11)*3000.0; lower[8]=0.0; upper[8]=self.W_dry*1.5; lower[9]=0.0; upper[9]=1.0; lower[10]=300.0; upper[10]=4000.0
            def func(x): return cell.residuals(x)

            if i == 0:
                guesses = [3000.0, 2000.0, 1500.0, 1000.0, 400.0]
                if solver_method == "newton_fd":
                    from .jax_solver import newton_solve_multistart_numpy
                    x0_list = [_make_cell0_x0(t) for t in guesses]
                    sol, _ = newton_solve_multistart_numpy(
                        x0_list,
                        func,
                        lower,
                        upper,
                        n_iter=40,
                        damper=0.5,
                        tol_residual=1e-8,
                        t_guesses=guesses,
                        jacobian_centered=(jacobian_mode == "centered_fd"),
                    )
                    if _needs_fallback(sol):
                        _mark_fallback()
                        sol = _least_squares_with_optional_jac(func, x0_list[0], (lower, upper))
                else:
                    best_sol, best_cost = None, 1e9
                    for t in guesses:
                        x0 = _make_cell0_x0(t)
                        sol_try = _least_squares_with_optional_jac(func, x0, (lower, upper))
                        if sol_try.success and sol_try.cost < best_cost:
                            best_sol, best_cost = sol_try, sol_try.cost
                    sol = best_sol if best_sol is not None else sol_try
            else:
                x0 = current_inlet.to_array()
                if solver_method == "newton_fd":
                    from .jax_solver import newton_solve_cell_numpy
                    sol = newton_solve_cell_numpy(
                        func,
                        x0,
                        lower,
                        upper,
                        jacobian_centered=(jacobian_mode == "centered_fd"),
                    )
                    if _needs_fallback(sol):
                        _mark_fallback()
                        sol = _least_squares_with_optional_jac(func, x0, (lower, upper))
                else:
                    sol = _least_squares_with_optional_jac(func, x0, (lower, upper))

            if _needs_fallback(sol):
                self.solve_stats["poor_convergence_count"] = int(self.solve_stats.get("poor_convergence_count", 0)) + 1

            sol_state = StateVector.from_array(sol.x, P=inlet.P, z=self.z_positions[i])
            phys = cell._calc_physics_props(sol_state, self.W_dry*self.Cd_total)
            tau = dz_list[i] / max(phys['v_g'], 0.1)
            _, Ts_out = cell._calc_particle_temperature(sol_state.T, current_inlet.T_solid_or_gas, tau, phys['d_p'], sol_state.solid_mass, sol_state.carbon_fraction)
            sol_state.T_solid = Ts_out; self.results.append(sol_state); current_inlet = sol_state
        return self._package_results()

    def _package_results(self):
        rows = [s.to_array() for s in self.results]; return np.array(rows), self.z_positions
