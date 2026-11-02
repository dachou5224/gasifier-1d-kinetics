import numpy as np
from scipy.optimize import least_squares
import math
import warnings
from .state import StateVector
from .cell import Cell
from .material import MaterialService, SPECIES_NAMES
from .physics import get_enthalpy_molar
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
    uses_fortran_default_model,
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
        self.geometry = geometry
        self.coal_props = coal_props.copy()  # copy to avoid mutating caller's dict (Hf_coal may be added below)
        self.op_conds = op_conds.copy()
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
        if op_conds.get("n2_flow", 0.0) < 0 or op_conds.get("co2_flow", 0.0) < 0:
            raise ValueError("Carrier gas flow rate cannot be negative.")
        if op_conds.get("carrier_gas_flow", 0.0) < 0:
            raise ValueError("Carrier gas flow rate cannot be negative.")
        carrier_gas_type = str(op_conds.get("carrier_gas_type", "N2")).upper()
        if carrier_gas_type not in {"N2", "CO2"}:
            raise ValueError("carrier_gas_type must be either 'N2' or 'CO2'.")
        if op_conds['P'] <= 0 or op_conds['T_in'] <= 0:
            raise ValueError("Pressure and Temperature must be positive.")
        if 'Cd' not in coal_props:
            raise ValueError("Coal property 'Cd' (Carbon dry basis) is required.")

    def _carrier_gas_molar_flows(self):
        """
        Return inlet carrier gas molar flows [mol/s] for mainline CO2 and N2 slots.

        Public op_conds contract:
        - ``n2_flow`` and ``co2_flow`` are legacy/direct mass flows [kg/s].
        - ``carrier_gas_flow`` is a mass flow [kg/s] assigned by
          ``carrier_gas_type`` in {"N2", "CO2"}.
        """
        f_n2 = self.op_conds.get("n2_flow", 0.0) / 28.013 * 1000.0
        f_co2 = self.op_conds.get("co2_flow", 0.0) / 44.01 * 1000.0
        carrier_flow = float(self.op_conds.get("carrier_gas_flow", 0.0))
        if carrier_flow:
            carrier_gas_type = str(self.op_conds.get("carrier_gas_type", "N2")).upper()
            if carrier_gas_type == "N2":
                f_n2 += carrier_flow / 28.013 * 1000.0
            elif carrier_gas_type == "CO2":
                f_co2 += carrier_flow / 44.01 * 1000.0
            else:
                raise ValueError("carrier_gas_type must be either 'N2' or 'CO2'.")
        return f_co2, f_n2

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
        F_O2 = (self.op_conds['o2_flow'] / 31.998) * 1000.0; F_CO2, F_N2 = self._carrier_gas_molar_flows(); F_st = W_steam / 18.015 * 1000.0
        gas = np.zeros(8); gas[0]=F_O2; gas[3]=F_CO2; gas[6]=F_N2; gas[7]=F_st
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
        mesh_family = str(self.op_conds.get('MeshFamily', 'legacy')).lower()
        allow_truncated = bool(self.op_conds.get('AllowTruncatedMesh', False))
        mesh_cfg = MeshConfig(
            total_length=L,
            n_cells=N_cells,
            ignition_zone_length=self.op_conds.get('FirstCellLength', 0.1),
            ignition_zone_res=self.op_conds.get('IgnitionZoneRes', 0.1),
            min_grid_size=0.001,
            mesh_family=mesh_family,
        )
        dz_list, self.z_positions = AdaptiveMeshGenerator(mesh_cfg).generate()
        # P1.5-A B1：网格必须铺满反应器。旧族在 3 <= N <= 20 时会被此断言拦下。
        self.mesh_family = mesh_family
        self.mesh_sum_dz = float(np.sum(dz_list))
        self.mesh_cover_fraction = self.mesh_sum_dz / L
        if abs(self.mesh_sum_dz - L) > 1e-9 * L and not allow_truncated:
            raise ValueError(
                "mesh does not cover the reactor: "
                f"sum(dz)={self.mesh_sum_dz:.6f} m vs L={L:.6f} m "
                f"(N_cells={N_cells}, family={mesh_family}). "
                "Use MeshFamily='uniform_full_length' for paper rounds, or set "
                "AllowTruncatedMesh=True to reproduce legacy (truncated) rounds."
            )
        # 每轮求解的接受状态与逐单元诊断（P1.5-A B2/B5/B6）
        self.solve_stats = {"fallback_count": 0, "poor_convergence_count": 0}
        self.cell_diagnostics = []
        self.execution_status = "completed"
        self.no_root_cell_index = None
        self.cell0_branches = []
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
            float(ref_f), float(ref_e), float(self.char_Xc0), float(f_s_coal),
            float(uses_fortran_default_model(self.op_conds)),
        )
        self.results = []
        for i in range(N):
            row = all_jnp[i]
            gas = np.asarray(jax9_to_mainline_gas8(row[:9]))
            self.results.append(StateVector(gas_moles=gas, solid_mass=float(row[9]), carbon_fraction=float(row[10]), T=float(row[11]), P=inlet.P, z=self.z_positions[i]))
        # jax_jit 是一次性打包求解，不逐个单元推进；这里按与顺序路径相同的阈值
        # 复核每一格是否为根，并在首个非根处截断。否则指定 SolverStrictRoot=True
        # 后切到 jax_jit 会把非根解当作对下游有效并报告 completed。
        self._enforce_strict_root_on_states(dz_list, inlet, A, N)
        return self._package_results()

    def _cell_ops_for_solve(self, dz_list):
        """与 _solve_sequential 相同的单元级 op_conds（热损归一化长度随网格族取值）。"""
        cell_ops = self.op_conds.copy()
        cell_ops["L_reactor"] = self.geometry["L"]
        if str(self.op_conds.get("MeshFamily", "legacy")).lower() == "uniform_full_length":
            cell_ops["L_heatloss_norm"] = max(float(self.geometry["L"]), 1e-9)
        else:
            cell_ops["L_heatloss_norm"] = max(float(np.sum(dz_list)), 1e-9)
        cell_ops["pyrolysis_done"] = True
        return cell_ops

    def _make_cell(self, i, dz_list, A, current_inlet, inlet, cell_ops, ref_f, ref_e):
        F_H2O_evap = (self.tmp_W_liq_evap / 18.015) * 1000.0
        L_evap = self.op_conds.get("L_evap_m", 1e-6)
        cell = Cell(
            i, self.z_positions[i], dz_list[i], A * dz_list[i], A, current_inlet,
            self.kinetics, self.coal_props, cell_ops,
            sources=[
                EvaporationSource(F_H2O_evap, L_evap_m=L_evap),
                PyrolysisSource(self.tmp_F_vol, self.tmp_W_vol, target_cell_idx=0, T_pyro=inlet.T),
            ],
        )
        cell.coal_flow_dry = self.W_dry
        cell.Cd_total = self.Cd_total
        cell.char_Xc0 = self.char_Xc0
        cell.ref_flow, cell.ref_energy = ref_f, ref_e
        return cell

    @staticmethod
    def _is_no_root(cell_cost: float) -> bool:
        """判定一个单元是否为「无根」。

        非有限成本（NaN/Inf）一律视为无根。`NaN > 1e-4` 为 False，
        若只用阈值比较，含 NaN 的状态会被当作合格的根保留下来并报告 completed。
        """
        return (not math.isfinite(float(cell_cost))) or float(cell_cost) > 1e-4

    def _enforce_strict_root_on_states(self, dz_list, inlet, A, N):
        """对 batch 求解结果执行严格根复核（P1.5-A B2 在 jax_jit 上的等价实现）。

        未开启 SolverStrictRoot 时只记录逐单元成本，不改变结果。
        """
        strict_root = bool(self.op_conds.get("SolverStrictRoot", False))
        cell_ops = self._cell_ops_for_solve(dz_list)
        ref_f = max(inlet.total_gas_moles, 1.0)
        ref_e = max(ref_f * 35.0 * 200.0, 5.0e5)
        current_inlet = inlet
        for i in range(N):
            cell = self._make_cell(i, dz_list, A, current_inlet, inlet, cell_ops, ref_f, ref_e)
            state = self.results[i]
            x = state.to_array()
            # 状态本身含 NaN 时残差必为 NaN，成本判定会走 _is_no_root 的同一分支。
            residuals_now = np.asarray(cell.residuals(x), dtype=float)
            cell_cost = float(0.5 * np.sum(residuals_now ** 2))
            diag = self._build_cell_diagnostic(i, cell, state, cell_cost, residuals_now, False)
            self.cell_diagnostics.append(diag)
            if self._is_no_root(cell_cost):
                self.solve_stats["poor_convergence_count"] = int(
                    self.solve_stats.get("poor_convergence_count", 0)) + 1
                if strict_root:
                    self.execution_status = "no_root"
                    self.no_root_cell_index = i
                    logger.warning(
                        "SolverStrictRoot (jax_jit): cell %d residual_cost=%s, truncating.",
                        i, cell_cost,
                    )
                    # 截断到首个非根之前：下游结果来自非根解，不能当作有效出口
                    self.results = self.results[:i]
                    return
            current_inlet = state

    def _solve_sequential(self, dz_list, inlet, A, solver_method, jacobian_mode):
        N_cells = len(dz_list); current_inlet = inlet; self.results = []; self.cells = []
        ref_f = max(inlet.total_gas_moles, 1.0); ref_e = max(ref_f * 35.0 * 200.0, 5.0e5)
        cell_ops = self._cell_ops_for_solve(dz_list)
        strict_root = bool(self.op_conds.get("SolverStrictRoot", False))
        cost_threshold = 1e-4

        def _newton_solver():
            return NewtonSolver(
                tol=1e-8,
                max_iter=100,
                damper=0.8,
                jacobian='centered' if jacobian_mode == "centered_fd" else 'finite_difference',
            )

        def _least_squares_with_optional_jac(func_local, x0, bounds):
            kwargs = dict(
                bounds=bounds,
                method='trf',
                xtol=float(self.op_conds.get("SolverXtol", 1e-12)),
                ftol=float(self.op_conds.get("SolverFtol", 1e-8)),
                max_nfev=int(self.op_conds.get("SolverMaxNfev", 1000)),
            )
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
                if str(self.op_conds.get("Cell0GuessMode", "pyrolysis_only")) == "oxidation_prestep":
                    xi1 = min(x0[1] * 0.99, x0[0] * 0.99 / 2.0)
                    x0[1] -= xi1
                    x0[0] -= 2.0 * xi1
                    x0[2] += xi1
                    x0[7] += 2.0 * xi1
                    xi2 = min(x0[5] * 0.99, x0[0] * 0.99 / 0.5)
                    x0[5] -= xi2
                    x0[0] -= 0.5 * xi2
                    x0[7] += xi2
                    xi3 = min(x0[2] * 0.99, x0[0] * 0.99 / 0.5)
                    x0[2] -= xi3
                    x0[0] -= 0.5 * xi3
                    x0[3] += xi3
            return x0

        # Solver bounds: constant across all cells, built once here
        lower = np.zeros(11)
        upper = np.ones(11) * 3000.0
        lower[8] = 0.0;  upper[8] = self.W_dry * 1.5
        lower[9] = 0.0;  upper[9] = 1.0
        lower[10] = 300.0; upper[10] = 4000.0

        for i in range(N_cells):
            cell = self._make_cell(i, dz_list, A, current_inlet, inlet, cell_ops, ref_f, ref_e)
            self.cells.append(cell)
            def func(x): return cell.residuals(x)

            if i == 0:
                guesses = list(self.op_conds.get("Cell0GuessTemperatures", [3000.0, 2000.0, 1500.0, 1000.0, 400.0]))
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
                    branches = []
                    for t in guesses:
                        x0 = _make_cell0_x0(t)
                        sol_try = _least_squares_with_optional_jac(func, x0, (lower, upper))
                        # 与 solver.cost 同一口径：0.5*||F_scaled||^2
                        c_try = float(0.5 * np.sum(np.asarray(func(sol_try.x), dtype=float) ** 2))
                        branches.append({
                            "guess_T": float(t),
                            "cost": c_try,
                            "T_cell0": float(sol_try.x[10]),
                            "success": bool(sol_try.success),
                        })
                        if sol_try.success and c_try < best_cost:
                            best_sol, best_cost = sol_try, c_try
                    sol = best_sol if best_sol is not None else sol_try
                    self.cell0_branches = branches
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

            if sol is None or not hasattr(sol, "x"):
                # 无任何可用解：无论是否严格模式都无法继续推进
                self.execution_status = "no_root"
                self.no_root_cell_index = i
                self.cell_diagnostics.append({
                    "cell_index": i, "dz_m": float(dz_list[i]),
                    "residual_cost": float("inf"), "solver_success": False,
                    "strict_no_root": True, "note": "solver returned no solution",
                })
                break

            sol_state = StateVector.from_array(sol.x, P=inlet.P, z=self.z_positions[i])
            residuals_now = np.asarray(func(sol.x), dtype=float)
            cell_cost = float(0.5 * np.sum(residuals_now ** 2))
            solver_failed = not bool(getattr(sol, "success", True))

            diag = self._build_cell_diagnostic(i, cell, sol_state, cell_cost, residuals_now, solver_failed)
            self.cell_diagnostics.append(diag)

            # 非有限成本同样是无根：NaN 与阈值比较为 False，若只比阈值会放过它。
            no_root = solver_failed or self._is_no_root(cell_cost)

            if no_root:
                self.solve_stats["poor_convergence_count"] = int(self.solve_stats.get("poor_convergence_count", 0)) + 1

            # P1.5-A B2：严格模式拒绝非根解，停止向下游推进，不把非根解当作下一格入口
            if strict_root and no_root:
                self.execution_status = "no_root"
                self.no_root_cell_index = i
                logger.warning(
                    "SolverStrictRoot: cell %d residual_cost=%.3e > %.0e, stopping.",
                    i, cell_cost, cost_threshold,
                )
                break

            phys = cell._calc_physics_props(sol_state, self.W_dry*self.Cd_total)
            tau = dz_list[i] / max(phys['v_g'], 0.1)
            _, Ts_out = cell._calc_particle_temperature(
                sol_state.T, current_inlet.T_solid_or_gas, tau, phys['d_p'],
                sol_state.solid_mass, sol_state.carbon_fraction,
                rho_p=phys.get('rho_p'),
            )
            sol_state.T_solid = Ts_out; self.results.append(sol_state); current_inlet = sol_state
        return self._package_results()

    def _build_cell_diagnostic(self, i, cell, sol_state, cell_cost, residuals_now, solver_failed):
        """P1.5-A B5/B6：逐单元记录残差成本、主残差分量与限幅激活布尔量。"""
        names = (
            tuple(f"gas_{n}" for n in SPECIES_NAMES)
            + ("solid_mass", "carbon_fraction", "energy")
        )
        idx_max = int(np.argmax(np.abs(residuals_now))) if residuals_now.size else -1
        row = {
            "cell_index": int(i),
            "dz_m": float(cell.dz),
            "residual_cost": float(cell_cost),
            "solver_success": not bool(solver_failed),
            "strict_no_root": False,
            "max_abs_scaled_residual": float(np.max(np.abs(residuals_now))) if residuals_now.size else float("nan"),
            "dominant_residual": names[idx_max] if idx_max >= 0 else "",
            "T_K": float(sol_state.T),
            "solid_mass_kg_s": float(sol_state.solid_mass),
            "carbon_fraction": float(sol_state.carbon_fraction),
        }
        try:
            snap = cell.snapshot_with_limits(sol_state)
            row.update({
                "is_combustion_zone": bool(snap["limits"].get("is_combustion_zone", False)),
                "d_p_m": float(snap["phys"]["d_p"]),
                "rho_p_kg_m3": float(snap["phys"]["rho_p"]),
                "S_total_m2": float(snap["phys"]["S_total"]),
                "X_total": float(snap["phys"]["X_total"]),
                "Ts_avg_K": float(snap["Ts_avg"]),
                "phi": float(snap["phi"]),
            })
            row.update({k: bool(v) for k, v in snap["limits"].items() if k != "is_combustion_zone"})
        except Exception as exc:  # 诊断失败不改变求解结果
            row["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        return row

    @property
    def cell0_n_distinct_roots(self) -> int:
        """cell 0 的不同根个数：cost < 1e-10 且温度相差 > 50 K 的初值分支。"""
        roots = [b for b in getattr(self, "cell0_branches", []) if float(b.get("cost", 1e18)) < 1e-10]
        distinct = []
        for b in roots:
            if all(abs(float(b["T_cell0"]) - float(o["T_cell0"])) > 50.0 for o in distinct):
                distinct.append(b)
        return len(distinct)

    @property
    def n_cells_above_cost_threshold(self) -> int:
        return sum(1 for d in getattr(self, "cell_diagnostics", []) if float(d.get("residual_cost", 0.0)) > 1e-4)

    @property
    def max_residual_cost(self) -> float:
        costs = [float(d.get("residual_cost", float("nan"))) for d in getattr(self, "cell_diagnostics", [])]
        costs = [c for c in costs if np.isfinite(c)]
        return float(max(costs)) if costs else float("nan")

    def _package_results(self):
        rows = [s.to_array() for s in self.results]; return np.array(rows), self.z_positions
