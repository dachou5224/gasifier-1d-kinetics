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
from .source_terms import EvaporationSource, OxygenMixingSource, PyrolysisSource
from .species_bridge import mainline_gas8_to_jax9, jax9_to_mainline_gas8
from .input_contract import (
    OXYGEN_MIXING_INSTANT,
    PYROLYSIS_HEAT_MODEL_ZERO,
    SOLID_ENTHALPY_MODEL_DEFAULT,
    ash_mass_fraction_dry,
    coal_flow_kg_s_for_heat_loss,
    coal_hhv_kj_kg,
    coal_lhv_kj_kg,
    heat_loss_norm_length_m,
    heat_loss_ref_temp_k,
    resolve_f_s_coal,
    resolve_oxygen_mixing_model,
    resolve_pyrolysis_heat_model,
    uses_fortran_default_model,
    validate_coal_heating_values,
)
from .solver import NewtonSolver
from .jax_residual_adapter import make_jacobian_fn
from .heat_loss import (
    HEAT_LOSS_MODEL_FORTRAN,
    HEAT_LOSS_MODEL_TOTAL,
    HEAT_LOSS_MODEL_TOTAL_UNIFORM,
    heat_loss_weight_relative_residual,
    resolve_heat_loss_model,
    temperature_weight,
    total_heat_loss_w,
)
import logging

logger = logging.getLogger(__name__)

# P1.12-A：整炉总量热损的外层不动点判据（权重相对变化）
TOL_DW = 1e-4

_LEGACY_SOLVER_ALIASES = {    "newton": "newton_fd",
    "jax_newton": "newton_fd",
    "jax_pure": "newton_fd",
}

# WGS 平衡常数式子 → jax_kinetics 的整数 id（P1.6-A）。在此重复一份是为了避免
# 在 gasifier_system 导入期把 jax 拉进来，那会绕过 jax_enable_x64 的时序。
_WGS_KEQ_MODEL_IDS = {"thermo": 0, "legacy_fit": 1, "fortran": 2}

def _hf_coal_from_pyrolysis_products(coal_props: dict, T_pyro: float) -> float:
    """``PyrolysisHeatModel="zero"``：由热解产物焓与焦炭焓反推煤的生成焓 [J/kg 干煤]。

    令 ``H_coal(T) = H_char(T) + H_vol(T)``（热解热恰为 0），而
    ``H_coal = m_coal·(cp·ΔT + Hf)``、``H_char = m_char·cp·ΔT``，于是

        Hf_coal = [H_vol(T) − (m_coal − m_char)·cp·ΔT] / m_coal
                = [H_vol(T) − m_vol·cp·ΔT] / m_coal

    不读取任何热值。``T_pyro`` 取该算例 cell 0 的热解温度（即 ``op_conds['T_in']``），
    因此这一口径下 cell 0 的热解热在定义上恰为 0，Fortran 的「煤无生成焓」记账与之一致。
    """
    service = PyrolysisService()
    molar_yields, w_vol = service.calc_yields(coal_props)  # mol/kg_dry, kg/kg_dry
    cp_s = float(coal_props.get("cp_char", 1300.0))
    dT = float(T_pyro) - 298.15
    h_vol = 0.0
    for idx, sp in enumerate(SPECIES_NAMES):
        if molar_yields[idx] > 0.0:
            h_vol += float(molar_yields[idx]) * float(get_enthalpy_molar(sp, float(T_pyro)))
    m_vol = float(w_vol)
    return float(h_vol - m_vol * cp_s * dT)


def compute_hf_coal_j_kg(coal_props: dict, op_conds: dict) -> tuple[float, str]:
    """返回 ``(Hf_coal [J/kg 干煤], 口径标记)``。

    * ``from_heating_value``：``Hf = Σ(元素→氧化物生成焓) + LHV_d``。
      给定 ``LHV_d`` 时直接用、不再扣潜热；只给 ``HHV_d`` 时保持修改前的行为
      （按高位处理、扣 ``9·Hd·2442``）；两者都给由
      ``validate_coal_heating_values`` 报错。
    * ``zero``：见 ``_hf_coal_from_pyrolysis_products``，不读任何热值。
    """
    model = resolve_pyrolysis_heat_model(op_conds)
    if model == PYROLYSIS_HEAT_MODEL_ZERO:
        return _hf_coal_from_pyrolysis_products(coal_props, float(op_conds["T_in"])), model
    cd = float(coal_props.get("Cd", 0.0)) / 100.0
    hd = float(coal_props.get("Hd", 0.0)) / 100.0
    sd = float(coal_props.get("Sd", 0.0)) / 100.0
    lhv_kj = coal_lhv_kj_kg(coal_props)
    # 每 kg 干煤完全氧化到 CO2(g)/H2O(g)/SO2(g) 的生成焓之和 [kJ/kg]
    comb_kj = (
        (cd / 0.012011) * (-393.51)
        + (hd / 0.001008) * 0.5 * (-241.83)
        + (sd / 0.03206) * (-296.81)
    )
    return float((comb_kj + lhv_kj) * 1000.0), model


class GasifierSystem:
    def __init__(self, geometry, coal_props, op_conds):
        self._validate_inputs(geometry, coal_props, op_conds)
        self.geometry = geometry
        self.coal_props = coal_props.copy()  # copy to avoid mutating caller's dict (Hf_coal may be added below)
        self.op_conds = op_conds.copy()
        # P1.7-A：低位与高位同时出现直接报错；热损基准另用 coal_hhv_kj_kg（只给低位时按定义推高位）。
        validate_coal_heating_values(self.coal_props)
        self.pyrolysis_heat_model = resolve_pyrolysis_heat_model(self.op_conds)
        if "Hf_coal" in self.coal_props and self.pyrolysis_heat_model != PYROLYSIS_HEAT_MODEL_ZERO:
            self.hf_coal_source = "given"
        else:
            if "Hf_coal" in self.coal_props:
                logger.warning(
                    "PyrolysisHeatModel='zero'：忽略调用方给出的 Hf_coal=%s，改由热解产物焓反推",
                    self.coal_props["Hf_coal"],
                )
            self.coal_props["Hf_coal"], _mf = compute_hf_coal_j_kg(self.coal_props, self.op_conds)
            self.hf_coal_source = _mf
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
                PyrolysisSource(
                    self.tmp_F_vol, self.tmp_W_vol, target_cell_idx=0, T_pyro=inlet.T,
                    coal_mass_flow_kg_s=self.W_dry, coal_props=self.coal_props,
                    solid_enthalpy_model=self.op_conds.get(
                        "SolidEnthalpyModel", SOLID_ENTHALPY_MODEL_DEFAULT),
                ),
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
        # P1.12-A 输入敏感性：Mt 单独扰动。只改「湿煤中有多少水」，
        # 保持干煤质量不变（W_wet 相应调整），因为干煤量由称重给煤机标定。
        _Mt_over = self.op_conds.get("Mt_override_pct")
        if _Mt_over is not None:
            _mt_new = float(_Mt_over)
            if not (0.0 <= _mt_new < 100.0):
                raise ValueError(f"Mt_override_pct 越界：{_mt_new}")
            W_wet = W_dry / (1.0 - _mt_new / 100.0)
            self.op_conds['coal_flow'] = W_wet
            Mt = _mt_new
            self.coal_props = {**self.coal_props, "Mt": _mt_new}
        slurry = self.op_conds.get('SlurryConcentration', 100.0); W_steam = self.op_conds.get('steam_flow', 0.0)
        W_coal_moist = W_wet * Mt/100.0
        W_slurry_h2o = (W_dry / (slurry/100.0)) - W_dry if slurry < 100.0 else 0.0
        self.tmp_W_h2o_total = W_coal_moist + W_slurry_h2o + W_steam
        self.tmp_W_liq_evap = W_coal_moist + W_slurry_h2o
        F_O2 = (self.op_conds['o2_flow'] / 31.998) * 1000.0; F_CO2, F_N2 = self._carrier_gas_molar_flows(); F_st = W_steam / 18.015 * 1000.0
        # P1.12-A 输入敏感性：O2 流量单独扰动（O/C 是温度与 X_C 的首要驱动量）。
        _o2s = self.op_conds.get("o2_flow_scale")
        if _o2s is not None:
            _f = float(_o2s)
            if not (0.0 < _f < 10.0):
                raise ValueError(f"o2_flow_scale 越界：{_f}")
            F_O2 = F_O2 * _f
        # P1.10-A：OxygenMixingModel != "instant" 时，入口 O2 作为**不反应的旁流**，
        # 反应流在 z = 0 不含 O2；旁流量按累计分数 F(z) 逐格并入（见 OxygenMixingSource）。
        # "instant"（缺省）不进入这一分支，gas[0] 仍为 F_O2 → 现行行为逐位不变。
        self.o2_mixing_model = resolve_oxygen_mixing_model(self.op_conds)
        self.o2_bypass_mol_s = 0.0
        self.o2_mixing_length_m = float("nan")
        if self.o2_mixing_model != OXYGEN_MIXING_INSTANT:
            L_mix_raw = self.op_conds.get("OxygenMixingLength_m")
            if L_mix_raw is None:
                raise ValueError(
                    f"OxygenMixingModel={self.o2_mixing_model!r} 需要 OxygenMixingLength_m（m），"
                    "未给出"
                )
            L_mix_val = float(L_mix_raw)
            if not np.isfinite(L_mix_val) or L_mix_val <= 0.0:
                raise ValueError(
                    f"OxygenMixingLength_m 必须为有限正数，得到 {L_mix_raw!r}"
                )
            # P1.10-A 审阅意见：L_mix 超过反应器长度时，F(z) 到出口仍 < 1，
            # 旁流会有一部分**永远没有并入**；但入口元素账本计的是全部旁流，
            # 于是 O2 静默丢失且守恒检查会显示一个假的"闭合"。这里直接拒绝。
            L_reactor = float(self.geometry["L"])
            if L_mix_val > L_reactor * (1.0 + 1e-12):
                raise ValueError(
                    f"OxygenMixingLength_m={L_mix_val} 超过反应器长度 L={L_reactor}；"
                    "此时 F(L) < 1，部分旁流 O2 永远不并入，元素守恒会被破坏。"
                    "请给出 L_mix <= L，或改用其它混合描述。"
                )
            self.o2_mixing_length_m = L_mix_val
        gas = np.zeros(8); gas[3]=F_CO2; gas[6]=F_N2; gas[7]=F_st
        if self.o2_mixing_model == OXYGEN_MIXING_INSTANT:
            gas[0] = F_O2
        else:
            self.o2_bypass_mol_s = F_O2
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
            graded_zone_length=self.op_conds.get('GradedZoneLength'),
            graded_zone_dz=self.op_conds.get('GradedZoneDz'),
            rest_cell_length=self.op_conds.get('RestCellLength'),
            wsr_length=self.op_conds.get('WSRLength_m'),
            pfr_dz=self.op_conds.get('PFRCellLength_m'),
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
        # P1.10-A 审阅意见（第二条防线）：即使 L_mix <= L，若网格在 L_mix 之前就结束
        # （例如截断网格或块覆盖不足），同样会留下未并入的旁流。这里核一次覆盖。
        if resolve_oxygen_mixing_model(self.op_conds) != OXYGEN_MIXING_INSTANT:
            L_mix_cfg = float(self.op_conds.get("OxygenMixingLength_m", float("nan")))
            z_end = float(np.sum(dz_list))
            if not np.isfinite(L_mix_cfg):
                raise ValueError("OxygenMixingLength_m 未给出或非有限")
            if z_end < L_mix_cfg * (1.0 - 1e-12):
                raise ValueError(
                    f"网格末端 z={z_end:.6f} m 小于 OxygenMixingLength_m="
                    f"{L_mix_cfg:.6f} m：旁流 O2 无法全部并入，元素守恒会被破坏。"
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
        # P1.12-A：整炉总量热损需要外层不动点迭代（逐格推进时不知道下游温度）。
        if resolve_heat_loss_model(self.op_conds) == HEAT_LOSS_MODEL_TOTAL:
            return self._solve_total_heat_loss(dz_list, inlet, A, solver_method, jacobian_mode)
        if resolve_heat_loss_model(self.op_conds) == HEAT_LOSS_MODEL_TOTAL_UNIFORM:
            # 均匀分配无需迭代：w_i = dz_i，ΣQ 逐位等于 Q_tot
            self.op_conds["HeatLossWeightSum"] = float(np.sum(np.asarray(dz_list, dtype=float)))
            self.op_conds["HeatLossIteration"] = 0
        return self._solve_sequential(dz_list, inlet, A, solver_method, jacobian_mode)

    def _solve_total_heat_loss(self, dz_list, inlet, A, solver_method, jacobian_mode):
        """``total_percent`` 的外层不动点迭代（P1.12-A 卡第 1 节）。

        逐格推进的求解器在解第 i 格时不知道下游温度，因此第 k+1 轮的权重
        ``w_i = dz_i·clip((T_i/1800)^4, 0.1, 5)`` 用第 k 轮的全炉温度场。
        第 0 轮按长度均匀（``w_i = dz_i``）。

        收敛：(a) ``|ΣQ_i − Q_tot| / Q_tot < 1e-6`` 且
        (b) 未松弛残差 ``max|(w_new − w)/w| < 1e-4``（``w_new = dz · temperature_weight(T)``）；
        最多 30 轮。欠松弛只用于推进迭代，不进入收敛判据。未收敛时
        ``hloss_status`` 置 ``hloss_not_converged``，但**保留**该轮解，供判读时如实报告。

        注意 (a) 在收敛前不成立：``Σ w_i/(Σw) = 1`` 只在权重与温度场自洽时才有意义，
        非自洽权重下 ΣQ 仍等于 Q_tot（因为分母就是同一轮权重的和），所以真正的
        判据是 (b)——(a) 在这里几乎恒成立，保留它只为符合卡面文字的核对。
        """
        dz = np.asarray(dz_list, dtype=float)
        hl_tot = float(self.op_conds.get('HeatLossPercentTotal', float("nan")))
        # 允许本轮脚本覆盖权重容差（仅用于容差敏感性对照；缺省用模块常量）
        _tol_dw = float(self.op_conds.get("HeatLossTolerance", TOL_DW))
        if not np.isfinite(hl_tot) or hl_tot < 0.0:
            raise ValueError(
                f"HeatLossModel='total_percent' 需要非负的 HeatLossPercentTotal（%），得到 {hl_tot!r}"
            )
        # 干煤口径的分母（P1.12-A）：与实测汽包吸热同口径
        m_dry = float(self.op_conds.get('coal_dry_flow_for_heat_loss', self.W_dry))
        hhv_kj = float(coal_hhv_kj_kg(self.coal_props))
        Q_tot = total_heat_loss_w(m_dry, hhv_kj, hl_tot)

        w = dz.copy()                      # 第 0 轮：按长度均匀
        alpha = 1.0
        last_raw_dw = float("inf")
        # P1.12-A 单元检查 4：允许外部指定第 0 轮温度场，用于验证迭代与初值无关
        _T0 = self.op_conds.get("HeatLossInitTemperatures")
        if _T0 is not None:
            _T0 = np.asarray(_T0, dtype=float).ravel()
            if _T0.size >= dz.size:
                w = dz * np.array(
                    [temperature_weight(_T0[i], self.op_conds.get('HeatLossRefTemp', 1800.0))
                     for i in range(dz.size)], dtype=float)
            else:
                # 长度不匹配时按最近邻重采样（P1.11-A 用 dz_down=0.02575，本卡用 0.0515）
                _z0 = np.linspace(0.0, 1.0, _T0.size)
                _z1 = np.linspace(0.0, 1.0, dz.size)
                _Tr = np.interp(_z1, _z0, _T0)
                w = dz * np.array(
                    [temperature_weight(_Tr[i], self.op_conds.get('HeatLossRefTemp', 1800.0))
                     for i in range(dz.size)], dtype=float)
        iters = 0
        converged = False
        hist: list[dict] = []
        prev_T: np.ndarray | None = None
        for it in range(30):
            iters = it
            self.op_conds["HeatLossWeights"] = w
            self.op_conds["HeatLossWeightSum"] = float(np.sum(w))
            self.op_conds["HeatLossIteration"] = int(it)
            # 每轮重置求解状态，避免把上一轮的解当作本轮入口
            self.results = []
            self.cells = []
            self.cell_diagnostics = []
            self.execution_status = "completed"
            self.no_root_cell_index = None
            self.cell0_branches = []
            self.solve_stats = {"fallback_count": 0, "poor_convergence_count": 0}
            profile, z = self._solve_sequential(dz_list, inlet, A, solver_method, jacobian_mode)

            n_solved = len(self.results)
            if self.execution_status != "completed" or n_solved != dz.size:
                # 求解未跑完：权重覆盖不全，本轮不可用作迭代基准，直接返回
                hist.append({"iter": int(it), "n_solved": int(n_solved),
                             "status": str(self.execution_status),
                             "sum_Q_over_Qtot": float("nan"), "max_rel_dw": float("nan")})
                break
            T = np.array([float(s.T) for s in self.results], dtype=float)
            w_new = dz * np.array(
                [temperature_weight(T[i], self.op_conds.get('HeatLossRefTemp', 1800.0))
                 for i in range(dz.size)], dtype=float)
            # 欠松弛：直接替换在 T⁴ 权重下会振荡（实测 L_wsr=0.2575、HL_tot=4% 时
            # 30 轮不收敛）。松弛只改变迭代路径，**不动点不变**：
            # 收敛时 w = w_new，与无松弛的方程组完全一致。
            # 自适应：本步的相对变化比上一步更大 → 判为振荡，收缩 0.6；否则缓慢放大。
            raw_dw = (heat_loss_weight_relative_residual(w, w_new)
                      if prev_T is not None else float("inf"))
            if prev_T is not None:
                if raw_dw > last_raw_dw:
                    alpha = max(0.1, alpha * 0.6)
                elif raw_dw < 0.5 * last_raw_dw:
                    alpha = min(1.0, alpha / 0.8)
            last_raw_dw = raw_dw
            w_target = w + alpha * (w_new - w)
            # ΣQ_i / Q_tot（用本轮权重）
            Q_sum = float(sum(float(d.get("hloss_W", 0.0)) for d in self.cell_diagnostics))
            sum_rel = abs(Q_sum - Q_tot) / max(abs(Q_tot), 1e-30)
            step_dw = (heat_loss_weight_relative_residual(w, w_target)
                       if prev_T is not None else float("inf"))
            hist.append({"iter": int(it), "n_solved": int(n_solved),
                         "status": "completed", "sum_Q_over_Qtot": float(sum_rel),
                         "max_rel_dw": float(raw_dw), "step_rel_dw": float(step_dw),
                         "alpha": float(alpha)})
            # 判据：ΣQ 残差（分子分母同权重，恒成立）+ 未松弛权重残差 max|w_new−w|/w < 1e-4。
            # 卡面两条都要求；后者是实现收敛的真正判据。松弛步长只用于推进，不用于判定。
            if it > 0 and sum_rel < 1e-6 and raw_dw < _tol_dw:
                converged = True
                break
            if np.array_equal(w_target, w):
                # 权重已逐位不变：不动点达成
                converged = True
                break
            w = w_target
            prev_T = T
        self.heat_loss_iterations = int(iters) + 1
        self.heat_loss_history = hist
        self.heat_loss_converged = bool(converged)
        last = hist[-1] if hist else {}
        self.heat_loss_raw_dw = float(last.get("max_rel_dw", float("nan")))
        self.heat_loss_step_dw = float(last.get("step_rel_dw", float("nan")))
        self.heat_loss_alpha = float(last.get("alpha", float("nan")))
        # ΣQ / Q_tot 最终值
        Q_sum_final = float(sum(float(d.get("hloss_W", 0.0)) for d in self.cell_diagnostics))
        self.heat_loss_Q_sum_W = Q_sum_final
        self.heat_loss_Q_tot_W = float(Q_tot)
        # 卡面：未收敛的行「标 hloss_not_converged」。这里**不覆盖** execution_status——
        # 热化学求解本身已完成，覆盖会让 run_case 不再计算出口量（X_C 变 NaN），
        # 反而丢掉必须报告的数值。用一个独立的标记列承载该状态。
        self.hloss_status = "converged" if converged else "hloss_not_converged"
        return self._package_results()

    def _pyrolysis_source(self) -> PyrolysisSource:
        """P1.6-A（B9）：NumPy/JAX 两条路径共用的热解源（含热解热口径）。"""
        return PyrolysisSource(
            self.tmp_F_vol, self.tmp_W_vol, target_cell_idx=0, T_pyro=self.op_conds["T_in"],
            coal_mass_flow_kg_s=self.W_dry, coal_props=self.coal_props,
            solid_enthalpy_model=self.op_conds.get("SolidEnthalpyModel", SOLID_ENTHALPY_MODEL_DEFAULT),
        )

    def _solve_jax_v4(self, dz_list, inlet, A):
        # P1.8-A：JAX 路径只实现了 hhv_percent 热损。fortran_wall 需要在本格内先算
        # 净反应热再取其 30%，这在 batch 求解里没有对应实现——必须报错，不得静默回退。
        if resolve_heat_loss_model(self.op_conds) == HEAT_LOSS_MODEL_FORTRAN:
            raise ValueError(
                "HeatLossModel='fortran_wall' is not implemented on the JAX batch path "
                "(GasifierSystem._solve_jax_v4). Use solver_method='minimize'/'newton_fd' "
                "(the NumPy sequential path), or set HeatLossModel='hhv_percent'."
            )
        # P1.12-A：整炉总量热损需要外层不动点迭代，JAX 批处理路径无此结构——必须报错。
        if resolve_heat_loss_model(self.op_conds) in (
            HEAT_LOSS_MODEL_TOTAL, HEAT_LOSS_MODEL_TOTAL_UNIFORM,
        ):
            raise ValueError(
                "HeatLossModel in ('total_percent', 'total_percent_uniform') is not implemented "
                "on the JAX batch path (GasifierSystem._solve_jax_v4); the outer fixed-point "
                "iteration over the temperature field has no batch equivalent. Use "
                "solver_method='minimize'/'newton_fd' or set HeatLossModel='hhv_percent'."
            )
        if float(self.op_conds.get("GasificationRateFactor", 1.0)) != 1.0:
            raise ValueError(
                "GasificationRateFactor is not implemented on the JAX batch path "
                "(GasifierSystem._solve_jax_v4); it would be silently ignored."
            )
        # P1.10-A：O2 分布式进入在 JAX 批处理路径上没有对应实现——必须报错，不得静默回退。
        if resolve_oxygen_mixing_model(self.op_conds) != OXYGEN_MIXING_INSTANT:
            raise ValueError(
                "OxygenMixingModel != 'instant' is not implemented on the JAX batch path "
                "(GasifierSystem._solve_jax_v4). Use solver_method='minimize'/'newton_fd' "
                "(the NumPy sequential path), or set OxygenMixingModel='instant'."
            )
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
                # P1.6-A（B9）：热解热与挥发分焓直接复用 NumPy 侧的 PyrolysisSource，
                # 保证两条路径共用同一份口径（legacy 口径下退化为仅挥发分焓）。
                e_src[i] += self._pyrolysis_source().energy_src_W

        solid_hf_inlet = np.zeros(N)
        solid_hf_inlet[0] = 1.0

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
        hhv = coal_hhv_kj_kg(self.coal_props); hhv_mj = hhv / 1000.0
        Ash_d = ash_mass_fraction_dry(self.coal_props)
        f_ash = Ash_d / (self.Cd_total + Ash_d + 1e-9)
        ref_f = self._ref_flow_scale(inlet); ref_e = max(ref_f * 35.0 * 200.0, 5.0e5)
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
            jnp.asarray(solid_hf_inlet, dtype=jnp.float64),
            float(self.op_conds.get("SolidEnthalpyModel", SOLID_ENTHALPY_MODEL_DEFAULT) == "legacy_coal_hf"),
            float(_WGS_KEQ_MODEL_IDS.get(str(self.op_conds.get("WGS_Keq_Model", "thermo")).lower(), 0)),
            float(self.op_conds.get("HetConversionCutoff", "smooth") == "hard_0999"),
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
        if str(self.op_conds.get("MeshFamily", "legacy")).lower() in (
            "uniform_full_length",
            "graded_full_length",
            "graded_block_tail",
            "wsr_plus_pfr",
        ):
            cell_ops["L_heatloss_norm"] = max(float(self.geometry["L"]), 1e-9)
        else:
            cell_ops["L_heatloss_norm"] = max(float(np.sum(dz_list)), 1e-9)
        cell_ops["pyrolysis_done"] = True
        return cell_ops

    def _make_cell(self, i, dz_list, A, current_inlet, inlet, cell_ops, ref_f, ref_e):
        F_H2O_evap = (self.tmp_W_liq_evap / 18.015) * 1000.0
        L_evap = self.op_conds.get("L_evap_m", 1e-6)
        sources = [
            EvaporationSource(F_H2O_evap, L_evap_m=L_evap),
            PyrolysisSource(
                self.tmp_F_vol, self.tmp_W_vol, target_cell_idx=0, T_pyro=inlet.T,
                coal_mass_flow_kg_s=self.W_dry, coal_props=self.coal_props,
                solid_enthalpy_model=cell_ops.get("SolidEnthalpyModel", SOLID_ENTHALPY_MODEL_DEFAULT),
            ),
        ]
        # P1.10-A：O2 旁流按 F(z) 逐格并入。源项进入 g_src[0]，因此
        # ① 气相 O2 平衡（cell._calc_balances 的 `inlet.gas_moles[0] + gas_src[0]`）
        # ② 燃烧区判据（同一表达式）③ 元素守恒 都会自动看到本格并入的 O2，
        # 无需另开口径 —— 这正是卡面「入口反应流 + 本格并入的 O2」。
        if self.o2_mixing_model != OXYGEN_MIXING_INSTANT:
            sources.append(OxygenMixingSource(
                o2_flow_mol_s=self.o2_bypass_mol_s,
                L_mix_m=float(self.op_conds.get("OxygenMixingLength_m", float("nan"))),
                model=self.o2_mixing_model,
                T_in_K=float(self.op_conds["T_in"]),
            ))
        cell = Cell(
            i, self.z_positions[i], dz_list[i], A * dz_list[i], A, current_inlet,
            self.kinetics, self.coal_props, cell_ops,
            sources=sources,
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
        ref_f = self._ref_flow_scale(inlet)
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

    def _ref_flow_scale(self, inlet) -> float:
        """残差缩放的参考摩尔流量。

        P1.10-A：``OxygenMixingModel != "instant"`` 时旁流 O2 不在 ``inlet.gas_moles`` 里，
        若仍按 ``inlet.total_gas_moles`` 定标，同一套方程会得到与 instant 不同的缩放，
        求解器停点随之漂移（实测残差同比放大 4.94 倍）。这里把旁流计入，
        使 instant 与 linear/exponential 在「全部 O2 落在首格」时共用同一缩放基准。
        ``instant`` 下 ``o2_bypass_mol_s`` 恒为 0，返回值与旧代码逐位相同。
        """
        return max(float(inlet.total_gas_moles) + float(self.o2_bypass_mol_s), 1.0)

    def _solve_sequential(self, dz_list, inlet, A, solver_method, jacobian_mode):
        N_cells = len(dz_list); current_inlet = inlet; self.results = []; self.cells = []
        ref_f = self._ref_flow_scale(inlet); ref_e = max(ref_f * 35.0 * 200.0, 5.0e5)
        cell_ops = self._cell_ops_for_solve(dz_list)
        # 首格会并入的旁流 O2 份额（初值用；instant 下为 0）
        _o2_frac_cell0 = 0.0
        if self.o2_mixing_model != OXYGEN_MIXING_INSTANT and len(dz_list) > 0:
            _probe = OxygenMixingSource(
                o2_flow_mol_s=self.o2_bypass_mol_s,
                L_mix_m=self.o2_mixing_length_m,
                model=self.o2_mixing_model,
                T_in_K=float(self.op_conds["T_in"]),
            )
            _o2_frac_cell0 = _probe.cell_fraction(self.z_positions[0], dz_list[0])
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
                # P1.10-A：旁流 O2 不在 current_inlet 里，但首格会并入一份。
                # 初值补上本格份额，使 instant 与 mixing 走同一初值路径——
                # 否则同一套方程会因起点不同而停在不同的收敛点（实测 0.018 K）。
                if self.o2_mixing_model != OXYGEN_MIXING_INSTANT:
                    x0[0] += self.o2_bypass_mol_s * _o2_frac_cell0
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
            # P1.10-A：本格 O2 并入量与判据用 pO2（审计用；instant 下并入量为 0）
            try:
                _src_tot = np.zeros(8)
                for _s in cell.sources:
                    _g, _sm, _e = _s.get_sources(cell.idx, cell.z, cell.dz)
                    _src_tot += _g
                _o2_in = max(float(cell.inlet.gas_moles[0]) + float(_src_tot[0]), 0.0)
                _f_tot = max(
                    float(np.sum([cell.inlet.gas_moles[_k] + _src_tot[_k] for _k in range(8)])),
                    1e-12,
                )
                row["o2_src_mol_s"] = float(_src_tot[0])
                row["pO2_criterion_atm"] = float(sol_state.P * (_o2_in / _f_tot) / 101325.0)
                # P1.11-A：全混流区尺寸是物理参数，需要逐格记录驻留时间与持料量，
                # 供 dz_dependence.md 量化「格长数米时是否仍是全混流」。
                _v_g = float(snap["phys"]["v_g"])
                _tau = float(cell.dz) / max(_v_g, 1e-12)
                row["v_g_m_s"] = _v_g
                row["tau_s"] = _tau
                row["m_hold_kg"] = float(sol_state.solid_mass) * _tau
                row["cell_volume_m3"] = float(cell.V)
            except Exception as exc:  # 诊断失败不改变求解结果
                row["diagnostic_o2_error"] = f"{type(exc).__name__}: {exc}"
            # P1.8-A：fortran_wall 下逐单元的 hloss / tw / tg（只记录，不改求解）
            if resolve_heat_loss_model(cell.op_conds) == HEAT_LOSS_MODEL_FORTRAN:
                try:
                    hl = cell.heat_loss_diagnostics(sol_state)
                    row.update({
                        "heat_loss_model": "fortran_wall",
                        "hloss_W": float(hl["hloss_W"]),
                        "hloss_zone": str(hl["zone"]),
                        "hloss_tw_K": float(hl["tw_K"]),
                        "hloss_tg_K": float(hl["tg_K"]),
                        "hloss_reaction_heat_W": float(hl["reaction_heat_W"]),
                    })
                except Exception as exc:  # 诊断失败不改变求解结果
                    row["heat_loss_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["heat_loss_model"] = resolve_heat_loss_model(cell.op_conds)
                # P1.9-A：hhv_percent 下也记录逐格 Q_loss，否则「HLP 不是整炉热损
                # 百分比」无法从产物核对（审阅意见）。与能量残差用同一算式。
                # P1.12-A：total_percent / total_percent_uniform 必须走各自的方法——
                # 用一个分支统一调 hhv_percent_hloss 会让审计列与方程不符
                # （实测 ΣQ/Q_tot 因此错报 0.768 而非 1）。
                try:
                    _model = resolve_heat_loss_model(cell.op_conds)
                    if _model in (HEAT_LOSS_MODEL_TOTAL, HEAT_LOSS_MODEL_TOTAL_UNIFORM):
                        hl = cell.total_percent_hloss(sol_state)
                        row.update({
                            "hloss_W": float(hl["hloss_W"]),
                            "hloss_zone": str(hl["hloss_zone"]),
                            "hloss_loss_pct": float(hl["hloss_loss_pct"]),
                            "hloss_Q_total_W": float(hl["hloss_Q_total_W"]),
                            "hloss_t_weight": float(hl["hloss_t_weight"]),
                            "hloss_w_cell": float(hl["hloss_w_cell"]),
                            "hloss_W_sum": float(hl["hloss_W_sum"]),
                            "hloss_iters": int(hl["hloss_iters"]),
                        })
                    else:
                        hl = cell.hhv_percent_hloss(sol_state)
                        row.update({
                            "hloss_W": float(hl["hloss_W"]),
                            "hloss_zone": str(hl["hloss_zone"]),
                            "hloss_loss_pct": float(hl["hloss_loss_pct"]),
                            "hloss_Q_total_W": float(hl["hloss_Q_total_W"]),
                            "hloss_t_weight": float(hl["hloss_t_weight"]),
                            "hloss_L_norm_m": float(hl["hloss_L_norm_m"]),
                            "hloss_dz_frac": float(hl["hloss_dz_frac"]),
                        })
                except Exception as exc:  # 诊断失败不改变求解结果
                    row["heat_loss_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
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
