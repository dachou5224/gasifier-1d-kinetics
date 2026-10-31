"""#4 审查意见的回归测试（2026-10-31）。

覆盖：
1. jax_jit 必须遵守 SolverStrictRoot（P1）
2. snapshot_with_limits 必须与 residuals 共享颗粒温度上下文（P2）
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT / "src"))


def _simple_system(**op_overrides):
    """最小可用系统，避免依赖外部数据。"""
    from model.gasifier_system import GasifierSystem

    coal = {
        "Cd": 70.0, "Hd": 4.5, "Od": 8.0, "Nd": 1.0, "Sd": 0.6,
        "Ad": 10.0, "Vd": 30.0, "FCd": 60.0, "Mt": 2.0, "HHV_d": 28000.0,
    }
    op = {
        "coal_flow": 1.0,
        "o2_flow": 0.9,
        "steam_flow": 0.2,
        "P": 2.0e6,
        "T_in": 500.0,
        "HeatLossPercent": 2.0,
        "ParticleSizeModel": "fortran_constant",
        "AllowTruncatedMesh": True,
    }
    op.update(op_overrides)
    return GasifierSystem({"L": 6.0, "D": 1.5}, coal, op)


# --------------------------------------------------- 1. jax_jit + SolverStrictRoot

def test_jax_path_has_equivalent_strict_root_enforcement():
    """jax_jit 不得静默忽略 SolverStrictRoot。

    回归审查意见（P1）：严格拒绝只在 _solve_sequential 实现，_solve_jax_v4 把
    所有单元打包返回，且 solve 开头把 execution_status 置为 completed，于是
    指定 SolverStrictRoot=True 后切到 jax_jit 仍会接受非根并报告 completed。
    """
    system = _simple_system(SolverStrictRoot=True)
    assert hasattr(system, "_enforce_strict_root_on_states"), \
        "jax path must expose the strict-root enforcement the batch solver uses"

    # 直接检验复核逻辑本身：造一个必然超过阈值的"结果"，应被标记 no_root 并截断。
    system._initialize_inlet()
    system.z_positions = np.array([0.5, 1.5])
    system.results = [
        __import__("model.state", fromlist=["StateVector"]).StateVector(
            gas_moles=np.full(8, 10.0), solid_mass=1.0, carbon_fraction=0.7, T=1500.0,
            P=2.0e6, z=0.5),
        __import__("model.state", fromlist=["StateVector"]).StateVector(
            gas_moles=np.full(8, 10.0), solid_mass=1.0, carbon_fraction=0.7, T=1500.0,
            P=2.0e6, z=1.5),
    ]
    system.execution_status = "completed"
    system.cell_diagnostics = []
    system.solve_stats = {"fallback_count": 0, "poor_convergence_count": 0}
    inlet = system.results[0]
    system._enforce_strict_root_on_states(np.array([1.0, 1.0]), inlet,
                                          math.pi * (1.5 / 2) ** 2, 2)
    # 这些人为状态显然不是根，严格模式下必须被拒
    assert system.execution_status == "no_root"
    assert system.no_root_cell_index == 0
    assert len(system.results) == 0   # 截断到首个非根之前


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_available(), reason="jax not installed")
def test_jax_jit_solve_actually_invokes_strict_root_enforcement(monkeypatch):
    """接线验证：jax_jit 求解必须真的走到严格根复核。

    仅测试 `_enforce_strict_root_on_states` 本身不够——还原修复只删掉
    `_solve_jax_v4` 里的那次调用，该方法仍在，单测照样通过。这里跑一次真实
    jax_jit 求解并计数调用。
    """
    from model.gasifier_system import GasifierSystem

    calls = {"n": 0}
    original = GasifierSystem._enforce_strict_root_on_states

    def spy(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GasifierSystem, "_enforce_strict_root_on_states", spy)

    system = _simple_system(SolverStrictRoot=True)
    system.solve(N_cells=3, solver_method="jax_jit", jax_warmup=False)
    assert calls["n"] == 1, "jax_jit must call the strict-root enforcement"


@pytest.mark.skipif(not _jax_available(), reason="jax not installed")
def test_jax_jit_reports_no_root_on_a_non_root_cell():
    """端到端：非根单元必须让 jax_jit 报 no_root 并截断结果。"""
    from model.gasifier_system import GasifierSystem

    system = _simple_system(SolverStrictRoot=True)
    rows, _z = system.solve(N_cells=3, solver_method="jax_jit", jax_warmup=False)
    assert system.execution_status == "no_root"
    assert system.no_root_cell_index is not None
    # 结果截断到首个非根之前
    assert len(rows) == system.no_root_cell_index
    assert system.cell_diagnostics, "per-cell diagnostics must be recorded on the jax path"


def test_jax_strict_root_does_not_truncate_when_disabled():
    """未开启严格模式时只记录成本，不改变结果长度。"""
    from model.state import StateVector

    system = _simple_system(SolverStrictRoot=False)
    system._initialize_inlet()
    system.z_positions = np.array([0.5, 1.5])
    system.results = [
        StateVector(gas_moles=np.full(8, 10.0), solid_mass=1.0, carbon_fraction=0.7,
                    T=1500.0, P=2.0e6, z=0.5),
        StateVector(gas_moles=np.full(8, 10.0), solid_mass=1.0, carbon_fraction=0.7,
                    T=1500.0, P=2.0e6, z=1.5),
    ]
    system.execution_status = "completed"
    system.cell_diagnostics = []
    system.solve_stats = {"fallback_count": 0, "poor_convergence_count": 0}
    system._enforce_strict_root_on_states(np.array([1.0, 1.0]), system.results[0],
                                          math.pi * (1.5 / 2) ** 2, 2)
    assert system.execution_status == "completed"
    assert len(system.results) == 2


def test_both_paths_build_cells_the_same_way():
    """顺序与 batch 路径必须共用同一 Cell 构造，避免再度分叉。"""
    system = _simple_system()
    system._initialize_inlet()
    system.z_positions = np.array([0.5])
    dz = np.array([1.0])
    A = math.pi * (1.5 / 2) ** 2
    ops = system._cell_ops_for_solve(dz)
    cell = system._make_cell(0, dz, A, system._initialize_inlet(),
                             system._initialize_inlet(), ops, 1.0, 1.0)
    assert cell.coal_flow_dry == system.W_dry
    assert cell.char_Xc0 == system.char_Xc0
    assert cell.ref_flow == 1.0
    assert ops["L_heatloss_norm"] == pytest.approx(float(np.sum(dz)))


# ------------------------------------------------ 2. snapshot shares residual ctx

def test_snapshot_particle_temperature_matches_residuals(monkeypatch):
    """snapshot_with_limits 与 residuals 必须得到同一 Ts_avg。

    回归审查意见（P2）：snapshot 遗漏 current/phys/gas_src/is_combustion_zone，
    使颗粒温度走另一条路径，进而让限幅标志不再是实际方程的值。

    注意：该缺陷在当前配置下是**潜伏**的——`USE_RK_GILL_COMBUSTION` 默认为
    False，两条路径恰好都走无反应热的指数弛豫，数值相同。因此这里临时把它
    打开，让燃烧区真正走 RK-Gill 分支，测试才具备区分能力。
    """
    from model.constants import PhysicalConstants
    from model.cell import Cell
    from model.state import StateVector

    monkeypatch.setattr(PhysicalConstants, "USE_RK_GILL_COMBUSTION", True)

    system = _simple_system()
    inlet = system._initialize_inlet()
    system.z_positions = np.array([0.5])
    dz = np.array([1.0])
    A = math.pi * (1.5 / 2) ** 2
    ops = system._cell_ops_for_solve(dz)
    cell = system._make_cell(0, dz, A, inlet, inlet, ops, 1.0, 1.0)

    state = StateVector(gas_moles=np.array([5.0, 0.5, 2.0, 1.0, 0.0, 1.5, 0.2, 2.0]),
                        solid_mass=1.0, carbon_fraction=0.7, T=1800.0, P=2.0e6, z=0.5)

    snap = cell.snapshot_with_limits(state)

    # 复算 residuals 内部所用的 Ts_avg（同一条调用链）
    g_src = np.zeros(8)
    for src in cell.sources:
        g, _sm, _e = src.get_sources(cell.idx, cell.z, cell.dz)
        g_src += g
    C_fed = cell.coal_flow_dry * (cell.coal_props.get("Cd", 60.0) / 100.0)
    phys = cell._calc_physics_props(state, C_fed)
    tau = cell.dz / max(phys["v_g"], 1.0e-12)
    F_in = max(sum(cell.inlet.gas_moles[i] + g_src[i] for i in range(8)), 1e-9)
    pO2 = state.P * (max(cell.inlet.gas_moles[0] + g_src[0], 0.0) / F_in)
    is_comb = pO2 > (cell.op_conds.get("P_O2_Combustion_atm", 0.05) * 101325.0)
    Ts_avg_ref, _ = cell._calc_particle_temperature(
        state.T, cell.inlet.T_solid_or_gas, tau, phys["d_p"],
        state.solid_mass, state.carbon_fraction,
        current=state, phys=phys, gas_src=g_src,
        is_combustion_zone=is_comb, rho_p=phys.get("rho_p"),
    )
    assert snap["Ts_avg"] == pytest.approx(float(Ts_avg_ref), rel=1e-12)


def test_snapshot_takes_the_combustion_zone_branch_like_residuals():
    """燃烧区上下文必须传入，否则颗粒温度选择分区路径时会与方程不一致。"""
    import inspect

    from model.cell import Cell

    src = inspect.getsource(Cell.snapshot_with_limits)
    for token in ("current=state", "phys=phys", "gas_src=g_src",
                  "is_combustion_zone=is_combustion"):
        assert token in src, f"snapshot_with_limits must pass {token}"
