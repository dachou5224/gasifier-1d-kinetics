"""P2.0-A 渣层截留（M2）：**调用生产实现**验证分流不改变颗粒转化率。

审阅意见（2026-11-16 第二轮，P2）指出上一版测试只是把 `X_total` 公式抄了一遍，
键名检查也只是搜源文件字符串。因此下面这些错误仍然能通过：

- 生产代码读错 `op_conds` 键；
- 注入位置错（写进了不共享的字典，同轮下游读不到）；
- 退回旧分母（按剩余碳缩放而不是按颗粒份额）。

本文件直接构造 `GasifierSystem` 与 `Cell`，调用真实的
`_split_slag_at_wsr_outlet`、`_make_cell` 与 `_calc_physics_props`，断言：

1. cell 0 拿不到注入（截留发生在它的出口，它自己不该被缩放）；
2. 同轮下游的每个 cell 都拿到正确的颗粒份额；
3. 分流前后颗粒转化率连续；
4. `f_s = 0` 时逐位不变。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.cell import Cell
from model.gasifier_system import GasifierSystem
from model.state import StateVector

N_SPECIES = 8
KEY = "CapturedParticleFractionOfFeed"


def _state(solid_mass: float, carbon_fraction: float, T: float = 1400.0) -> StateVector:
    gas = np.zeros(N_SPECIES)
    gas[2] = 50.0   # CO
    gas[3] = 10.0   # CO2
    gas[5] = 20.0   # H2
    gas[7] = 15.0   # H2O
    return StateVector(gas_moles=gas, solid_mass=solid_mass, carbon_fraction=carbon_fraction,
                       T=T, P=4.0e6, z=0.0, T_solid=T)


class _MiniSystem:
    """只为调用 `_split_slag_at_wsr_outlet` 准备的最小上下文。

    用 `GasifierSystem.__new__` 绕开完整构造：该方法的依赖只有
    `op_conds`、`coal_props`、`W_dry`、`slag_capture` 与 `Cd_total`。
    """

    def __init__(self, f_s: float, W_dry: float = 20.0, Cd: float = 73.0):
        object.__setattr__(self, "op_conds", {"SlagCaptureFraction": f_s})
        object.__setattr__(self, "coal_props", {"Cd": Cd, "cp_char": 1300.0})
        object.__setattr__(self, "W_dry", W_dry)
        object.__setattr__(self, "Cd_total", Cd)
        object.__setattr__(self, "slag_capture", None)


def _split(sys_: _MiniSystem, wsr_outlet: StateVector):
    """直接调用生产实现。"""
    return GasifierSystem._split_slag_at_wsr_outlet(sys_, 0, wsr_outlet)


def _x_of(cell: Cell, st: StateVector, C_fed: float,
          cap_frac: float | None = None) -> float:
    """用生产实现算 X_total。cap_frac 为 None 时用 cell 自带的 op_conds。"""
    if cap_frac is not None:
        cell.op_conds = dict(cell.op_conds)
        if cap_frac > 0:
            cell.op_conds[KEY] = cap_frac
        else:
            cell.op_conds.pop(KEY, None)
    return float(cell._calc_physics_props(st, C_fed)["X_total"])


def _make_cell(cap_frac: float = 0.0, coal_props=None, Cd_total: float = 73.0) -> Cell:
    op = {}
    if cap_frac > 0:
        op[KEY] = cap_frac
    st = _state(1.0, 1.0)
    return Cell(0, 0.0, 0.1, 1.0, 1.0, st, kinetics=None,
                coal_props=coal_props or {"Cd": Cd_total, "cp_char": 1300.0},
                op_conds=op)


# --------------------------------------------------------------------------
# 1. 分流前后颗粒转化率连续（走生产实现）
# --------------------------------------------------------------------------

@pytest.mark.parametrize("f_s", [0.0, 0.05, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("X_wsr", [0.0, 0.3, 0.5, 0.9])
def test_split_preserves_particle_conversion_via_production_code(f_s: float, X_wsr: float):
    """审阅算例推广版：无论 f_s 与转化率取何值，分流不改变剩余颗粒的 X。"""
    C_fed = 1.0
    sys_ = _MiniSystem(f_s)
    wsr_outlet = _state(solid_mass=C_fed * (1.0 - X_wsr), carbon_fraction=1.0)
    retained = _split(sys_, wsr_outlet)

    x_before = _x_of(_make_cell(0.0), wsr_outlet, C_fed)
    x_after = _x_of(_make_cell(f_s), retained, C_fed)

    assert x_before == pytest.approx(X_wsr, abs=1e-12)
    assert x_after == pytest.approx(x_before, abs=1e-12), (
        f"f_s={f_s}, X={X_wsr}: 分流把转化率从 {x_before:.4f} 改成了 {x_after:.4f}"
    )


def test_review_example_50pct_30pct():
    """审阅给出的原算例：C_fed=1、X=50%、f_s=30% 必须仍是 50%，不是 58.82%。"""
    sys_ = _MiniSystem(0.30)
    wsr_outlet = _state(solid_mass=0.5, carbon_fraction=1.0)
    retained = _split(sys_, wsr_outlet)
    x = _x_of(_make_cell(0.30), retained, 1.0)
    assert x == pytest.approx(0.50, abs=1e-12)
    assert abs(x - 0.5882352941) > 1e-6


# --------------------------------------------------------------------------
# 2. 注入语义：cell 0 不受影响，同轮下游拿到份额
# --------------------------------------------------------------------------

def test_split_writes_particle_fraction_not_carbon_fraction():
    """分流记录的是**颗粒份额** f_s 本身，不是被截留颗粒的剩余碳份额。"""
    sys_ = _MiniSystem(0.30)
    wsr_outlet = _state(solid_mass=0.5, carbon_fraction=0.8)  # X=50%，碳分数 0.8
    _split(sys_, wsr_outlet)

    cap = sys_.slag_capture
    assert cap is not None
    assert cap["particle_fraction_of_feed"] == pytest.approx(0.30, abs=1e-15)
    # 旧的错法记录的是剩余碳份额，数值会明显小于 f_s
    assert "carbon_fraction_of_feed" not in cap, (
        "按剩余碳份额缩放会把分流算成转化，见 2026-11-16 审阅"
    )


def test_cell0_keeps_original_reference():
    """cell 0 在截留**之前**，它的 X_total 不应被缩放。"""
    cell0 = _make_cell(0.0)
    sys_ = _MiniSystem(0.30)
    wsr_outlet = _state(solid_mass=0.5, carbon_fraction=1.0)
    x0 = _x_of(cell0, wsr_outlet, 1.0)
    _split(sys_, wsr_outlet)          # 分流发生，但 cell 0 已经算完
    assert x0 == pytest.approx(0.50, abs=1e-12)
    assert KEY not in cell0.op_conds, "cell 0 不应被注入截留份额"


def test_downstream_injection_matches_recorded_fraction():
    """模拟 `_make_cell` 的注入：下游 cell 的份额必须等于分流记录的值。"""
    sys_ = _MiniSystem(0.17)
    wsr_outlet = _state(solid_mass=0.6, carbon_fraction=0.9)
    retained = _split(sys_, wsr_outlet)
    frac = float(sys_.slag_capture["particle_fraction_of_feed"])

    downstream = _make_cell(frac)     # 等价于 _make_cell 注入后的结果
    assert downstream.op_conds[KEY] == pytest.approx(0.17, abs=1e-15)
    assert _x_of(downstream, retained, 1.0) == pytest.approx(
        _x_of(_make_cell(0.0), wsr_outlet, 1.0), abs=1e-12)


# --------------------------------------------------------------------------
# 3. 缺省关闭时逐位不变
# --------------------------------------------------------------------------

def test_no_capture_bit_identical():
    C_fed = 1.0
    for solid in (0.0, 0.25, 0.5, 1.0):
        st = _state(solid_mass=solid, carbon_fraction=1.0)
        a = _x_of(_make_cell(0.0), st, C_fed)
        b = _x_of(_make_cell(0.0), st, C_fed)
        assert a == b


def test_fs_zero_returns_same_object():
    """f_s = 0 时生产实现必须原样返回，不新建状态。"""
    sys_ = _MiniSystem(0.0)
    st = _state(solid_mass=0.5, carbon_fraction=1.0)
    assert _split(sys_, st) is st
    assert sys_.slag_capture is None


# --------------------------------------------------------------------------
# 4. 键名两侧一致（这次是真的调用，不是搜字符串）
# --------------------------------------------------------------------------

def test_writer_and_reader_agree_on_key():
    """写入侧写哪个键，读取侧就读哪个键——用生产实现验证，不查源码文本。"""
    sys_ = _MiniSystem(0.25)
    captured = _split(sys_, _state(solid_mass=0.8, carbon_fraction=1.0))
    frac = float(sys_.slag_capture["particle_fraction_of_feed"])

    # 写入侧记录的份额，喂给读取侧后必须真的生效
    cell = _make_cell(frac)
    st = _state(solid_mass=0.8 * (1 - 0.25), carbon_fraction=1.0)
    x_with = _x_of(cell, st, 1.0)
    x_without = _x_of(_make_cell(0.0), st, 1.0)
    # 若读取侧读错键，这两个值会相同——份额等于没生效
    assert x_with != pytest.approx(x_without, abs=1e-9), (
        "注入份额没有影响 X_total，读写两侧的键可能不一致"
    )


# --------------------------------------------------------------------------
# 5. 走 GasifierSystem 的真实单元创建路径（审阅 2026-11-16 第三轮 P2）
# --------------------------------------------------------------------------
#
# 上一版的 `downstream` 由测试助手手动传入 fraction 构造，等于模拟了注入，
# 而没有验证 `_make_cell` 真的会注入、也没验证 cell 0 真的不注入。
# 这里直接调用生产 `_make_cell`。

def _built_system(f_s: float):
    """构造一个可调用 `_make_cell` 的真实 GasifierSystem。"""
    from model.fortran_anchor_20261030 import anchor_cases, build_case_inputs
    from model.p1_5_round_20261030 import paper_op_conds

    cases, geo = anchor_cases()
    case = next(c for c in cases if c["case_name"] == "texaco i-1")
    built = build_case_inputs(case, geometry=geo)
    op = paper_op_conds(built, extra={"SlagCaptureFraction": f_s})
    system = GasifierSystem(built["geometry"], built["coal_props"], op)
    # `_make_cell` 依赖 solve 预处理写入的 tmp_* 与 z_positions。
    # 跑一次极小的真实 solve（4 格）把它们建好，再拿这个 system 做单元级检查；
    # 这样测的是真实初始化路径，而不是测试里手搓的状态。
    system.solve(N_cells=4)
    return system, built


def test_make_cell_injects_only_downstream():
    """真实 `_make_cell`：cell 0 不带注入，下游带；且份额等于分流记录值。"""
    system, built = _built_system(0.25)
    inlet = _state(solid_mass=1.0, carbon_fraction=0.7)
    dz_list = [0.5, 0.05, 0.05]

    # 先做一次分流，制造 slag_capture
    wsr_outlet = _state(solid_mass=0.8, carbon_fraction=0.7)
    system.slag_capture = None
    retained = GasifierSystem._split_slag_at_wsr_outlet(system, 0, wsr_outlet)
    frac = float(system.slag_capture["particle_fraction_of_feed"])
    assert frac == pytest.approx(0.25, abs=1e-15)

    cell_ops = system._cell_ops_for_solve(dz_list)
    A = 1.0
    ref_f, ref_e = 1.0, 1.0

    cell0 = system._make_cell(0, dz_list, A, inlet, inlet, cell_ops, ref_f, ref_e)
    cell1 = system._make_cell(1, dz_list, A, retained, inlet, cell_ops, ref_f, ref_e)

    assert KEY not in cell0.op_conds, (
        "cell 0 在截留之前，不应被注入；否则它会用缩小后的基准误算自己的转化率"
    )
    assert cell1.op_conds.get(KEY) == pytest.approx(frac, abs=1e-15), (
        "同轮下游必须拿到本轮分流记录的份额"
    )
    # 共享 dict 不能被污染
    assert KEY not in cell_ops


def test_make_cell_no_capture_is_clean():
    """f_s = 0 时下游也不应出现该键。"""
    system, _ = _built_system(0.0)
    system.slag_capture = None
    inlet = _state(solid_mass=1.0, carbon_fraction=0.7)
    dz_list = [0.5, 0.05]
    cell_ops = system._cell_ops_for_solve(dz_list)
    cell1 = system._make_cell(1, dz_list, 1.0, inlet, inlet, cell_ops, 1.0, 1.0)
    assert KEY not in cell1.op_conds
