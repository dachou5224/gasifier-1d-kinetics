"""
coal_props / op_conds 基础输入契约：消除 Ad/Ashd、热损与硫参数等的静默错读。
"""
from __future__ import annotations

from typing import Any, Mapping


def ash_mass_fraction_dry(coal_props: Mapping[str, Any]) -> float:
    """
    干基灰分质量分数 [0,1]。
    优先 ``Ad``（与工业煤分析常用符号一致），否则 ``Ashd``（历史字段）。
    """
    if "Ad" in coal_props:
        return float(coal_props["Ad"]) / 100.0
    return float(coal_props.get("Ashd", 6.0)) / 100.0


def coal_flow_kg_s_for_heat_loss(op_conds: Mapping[str, Any]) -> float:
    """
    与 ``Cell._calc_energy_balance`` 一致：热损基准功率 = 湿煤流量 × HHV。
    """
    return float(op_conds["coal_flow"])


def heat_loss_norm_length_m(op_conds: Mapping[str, Any], geometry_L: float, mesh_sum_dz: float) -> float:
    """
    热损沿程归一化长度 [m]。
    与 cell：``L_heatloss_norm`` → ``L_reactor`` → 默认几何长 一致；若均未给则用 ``sum(dz)``。
    """
    if "L_heatloss_norm" in op_conds:
        return max(float(op_conds["L_heatloss_norm"]), 1e-9)
    if "L_reactor" in op_conds:
        return max(float(op_conds["L_reactor"]), 1e-9)
    return max(float(geometry_L), float(mesh_sum_dz), 1e-9)


def heat_loss_ref_temp_k(op_conds: Mapping[str, Any]) -> float:
    """辐射权重参考温度 [K]，与 ``Cell._calc_energy_balance`` 默认 1800 一致。"""
    return float(op_conds.get("HeatLossRefTemp", 1800.0))


def resolve_f_s_coal(coal_props: Mapping[str, Any], op_conds: Mapping[str, Any]) -> float:
    """
    颗粒表面碳氧化时硫向气相释放的有效系数（JAX 残差中 ``S_release`` 用）。

    默认 **0**：与当前主线 ``Cell``（无显式 ``S_release`` 项）短期等价；
    可通过 ``op_conds['f_s_coal']`` 或 ``coal_props['f_s_coal']`` 显式开启。
    """
    if "f_s_coal" in op_conds:
        return float(op_conds["f_s_coal"])
    if "f_s_coal" in coal_props:
        return float(coal_props["f_s_coal"])
    return 0.0
