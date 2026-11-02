"""
coal_props / op_conds 基础输入契约：消除 Ad/Ashd、热损与硫参数等的静默错读。
"""
from __future__ import annotations

from typing import Any, Mapping

# 颗粒尺寸模型的唯一缺省值。2026-11-02 起缺省为 fortran_constant（与 Wen 1979 /
# legacy Fortran 一致）。此前的调用默认是 shrinking_legacy，靠 paper_op_conds
# 逐个显式覆盖；一旦有代码绕过 paper_op_conds 就会静默拿到旧模型。
PARTICLE_MODEL_DEFAULT = "fortran_constant"
PARTICLE_MODEL_LEGACY = "shrinking_legacy"
VALID_PARTICLE_MODELS = (PARTICLE_MODEL_DEFAULT, PARTICLE_MODEL_LEGACY)


def resolve_particle_model(op_conds: Mapping[str, Any]) -> str:
    """解析 op_conds 里的颗粒尺寸模型，缺省与非法值都走同一处判断。

    非法值直接报错，而不是静默退回缺省：写错模型名会让整轮结果用错物理，
    却看不出任何异常。
    """
    raw = op_conds.get("ParticleSizeModel", PARTICLE_MODEL_DEFAULT)
    model = str(raw).lower()
    if model not in VALID_PARTICLE_MODELS:
        raise ValueError(
            f"ParticleSizeModel must be one of {VALID_PARTICLE_MODELS}, got {raw!r}"
        )
    return model


def uses_fortran_default_model(op_conds: Mapping[str, Any]) -> bool:
    """该轮是否走缺省的 fortran_constant 模型。

    用于给复现产物打标记：结果为 False 的行明确说明它用的是旧模型，
    读数时不能与 fortran 口径混用。
    """
    return resolve_particle_model(op_conds) == PARTICLE_MODEL_DEFAULT


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
    颗粒表面碳氧化时硫向气相释放的系数。

    P1.5-B（2026-10-30）起：``PyrolysisService`` 已把煤中全部硫作为 H2S 写入气相源项，
    JAX 残差里的 ``S_release`` 会造成**重复释放**，该项已删除。为避免调用方仍以为
    可以打开表面硫释放，显式设置非零值直接报错。

    默认返回 **0.0**。
    """
    value = 0.0
    if "f_s_coal" in op_conds:
        value = float(op_conds["f_s_coal"])
    elif "f_s_coal" in coal_props:
        value = float(coal_props["f_s_coal"])
    if value != 0.0:
        raise ValueError(
            "f_s_coal != 0 is not supported: sulfur is fully released as H2S by "
            "PyrolysisService, and the JAX S_release term was removed in P1.5-B "
            "to avoid double counting."
        )
    return 0.0
