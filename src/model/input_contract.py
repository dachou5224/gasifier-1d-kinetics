"""
coal_props / op_conds 基础输入契约：消除 Ad/Ashd、热损与硫参数等的静默错读。
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# P1.6-A（2026-11-02）方程修正开关
# ---------------------------------------------------------------------------

# 1a 固相焓口径（审计 B9）。
#   char_formation_zero : 入炉煤按 LHV 口径的 Hf_coal 计焓；热解后的焦炭碳取石墨生成焓 0、
#                         灰只计显热；入炉煤焓与（焦炭焓 + 挥发分焓）之差作为热解热进源项。
#   legacy_coal_hf      : 修改前的口径——焦炭沿用每 kg 固体的 Hf_coal，热解热不显式计入。
SOLID_ENTHALPY_MODEL_DEFAULT = "char_formation_zero"
SOLID_ENTHALPY_MODEL_LEGACY = "legacy_coal_hf"
VALID_SOLID_ENTHALPY_MODELS = (SOLID_ENTHALPY_MODEL_DEFAULT, SOLID_ENTHALPY_MODEL_LEGACY)

# 2 异相速率里的碳转化率硬截断。
#   smooth     : 去掉 X_total >= 0.999 的硬截断，让 1/(k_s Y^2) 在 Y -> 0 时把速率按 Y^2 压到 0。
#   hard_0999  : 修改前的行为，X_total >= 0.999 时四条异相速率直接置 0（含 C+H2 的 0.999 判据）。
HET_CONVERSION_CUTOFF_DEFAULT = "smooth"
HET_CONVERSION_CUTOFF_LEGACY = "hard_0999"
VALID_HET_CONVERSION_CUTOFFS = (HET_CONVERSION_CUTOFF_DEFAULT, HET_CONVERSION_CUTOFF_LEGACY)


def _resolve_choice(op_conds: Mapping[str, Any], key: str, default: str, valid: tuple[str, ...]) -> str:
    raw = op_conds.get(key, default)
    value = str(raw).lower()
    if value not in valid:
        raise ValueError(f"{key} must be one of {valid}, got {raw!r}")
    return value


def resolve_solid_enthalpy_model(op_conds: Mapping[str, Any]) -> str:
    """解析 1a 的固相焓口径。非法值直接报错，避免静默退回缺省。"""
    return _resolve_choice(
        op_conds, "SolidEnthalpyModel", SOLID_ENTHALPY_MODEL_DEFAULT, VALID_SOLID_ENTHALPY_MODELS
    )


def resolve_het_conversion_cutoff(op_conds: Mapping[str, Any]) -> str:
    """解析 2 的碳转化率截断口径。非法值直接报错。"""
    return _resolve_choice(
        op_conds, "HetConversionCutoff", HET_CONVERSION_CUTOFF_DEFAULT, VALID_HET_CONVERSION_CUTOFFS
    )


# ---------------------------------------------------------------------------
# P1.7-A（2026-11-03）热值口径与热解热模型
# ---------------------------------------------------------------------------
#
# 缺陷（审计 C2）：数据合同的 ``heating_value_dry_kj_kg`` 标注为**干燥基低位发热量**
# （P1 合同 27720 kJ/kg），适配器却填进 ``HHV_d``，而 ``Hf_coal`` 公式把 ``HHV_d``
# 当高位处理、再减去 ``9·Hd·2442`` 换成低位——汽化潜热被扣了两次（约 950 kJ/kg 干煤）。
#
# 口径约定（本轮起唯一口径）：
#   * ``LHV_d``：干燥基低位发热量 [kJ/kg]（允许历史上按 MJ/kg 写的 < 1000 的数值）。
#   * ``HHV_d``：干燥基高位发热量 [kJ/kg]（同上允许 MJ/kg）。
#   * 两者同时出现 → 直接报错，不做任何猜测。
#   * ``Hf_coal`` 用**低位**：煤的生成焓基准与产物侧的气态水一致（Shomate 的 H2O(g)），
#     用高位会把水的汽化潜热重复计入。
#   * 热损基准（文献口径「入炉煤 HHV 的 X%」）用**高位**：只给 ``LHV_d`` 时按定义
#     加回 ``9·Hd·2442`` 得到 HHV，不引入新的假设。
LHV_KEY = "LHV_d"
HHV_KEY = "HHV_d"
_MJ_PER_KG_THRESHOLD = 1000.0  # 小于该值视为历史输入的 MJ/kg

# 热解热口径。
#   from_heating_value（缺省）：由热值算 Hf_coal，热解热 = H_coal − H_char − H_vol（可正可负）。
#   zero：Hf_coal 由热解产物焓与焦炭焓反推，使 cell 0 的热解热恰为 0，不读取任何热值。
#         Fortran 采用反应热记账、煤本身无生成焓，其能量自洽于自身热解产物，等价于此口径。
PYROLYSIS_HEAT_MODEL_DEFAULT = "from_heating_value"
PYROLYSIS_HEAT_MODEL_ZERO = "zero"
VALID_PYROLYSIS_HEAT_MODELS = (PYROLYSIS_HEAT_MODEL_DEFAULT, PYROLYSIS_HEAT_MODEL_ZERO)

# 热损基准的缺省 HHV：仅用于「既没给 LHV_d 也没给 HHV_d、且 Hf_coal 已由调用方显式给出」
# 的历史路径（与原代码 ``coal_props.get('HHV_d', 30.0)`` 等价）。
DEFAULT_COAL_HHV_KJ_KG = 30000.0

# H 的汽化潜热当量：约 2442 kJ/kg 水 × 9 kg 水/kg H
WATER_LATENT_HEAT_KJ_PER_KG = 2442.0
WATER_H_PER_KG = 9.0


def resolve_pyrolysis_heat_model(op_conds: Mapping[str, Any]) -> str:
    """解析热解热口径。非法值直接报错，避免静默退回缺省。"""
    return _resolve_choice(
        op_conds,
        "PyrolysisHeatModel",
        PYROLYSIS_HEAT_MODEL_DEFAULT,
        VALID_PYROLYSIS_HEAT_MODELS,
    )


def heating_value_kj_kg(coal_props: Mapping[str, Any], key: str) -> float | None:
    """读取一个热值字段并归一到 kJ/kg；缺失返回 ``None``（不落到默认值）。"""
    raw = coal_props.get(key)
    if raw is None:
        return None
    value = float(raw)
    return value * 1000.0 if value < _MJ_PER_KG_THRESHOLD else value


def validate_coal_heating_values(coal_props: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """返回 ``(lhv_kj_kg, hhv_kj_kg)``；两者同时给出直接报错。"""
    lhv = heating_value_kj_kg(coal_props, LHV_KEY)
    hhv = heating_value_kj_kg(coal_props, HHV_KEY)
    if lhv is not None and hhv is not None:
        raise ValueError(
            f"coal_props 同时提供 {LHV_KEY}={lhv} 与 {HHV_KEY}={hhv} kJ/kg："
            "低位与高位热值的口径不同（Hf_coal 用低位、热损基准用高位），"
            "不能靠猜决定用哪一个。请只保留合同实际给出的那一个。"
        )
    return lhv, hhv


def coal_lhv_kj_kg(coal_props: Mapping[str, Any]) -> float:
    """干燥基低位发热量 [kJ/kg]（Hf_coal 用）。只给 HHV 时按定义扣潜热换成低位。

    两者都没有时退回历史缺省 ``HHV = 30 MJ/kg``（与修改前 ``coal_props.get('HHV_d', 30.0)``
    完全一致），只打一条 warning；历史脚本与夹具靠这条缺省仍在跑，新轮次一律显式给热值。
    """
    lhv, hhv = validate_coal_heating_values(coal_props)
    if lhv is not None:
        return float(lhv)
    hd = float(coal_props.get("Hd", 0.0)) / 100.0
    if hhv is not None:
        return float(hhv - WATER_H_PER_KG * hd * WATER_LATENT_HEAT_KJ_PER_KG)
    logger.warning(
        "coal_props 既没有 %s 也没有 %s：退回历史缺省 HHV=%s kJ/kg 再扣潜热（Hf_coal 用低位）。",
        LHV_KEY, HHV_KEY, DEFAULT_COAL_HHV_KJ_KG,
    )
    return float(DEFAULT_COAL_HHV_KJ_KG - WATER_H_PER_KG * hd * WATER_LATENT_HEAT_KJ_PER_KG)


def coal_hhv_kj_kg(coal_props: Mapping[str, Any]) -> float:
    """干燥基高位发热量 [kJ/kg]（热损基准用）。只给 LHV 时按定义加回潜热。"""
    lhv, hhv = validate_coal_heating_values(coal_props)
    if hhv is not None:
        return float(hhv)
    if lhv is not None:
        hd = float(coal_props.get("Hd", 0.0)) / 100.0
        return float(lhv + WATER_H_PER_KG * hd * WATER_LATENT_HEAT_KJ_PER_KG)
    return DEFAULT_COAL_HHV_KJ_KG


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

    P1.7-A：HHV 由 ``input_contract.coal_hhv_kj_kg`` 取/推（只给 LHV 时按定义加回潜热），
    基准仍是高位（文献口径「入炉煤 HHV 的 X%」）。
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
