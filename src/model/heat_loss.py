"""P1.8-A（2026-11-04）壁面热损模型：``HeatLossModel`` 与 Fortran 壁面热损的逐字转写。

背景（验证文档 E5）：Python 一直用「入炉煤高位热值的 2%，按长度摊分」这一套热损，
而 ``reference_fortran/Source1_副本.for`` 用的是两区分段模型：
燃烧区按反应热的 30% 传给壁面，气化区按气-壁换热系数与壁温剖面算。
P1.5-C 以来的锚点比较因此是两套能量模型之间的比较。本模块把 Fortran 的两条式子
搬进 Python，供锚点（模式 F′）使用；装置九窗计算仍用 ``hhv_percent``。

Fortran 原文（行号为 ``Source1_副本.for`` 去 CR 后的行号）::

    30   data hl/30./ !combustion heat percent transfered to the wall
    32   data vis,at,dt,pt,height/6.e-4,18241.5,152.4,24.,330./
    96   delh = 25.
    146  nc = 3
    214  u0 = 3.4e-3        ! cal/(sec.cm^2.K)
    341-350  if(poxyin .gt. 0.05) then
                hloss = fcoal*abs(rheat1)*hl/100.
             else
                tw = 2100.-600.*hs/height
                hloss = u0*3.14*dt*float(nc)*delh*(tg-tw)
             endif
    772-781  rheat = rheat1+rheat2+rheat3 - rhts
             if(poxyin. gt. 0.05) goto 736
             tw = 2100. - 600.*hs/height
             hloss = u0*3.14*dt*float(nc)*delh*(tg-tw)
             goto 737
    736      hloss = fcoal*abs(rheat)*hl/100.
    737      htexc = -rheat-tohup-hloss/fcoal

注意两条分支用的被乘数不同：第 346 行用 ``rheat1``（``pyroly`` 写入的挥发分项），
第 780 行用 ``rheat``（``rheat1+rheat2+rheat3-rhts`` 的净反应热）。本模块按卡面
第 1.1 节采用 ``rheat``（净反应热），因为它与主循环 515→737 的能量式子同处一处。

单位与几何口径
--------------
* ``dt`` = 152.4 cm、``height`` = 330 cm 取自第 32 行的 ``data`` 语句，**不取 Python 几何**
  （锚点算例两者数值相同，但口径必须写明是从 Fortran 读的）。
* ``nc*delh`` 是「一个气室沿轴向占的长度」：``delh`` 在每个气室内累进 ``hs = hs + delh``
  （第 277、763 行），共 ``nc`` 次，因此 ``nc*delh`` 就是一个气室的轴向长度 [cm]。
  Python 侧一个单元就是一个气室，故映射 ``nc*delh -> dz``。
* ``hs`` 在调用处已经是**气室出口**的轴向位置（第 277、763 行先累进再算 hloss），
  故 Python 侧取 ``z + dz/2``（单元出口）。
* ``hl``、``u0``、``tw`` 的系数原样保留；唯一改动是把 ``3.14`` 写成 ``π``，把 cal 换成 J。
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

from .physics import get_enthalpy_molar

# 炭颗粒比热与碳摩尔质量：用于把固碳显焓并入异相反应焓。
# 数值与 constants.PhysicalConstants.HEAT_CAPACITY_SOLID 缺省一致（1500 J/(kg·K)）；
# 此处内联是为了不改变 reaction_heat_release_w 的签名（该函数不收 coal_props）。
HEAT_CAPACITY_SOLID = 1500.0
CARBON_MOLAR_MASS_KG = 0.012011

logger = logging.getLogger(__name__)

# 每 cal 的焦耳数（Fortran 全程用 cal；Python 的能量残差用 W）
CAL_TO_J = 4.184

# 第 30 行：data hl/30./
FORTRAN_HL_PERCENT = 30.0
# 第 214 行：u0 = 3.4e-3 cal/(sec.cm^2.K)
FORTRAN_U0_CAL_PER_S_CM2_K = 3.4e-3
# 第 32 行：data ... dt ... height
FORTRAN_DT_CM = 152.4
FORTRAN_HEIGHT_CM = 330.0
# 第 348 / 777 行：tw = 2100. - 600.*hs/height
FORTRAN_TW_BASE_K = 2100.0
FORTRAN_TW_SLOPE_K = 600.0

# 热损模型开关
HEAT_LOSS_MODEL_DEFAULT = "hhv_percent"
HEAT_LOSS_MODEL_FORTRAN = "fortran_wall"
VALID_HEAT_LOSS_MODELS = (HEAT_LOSS_MODEL_DEFAULT, HEAT_LOSS_MODEL_FORTRAN)


def resolve_heat_loss_model(op_conds: Mapping[str, Any]) -> str:
    """解析 ``op_conds['HeatLossModel']``；缺省为 ``hhv_percent``（行为逐位不变）。

    非法值直接报错，不静默退回缺省：写错模型名会让整轮结果用错热损，却看不出异常。
    """
    raw = op_conds.get("HeatLossModel", HEAT_LOSS_MODEL_DEFAULT)
    model = str(raw).lower()
    if model not in VALID_HEAT_LOSS_MODELS:
        raise ValueError(f"HeatLossModel must be one of {VALID_HEAT_LOSS_MODELS}, got {raw!r}")
    return model


def combustion_zone_heat_loss_w(
    reaction_heat_w: float,
    *,
    hl_percent: float = FORTRAN_HL_PERCENT,
) -> float:
    """燃烧区：``hloss = fcoal·|rheat|·hl/100``（第 780 行）。

    在 Python 侧 ``fcoal·rheat`` 就是本单元净反应热 [W]，``hl/100`` = 30%，
    故等价于 ``0.30 × |本单元净反应热|``。取绝对值（Fortran 原文即 ``abs``）。
    """
    return float(hl_percent) / 100.0 * abs(float(reaction_heat_w))


def gasification_zone_heat_loss_w(
    T_gas_K: float,
    *,
    z_outlet_m: float,
    dz_m: float,
    u0_cal_per_s_cm2_k: float = FORTRAN_U0_CAL_PER_S_CM2_K,
    dt_cm: float = FORTRAN_DT_CM,
    height_cm: float = FORTRAN_HEIGHT_CM,
) -> dict[str, float]:
    """气化区：``hloss = u0·π·dt·nc·delh·(tg − tw)``（第 349 / 778 行）。

    ``nc·delh`` 用 ``dz``（一个单元 = 一个气室），``hs`` 用单元出口位置。
    ``tg < tw`` 时返回负值——壁面向气体供热，**保留符号，不 clamp 到 0**。
    """
    hs_cm = float(z_outlet_m) * 100.0
    dz_cm = float(dz_m) * 100.0
    tw_k = FORTRAN_TW_BASE_K - FORTRAN_TW_SLOPE_K * hs_cm / float(height_cm)
    area_cm2 = float(np.pi) * float(dt_cm) * dz_cm  # π·dt·(nc·delh)
    hloss_cal_s = float(u0_cal_per_s_cm2_k) * area_cm2 * (float(T_gas_K) - tw_k)
    return {
        "hloss_W": hloss_cal_s * CAL_TO_J,
        "tw_K": tw_k,
        "hs_cm": hs_cm,
        "area_cm2": area_cm2,
    }


def fortran_wall_heat_loss(
    *,
    is_combustion_zone: bool,
    reaction_heat_w: float,
    T_gas_K: float,
    z_outlet_m: float,
    dz_m: float,
    hl_percent: float = FORTRAN_HL_PERCENT,
    u0_cal_per_s_cm2_k: float = FORTRAN_U0_CAL_PER_S_CM2_K,
    dt_cm: float = FORTRAN_DT_CM,
    height_cm: float = FORTRAN_HEIGHT_CM,
) -> dict[str, Any]:
    """按第 341–350 / 772–781 行的两区分支给出一格的 ``hloss`` [W]。

    返回字典里带 ``hloss_W``（进能量方程的减项，可为负）、分区标签、``tw``（仅气化区有值）、
    ``tg``，以及该格的换热面积，供逐单元证据表使用。
    """
    if is_combustion_zone:
        hloss_w = combustion_zone_heat_loss_w(reaction_heat_w, hl_percent=hl_percent)
        return {
            "hloss_W": hloss_w,
            "zone": "combustion",
            "tw_K": float("nan"),
            "tg_K": float(T_gas_K),
            "hs_cm": float(z_outlet_m) * 100.0,
            "area_cm2": float("nan"),
            "reaction_heat_W": float(reaction_heat_w),
            "hl_percent": float(hl_percent),
        }
    gas = gasification_zone_heat_loss_w(
        T_gas_K,
        z_outlet_m=z_outlet_m,
        dz_m=dz_m,
        u0_cal_per_s_cm2_k=u0_cal_per_s_cm2_k,
        dt_cm=dt_cm,
        height_cm=height_cm,
    )
    return {
        "hloss_W": gas["hloss_W"],
        "zone": "gasification",
        "tw_K": gas["tw_K"],
        "tg_K": float(T_gas_K),
        "hs_cm": gas["hs_cm"],
        "area_cm2": gas["area_cm2"],
        "reaction_heat_W": float(reaction_heat_w),
        "hl_percent": float("nan"),
    }


# ---------------------------------------------------------------------------
# 单元净反应热（Fortran 的 rheat = rheat1 + rheat2 + rheat3 − rhts）
# ---------------------------------------------------------------------------

def reaction_heat_release_w(
    r_het: Mapping[str, float],
    r_homo: Mapping[str, float],
    *,
    phi: float,
    f_co2: float,
    T_gas_K: float,
) -> float:
    """一格内所有已解析反应的**净**放热速率 [W]（正 = 放热）。

    Fortran 的 ``rheat`` 是「本气室内净产生的热量」，由四块相加：挥发分项 ``rheat1``、
    气相燃烧项 ``rheat2``、WGS 项 ``rheat3``、以及被固相吸收的 ``rhts``。Python 侧
    没有单独记账的固相吸收项（C+O2/C+H2O/C+CO2/C+H2 的反应焓本身就是「气相 + 固相」
    的总和），所以这里直接用每个反应的化学计量与模型自身的 Shomate 生成焓求
    ``Q = −Σ rate·ΔH(T)``：

    - 与 ``Cell._calc_balances`` 里的化学计量严格一致（含 ``phi`` 与 ``f_co2``）；
    - 反应焓在**单元气相温度**下求值，与 Fortran 用 ``qrh``（颗粒温度）与 ``ht``
      （气相温度）两套多项式分别求值的做法同构，差别只在热容关联式；
    - 碳（焦炭）的生成焓按定义取 0（模型其余部分亦如此，见
      ``SolidEnthalpyModel="char_formation_zero"``）。

    与 Fortran 的一处已知差别记在 ``interpretation.md``：Fortran 的 ``rheat1`` 是
    挥发分的显热当量（``ht(n,tg)`` 乘产率），而我们的挥发分能量已经通过
    ``PyrolysisSource.energy_src_W`` 进能量方程，不再重复计入反应热。
    """
    t = float(T_gas_K)
    h = {
        sp: float(get_enthalpy_molar(sp, t))
        for sp in ("CO", "CO2", "H2", "H2O", "O2", "CH4")
    }
    # 碳（固）的显焓。``char_formation_zero`` 只把**生成焓**归零，显焓
    # ``cp_char·(T − 298.15)`` 仍在（见 MaterialService.get_solid_enthalpy），
    # 而异相反应消耗 1 mol 固碳。漏掉它会让反应热与能量方程口径不一致：
    # 2000 K 时 cp_char 取缺省 1500 J/(kg·K) 约 30 kJ/mol，直接影响 30% 壁损项。
    # 2026-11-04 更正（审阅意见）。
    h_c_solid = HEAT_CAPACITY_SOLID * CARBON_MOLAR_MASS_KG * (t - 298.15)  # J/mol C

    f = float(f_co2)
    ph = float(phi)

    rates = {
        "C+O2": float(r_het.get("C+O2", 0.0)),
        "C+H2O": float(r_het.get("C+H2O", 0.0)),
        "C+CO2": float(r_het.get("C+CO2", 0.0)),
        "C+H2": float(r_het.get("C+H2", 0.0)),
        "CO_Ox": float(r_homo.get("CO_Ox", 0.0)),
        "H2_Ox": float(r_homo.get("H2_Ox", 0.0)),
        "CH4_Ox": float(r_homo.get("CH4_Ox", 0.0)),
        "WGS": float(r_homo.get("WGS", 0.0)),
        "RWGS": float(r_homo.get("RWGS", 0.0)),
        "MSR": float(r_homo.get("MSR", 0.0)),
    }

    # 每个反应的 ΔH [J/mol reaction]；碳（固）焓记 0
    dh = {
        # C + (1/phi) O2 -> (2-2/phi) CO + (2/phi-1) CO2
        # 以下四个异相反应各消耗 1 mol 固碳，故减 h_c_solid（见上文说明）。
        "C+O2": (
            (2.0 - 2.0 / ph) * h["CO"]
            + (2.0 / ph - 1.0) * h["CO2"]
            - (1.0 / ph) * h["O2"]
            - h_c_solid
        ),
        # C + H2O -> CO + H2
        "C+H2O": h["CO"] + h["H2"] - h["H2O"] - h_c_solid,
        # C + CO2 -> 2 CO
        "C+CO2": 2.0 * h["CO"] - h["CO2"] - h_c_solid,
        # C + 2 H2 -> CH4
        "C+H2": h["CH4"] - 2.0 * h["H2"] - h_c_solid,
        # CO + 0.5 f O2 -> f CO2
        "CO_Ox": f * (h["CO2"] - h["CO"] - 0.5 * h["O2"]),
        # H2 + 0.5 O2 -> H2O
        "H2_Ox": h["H2O"] - h["H2"] - 0.5 * h["O2"],
        # CH4 + (1.5+0.5f) O2 -> f CO2 + (1-f) CO + 2 H2O
        "CH4_Ox": (
            f * h["CO2"]
            + (1.0 - f) * h["CO"]
            + 2.0 * h["H2O"]
            - h["CH4"]
            - (1.5 + 0.5 * f) * h["O2"]
        ),
        # CO + H2O -> CO2 + H2
        "WGS": h["CO2"] + h["H2"] - h["CO"] - h["H2O"],
        # CO2 + H2 -> CO + H2O
        "RWGS": h["CO"] + h["H2O"] - h["CO2"] - h["H2"],
        # CH4 + H2O -> CO + 3 H2
        "MSR": h["CO"] + 3.0 * h["H2"] - h["CH4"] - h["H2O"],
    }

    total = 0.0
    for key, rate in rates.items():
        total += -rate * dh[key]
    return float(total)


__all__ = [
    "CAL_TO_J",
    "FORTRAN_HL_PERCENT",
    "FORTRAN_U0_CAL_PER_S_CM2_K",
    "FORTRAN_DT_CM",
    "FORTRAN_HEIGHT_CM",
    "FORTRAN_TW_BASE_K",
    "FORTRAN_TW_SLOPE_K",
    "HEAT_LOSS_MODEL_DEFAULT",
    "HEAT_LOSS_MODEL_FORTRAN",
    "VALID_HEAT_LOSS_MODELS",
    "resolve_heat_loss_model",
    "combustion_zone_heat_loss_w",
    "gasification_zone_heat_loss_w",
    "fortran_wall_heat_loss",
    "reaction_heat_release_w",
]
