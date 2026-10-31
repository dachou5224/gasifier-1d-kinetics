"""P1.5-C Wen/Fortran 锚点：Fortran 输出解析 + Python 输入映射（2026-10-30）。

Fortran 源与输入：``reference_fortran/Source1_副本.for``、``reference_fortran/input_副本.txt``。
本模块只做 I/O 映射与对比，不改任何物理常数。
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .gasifier_system import GasifierSystem
from .p1_5_round_20261030 import MESH_FAMILY, PARTICLE_MODEL_MAIN, paper_op_conds

# Fortran 常量（行号指 reference_fortran/Source1_副本.for 去 CR 后的行号）
FORTRAN_L_M = 3.30          # height = 330 cm, line 35
FORTRAN_D_M = 1.524         # dt = 152.4 cm, line 35
FORTRAN_P_ATM = 24.0        # pt = 24 atm, line 35
FORTRAN_RI_CM = 1.75e-2     # ri = 1.75e-2 cm, line 57
FORTRAN_CPOXY = 8.724       # cal/mol/K, line 60
FORTRAN_CPSTM = 11.34       # cal/mol/K, line 61
EXPECTED_CASE_NAMES = (
    "texaco i-1",
    "texaco i-2",
    "texaco i-5c",
    "texaco i-10",
    "texaco exxon",
    "slurry western",
    "slurry eastern",
)
DRY_POWDER_CASES = ("texaco i-1", "texaco i-2", "texaco i-5c", "texaco i-10", "texaco exxon")

_NAME_RE = re.compile(r'NAME="([^"]*)"')
_PARAM_RE = re.compile(r"^\s*([A-Z]+)=\s*([-\d.E+]+)\s*,?\s*$")
_NUM = r"([-\d.eE+]+)"


def _num(text: str) -> float:
    return float(text)


def parse_fortran_log(log_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 Fortran stdout 解析成逐算例出口表与逐单元剖面表。"""
    text = log_path.read_text(errors="replace")
    blocks = text.split("&READDATA")
    outlets: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []

    for block in blocks[1:]:
        name_m = _NAME_RE.search(block)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        params: dict[str, float] = {}
        for line in block.splitlines()[:40]:
            m = _PARAM_RE.match(line)
            if m:
                try:
                    params[m.group(1).lower()] = _num(m.group(2))
                except ValueError:
                    pass

        comp_blocks = block.split("for gaseous compartment")
        # comp_blocks[0] 是输入与初始化噪声，之后每段对应一个气室
        last_complete: dict[str, Any] | None = None
        for idx, cb in enumerate(comp_blocks[1:], start=1):
            get = lambda pat: (re.search(pat, cb).group(1) if re.search(pat, cb) else None)  # noqa: E731
            tg = get("corrected gas temp=" + r"\s*" + _NUM)
            ts = get("ts outlet=" + r"\s*" + _NUM)
            gco = get("co flow rate=" + r"\s*" + _NUM)
            gco2 = get("co2 flow rate=" + r"\s*" + _NUM)
            gh2 = get("h2 flow rate=" + r"\s*" + _NUM)
            gch4 = get("gch4=" + r"\s*" + _NUM)
            gsteam = get("steam flow rate=" + r"\s*" + _NUM)
            goxy = get("oxygen flow rate=" + r"\s*" + _NUM)
            gn2 = get("n2 flow rate=" + r"\s*" + _NUM)
            gh2s = get("h2s flow rate=" + r"\s*" + _NUM)
            if tg is None or ts is None or gco is None:
                continue
            wl = get("total weight loss=" + r"\s*" + _NUM)
            hs = get("entrained height=" + r"\s*" + _NUM)
            co_pct = get("co mol %=" + r"\s*" + _NUM)
            h2_pct = get("h2 mol %=" + r"\s*" + _NUM)
            co2_pct = get("co2 mol %=" + r"\s*" + _NUM)
            ch4_pct = get("ch4 mol %=" + r"\s*" + _NUM)
            n2_pct = get("n2 mol %=" + r"\s*" + _NUM)
            h2s_pct = get("h2s mol %=" + r"\s*" + _NUM)
            row = {
                "case_name": name,
                "compartment": idx,
                "hs_cm": float(hs) if hs else np.nan,
                "T_gas_K": float(tg),
                "T_solid_K": float(ts),
                "wl_pct": float(wl) if wl else np.nan,
                "gco_g_s": float(gco),
                "gco2_g_s": float(gco2),
                "gh2_g_s": float(gh2) if gh2 else np.nan,
                "gch4_g_s": float(gch4) if gch4 else np.nan,
                "gsteam_g_s": float(gsteam) if gsteam else np.nan,
                "goxy_g_s": float(goxy) if goxy else np.nan,
                "gn2_g_s": float(gn2) if gn2 else np.nan,
                "gh2s_g_s": float(gh2s) if gh2s else np.nan,
                "CO_vol_pct": float(co_pct) if co_pct else np.nan,
                "H2_vol_pct": float(h2_pct) if h2_pct else np.nan,
                "CO2_vol_pct": float(co2_pct) if co2_pct else np.nan,
                "CH4_vol_pct": float(ch4_pct) if ch4_pct else np.nan,
                "N2_vol_pct": float(n2_pct) if n2_pct else np.nan,
                "H2S_vol_pct": float(h2s_pct) if h2s_pct else np.nan,
            }
            profiles.append(row)
            last_complete = row

        if last_complete is None:
            continue

        yc = params.get("yc", math.nan)
        yash = params.get("yash", math.nan)
        xmois = params.get("xmois", math.nan)
        tfcoal = params.get("tfcoal", math.nan)
        fcoal = tfcoal * (1.0 - yash) * (1.0 - xmois)  # 行 70
        xcarb = 100.0 * (
            last_complete["gco_g_s"] / 28.0
            + last_complete["gco2_g_s"] / 44.0
            + (last_complete["gch4_g_s"] or 0.0) / 16.0
        ) * 12.0 / (fcoal * yc)  # 行 336 的口径（出口 tar=0）
        out = dict(last_complete)
        out.update(
            {
                "ta_K": params.get("ta"),
                "tsteam_K": params.get("tsteam"),
                "toxy_K": params.get("toxy"),
                "yoxy_coal_O": params.get("yoxy"),
                "yh_dmmf": params.get("yh"),
                "yc_dmmf": yc,
                "ys_dmmf": params.get("ys"),
                "yn_dmmf": params.get("yn"),
                "yash_ar": yash,
                "xmois": xmois,
                "foxy_g_per_g": params.get("foxy"),
                "fsteam_g_per_g": params.get("fsteam"),
                "tfcoal_g_s": tfcoal,
                "fcoal_dmmf_g_s": fcoal,
                "X_C_pct": float(xcarb),
            }
        )
        outlets.append(out)
    return outlets, profiles


def fortran_gas_inlet_temperature(case: Mapping[str, Any]) -> float:
    """行 185-187：tg1 为 O2 与蒸汽的质量加权入口温度。"""
    goxyin = float(case["foxy_g_per_g"]) * float(case["tfcoal_g_s"])   # 行 134
    gh2oin = (float(case["fsteam_g_per_g"]) + float(case["xmois"])) * float(case["tfcoal_g_s"])  # 行 135
    num = (
        goxyin * FORTRAN_CPOXY * float(case["toxy_K"]) / 32.0
        + gh2oin * FORTRAN_CPSTM * float(case["tsteam_K"]) / 18.0
    )
    den = goxyin * FORTRAN_CPOXY / 32.0 + gh2oin * FORTRAN_CPSTM / 18.0
    return float(num / den)


def build_case_inputs(case: Mapping[str, Any]) -> dict[str, Any]:
    """把 Fortran 算例参数映射成 GasifierSystem 的 geometry / coal_props / op_conds。

    口径说明（写进 protocol.json）：
    - yc/yh/ys/yn/yoxy 是 dmmf 煤的元素质量分数；yash 为收到基灰分，fcoal 已扣除灰与水分。
    - 干煤质量 = tfcoal*(1-xmois)；干基元素分数 = (1-yash)*y。
    - 氧化剂按纯 O2 处理（foxy g O2 / g 煤），蒸汽按 fsteam g / g 煤；煤中水分由 Mt 承担。
    """
    yash = float(case["yash_ar"])
    tfcoal = float(case["tfcoal_g_s"])
    xmois = float(case["xmois"])
    dry = tfcoal * (1.0 - xmois)
    dmmf_to_dry = 1.0 - yash
    coal_props = {
        "Cd": dmmf_to_dry * float(case["yc_dmmf"]) * 100.0,
        "Hd": dmmf_to_dry * float(case["yh_dmmf"]) * 100.0,
        "Od": dmmf_to_dry * float(case["yoxy_coal_O"]) * 100.0,
        "Nd": dmmf_to_dry * float(case["yn_dmmf"]) * 100.0,
        "Sd": dmmf_to_dry * float(case["ys_dmmf"]) * 100.0,
        "Ad": yash * 100.0,
        "Mt": xmois * 100.0,
        # Fortran 未给 HHV；热损在 Fortran 里按反应热比例处理（见台账 none 项）
        "HHV_d": 30.0,
    }
    geometry = {"L": FORTRAN_L_M, "D": FORTRAN_D_M}
    op_conds = {
        "coal_flow": tfcoal / 1000.0,
        "o2_flow": float(case["foxy_g_per_g"]) * tfcoal / 1000.0,
        "steam_flow": float(case["fsteam_g_per_g"]) * tfcoal / 1000.0,
        "carrier_gas_type": "CO2",
        "carrier_gas_flow": 0.0,
        "P": FORTRAN_P_ATM * 101325.0,
        "T_in": fortran_gas_inlet_temperature(case),
        "particle_diameter": 2.0 * FORTRAN_RI_CM * 1e-2,
        "epsilon": 1.0,
        "P_O2_Combustion_atm": 0.05,
    }
    return {"geometry": geometry, "coal_props": coal_props, "op_conds": op_conds, "dry_coal_g_s": dry}


def solve_fortran_case(
    case: Mapping[str, Any],
    *,
    n_cells: int,
    mode: str,
    switches: Mapping[str, Any] | None = None,
    coal_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """mode: 'M'（论文主线）或 'F'（Fortran 兼容）。"""
    built = build_case_inputs(case)
    if coal_overrides:
        built["coal_props"] = {**built["coal_props"], **dict(coal_overrides)}
    extra: dict[str, Any] = {}
    if mode == "F":
        extra.update(
            {
                "UseFortranDiffusion": True,
                "WGS_RatFactor": True,
                "CharCombustionRateFactor": 1.0,
            }
        )
    if switches:
        extra.update(switches)
    op = paper_op_conds(built, extra=extra)  # type: ignore[arg-type]
    system = GasifierSystem(built["geometry"], built["coal_props"], op)
    profile, z = system.solve(
        N_cells=int(n_cells), solver_method="minimize", jacobian_mode="scipy", jax_warmup=False
    )
    from .revision_evidence_common import _dry_syngas_kpis

    kpis = _dry_syngas_kpis(np.asarray(profile, dtype=float)) if len(profile) else {}
    carbon_inlet = float(system.W_dry * float(system.coal_props["Cd"]) / 100.0 / 0.012011)
    out: dict[str, Any] = {
        "case_name": str(case["case_name"]),
        "mode": mode,
        "N_cells": int(n_cells),
        "execution_status": str(system.execution_status),
        "no_root_cell_index": system.no_root_cell_index,
        "max_residual_cost": float(system.max_residual_cost),
        "n_cells_above_1e-4": int(system.n_cells_above_cost_threshold),
        "C_inlet_coal_mol_s": carbon_inlet,
        "switches": {k: extra[k] for k in sorted(extra)},
    }
    if system.execution_status == "completed" and len(system.results):
        last = system.results[-1]
        gas = np.asarray(last.gas_moles, dtype=float)
        dry_sum = float(np.sum(gas[:7]))
        out.update(
            {
                "T_gas_K": float(last.T),
                "T_solid_K": float(getattr(last, "T_solid", last.T)),
                "CO_vol_pct": float(gas[2] / dry_sum * 100.0),
                "H2_vol_pct": float(gas[5] / dry_sum * 100.0),
                "CO2_vol_pct": float(gas[3] / dry_sum * 100.0),
                "CH4_vol_pct": float(gas[1] / dry_sum * 100.0),
                "N2_vol_pct": float(gas[6] / dry_sum * 100.0),
                "H2S_vol_pct": float(gas[4] / dry_sum * 100.0),
                "outlet_solid_carbon_mol_s": float(last.solid_mass * last.carbon_fraction / 0.012011),
            }
        )
        out["X_C_pct"] = float(
            100.0 * (1.0 - out["outlet_solid_carbon_mol_s"] / max(carbon_inlet, 1e-12))
        )
        out["O2_drop_to_1pct_z_m"] = _o2_drop_position(system)
    else:
        for key in ("T_gas_K", "T_solid_K", "CO_vol_pct", "H2_vol_pct", "CO2_vol_pct",
                    "CH4_vol_pct", "N2_vol_pct", "H2S_vol_pct", "X_C_pct",
                    "outlet_solid_carbon_mol_s", "O2_drop_to_1pct_z_m"):
            out[key] = float("nan")
    if n_cells == 160:
        out["_profile"] = profile
        out["_z"] = z
        out["_system"] = system
    return out


def _o2_drop_position(system: GasifierSystem) -> float:
    """O2 降到入口值 1% 的轴向位置（用单元出口状态插值到 1% 阈值）。"""
    inlet_o2 = float(system.cells[0].inlet.gas_moles[0]) if system.cells else 0.0
    if inlet_o2 <= 0:
        return float("nan")
    threshold = 0.01 * inlet_o2
    for state, z in zip(system.results, system.z_positions):
        if float(state.gas_moles[0]) <= threshold:
            return float(z)
    return float("nan")


__all__ = [
    "FORTRAN_L_M",
    "FORTRAN_D_M",
    "EXPECTED_CASE_NAMES",
    "DRY_POWDER_CASES",
    "parse_fortran_log",
    "build_case_inputs",
    "solve_fortran_case",
    "fortran_gas_inlet_temperature",
]
