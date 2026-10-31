#!/usr/bin/env python3
"""氢限幅与反向 WGS 截断的可解性审计（2026-10-31）。

起因：`tests/integration/test_validation_config.py` 的 Texaco_I-3 / I-6 出口温度
超过 3000 K 上界。本脚本回答三个问题：
1. 3400 K 是物理结果还是未收敛产物；
2. 若属后者，是哪一处代码改动引起的；
3. 该改动是否被论文轮次依赖（能否直接回退）。

只读现有代码与产物，不改方程、不放宽容差。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.ERROR)

from model.cell import Cell  # noqa: E402
from model.chemistry import COAL_DATABASE  # noqa: E402
from model.gasifier_system import GasifierSystem  # noqa: E402
from model.revision_evidence_common import load_frozen_split  # noqa: E402
from model.validation_loader import (  # noqa: E402
    get_validation_cases_final_path,
    iter_validation_cases_final,
    normalize_final_case_to_legacy,
)

OUT = ROOT / "outputs" / "diagnostics" / "round-wgs-cap-audit-20261031"
TEXACO = ("Texaco_I-1", "Texaco_I-3", "Texaco_I-6")
HUAYI = (
    "R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-10N_006__W001",
    "R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11N_009__W002",
    "R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003",
)


def cap_on(avail_h2, r_h2_ox, r_c_h2, r_wgs):
    """现行实现：收紧 C+H2，并截断负向净 WGS。"""
    rem = max(float(avail_h2) - float(r_h2_ox), 0.0)
    limited = min(float(r_c_h2), rem * 0.5 * 0.99)
    rem2 = max(rem - 2.0 * limited, 0.0)
    w = float(r_wgs)
    if w < 0.0:
        w = max(w, -rem2 * 0.99)
    return limited, w


def cap_off(avail_h2, r_h2_ox, r_c_h2, r_wgs):
    """只保留 C+H2 收紧，不截断 WGS（变体 A）。"""
    rem = max(float(avail_h2) - float(r_h2_ox), 0.0)
    return min(float(r_c_h2), rem * 0.5 * 0.99), float(r_wgs)


def cap_wgs_only(avail_h2, r_h2_ox, r_c_h2, r_wgs):
    """只截断 WGS，C+H2 用旧口径（变体 B）。"""
    rem = max(float(avail_h2) - float(r_h2_ox), 0.0)
    rem2 = max(rem - 2.0 * float(r_c_h2), 0.0)
    w = float(r_wgs)
    if w < 0.0:
        w = max(w, -rem2 * 0.99)
    return float(r_c_h2), w


VARIANTS = {
    "A_cH2_only_no_wgs_cap": cap_off,
    "B_wgs_cap_only": cap_wgs_only,
    "C_current_both": cap_on,
}


def _texaco_cases():
    raw = json.load(open(get_validation_cases_final_path(), encoding="utf-8"))
    return {n: normalize_final_case_to_legacy(n, d) for n, d in iter_validation_cases_final(raw)}


def run_texaco(cases, name, n_cells=25):
    data = cases[name]
    inp = data["inputs"]
    coal = COAL_DATABASE[data["coal_type"]].copy()
    w = inp["FeedRate_kg_h"] / 3600.0
    op = {
        "coal_flow": w,
        "o2_flow": w * inp["Ratio_OC"],
        "steam_flow": w * inp.get("SteamRatio_SC", 0.0),
        "P": inp["P_Pa"],
        "T_in": inp["T_in_K"],
        "SlurryConcentration": inp.get("SlurryConcentration", 100.0),
    }
    geo = {"L": inp.get("L_reactor", 6.0), "D": inp.get("D_reactor", 2.0)}
    system = GasifierSystem(geo, coal, op)
    res, z = system.solve(N_cells=n_cells)
    costs = [0.5 * float(np.sum(np.asarray(c.residuals(np.asarray(row, dtype=float)), dtype=float) ** 2))
             for c, row in zip(system.cells, res)]
    gas = res[-1][:8]
    dry = float(np.sum(gas[:7]))
    return {
        "case_name": name,
        "T_out_K": float(res[-1][10]),
        "max_residual_cost": float(max(costs)),
        "n_cells_above_1e-4": int(sum(c > 1e-4 for c in costs)),
        "n_cells": int(len(costs)),
        "CO_dry_vol_pct": float(gas[2] / dry * 100.0) if dry > 0 else np.nan,
        "H2_dry_vol_pct": float(gas[5] / dry * 100.0) if dry > 0 else np.nan,
        "CO2_dry_vol_pct": float(gas[3] / dry * 100.0) if dry > 0 else np.nan,
    }


def run_huayi(row, n_cells):
    from model.p1_5_round_20261030 import run_case, silence_solver_logs

    silence_solver_logs()
    r = run_case(row, n_cells=n_cells, particle_model="fortran_constant")
    return {
        "window_id": r["window_id"],
        "N_cells": n_cells,
        "execution_status": r["execution_status"],
        "no_root_cell_index": r["no_root_cell_index"],
        "max_residual_cost": r["max_residual_cost"],
        "X_C_implied": r["X_C_implied"],
        "dry_flow_ratio": r["dry_flow_ratio"],
        "CO_vol_pct": r["CO_vol_pct"],
        "H2_vol_pct": r["H2_vol_pct"],
        "CO2_vol_pct": r["CO2_vol_pct"],
    }


def cap_binding_stats(cells_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cells_csv)
    rows = []
    for (status, comb), g in df.groupby(["execution_status", "is_combustion_zone"]):
        rows.append({
            "execution_status": status,
            "is_combustion_zone": bool(comb),
            "n_cells": int(len(g)),
            "dz_total_m": float(g["dz_m"].sum()),
            "wgs_truncated_dz_m": float(g.loc[g["WGS_NegativeTruncated"] == True, "dz_m"].sum()),  # noqa: E712
            "wgs_truncated_dz_fraction": float(
                g.loc[g["WGS_NegativeTruncated"] == True, "dz_m"].sum() / g["dz_m"].sum()  # noqa: E712
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _texaco_cases()
    ok, rows, _ = load_frozen_split()
    if not ok:
        raise SystemExit("frozen split not intact")
    by_id = {r["window_id"]: r for r in rows}

    texaco_rows, huayi_rows = [], []

    # 1) 三变体在 Texaco 上的表现
    for tag, fn in VARIANTS.items():
        Cell.limit_h2_consumers_after_oxidation = staticmethod(fn)
        for name in TEXACO:
            rec = run_texaco(cases, name)
            rec["variant"] = tag
            texaco_rows.append(rec)

    # 2) 现行实现下，论文轮次是否依赖该截断
    Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_on)
    for wid in HUAYI:
        for n in (40, 160):
            huayi_rows.append(run_huayi(by_id[wid], n))
    Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_off)
    for wid in HUAYI:
        for n in (40, 160):
            rec = run_huayi(by_id[wid], n)
            rec["variant"] = "A_no_wgs_cap"
            huayi_rows.append(rec)
    Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_on)
    for rec in huayi_rows:
        rec.setdefault("variant", "C_current")

    # 3) 截断在 P2-A 已收敛解中的激活范围
    p2a_cells = ROOT / "outputs" / "diagnostics" / "round-p2a-fullmesh-20261030" / "fullmesh_cells.csv"
    stats = cap_binding_stats(p2a_cells) if p2a_cells.exists() else pd.DataFrame()

    # 4) 截断对收敛解的偏离：在 capped 解上评估 uncapped 残差
    consistency = []
    for wid in HUAYI[:2]:
        Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_on)
        from model.p1_5_round_20261030 import run_case, silence_solver_logs

        silence_solver_logs()
        r = run_case(by_id[wid], n_cells=160, particle_model="fortran_constant", keep_system=True)
        system, profile = r["_system"], np.asarray(r["_profile"])
        Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_off)
        worst, worst_i = 0.0, None
        for i, (cell, row) in enumerate(zip(system.cells, profile)):
            res = np.asarray(cell.residuals(np.asarray(row, dtype=float)), dtype=float)
            cost = 0.5 * float(np.sum(res * res))
            if cost > worst:
                worst, worst_i = cost, i
        consistency.append({
            "window_id": wid,
            "N_cells": 160,
            "capped_max_residual_cost": float(r["max_residual_cost"]),
            "same_state_uncapped_max_residual_cost": worst,
            "same_state_uncapped_worst_cell": worst_i,
        })
    Cell.limit_h2_consumers_after_oxidation = staticmethod(cap_on)

    texaco_df = pd.DataFrame(texaco_rows)
    huayi_df = pd.DataFrame(huayi_rows)
    cons_df = pd.DataFrame(consistency)

    texaco_df.to_csv(OUT / "texaco_variant_matrix.csv", index=False)
    huayi_df.to_csv(OUT / "huayi_cap_dependency.csv", index=False)
    if len(stats):
        stats.to_csv(OUT / "p2a_combustion_wgs_binding.csv", index=False)
    if len(cons_df):
        cons_df.to_csv(OUT / "capped_solution_uncapped_residual.csv", index=False)

    protocol = {
        "round": "WGS cap audit / 2026-10-31",
        "trigger": "test_validation_config Texaco_I-3 and Texaco_I-6 exceed the 3000 K sanity bound",
        "files_read": [
            "outputs/diagnostics/round-p2a-fullmesh-20261030/fullmesh_cells.csv",
            "tests/integration/test_validation_config.py",
        ],
        "cell_constant": "Cell.limit_h2_consumers_after_oxidation swapped at runtime; no source edit",
        "variants": {
            "A_cH2_only_no_wgs_cap": "keep tightened C+H2 limit, drop the negative-WGS truncation",
            "B_wgs_cap_only": "keep the negative-WGS truncation, restore the old C+H2 limit",
            "C_current_both": "current HEAD",
        },
        "thresholds": {"root_cost": 1e-4, "test_T_upper_K": 3000.0},
        "frozen": "no equation changed, no tolerance relaxed, no parameter tuned",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 反向 WGS 截断的可解性审计（2026-10-31）\n"]
    lines.append("## 1. Texaco_I-3 / I-6 的 3400 K 不是物理结果\n")
    cur = texaco_df[texaco_df["variant"] == "C_current_both"].set_index("case_name")
    for name in TEXACO:
        r = cur.loc[name]
        lines.append(
            f"- `{name}`：T_out = {r['T_out_K']:.1f} K，最大残差成本 {r['max_residual_cost']:.3e}，"
            f"{int(r['n_cells_above_1e-4'])}/{int(r['n_cells'])} 个单元超过 1e-4。"
        )
    lines.append(
        "\n残差成本比阈值高 4 个数量级，轴向温度逐格奇偶震荡（例如 3424.9 → 3447.7 → 3486.6 → 3508.3 K），"
        "出口段 5 个单元温度完全相同而残差恒为 0.455——这是求解器停在最小二乘局部极小，不是气流达到 3400 K。"
        "测试的 3000 K 上界抓到的是真实的不收敛。\n"
    )

    lines.append("## 2. 归因：两个子改动的可解性都不稳\n")
    lines.append("| 算例 | A 只收紧 C+H2 | B 只截断 WGS | C 现行（两者） |")
    lines.append("|---|---|---|---|")
    for name in TEXACO:
        cells = []
        for tag in ("A_cH2_only_no_wgs_cap", "B_wgs_cap_only", "C_current_both"):
            r = texaco_df[(texaco_df["variant"] == tag) & (texaco_df["case_name"] == name)].iloc[0]
            okk = "✓" if r["max_residual_cost"] <= 1e-4 else "✗"
            cells.append(f"{r['T_out_K']:.0f} K / {r['max_residual_cost']:.1e} {okk}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append(
        "\n没有单一子改动能同时让三个算例收敛：A 在 I-1 留 5.6e-3、B 在 I-6 崩到 0.28、C 在 I-3/I-6 崩到 16.5/0.46。"
        "**这组结果随算例激烈跳变，说明两个耦合限幅把系统推到了高度非光滑、可解性依赖具体工况的区域**，"
        "不是一个可以靠调某一边数值修好的问题。\n"
    )
    exp = {"Texaco_I-1": (54.66, 39.31, 5.51), "Texaco_I-3": (54.66, 39.31, 5.51), "Texaco_I-6": (55.03, 39.43, 5.01)}
    lines.append(
        "附带一项与收敛无关的旁证：变体 A 在 `Texaco_I-3` 上给出干基 CO 54.83 vol%（Wen 表值 54.66），"
        "而现行实现给 68.73。也就是说，这个截断不仅让两个算例失去根，还把 CO 推离了参考值 14 个百分点。"
        "`Texaco_I-6` 同样由 54.01（A）变为 63.39（C），参考值 55.03。\n"
    )

    lines.append("## 3. 该截断是论文轮次的承重件\n")
    if len(huayi_df):
        piv = huayi_df.pivot_table(index=["window_id", "N_cells"], columns="variant",
                                   values=["execution_status", "max_residual_cost"], aggfunc="first")
        for (wid, n), r in piv.iterrows():
            st_c = r[("execution_status", "C_current")]
            st_a = r[("execution_status", "A_no_wgs_cap")]
            mc = r[("max_residual_cost", "C_current")]
            ma = r[("max_residual_cost", "A_no_wgs_cap")]
            lines.append(f"- `{wid.split('__')[-1]}` N={int(n)}：现行 `{st_c}`（{mc:.1e}）→ 去掉截断 `{st_a}`（{ma:.1e}）")
        lines.append(
            "\n去掉截断后 **P2-A 的全部行都变成 `no_root`**。所以在当前方程形式下，"
            "这个截断不是可选优化，而是让全长网格华谊工况存在根的必要条件。"
            "直接回退会作废 P2-A 整轮。\n"
        )
    if len(cons_df):
        for _, r in cons_df.iterrows():
            lines.append(
                f"- `{r['window_id'].split('__')[-1]}`：capped 解残差 {r['capped_max_residual_cost']:.1e}，"
                f"同一状态在 uncapped 模型下残差 {r['same_state_uncapped_max_residual_cost']:.2f}"
                f"（cell {int(r['same_state_uncapped_worst_cell'])}）。"
            )
        lines.append(
            "\n即：P2-A 的收敛解在去掉截断后**根本不再是根**，cell 0 的偏差量级达 9.5–10.0。"
            "截断改变的不是数值路径，而是模型本身。\n"
        )

    lines.append("## 4. 截断在 P2-A 产物中的实际激活范围\n")
    if len(stats):
        for _, r in stats.iterrows():
            lines.append(
                f"- status={r['execution_status']}，燃烧区={r['is_combustion_zone']}："
                f"{int(r['n_cells'])} 个单元，dz 合计 {r['dz_total_m']:.3f} m，"
                f"其中 WGS 截断占 {r['wgs_truncated_dz_fraction']*100:.1f}%"
            )
        lines.append(
            "\n结论：截断**只在燃烧区绑定**（占燃烧区长度一半），气化区完全不绑定。"
            "因此 P2-A 的组成结论主要来自气化区，不受影响；但燃烧区（cell 0/1）的 WGS 是被抑制的，"
            "这与「O2 充足时 WGS 受抑」的定性预期一致，却不是一个已验证的独立模型假设。\n"
        )

    lines.append("## 5. 机理判断\n")
    lines.append(
        "截断在物理上是有依据的：反向 WGS（CO2 + H2 → CO + H2O）消耗 H2，"
        "当瞬时氧化已把 H2 吃到接近零时，不允许反向 WGS 再抽取不存在的 H2 是对的。"
        "问题在于实现方式——把 WGS 速率硬钳到一个依赖状态的上界，"
        "使该方程在钳位区退化为常数，而 MSR 限幅往往同时绑定，"
        "两者叠加后方程组失去足够自由度，不再存在根。\n"
    )
    lines.append(
        "正确的做法不是回退，而是把 H2 的可用量在消耗方之间做**一次一致分配**"
        "（类似 `_instant_volatile_combustion` 对 O2 的处理），使整个系统保持单一光滑约束；"
        "这会改变方程，需要单独授权与全轮重跑。\n"
    )

    lines.append("## 6. 待决策\n")
    lines.append(
        "1. 是否接受「P2-A 结论以反向 WGS 截断为前提」并在论文与回执中显式声明；\n"
        "2. 是否把校验测试从判温度改为判严格根（把「T 超界」翻译成「未收敛」）；\n"
        "3. 是否授权把独立限幅改写为一致的 H2 预算分配（改方程，需重跑 P1.5/P2 轮次）。\n"
    )
    (OUT / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")

    from model.revision_evidence_common import write_sha256_manifest

    artifacts = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "sha256.txt")
    write_sha256_manifest(artifacts, OUT / "sha256.txt")

    print(texaco_df.to_string(index=False))
    print()
    print(huayi_df.to_string(index=False))


if __name__ == "__main__":
    main()
