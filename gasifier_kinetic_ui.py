"""
gasifier_kinetic_ui.py - Streamlit Web UI for 1D Kinetic Gasifier (Chem Portal Integration)

供 Chem Portal 调用的 run() 接口，与 gasifier-model/gasifier_ui.py 模式一致。
基于 GasifierSystem（Wen & Chaung 1979 1D 动力学模型）。
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import logging
import json
import time

st.set_page_config(layout="wide")

# 添加 src 到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

logging.getLogger("model").setLevel(logging.WARNING)

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE, VALIDATION_CASES

# 工业工况预设（与 chemistry.py VALIDATION_CASES 核心物理标定对齐）
INDUSTRIAL_CASES = {
    "Paper_Case_6": {
        "coal": "Paper_Base_Coal",
        "FeedRate_kg_h": 41670.0,
        "SlurryConc": 100.0,   # FIXED: Dry-feed for baseline calibration
        "Ratio_OC": 1.019,     # Tuned from 1.05 (0.97x)
        "Ratio_SC": 0.35,      # TUNED: Steam required for H2 balance
        "P_MPa": 4.08,
        "T_in_K": 300.0,
        "HeatLossPercent": 4.5, # CALIBRATED: Radiation/Wall loss
        "L_m": 6.0,
        "D_m": 2.0,
    },
    "LuNan_Texaco_Slurry": {
        "coal": "LuNan_Coal",
        "FeedRate_kg_h": 40000.0,
        "SlurryConc": 60.0,     # Slurry-fed plant
        "Ratio_OC": 1.15,
        "Ratio_SC": 0.0,       # Water from slurry
        "P_MPa": 4.0,
        "T_in_K": 300.0,
        "HeatLossPercent": 1.2, # Larger scale radiation factor
        "L_m": 6.87,
        "D_m": 1.68,
    },
}

# 仅使用已完成有限差分/newton_fd 验证的工况集合
NEWTON_FD_VALIDATED_CASE_KEYS = [
    "Paper_Case_6",
    "Paper_Case_1",
    "Paper_Case_2",
    "LuNan_Texaco_Slurry",
]

# UI 展示层修正：网站体验优先，避免 Paper 干粉工况因高蒸汽比导致温度偏低。
# 不修改 src/model/chemistry.py 的验证数据源，只在 UI 预填阶段覆盖。
UI_CASE_INPUT_OVERRIDES = {
    "Paper_Case_1": {"Ratio_SC": 0.08, "HeatLossPercent": 2.0},
    "Paper_Case_2": {"Ratio_SC": 0.08, "HeatLossPercent": 2.0},
    "Paper_Case_6": {"Ratio_SC": 0.08, "HeatLossPercent": 2.0},
}


def _profile_dataframe(results_arr: np.ndarray, z: np.ndarray, coal_flow: float, cd_wt_pct: float) -> pd.DataFrame:
    df = pd.DataFrame(
        results_arr,
        columns=["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O", "W_solid", "X_C", "T"],
    )
    df["z_m"] = z
    f_dry_arr = np.sum(results_arr[:, :7], axis=1) + 1e-12
    df["yCO_dry_pct"] = results_arr[:, 2] / f_dry_arr * 100.0
    df["yCO2_dry_pct"] = results_arr[:, 3] / f_dry_arr * 100.0
    df["yH2_dry_pct"] = results_arr[:, 5] / f_dry_arr * 100.0
    df["yH2O_wet_pct"] = results_arr[:, 7] / (f_dry_arr + results_arr[:, 7] + 1e-12) * 100.0
    df["T_C"] = results_arr[:, 10] - 273.15
    df["Carbon_Conv"] = 1 - (df["W_solid"] * df["X_C"]) / (coal_flow * (cd_wt_pct / 100.0) + 1e-9)
    return df


def _exit_kpis(results_arr: np.ndarray) -> dict:
    last = results_arr[-1]
    gas = last[:8]
    f_dry = float(np.sum(gas[:7]) + 1e-12)
    return {
        "T_out_C": float(last[10] - 273.15),
        "yCO": float(gas[2] / f_dry * 100.0),
        "yH2": float(gas[5] / f_dry * 100.0),
        "yCO2": float(gas[3] / f_dry * 100.0),
        "syngas": float((gas[2] + gas[5]) / f_dry * 100.0),
        "gas": gas,
    }


def _expected_kpis_for_case(case_name: str) -> dict:
    case = VALIDATION_CASES.get(case_name, {})
    return dict(case.get("expected", {}))


def _solver_recommendation(solver_method: str, compare_enabled: bool) -> tuple[str, str]:
    if solver_method == "jax_jit":
        return (
            "热态服务优先",
            "适合在线重复计算；线上默认应使用已预热的 hot path。"
            if not compare_enabled
            else "当前已启用 minimize 基线对比，适合同时检查精度贴合度与热态收益。",
        )
    if solver_method == "newton_fd":
        return (
            "工程折中优先",
            "适合日常交互与 CPU 路线验证，通常比 minimize 更快，但稳健性仍依赖具体工况。",
        )
    return (
        "稳健基线优先",
        "适合确认基线结果、排查异常工况，通常是最稳但不是最快。",
    )


def _solver_run(system: GasifierSystem, n_cells: int, solver_method: str):
    t0 = time.perf_counter()
    arr, z = system.solve(
        N_cells=n_cells,
        solver_method=solver_method,
        jacobian_mode=("centered_fd" if solver_method == "newton_fd" else "scipy"),
        jax_warmup=True,
    )
    elapsed = time.perf_counter() - t0
    return arr, z, elapsed


def _build_runtime_badge(current_solver: str, elapsed_s: float, baseline_summary, jax_hot_summary) -> str:
    if current_solver == "jax_jit" and jax_hot_summary is not None:
        hot_s = jax_hot_summary["elapsed_s"]
        if baseline_summary is not None and baseline_summary["elapsed_s"] > 0:
            ratio = hot_s / baseline_summary["elapsed_s"]
            return f"热态 `jax_jit` / `minimize` 时间比: `{ratio:.2f}x`"
        return f"热态 `jax_jit` 耗时: `{hot_s:.2f}s`"

    if baseline_summary is not None and baseline_summary["elapsed_s"] > 0:
        ratio = elapsed_s / baseline_summary["elapsed_s"]
        return f"当前 solver / `minimize` 时间比: `{ratio:.2f}x`"

    return f"当前求解耗时: `{elapsed_s:.2f}s`"


def _performance_summary_lines(current_solver: str, elapsed_s: float, baseline_summary, jax_hot_summary) -> list[str]:
    lines = []

    if current_solver == "jax_jit":
        lines.append(f"冷启动展示耗时: `{elapsed_s:.2f}s`")
        if jax_hot_summary is not None:
            hot_s = jax_hot_summary["elapsed_s"]
            lines.append(f"热态再次求解: `{hot_s:.2f}s`")
            if elapsed_s > 0:
                lines.append(f"热态相对冷启动: `{hot_s / elapsed_s:.2f}x`")
        if baseline_summary is not None and jax_hot_summary is not None and baseline_summary["elapsed_s"] > 0:
            lines.append(f"热态 `jax_jit` / `minimize`: `{jax_hot_summary['elapsed_s'] / baseline_summary['elapsed_s']:.2f}x`")
        elif baseline_summary is not None and baseline_summary["elapsed_s"] > 0:
            lines.append(f"当前 `jax_jit` / `minimize`: `{elapsed_s / baseline_summary['elapsed_s']:.2f}x`")
        lines.append("解释：线上服务应隐藏 compile 成本，只向用户暴露热态时间。")
        return lines

    lines.append(f"当前 solver 耗时: `{elapsed_s:.2f}s`")
    if baseline_summary is not None and baseline_summary["elapsed_s"] > 0:
        lines.append(f"相对 `minimize` 时间比: `{elapsed_s / baseline_summary['elapsed_s']:.2f}x`")
    lines.append("解释：当前结果更适合作为交互或基线参考。")
    return lines




def run():
    """Chem Portal 调用的主入口"""
    st.title("🏭 气化炉 1D 动力学模型")
    st.markdown("1D 塞流稳态动力学模拟，基于 Wen & Chaung (1979)，含异相/均相反应与 WGS。")
    st.markdown("[📖 gasifier-1d-kinetic README](https://github.com/dachou5224/gasifier-1d-kinetics)")
    st.divider()

    with st.expander("📖 本地中文操作手册 (使用前必读)", expanded=False):
        st.markdown("""
        **【模型定位】**：本页面为 **一维(1D)管状动力学模型**（基于经典 Wen & Chaung 1979 机制）。它能带你观察煤粉颗粒是如何在管流空间($z$)上一路上演：**脱散挥发分 -> 疯狂争夺氧气燃烧 -> 陷入缺氧开始缓慢吸热气化** 的完整生命周期。由于耦合了常微分动力学网络，它比平衡模型更接近工业现实空间。

        **【预设工况小贴士】**
        *   **Paper Base (文献基准)**：适合学术与机制研究者对标文献数据，观察反应网络机理在标准下的反馈。
        *   **LuNan / Texaco**：适合工业工程师模拟。它们自带了厂里的特征管径、真实的重载运行吃煤量以及高达两千度的高压氛围。

        **【智能化操纵杆：所见即所得】**
        *   **隐藏的流量换算**：在一维动力学求解器底层，其实需要极其严苛复杂的组分质量流率（kg/s）。但你在本应用里**只需微调外围熟悉的宏观“摇杆”**（如水煤浆浓度wt、干煤投料量、O/C）。底层的桥接代码会在运算拉起瞬间，自动帮你算明白水和氧到底各自需要多少 kg/s。
        *   **寻找极端点**：气化炉空间的一大看点是“火焰锋面”。计算完毕后请留意右侧的主图。图上**温度曲线的最高峰**通常对应着氧气($O_2$)浓度被彻底耗尽的位置（即气相氧化区结束，转入吸热气化区的拐点）。通过调整炉长或氧煤比，你能观测到这个驻点在管道内的推移！
        """)

    all_case_names = list(dict.fromkeys(list(VALIDATION_CASES.keys()) + list(INDUSTRIAL_CASES.keys())))
    curated_case_names = [k for k in NEWTON_FD_VALIDATED_CASE_KEYS if (k in VALIDATION_CASES or k in INDUSTRIAL_CASES)]
    preset_names = ["自定义"]

    def _get_case(name):
        if name in VALIDATION_CASES:
            base = dict(VALIDATION_CASES[name]["inputs"])
            if name in UI_CASE_INPUT_OVERRIDES:
                base.update(UI_CASE_INPUT_OVERRIDES[name])
            return base
        if name in INDUSTRIAL_CASES:
            return INDUSTRIAL_CASES[name]
        return None

    with st.sidebar:
        st.header("⚙️ 参数配置")

        show_all_templates = st.toggle(
            "显示全部模板",
            value=False,
            help="默认只显示已做 newton_fd 验证的模板；打开后显示全部 VALIDATION_CASES 与工业模板。",
        )
        preset_pool = all_case_names if show_all_templates else curated_case_names
        preset_names = ["自定义"] + preset_pool

        with st.container(border=True):
            st.markdown("#### 📂 步骤 1. 选择参考模板")
            st.caption("默认仅显示已做 newton_fd 验证的模板；可切换“显示全部模板”。")
            case_name = st.selectbox("选择工况:", preset_names, label_visibility="collapsed")
            case = _get_case(case_name) if case_name != "自定义" else None

            # 自动预填：只要切换到某个模板，就立刻把模板参数写入 session_state
            # 避免“忘记点预填按钮”导致仍使用默认 S/C=0.08、HeatLoss=8% 从而出现异常低温。
            last_loaded = st.session_state.get("_last_loaded_case_k")
            should_autoload = case is not None and last_loaded != case_name

            if case and (should_autoload or st.button("预填该工况", type="secondary", use_container_width=True)):
                coal = COAL_DATABASE.get(case.get("coal", "Paper_Base_Coal"), {})
                feed_kg_h = case.get("FeedRate", case.get("FeedRate_kg_h", 41670.0))
                ratio_oc = case.get("Ratio_OC", 1.05)
                ratio_sc = case.get("Ratio_SC", 0.08)
                P_val = case.get("P_MPa", 4.08) * 1e6 if "P_MPa" in case else case.get("P", 4.08e6)
                
                st.session_state.update({
                    "L_k": float(case.get("L_m", case.get("L", 6.0))),
                    "D_k": float(case.get("D_m", case.get("D", 2.0))),
                    "Cd_k": float(coal.get("Cd", 80.19)), "Hd_k": float(coal.get("Hd", 4.83)),
                    "Od_k": float(coal.get("Od", 9.76)), "Ad_k": float(coal.get("Ad", 7.35)),
                    "feed_kg_h_k": float(feed_kg_h),
                    "ratio_oc_k": float(ratio_oc),
                    "ratio_sc_k": float(ratio_sc),
                    "P_k": float(P_val),
                    "T_in_k": float(case.get("TIN", case.get("T_in_K", 300.0))),
                    "SlurryConc_k": float(case.get("SlurryConcentration", case.get("SlurryConc", 60.0))),
                    "HeatLossPercent_k": float(case.get("HeatLossPercent", 3.0)),
                    "Combustion_CO2_Fraction_k": float(case.get("Combustion_CO2_Fraction", 0.15)),
                    "WGS_CatalyticFactor_k": float(case.get("WGS_CatalyticFactor", 1.5)),
                    "N_cells_k": 40,
                    "_last_loaded_case_k": case_name,
                })
                st.rerun()

        feed_kg_h = st.session_state.get("feed_kg_h_k", 41670.0)
        ratio_oc = st.session_state.get("ratio_oc_k", 1.05)
        ratio_sc = st.session_state.get("ratio_sc_k", 0.08)

        st.markdown("#### 🎛️ 步骤 2. 核心操纵杆")
        st.session_state.solver_method_k = st.selectbox(
            "求解器",
            options=["newton_fd", "minimize", "jax_jit"],
            index=["newton_fd", "minimize", "jax_jit"].index(
                st.session_state.get("solver_method_k", "newton_fd")
            ),
            help="默认 newton_fd：有限差分 Jacobian + 阻尼 Newton；minimize 用于稳健基线对照；jax_jit 为新的全 JAX/JIT 路径。",
        )
        st.caption("验证工况参数来自 model.chemistry.VALIDATION_CASES（含最新 O/C 复标定）。")

        c_feed, c_slurry = st.columns(2)
        st.session_state.feed_kg_h_k = c_feed.number_input("干煤投料 (kg/h)", value=float(feed_kg_h), step=100.0)
        st.session_state.SlurryConc_k = c_slurry.number_input("水煤浆浓度 (wt%)", value=float(st.session_state.get("SlurryConc_k", 60.0)), step=1.0)

        c_o2, c_stm = st.columns(2)
        st.session_state.ratio_oc_k = c_o2.number_input("氧煤比 (O/C)", value=float(ratio_oc), format="%.3f", step=0.01)
        st.session_state.ratio_sc_k = c_stm.number_input("汽煤比 (S/C)", value=float(ratio_sc), format="%.3f", step=0.01)

        c_p, c_loss = st.columns(2)
        st.session_state.P_k = c_p.number_input("压力 (MPa)", value=float(st.session_state.get("P_k", 4.08e6))/1e6, step=0.1) * 1e6
        st.session_state.HeatLossPercent_k = c_loss.number_input("热损估测 (%)", value=float(st.session_state.get("HeatLossPercent_k", 8.0)), step=0.5)

        st.session_state.compare_minimize_k = st.toggle(
            "同时跑 minimize 基线对比",
            value=st.session_state.get("compare_minimize_k", False),
            help="会额外运行一次 minimize，适合检查当前参数下 jax_jit / newton_fd 是否仍贴近稳健基线。",
        )

        rec_title, rec_desc = _solver_recommendation(
            st.session_state.get("solver_method_k", "newton_fd"),
            st.session_state.get("compare_minimize_k", False),
        )
        st.caption(f"推荐模式：{rec_title}")
        st.caption(rec_desc)

        if st.session_state.get("solver_method_k") == "jax_jit":
            st.caption("线上语义：默认假设服务启动时已完成预热；本页面展示的是本地诊断与热态收益评估。")

        st.markdown("---")
        advanced_mode = st.toggle("🛠️ 显示高级配置 (几何/微观)", value=st.session_state.get("advanced_mode_k", False))
        st.session_state.advanced_mode_k = advanced_mode

        if advanced_mode:
            with st.expander("📐 反应器网格与几何", expanded=True):
                st.session_state.L_k = st.number_input("炉长 L (m)", value=float(st.session_state.get("L_k", 6.0)))
                st.session_state.D_k = st.number_input("炉径 D (m)", value=float(st.session_state.get("D_k", 2.0)))
                st.session_state.N_cells_k = st.slider("网格数 N_cells", 10, 80, st.session_state.get("N_cells_k", 40))

            with st.expander("🪨 煤质属性 (Ultimate)", expanded=False):
                st.session_state.Cd_k = st.number_input("C (wt%, d)", value=float(st.session_state.get("Cd_k", 80.19)))
                st.session_state.Hd_k = st.number_input("H (wt%, d)", value=float(st.session_state.get("Hd_k", 4.83)))
                st.session_state.Od_k = st.number_input("O (wt%, d)", value=float(st.session_state.get("Od_k", 9.76)))
                st.session_state.Ad_k = st.number_input("Ash (wt%, d)", value=float(st.session_state.get("Ad_k", 7.35)))

            with st.expander("🏭 入口管道条件", expanded=False):
                st.session_state.T_in_k = st.number_input("入口温度 (K)", value=float(st.session_state.get("T_in_k", 300.0)))

        coal_flow = st.session_state.feed_kg_h_k / 3600.0
        o2_flow = coal_flow * st.session_state.ratio_oc_k
        steam_flow = coal_flow * st.session_state.ratio_sc_k

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 运行动力学模拟", type="primary", use_container_width=True)

    if run_btn:
            geometry = {"L": st.session_state.get("L_k", 6.0), "D": st.session_state.get("D_k", 2.0)}
            coal_props = {
                "Cd": st.session_state.get("Cd_k", 80.19), 
                "Hd": st.session_state.get("Hd_k", 4.83), 
                "Od": st.session_state.get("Od_k", 9.76), 
                "Ad": st.session_state.get("Ad_k", 7.35), 
                "HHV_d": 29800.0
            }
            op_conds = {
                "coal_flow": coal_flow, "o2_flow": o2_flow, "steam_flow": steam_flow,
                "P": st.session_state.get("P_k", 4.08e6), 
                "T_in": st.session_state.get("T_in_k", 300.0),
                "SlurryConcentration": st.session_state.get("SlurryConc_k", 60.0),
                "HeatLossPercent": st.session_state.get("HeatLossPercent_k", 8.0),
                "AdaptiveFirstCellLength": True,
                "Combustion_CO2_Fraction": st.session_state.get("Combustion_CO2_Fraction_k", 0.15),
                "WGS_CatalyticFactor": st.session_state.get("WGS_CatalyticFactor_k", 1.5),
            }
            N_cells = st.session_state.get("N_cells_k", 40)
            solver_method = st.session_state.get("solver_method_k", "newton_fd")

            try:
                system = GasifierSystem(geometry, coal_props, op_conds)
                with st.spinner(f"计算中... (solver={solver_method}, 逐 cell 求解非线性方程组)"):
                    results_arr, z, elapsed_s = _solver_run(system, N_cells, solver_method)

                st.success("收敛成功！")

                Cd_val = st.session_state.get("Cd_k", 80.19)
                df = _profile_dataframe(results_arr, z, coal_flow, Cd_val)
                kpi = _exit_kpis(results_arr)
                gas = kpi["gas"]
                y_co = kpi["yCO"]
                y_h2 = kpi["yH2"]
                y_co2 = kpi["yCO2"]
                T_out_C = kpi["T_out_C"]
                y_co_arr = df["yCO_dry_pct"].to_numpy()
                y_co2_arr = df["yCO2_dry_pct"].to_numpy()
                y_h2_arr = df["yH2_dry_pct"].to_numpy()
                y_h2o_arr = df["yH2O_wet_pct"].to_numpy()
                T_arr = df["T_C"].to_numpy()
                expected = _expected_kpis_for_case(case_name) if case_name != "自定义" else {}

                baseline_summary = None
                jax_hot_summary = None
                if st.session_state.get("compare_minimize_k") and solver_method != "minimize":
                    with st.spinner("额外计算 minimize 基线，用于结果对比..."):
                        sys_base = GasifierSystem(geometry, coal_props, op_conds)
                        base_arr, _base_z, base_elapsed_s = _solver_run(sys_base, N_cells, "minimize")
                    base_kpi = _exit_kpis(base_arr)
                    baseline_summary = {
                        "solver": "minimize",
                        "elapsed_s": base_elapsed_s,
                        "T_out_C": base_kpi["T_out_C"],
                        "yCO": base_kpi["yCO"],
                        "yH2": base_kpi["yH2"],
                        "yCO2": base_kpi["yCO2"],
                        "max_dT_K": float(np.max(np.abs(base_arr[:, 10] - results_arr[:, 10]))),
                        "max_dyCO_pct": float(np.max(np.abs(_profile_dataframe(base_arr, z, coal_flow, Cd_val)["yCO_dry_pct"].to_numpy() - y_co_arr))),
                    }

                if solver_method == "jax_jit":
                    with st.spinner("补充热态 jax_jit 耗时，用于展示在线服务优势..."):
                        sys_hot = GasifierSystem(geometry, coal_props, op_conds)
                        _cold_arr, _cold_z, _cold_elapsed_s = _solver_run(sys_hot, N_cells, "jax_jit")
                        hot_arr, _hot_z, hot_elapsed_s = _solver_run(sys_hot, N_cells, "jax_jit")
                    hot_kpi = _exit_kpis(hot_arr)
                    jax_hot_summary = {
                        "elapsed_s": hot_elapsed_s,
                        "T_out_C": hot_kpi["T_out_C"],
                        "yCO": hot_kpi["yCO"],
                        "yH2": hot_kpi["yH2"],
                        "yCO2": hot_kpi["yCO2"],
                        "max_dT_K_vs_current": float(np.max(np.abs(hot_arr[:, 10] - results_arr[:, 10]))),
                    }

                # --- 核心可视化：学术风格双轴图 ---
                fig_main = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 字体与配色
                font_style = dict(family="Arial, sans-serif", size=14, color="black")
                colors = {"CO": "#E41A1C", "H2": "#4DAF4A", "CO2": "#377EB8", "H2O": "#FF7F00", "Conv": "#000000", "T": "#E7298A"}
                
                fig_main.add_trace(go.Scatter(x=z, y=y_co_arr, mode="lines", name="CO (dry%)", line=dict(color=colors["CO"], width=3)), secondary_y=False)
                fig_main.add_trace(go.Scatter(x=z, y=y_h2_arr, mode="lines", name="H₂ (dry%)", line=dict(color=colors["H2"], width=3)), secondary_y=False)
                fig_main.add_trace(go.Scatter(x=z, y=y_co2_arr, mode="lines", name="CO₂ (dry%)", line=dict(color=colors["CO2"], width=3)), secondary_y=False)
                fig_main.add_trace(go.Scatter(x=z, y=y_h2o_arr, mode="lines", name="H₂O (wet%)", line=dict(color=colors["H2O"], width=1.5, dash="dot")), secondary_y=False)
                fig_main.add_trace(go.Scatter(x=z, y=df["Carbon_Conv"] * 100, mode="lines", name="Carbon Conv. (%)", line=dict(color=colors["Conv"], width=3, dash="dash")), secondary_y=False)
                
                fig_main.add_trace(go.Scatter(x=z, y=T_arr, mode="lines", name="Temperature (°C)", line=dict(color=colors["T"], width=4)), secondary_y=True)

                L_val = st.session_state.get("L_k", 6.0)
                T_max = np.max(T_arr)
                T_range = [800, max(2000, T_max + 200)]

                fig_main.update_layout(
                    title=dict(text=f"<b>Axial Profiles: Temperature and Species Concentration</b>", font=dict(size=18)),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, bordercolor="Black", borderwidth=1),
                    height=550,
                    margin=dict(t=80, b=100, l=60, r=60),
                    font=font_style
                )

                fig_main.update_xaxes(title_text="<b>Distance from Inlet (z) [m]</b>", range=[0, L_val], showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='lightgrey', gridwidth=0.5)
                fig_main.update_yaxes(title_text="<b>Conc. / Conversion [%]</b>", range=[0, 105], showline=True, linewidth=2, linecolor='black', secondary_y=False, gridcolor='lightgrey', gridwidth=0.5)
                fig_main.update_yaxes(title_text="<b>Gas Temperature [°C]</b>", range=T_range, showline=True, linewidth=2, linecolor=colors["T"], secondary_y=True, showgrid=False)

                st.plotly_chart(fig_main, use_container_width=True)

                tab_overview, tab_compare, tab_data = st.tabs(["结果总览", "对比分析", "数据与下载"])

                with tab_overview:
                    service_c1, service_c2, service_c3 = st.columns(3)
                    service_c1.metric("当前 solver", solver_method)
                    service_c2.metric("服务定位", rec_title)
                    service_c3.metric("网格规模", f"{N_cells} cells")

                    if solver_method == "jax_jit":
                        st.info("当前页面是本地诊断视图，会额外展示冷启动与热态差异；线上部署默认应直接提供热态 `jax_jit`。")
                        if jax_hot_summary is not None:
                            hot_msg = (
                                f"当前展示耗时 `{elapsed_s:.2f}s`，同进程热态再次求解约 `{jax_hot_summary['elapsed_s']:.2f}s`。"
                            )
                            if baseline_summary is not None and baseline_summary["elapsed_s"] > 0:
                                ratio = jax_hot_summary["elapsed_s"] / baseline_summary["elapsed_s"]
                                hot_msg += f" 相对 `minimize` 基线时间比约为 `{ratio:.2f}x`。"
                            st.success(hot_msg)

                    st.caption(_build_runtime_badge(solver_method, elapsed_s, baseline_summary, jax_hot_summary))

                    perf_lines = _performance_summary_lines(solver_method, elapsed_s, baseline_summary, jax_hot_summary)
                    st.markdown("#### ⚡ Performance Summary")
                    for line in perf_lines:
                        st.markdown(f"- {line}")

                    c_kpi, c_pie = st.columns([1.2, 1])

                    with c_kpi:
                        st.markdown("#### 🎯 出口关键指标 (KPIs)")
                        mk1, mk2 = st.columns(2)
                        mk1.metric("出口温度", f"{T_out_C:.1f} °C")
                        mk2.metric("碳转化率", f"{df['Carbon_Conv'].iloc[-1]*100:.2f}%")

                        mk3, mk4 = st.columns(2)
                        mk3.metric("当前 solver 耗时", f"{elapsed_s:.2f} s")
                        if solver_method == "jax_jit" and jax_hot_summary is not None:
                            mk4.metric("热态 jax_jit 耗时", f"{jax_hot_summary['elapsed_s']:.2f} s")
                        elif baseline_summary is not None:
                            mk4.metric("minimize 基线耗时", f"{baseline_summary['elapsed_s']:.2f} s")
                        else:
                            mk4.metric("服务建议", rec_title)

                        mk5, mk6, mk7 = st.columns(3)
                        mk5.metric("CO (dry)", f"{y_co:.2f} %")
                        mk6.metric("H₂ (dry)", f"{y_h2:.2f} %")
                        mk7.metric("CO₂ (dry)", f"{y_co2:.2f} %")

                        if expected:
                            st.markdown("#### 📏 与预期值的出口偏差")
                            ex1, ex2, ex3 = st.columns(3)
                            exp_t = expected.get("TOUT_C")
                            exp_yco = expected.get("YCO")
                            exp_yh2 = expected.get("YH2")
                            ex1.metric("ΔT vs expected", "-" if exp_t is None else f"{T_out_C - float(exp_t):+.1f} K")
                            ex2.metric("ΔyCO vs expected", "-" if exp_yco is None else f"{y_co - float(exp_yco):+.2f} %")
                            ex3.metric("ΔyH₂ vs expected", "-" if exp_yh2 is None else f"{y_h2 - float(exp_yh2):+.2f} %")

                    with c_pie:
                        st.markdown("#### 🍕 出口干基组分占比")
                        labels_pie = ["CO", "H₂", "CO₂", "CH₄", "N₂"]
                        values_pie = [gas[2], gas[5], gas[3], gas[1], gas[6]]

                        fig_pie = go.Figure(data=[go.Pie(
                            labels=labels_pie,
                            values=values_pie,
                            hole=.4,
                            marker=dict(colors=[colors["CO"], colors["H2"], colors["CO2"], "#999999", "#CCCCCC"])
                        )])
                        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=240, showlegend=True, legend=dict(orientation="v", x=1.1, y=0.5))
                        st.plotly_chart(fig_pie, use_container_width=True)

                with tab_compare:
                    st.markdown("#### ⚖️ Solver / 结果判读")
                    if baseline_summary is None:
                        st.info("打开左侧“同时跑 minimize 基线对比”后，这里会显示当前 solver 与稳健基线的出口和 profile 偏差。")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("基线耗时", f"{baseline_summary['elapsed_s']:.2f} s")
                        c2.metric("ΔT_out vs minimize", f"{T_out_C - baseline_summary['T_out_C']:+.2f} K")
                        c3.metric("max|ΔT| profile", f"{baseline_summary['max_dT_K']:.2f} K")
                        c4, c5, c6 = st.columns(3)
                        c4.metric("ΔyCO vs minimize", f"{y_co - baseline_summary['yCO']:+.3f} %")
                        c5.metric("ΔyH₂ vs minimize", f"{y_h2 - baseline_summary['yH2']:+.3f} %")
                        c6.metric("max|ΔyCO| profile", f"{baseline_summary['max_dyCO_pct']:.3f} %")

                        summary_rows = pd.DataFrame(
                            [
                                {"solver": solver_method, "time_s": elapsed_s, "T_out_C": T_out_C, "yCO_dry_pct": y_co, "yH2_dry_pct": y_h2, "yCO2_dry_pct": y_co2},
                                {"solver": baseline_summary["solver"], "time_s": baseline_summary["elapsed_s"], "T_out_C": baseline_summary["T_out_C"], "yCO_dry_pct": baseline_summary["yCO"], "yH2_dry_pct": baseline_summary["yH2"], "yCO2_dry_pct": baseline_summary["yCO2"]},
                            ]
                        )
                        if solver_method == "jax_jit" and jax_hot_summary is not None:
                            summary_rows = pd.concat(
                                [
                                    summary_rows,
                                    pd.DataFrame(
                                        [
                                            {
                                                "solver": "jax_jit_hot",
                                                "time_s": jax_hot_summary["elapsed_s"],
                                                "T_out_C": jax_hot_summary["T_out_C"],
                                                "yCO_dry_pct": jax_hot_summary["yCO"],
                                                "yH2_dry_pct": jax_hot_summary["yH2"],
                                                "yCO2_dry_pct": jax_hot_summary["yCO2"],
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

                        if solver_method == "jax_jit" and jax_hot_summary is not None:
                            st.markdown("#### ⚡ jax_jit 热态耗时")
                            hot_c1, hot_c2, hot_c3 = st.columns(3)
                            hot_c1.metric("当前展示耗时", f"{elapsed_s:.2f} s")
                            hot_c2.metric("热态再次求解", f"{jax_hot_summary['elapsed_s']:.2f} s")
                            hot_c3.metric(
                                "热态 / minimize",
                                f"{jax_hot_summary['elapsed_s'] / baseline_summary['elapsed_s']:.2f}x",
                            )

                        compare_fig = make_subplots(
                            rows=1,
                            cols=2,
                            subplot_titles=("Temperature Profile", "Dry Gas Profile (CO / H₂ / CO₂)"),
                        )
                        compare_fig.add_trace(
                            go.Scatter(x=z, y=T_arr, mode="lines", name=f"{solver_method} T", line=dict(width=3, color="#E7298A")),
                            row=1, col=1,
                        )
                        base_df = _profile_dataframe(base_arr, z, coal_flow, Cd_val)
                        compare_fig.add_trace(
                            go.Scatter(x=z, y=base_df["T_C"], mode="lines", name="minimize T", line=dict(width=2, dash="dash", color="#222222")),
                            row=1, col=1,
                        )
                        for col_name, color, label in [
                            ("yCO_dry_pct", "#E41A1C", "CO"),
                            ("yH2_dry_pct", "#4DAF4A", "H₂"),
                            ("yCO2_dry_pct", "#377EB8", "CO₂"),
                        ]:
                            compare_fig.add_trace(
                                go.Scatter(x=z, y=df[col_name], mode="lines", name=f"{solver_method} {label}", line=dict(width=3, color=color)),
                                row=1, col=2,
                            )
                            compare_fig.add_trace(
                                go.Scatter(x=z, y=base_df[col_name], mode="lines", name=f"minimize {label}", line=dict(width=2, dash="dash", color=color)),
                                row=1, col=2,
                            )
                        compare_fig.update_layout(height=460, legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
                        compare_fig.update_xaxes(title_text="z (m)", row=1, col=1)
                        compare_fig.update_xaxes(title_text="z (m)", row=1, col=2)
                        compare_fig.update_yaxes(title_text="T (°C)", row=1, col=1)
                        compare_fig.update_yaxes(title_text="mol%", row=1, col=2)
                        st.plotly_chart(compare_fig, use_container_width=True)

                with tab_data:
                    st.markdown("#### 📦 下载与 profile 数据")
                    export_cols = [
                        "z_m", "T_C", "Carbon_Conv", "yCO_dry_pct", "yH2_dry_pct", "yCO2_dry_pct", "yH2O_wet_pct",
                        "O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O", "W_solid", "X_C",
                    ]
                    export_df = df[export_cols].copy()
                    d1, d2 = st.columns(2)
                    d1.download_button(
                        "下载 profile CSV",
                        data=export_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"gasifier_profile_{solver_method}_{case_name if case_name != '自定义' else 'custom'}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    payload = {
                        "case_name": case_name,
                        "solver_method": solver_method,
                        "elapsed_s": elapsed_s,
                        "geometry": geometry,
                        "coal_props": coal_props,
                        "op_conds": op_conds,
                        "exit_kpis": {k: v for k, v in kpi.items() if k != "gas"},
                        "expected": expected,
                    }
                    d2.download_button(
                        "下载本次输入 JSON",
                        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"gasifier_run_{solver_method}_{case_name if case_name != '自定义' else 'custom'}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    st.dataframe(export_df.round(4), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"计算失败: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("💡 **提示**: 1D 动力学模型含燃烧区起燃、WGS 判据等。建议从 **Paper_Case_6** 或 **LuNan_Texaco_Slurry** 预设开始。")
        with st.expander("📐 气化炉小室划分示意图", expanded=True):
            st.components.v1.html(get_chamber_schematic_html(), height=500, scrolling=False)


def get_chamber_schematic_html() -> str:
    """返回 HTML 字符串，包含完整 SVG（工程蓝图风格：深色背景 #1a2332，蓝白线条）"""
    font_family = '"Microsoft YaHei", "SimHei", sans-serif'
    stroke = "#4FA8FB"
    bg = "#1a2332"
    text_color = "#FFFFFF"
    
    # SVG 1
    svg1 = f'''
    <svg viewBox="0 0 400 500" width="100%" height="100%" style="background:{bg}; border-radius:4px;">
        <text x="200" y="30" font-family="{font_family}" font-size="16" font-weight="bold" fill="{text_color}" text-anchor="middle">图2-2：气化炉小室划分示意图</text>
        
        <!-- 向下箭头（进料方向） -->
        <line x1="200" y1="40" x2="200" y2="70" stroke="{stroke}" stroke-width="2" marker-end="url(#arrow)"/>
        <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="{stroke}" />
            </marker>
        </defs>

        <!-- 气化炉轮廓 -->
        <!-- 喷嘴: 80-100 -->
        <!-- 拱顶: 100-150 -->
        <!-- 主直筒: 150-435 -->
        <!-- 收缩渣口: 435-465 -->
        <!-- 出口直通: 465-480 -->
        <polygon points="
            185,80 215,80 
            215,100 
            240,150 
            240,435 
            215,465 
            215,480 
            185,480 
            185,465 
            160,435 
            160,150 
            185,100" 
            fill="none" stroke="{stroke}" stroke-width="2"/>
        
        <!-- 虚线分隔段 -->
        <!-- 喷嘴 100 -->
        <line x1="140" y1="100" x2="260" y2="100" stroke="{stroke}" stroke-dasharray="4" stroke-width="1"/>
        <!-- 上拱顶 150 -->
        <line x1="140" y1="150" x2="260" y2="150" stroke="{stroke}" stroke-dasharray="4" stroke-width="1"/>
        <!-- 射流段 230 -->
        <line x1="140" y1="230" x2="260" y2="230" stroke="{stroke}" stroke-dasharray="4" stroke-width="1"/>
        <!-- 上直筒段 330 -->
        <line x1="140" y1="330" x2="260" y2="330" stroke="{stroke}" stroke-dasharray="4" stroke-width="1"/>
        <!-- 下直筒段 400 -->
        <line x1="140" y1="400" x2="260" y2="400" stroke="{stroke}" stroke-dasharray="4" stroke-width="1"/>
        <!-- 管流段 435 -->
        <line x1="140" y1="435" x2="260" y2="435" stroke="{stroke}" stroke-dasharray="4" stroke-width="1"/>

        <!-- 左侧名称 -->
        <text x="135" y="95" font-family="{font_family}" font-size="13" fill="{text_color}" text-anchor="end">喷嘴</text>
        <text x="135" y="130" font-family="{font_family}" font-size="13" fill="{text_color}" text-anchor="end">上拱顶</text>
        <text x="135" y="195" font-family="{font_family}" font-size="13" fill="{text_color}" text-anchor="end">射流段</text>
        <text x="135" y="285" font-family="{font_family}" font-size="13" fill="{text_color}" text-anchor="end">上直筒段</text>
        <text x="135" y="370" font-family="{font_family}" font-size="13" fill="{text_color}" text-anchor="end">下直筒段</text>
        <text x="135" y="422" font-family="{font_family}" font-size="13" fill="{text_color}" text-anchor="end">管流段</text>
        <text x="135" y="462" font-family="{font_family}" font-size="13" fill="{text_color}" text-anchor="end">渣口及出口</text>

        <!-- 右侧序号 -->
        <text x="265" y="95" font-family="{font_family}" font-size="13" fill="{text_color}">1</text>
        <text x="265" y="130" font-family="{font_family}" font-size="13" fill="{text_color}">2</text>
        <text x="265" y="300" font-family="{font_family}" font-size="13" fill="{text_color}">... i ...</text>
        <text x="265" y="462" font-family="{font_family}" font-size="13" fill="{text_color}">N</text>
    </svg>
    '''

    # SVG 2
    svg2 = f'''
    <svg viewBox="0 0 400 500" width="100%" height="100%" style="background:{bg}; border-radius:4px;">
        <defs>
            <marker id="arrow_solid" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L0,8 L8,4 z" fill="{stroke}" />
            </marker>
            <marker id="arrow_gas" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
              <path d="M0,0 L0,6 L9,3 z" fill="{stroke}" />
            </marker>
        </defs>

        <text x="200" y="30" font-family="{font_family}" font-size="16" font-weight="bold" fill="{text_color}" text-anchor="middle">图2-3：相邻小室相互关系示意图</text>
        
        <!-- I-1 小室 -->
        <rect x="120" y="100" width="160" height="100" fill="none" stroke="{stroke}" stroke-width="2" rx="4"/>
        <text x="200" y="90" font-family="{font_family}" font-size="14" fill="{text_color}" text-anchor="middle">I-1 小室</text>
        
        <!-- I-1 颗粒 -->
        <circle cx="170" cy="130" r="5" fill="{stroke}"/>
        <circle cx="230" cy="130" r="5" fill="{stroke}"/>
        <circle cx="170" cy="170" r="5" fill="{stroke}"/>
        <circle cx="230" cy="170" r="5" fill="{stroke}"/>

        <!-- I 小室 -->
        <rect x="120" y="280" width="160" height="100" fill="none" stroke="{stroke}" stroke-width="2" rx="4"/>
        <text x="200" y="270" font-family="{font_family}" font-size="14" fill="{text_color}" text-anchor="middle">I 小室</text>
        
        <!-- I 颗粒 -->
        <circle cx="170" cy="310" r="5" fill="{stroke}"/>
        <circle cx="230" cy="310" r="5" fill="{stroke}"/>
        <circle cx="170" cy="350" r="5" fill="{stroke}"/>
        <circle cx="230" cy="350" r="5" fill="{stroke}"/>

        <!-- 左侧固体流动箭头 (粗) -->
        <line x1="80" y1="150" x2="80" y2="330" stroke="{stroke}" stroke-width="4" marker-end="url(#arrow_solid)"/>
        <text x="70" y="240" font-family="{font_family}" font-size="14" fill="{text_color}" text-anchor="end" dominant-baseline="middle">固体流动</text>

        <!-- 右侧气体流动箭头 (细) -->
        <line x1="320" y1="150" x2="320" y2="330" stroke="{stroke}" stroke-width="1" marker-end="url(#arrow_gas)"/>
        <text x="330" y="240" font-family="{font_family}" font-size="14" fill="{text_color}" text-anchor="start" dominant-baseline="middle">气体流动</text>
    </svg>
    '''

    return f'<div style="display:flex; width:100%; gap:20px;">{svg1}{svg2}</div>'


if __name__ == "__main__":
    run()
