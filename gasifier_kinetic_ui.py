"""
gasifier_kinetic_ui.py - Streamlit Web UI for 1D Kinetic Gasifier (Chem Portal Integration)

供 Chem Portal 调用的 run() 接口，与 gasifier-model/gasifier_ui.py 模式一致。
基于 GasifierSystem（Wen & Chaung 1979 1D 动力学模型）。
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import logging

# 添加 src 到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

logging.getLogger("model").setLevel(logging.WARNING)

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE, VALIDATION_CASES

# 工业工况预设（与 run_gasifier_model_cases.py 一致）
INDUSTRIAL_CASES = {
    "Paper_Case_6": {
        "coal": "Paper_Base_Coal",
        "FeedRate_kg_h": 41670.0,
        "SlurryConc": 60.0,
        "Ratio_OC": 1.05,
        "Ratio_SC": 0.08,
        "P_MPa": 4.08,
        "T_in_K": 300.0,
        "HeatLossPercent": 8.0,
        "L_m": 6.0,
        "D_m": 2.0,
    },
    "LuNan_Texaco": {
        "coal": "LuNan_Coal",
        "FeedRate_kg_h": 17917.0,
        "SlurryConc": 66.0,
        "Ratio_OC": 0.872,
        "Ratio_SC": 0.0,
        "P_MPa": 4.0,
        "T_in_K": 400.0,
        "HeatLossPercent": 5.0,
        "L_m": 6.87,
        "D_m": 1.68,
    },
}


def run():
    """Chem Portal 调用的主入口"""
    st.title("🏭 气化炉 1D 动力学模型")
    st.markdown("1D 塞流稳态动力学模拟，基于 Wen & Chaung (1979)，含异相/均相反应与 WGS。")
    st.markdown("[📖 gasifier-1d-kinetic README](https://github.com/dachou5224/gasifier-1d-kinetics)")
    st.divider()

    # 预设工况：合并 VALIDATION_CASES 与 INDUSTRIAL_CASES
    preset_names = ["自定义"] + list(VALIDATION_CASES.keys()) + list(INDUSTRIAL_CASES.keys())
    def _get_case(name):
        if name in VALIDATION_CASES:
            return VALIDATION_CASES[name]["inputs"]
        if name in INDUSTRIAL_CASES:
            return INDUSTRIAL_CASES[name]
        return None

    col_input, col_plot = st.columns([1, 2.2], gap="large")

    with col_input:
        st.header("⚙️ 参数配置")

        with st.expander("📂 加载预设工况", expanded=True):
            case_name = st.selectbox("选择工况", preset_names)
            case = _get_case(case_name) if case_name != "自定义" else None
            if case:
                coal = COAL_DATABASE.get(case.get("coal", "Paper_Base_Coal"), {})
                feed_kg_h = case.get("FeedRate", case.get("FeedRate_kg_h", 41670.0))
                feed = feed_kg_h / 3600.0
                ratio_oc = case.get("Ratio_OC", 1.05)
                ratio_sc = case.get("Ratio_SC", 0.08)
                P_val = case.get("P_MPa", 4.08) * 1e6 if "P_MPa" in case else case.get("P", 4.08e6)
                if st.button("加载数据"):
                    st.session_state.update({
                        "L": case.get("L_m", case.get("L", 6.0)),
                        "D": case.get("D_m", case.get("D", 2.0)),
                        "Cd": coal.get("Cd", 80.19), "Hd": coal.get("Hd", 4.83),
                        "Od": coal.get("Od", 9.76), "Ad": coal.get("Ad", 7.35),
                        "coal_flow": feed,
                        "o2_flow": feed * ratio_oc,
                        "steam_flow": feed * ratio_sc,
                        "P": P_val,
                        "T_in": case.get("TIN", case.get("T_in_K", 300.0)),
                        "SlurryConc": case.get("SlurryConcentration", case.get("SlurryConc", 60.0)),
                        "HeatLossPercent": case.get("HeatLossPercent", 3.0),
                    })
                    st.rerun()

        with st.expander("📐 几何参数", expanded=False):
            L = st.number_input("炉长 L (m)", value=st.session_state.get("L", 6.0), key="L_val")
            D = st.number_input("炉径 D (m)", value=st.session_state.get("D", 2.0), key="D_val")
            N_cells = st.slider("网格数 N_cells", 10, 60, st.session_state.get("N_cells", 40), key="N_cells_val")

        with st.expander("🪨 煤质属性", expanded=False):
            Cd = st.number_input("C (wt%, d)", value=st.session_state.get("Cd", 80.19), key="Cd_val")
            Hd = st.number_input("H (wt%, d)", value=st.session_state.get("Hd", 4.83), key="Hd_val")
            Od = st.number_input("O (wt%, d)", value=st.session_state.get("Od", 9.76), key="Od_val")
            Ad = st.number_input("Ash (wt%, d)", value=st.session_state.get("Ad", 7.35), key="Ad_val")

        with st.expander("🏭 工艺条件", expanded=True):
            coal_flow = st.number_input("煤投料 (kg/s)", value=st.session_state.get("coal_flow", 11.58), key="coal_flow_val")
            o2_flow = st.number_input("氧气流量 (kg/s)", value=st.session_state.get("o2_flow", 12.16), key="o2_flow_val")
            steam_flow = st.number_input("蒸汽流量 (kg/s)", value=st.session_state.get("steam_flow", 0.93), key="steam_flow_val")
            P = st.number_input("压力 (Pa)", value=st.session_state.get("P", 4.08e6), format="%.2e", key="P_val")
            T_in = st.number_input("入口温度 (K)", value=st.session_state.get("T_in", 300.0), key="T_in_val")
            SlurryConc = st.number_input("水煤浆浓度 (%)", value=st.session_state.get("SlurryConc", 60.0), key="SlurryConc_val")
            HeatLossPercent = st.number_input("热损 (%)", value=st.session_state.get("HeatLossPercent", 8.0), key="HeatLoss_val")

        run_btn = st.button("🚀 运行模拟", type="primary", use_container_width=True)

    with col_plot:
        if run_btn:
            geometry = {"L": L, "D": D}
            coal_props = {"Cd": Cd, "Hd": Hd, "Od": Od, "Ad": Ad, "HHV_d": 29800.0}
            op_conds = {
                "coal_flow": coal_flow, "o2_flow": o2_flow, "steam_flow": steam_flow,
                "P": P, "T_in": T_in,
                "SlurryConcentration": SlurryConc,
                "HeatLossPercent": HeatLossPercent,
                "AdaptiveFirstCellLength": True,
            }

            try:
                system = GasifierSystem(geometry, coal_props, op_conds)
                with st.spinner("计算中... (逐 cell 求解非线性方程组)"):
                    results_arr, z = system.solve(N_cells=N_cells)

                st.success("收敛成功！")

                # 列: O2, CH4, CO, CO2, H2S, H2, N2, H2O, W_solid, X_C, T
                df = pd.DataFrame(results_arr, columns=["O2", "CH4", "CO", "CO2", "H2S", "H2", "N2", "H2O", "W_solid", "X_C", "T"])
                df["z_m"] = z

                # 出口干基组成
                last = results_arr[-1]
                gas = last[:8]
                F_dry = np.sum(gas[:7]) + 1e-12
                y_co = gas[2] / F_dry * 100
                y_h2 = gas[5] / F_dry * 100
                y_co2 = gas[3] / F_dry * 100
                T_out_C = last[10] - 273.15

                st.metric("出口温度", f"{T_out_C:.1f} °C")
                st.caption(f"干基: CO={y_co:.1f}%  H2={y_h2:.1f}%  CO2={y_co2:.1f}%")

                # 温度分布
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=df["z_m"], y=df["T"] - 273.15, mode="lines+markers", name="T (°C)", line=dict(color="orangered", width=2)))
                fig_t.update_layout(title="轴向温度分布", xaxis_title="z (m)", yaxis_title="T (°C)", height=350)
                st.plotly_chart(fig_t, use_container_width=True)

                # 气相组分
                fig_gas = go.Figure()
                for s in ["CO", "H2", "CO2", "CH4", "H2O"]:
                    fig_gas.add_trace(go.Scatter(x=df["z_m"], y=df[s], mode="lines", name=s))
                fig_gas.update_layout(title="轴向气相摩尔流 (mol/s)", xaxis_title="z (m)", yaxis_title="Flow", height=350)
                st.plotly_chart(fig_gas, use_container_width=True)

                # 碳转化率与固体
                c1, c2 = st.columns(2)
                with c1:
                    df["Carbon_Conv"] = 1 - (df["W_solid"] * df["X_C"]) / (coal_flow * (Cd / 100.0) + 1e-9)
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(x=df["z_m"], y=df["Carbon_Conv"], name="碳转化率", line=dict(dash="dot")))
                    fig_conv.update_layout(title="碳转化率", yaxis_title="Conversion (-)", height=280)
                    st.plotly_chart(fig_conv, use_container_width=True)
                with c2:
                    fig_solid = go.Figure()
                    fig_solid.add_trace(go.Scatter(x=df["z_m"], y=df["W_solid"], name="固体流量", fill="tozeroy"))
                    fig_solid.update_layout(title="固体质量流量 (kg/s)", yaxis_title="W", height=280)
                    st.plotly_chart(fig_solid, use_container_width=True)

            except Exception as e:
                st.error(f"计算失败: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("💡 **提示**: 1D 动力学模型含燃烧区起燃、WGS 判据等。建议从 **Paper_Case_6** 或 **LuNan_Texaco** 预设开始。")
