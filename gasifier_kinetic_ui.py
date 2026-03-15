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
        "Ratio_OC": 1.05,
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

        with st.container(border=True):
            st.markdown("#### 📂 步骤 1. 选择参考模板")
            st.caption("加载实验或工业基准，隐藏底层几何代码直接起跑。")
            case_name = st.selectbox("选择工况:", preset_names, label_visibility="collapsed")
            case = _get_case(case_name) if case_name != "自定义" else None
            
            if case and st.button("预填该工况", type="secondary", use_container_width=True):
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
                    "N_cells_k": 40
                })
                st.rerun()

        feed_kg_h = st.session_state.get("feed_kg_h_k", 41670.0)
        ratio_oc = st.session_state.get("ratio_oc_k", 1.05)
        ratio_sc = st.session_state.get("ratio_sc_k", 0.08)

        st.markdown("#### 🎛️ 步骤 2. 核心操纵杆")
        c_feed, c_slurry = st.columns(2)
        st.session_state.feed_kg_h_k = c_feed.number_input("干煤投料 (kg/h)", value=float(feed_kg_h), step=100.0)
        st.session_state.SlurryConc_k = c_slurry.number_input("水煤浆浓度 (wt%)", value=float(st.session_state.get("SlurryConc_k", 60.0)), step=1.0)

        c_o2, c_stm = st.columns(2)
        st.session_state.ratio_oc_k = c_o2.number_input("氧煤比 (O/C)", value=float(ratio_oc), format="%.3f", step=0.01)
        st.session_state.ratio_sc_k = c_stm.number_input("汽煤比 (S/C)", value=float(ratio_sc), format="%.3f", step=0.01)

        c_p, c_loss = st.columns(2)
        st.session_state.P_k = c_p.number_input("压力 (MPa)", value=float(st.session_state.get("P_k", 4.08e6))/1e6, step=0.1) * 1e6
        st.session_state.HeatLossPercent_k = c_loss.number_input("热损估测 (%)", value=float(st.session_state.get("HeatLossPercent_k", 8.0)), step=0.5)

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

    with col_plot:
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

                Cd_val = st.session_state.get("Cd_k", 80.19)
                df["Carbon_Conv"] = 1 - (df["W_solid"] * df["X_C"]) / (coal_flow * (Cd_val / 100.0) + 1e-9)
                # 干基体积分数 (%)：每 cell 的 F_dry 与 y
                F_dry_arr = np.sum(results_arr[:, :7], axis=1) + 1e-12
                y_co_arr = results_arr[:, 2] / F_dry_arr * 100   # CO
                y_co2_arr = results_arr[:, 3] / F_dry_arr * 100  # CO2
                y_h2_arr = results_arr[:, 5] / F_dry_arr * 100  # H2
                y_h2o_arr = results_arr[:, 7] / (F_dry_arr + results_arr[:, 7] + 1e-12) * 100  # H2O 湿基
                T_arr = results_arr[:, 10] - 273.15

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

                # --- 下方仪表盘：出口指标与组成饼图 ---
                st.divider()
                c_kpi, c_pie = st.columns([1.2, 1])
                
                with c_kpi:
                    st.markdown("#### 🎯 出口关键指标 (KPIs)")
                    mk1, mk2 = st.columns(2)
                    mk1.metric("出口温度", f"{T_out_C:.1f} °C")
                    mk2.metric("碳转化率", f"{df['Carbon_Conv'].iloc[-1]*100:.2f}%")
                    
                    mk3, mk4 = st.columns(2)
                    mk3.metric("有效气 (CO+H₂)", f"{y_co + y_h2:.1f} %")
                    mk4.metric("CO₂ 浓度", f"{y_co2:.1f} %")
                
                with c_pie:
                    st.markdown("#### 🍕 出口干基组分占比")
                    labels_pie = ["CO", "H₂", "CO₂", "CH₄", "N₂"]
                    values_pie = [gas[2], gas[5], gas[3], gas[1], gas[6]] # gas = [O2, CH4, CO, CO2, H2S, H2, N2, H2O]
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=labels_pie, 
                        values=values_pie, 
                        hole=.4,
                        marker=dict(colors=[colors["CO"], colors["H2"], colors["CO2"], "#999999", "#CCCCCC"])
                    )])
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=240, showlegend=True, legend=dict(orientation="v", x=1.1, y=0.5))
                    st.plotly_chart(fig_pie, use_container_width=True)

            except Exception as e:
                st.error(f"计算失败: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("💡 **提示**: 1D 动力学模型含燃烧区起燃、WGS 判据等。建议从 **Paper_Case_6** 或 **LuNan_Texaco** 预设开始。")
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
