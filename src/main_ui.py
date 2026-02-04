import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# 添加 src 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# If running from src, current_dir IS src. 
# We need 'model' to be importable. 'model' is in 'src'.
# So we need 'src' in sys.path.
src_path = current_dir
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.solver import GasifierSolver1D
from model.chemistry import VALIDATION_CASES, COAL_DATABASE

def run_gui():
    st.set_page_config(page_title="1D Kinetic Gasifier Model", layout="wide")
    st.title("🏭 气化炉一维动力学模型 (1D Kinetic)")
    st.markdown("Validated axial distribution simulation with strict stoichiometry and elemental conservation.")
    
    # Session state for inputs
    if 'Cd' not in st.session_state:
        st.session_state.update({
            'L': 6.0, 'D': 2.0, 'N_cells': 20,
            'Cd': 80.19, 'Hd': 4.83, 'Od': 9.76, 'Ad': 7.35, 'Hf': -0.6e6,
            'coal_flow': 41670.0/3600.0, 'o2_flow': (41670.0*1.05)/3600.0, 
            'steam_flow': (41670.0*0.08)/3600.0, 'P': 4.08e6, 'T_in': 300.0
        })

    st.divider()
    
    # 侧边栏改为右侧布局或分栏
    col_input, col_plot = st.columns([1, 2.2], gap="large")
    
    with col_input:
        st.header("⚙️ 参数配置")
        
        with st.expander("📂 加载验证工况 (Validation Cases)", expanded=True):
            case_name = st.selectbox("选择工况", ["自定义"] + list(VALIDATION_CASES.keys()))
            if case_name != "自定义" and st.button("加载数据"):
                case = VALIDATION_CASES[case_name]["inputs"]
                coal = COAL_DATABASE[case["coal"]]
                st.session_state.update({
                    'Cd': coal['Cd'], 'Hd': coal['Hd'], 'Od': coal['Od'], 'Ad': coal['Ad'],
                    'coal_flow': case['FeedRate'] / 3600.0,
                    'o2_flow': (case['FeedRate'] * case['Ratio_OC']) / 3600.0,
                    'steam_flow': (case['FeedRate'] * case['Ratio_SC']) / 3600.0,
                    'P': case['P'], 'T_in': case['TIN']
                })
                st.rerun()

        with st.expander("📐 几何参数", expanded=False):
            L = st.number_input("炉长 (m)", value=st.session_state.L, key='L_val')
            D = st.number_input("炉径 (m)", value=st.session_state.D, key='D_val')
            N_cells = st.slider("网格分辨率", 5, 50, st.session_state.N_cells, key='N_cells_val')
            
        with st.expander("🪨 煤质属性", expanded=False):
            c1, c2 = st.columns(2)
            Cd = c1.number_input("C (wt%, d)", value=st.session_state.Cd, key='Cd_val')
            Hd = c2.number_input("H (wt%, d)", value=st.session_state.Hd, key='Hd_val')
            Od = c1.number_input("O (wt%, d)", value=st.session_state.Od, key='Od_val')
            Ad = c2.number_input("Ash (wt%, d)", value=st.session_state.Ad, key='Ad_val')
            Hf = st.number_input("生成焓 (J/kg)", value=st.session_state.Hf, key='Hf_val')
            
        with st.expander("🏭 工艺条件", expanded=True):
            coal_flow = st.number_input("煤投料 (kg/s)", value=st.session_state.coal_flow, key='coal_flow_val')
            o2_flow = st.number_input("氧气流量 (kg/s)", value=st.session_state.o2_flow, key='o2_flow_val')
            steam_flow = st.number_input("蒸汽流量 (kg/s)", value=st.session_state.steam_flow, key='steam_flow_val')
            P = st.number_input("压力 (Pa)", value=st.session_state.P, format="%.1e", key='P_val')
            T_in = st.number_input("入口温度 (K)", value=st.session_state.T_in, key='T_in_val')
            
        run_btn = st.button("🚀 运行模拟", type="primary", use_container_width=True)
        
    with col_plot:
        if run_btn:
            geometry = {'L': L, 'D': D}
            coal_props = {'Cd': Cd, 'Hd': Hd, 'Od': Od, 'Ad': Ad, 'Hf': Hf}
            op_conds = {
                'coal_flow': coal_flow, 'o2_flow': o2_flow, 'steam_flow': steam_flow,
                'P': P, 'T_in': T_in
            }
            
            solver = GasifierSolver1D(geometry, coal_props, op_conds)
            
            with st.spinner("计算中... (Solving non-linear equations)"):
                results = solver.solve(N_cells=N_cells)
            
            st.success("收敛成功！")
            
            # 数据处理
            z = np.linspace(0, L, N_cells)
            df = pd.DataFrame(results, columns=['O2', 'CH4', 'CO', 'CO2', 'H2S', 'H2', 'N2', 'H2O', 'W_solid', 'X_C', 'T'])
            df['Distance'] = z
            
            # 平滑数据绘图
            # 1. 温度分布
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=df['Distance'], y=df['T'], mode='lines+markers', name='T (K)', line=dict(color='orangered', width=3)))
            fig_t.update_layout(title="轴向温度分布 (Axial Temperature Profile)", xaxis_title="Position (m)", yaxis_title="T (K)", height=400)
            st.plotly_chart(fig_t, use_container_width=True)
            
            # 2. 气相组成
            fig_gas = go.Figure()
            for s in ['CO', 'H2', 'CO2', 'CH4', 'H2O']:
                fig_gas.add_trace(go.Scatter(x=df['Distance'], y=df[s], mode='lines', name=s))
            fig_gas.update_layout(title="轴向气相组分流率 (Mole Flow Profiles)", xaxis_title="Position (m)", yaxis_title="Flow (mol/s)", height=400)
            st.plotly_chart(fig_gas, use_container_width=True)
            
            # 3. 碳转换与固体
            c1, c2 = st.columns(2)
            with c1:
                df['Carbon_Conv'] = 1 - (df['W_solid'] * df['X_C']) / (coal_flow * (Cd/100.0) + 1e-9)
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(x=df['Distance'], y=df['Carbon_Conv'], name='Conv', line=dict(dash='dot')))
                fig_conv.update_layout(title="碳转化率", yaxis_title="Conversion (-)")
                st.plotly_chart(fig_conv, use_container_width=True)
            with c2:
                fig_solid = go.Figure()
                fig_solid.add_trace(go.Scatter(x=df['Distance'], y=df['W_solid'], name='Solid Flow', fill='tozeroy'))
                fig_solid.update_layout(title="固体质量流量", yaxis_title="W (kg/s)")
                st.plotly_chart(fig_solid, use_container_width=True)
            
        else:
            st.info("💡 **提示**: 煤气化炉一维模型计算通常受燃烧放热起燃控制。目前的模型已包含自动起燃补偿算法。建议从 **Paper_Case_6** 开始尝试。")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Gasifier_Generic.svg/1024px-Gasifier_Generic.svg.png", caption="Generic Entrained Flow Gasifier Schematic", width=300)

if __name__ == "__main__":
    run_gui()
