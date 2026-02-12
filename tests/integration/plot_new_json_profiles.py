"""
绘制新 JSON 验证工况（Illinois_No6, Australia_UBE, Fluid_Coke）的轴向分布：
- 温度 T (°C) - 左 y 轴
- 干基合成气组分 (CO, H2, CO2, CH4) (%) - 右 y 轴
- 碳转化率 Xc (%) - 右 y 轴
所有曲线在一张图上，每个工况用不同颜色/线型。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
import matplotlib.pyplot as plt
from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.chemistry import COAL_DATABASE
from model.validation_loader import load_validation_cases_from_json, get_validation_cases_new_path


def dry_mole_fraction_pct(gas_moles):
    """计算干基体积分数（不含 H2O）。"""
    F_dry = sum(gas_moles[:7]) + 1e-12
    return {
        "O2": gas_moles[0] / F_dry * 100,
        "CH4": gas_moles[1] / F_dry * 100,
        "CO": gas_moles[2] / F_dry * 100,
        "CO2": gas_moles[3] / F_dry * 100,
        "H2S": gas_moles[4] / F_dry * 100,
        "H2": gas_moles[5] / F_dry * 100,
        "N2": gas_moles[6] / F_dry * 100,
    }


def main():
    path = get_validation_cases_new_path()
    if not os.path.isfile(path):
        print("Not found:", path)
        return

    coal_db_new, cases_new = load_validation_cases_from_json(path)
    coal_db = {**COAL_DATABASE, **coal_db_new}

    geometry = {"L": 8.0, "D": 2.6}
    n_cells = 50

    # 存储每个工况的轴向数据
    profiles = {}

    print("运行工况并收集轴向数据...")
    for name, data in cases_new.items():
        inp = data["inputs"]
        coal_props = coal_db[inp["coal"]]
        op_conds = {
            "coal_flow": inp["FeedRate"] / 3600.0,
            "o2_flow": (inp["FeedRate"] * inp["Ratio_OC"]) / 3600.0,
            "steam_flow": (inp["FeedRate"] * inp["Ratio_SC"]) / 3600.0,
            "P": inp["P"],
            "T_in": inp["TIN"],
            "HeatLossPercent": inp.get("HeatLossPercent", 2.0),
            "SlurryConcentration": inp.get("SlurryConcentration", 100.0),
            "pilot_heat": 5.0e6,
        }

        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            results, z_grid = system.solve(N_cells=n_cells)

            # 计算轴向温度（K -> °C）
            T_prof = results[:, 10] - 273.15
            z_full = np.concatenate(([0.0], z_grid))
            T_full = np.concatenate(([op_conds['T_in'] - 273.15], T_prof))

            # 计算轴向干基组分（排除 H2O，索引 7）
            dry_moles = np.sum(results[:, 0:7], axis=1)  # 只算前 7 个组分（不含 H2O）
            CO_prof = (results[:, 2] / (dry_moles + 1e-9)) * 100.0
            H2_prof = (results[:, 5] / (dry_moles + 1e-9)) * 100.0
            CO2_prof = (results[:, 3] / (dry_moles + 1e-9)) * 100.0
            CH4_prof = (results[:, 1] / (dry_moles + 1e-9)) * 100.0

            # 入口组分（近似为 0）
            CO_full = np.concatenate(([0.0], CO_prof))
            H2_full = np.concatenate(([0.0], H2_prof))
            CO2_full = np.concatenate(([0.0], CO2_prof))
            CH4_full = np.concatenate(([0.0], CH4_prof))

            # 计算轴向碳转化率
            W_dry_kg_s = op_conds['coal_flow'] * (1 - coal_props.get('Mt', 0.0)/100.0)
            C_fed = W_dry_kg_s * (coal_props['Cd']/100.0)
            C_sol_prof = results[:, 8] * results[:, 9]  # solid_mass * carbon_fraction
            Xc_prof = (C_fed - C_sol_prof) / (C_fed + 1e-9) * 100.0
            Xc_full = np.concatenate(([0.0], Xc_prof))

            profiles[name] = {
                'z': z_full,
                'T': T_full,
                'CO': CO_full,
                'H2': H2_full,
                'CO2': CO2_full,
                'CH4': CH4_full,
                'Xc': Xc_full,
            }
            print(f"  {name}: 完成")

        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    if not profiles:
        print("无成功工况，无法绘图")
        return

    # 绘图
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(figsize=(12, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿
    linestyles = ['-', '--', '-.']
    case_names = list(profiles.keys())

    # 左 y 轴：温度
    for i, (name, prof) in enumerate(profiles.items()):
        ax1.plot(prof['z'], prof['T'], 
                label=f"{name} - T", 
                color=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2.0)

    ax1.set_xlabel('轴向位置 z (m)', fontsize=12)
    ax1.set_ylabel('温度 T (°C)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # 右 y 轴：组分和转化率
    ax2 = ax1.twinx()

    # 组分线型：实线、虚线、点划线
    comp_styles = ['-', '--', '-.']
    for i, (name, prof) in enumerate(profiles.items()):
        c = colors[i % len(colors)]
        # CO: 实线
        ax2.plot(prof['z'], prof['CO'], 
                label=f"{name} CO", 
                color=c, linestyle='-', linewidth=2.0, alpha=0.9)
        # H2: 虚线
        ax2.plot(prof['z'], prof['H2'], 
                label=f"{name} H2", 
                color=c, linestyle='--', linewidth=2.0, alpha=0.9)
        # CO2: 点划线
        ax2.plot(prof['z'], prof['CO2'], 
                label=f"{name} CO2", 
                color=c, linestyle='-.', linewidth=2.0, alpha=0.9)
        # Xc: 点线
        ax2.plot(prof['z'], prof['Xc'], 
                label=f"{name} Xc", 
                color=c, linestyle=':', linewidth=2.5, alpha=0.9)

    ax2.set_ylabel('干基组分 (%) / 碳转化率 Xc (%)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 图例：分两列，温度在左轴，组分/转化率在右轴
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 合并图例，温度在前
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    ax1.legend(all_lines, all_labels, loc='upper left', fontsize=9, ncol=2, framealpha=0.9)

    plt.title('新 JSON 验证工况轴向分布：温度、合成气组分与碳转化率', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), '../../results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'new_json_profiles.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
