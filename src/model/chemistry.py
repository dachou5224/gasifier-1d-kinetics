# chemistry.py
# Updated Coal Database synchronized from gasifier-model and calibration data

COAL_DATABASE = {
    "Paper_Base_Coal": {
        # Normalized to C+H+O+N+S+A=100 (from original 103.39%)
        "Cd": 77.56, "Hd": 4.67, "Od": 9.44, "Nd": 0.82, "Sd": 0.40,
        "Ad": 7.11,  "Vd": 31.24, "FCd": 61.41, 
        "Mt": 4.53,
        "HHV_d": 29200.0, # kJ/kg (Fine-tuned benchmark)
    },
    "ShenYou_1": {
        "Mt": 14.3, "Ad": 7.22, "Vd": 34.96, "FCd": 57.82,
        "Cd": 75.83, "Hd": 4.58, "Od": 10.89, "Nd": 1.19, "Sd": 0.29,
        "HHV_d": 30720.0  # kJ/kg
    },
    "ShenYou_2": {
        "Mt": 14.2, "Ad": 7.62, "Vd": 31.60, "FCd": 60.78,
        "Cd": 75.68, "Hd": 4.12, "Od": 11.12, "Nd": 1.04, "Sd": 0.44,
        "HHV_d": 29740.0
    },
    "LuNan_Coal": {
        "Cd": 71.5, "Hd": 4.97, "Od": 11.15, "Nd": 1.07, "Sd": 2.16,
        "Ad": 9.15, "Vd": 32.0, "FCd": 58.85,
        "Mt": 0.0,
        "HHV_d": 27800.0,
    },
    "Illinois_No6": {
        "Cd": 77.36, "Hd": 4.66, "Od": 9.42, "Nd": 0.82, "Sd": 0.40,
        "Ad": 7.35,  "Vd": 31.24, "FCd": 61.41, 
        "Mt": 0.0,
        "HHV_d": 31000.0,
    },
    # --- Wen & Chaung (1979) Texaco 小试的三类进料 ---
    #
    # Ultimate 来自 data/archive/validation_cases_OriginalPaper.json 的 coal_database，
    # 与 reference_fortran/Source1_副本.for 的 namelist 一致：
    #   C=74.05/(C+H+O+N+S=84.10) = 0.8805 vs yc = 0.88026（差 0.03%，源表取整）
    #   O=1.32 /84.10 = 0.01570 vs yoxy = 0.015696
    # 注意：三者是加氢液化残渣 / 减压塔底，不是原煤，氧含量远低于原煤。
    # 不要用 Illinois_No6（原煤，Od=9.42）代替 H-coal residue（Od=1.32）：
    # 氧差 7 倍、灰分差一半，会让算例根本不可解。
    #
    # ---------------------------------------------------------------------------
    # 逐字段溯源：哪些是实测，哪些是假设
    # ---------------------------------------------------------------------------
    # 归档 JSON 只给 H-coal_residue_Illinois_No6 提供了 proximate 与发热量；
    # Wyodak 与 Exxon 只有 ultimate。因此：
    #
    #   字段        Illinois residue              Wyodak / Exxon
    #   -------------------------------  ------------------------------  ------------------------------
    #   C/H/O/N/S   JSON ultimate           实测  JSON ultimate           实测
    #   Ad          JSON ultimate           实测  JSON ultimate           实测
    #   FCd / Vd    fc=65.0 dmmf 推导       假设  fc=65.0 dmmf 推导       假设
    #   Mt          4.53（JSON moisture）   实测  4.53                      假设（借用 Illinois）
    #   HHV_d       29800（JSON HHV）       实测  29800                     假设（借用 Illinois）
    #
    # 关键更正（2026-11-02 复审）：Illinois 的 FCd/Vd **也是假设，不是实测**。
    # 早先注释写成"与 JSON proximate 吻合"是错的：
    #   归档 JSON proximate = FC 61.41 / VM 31.24 / ash 7.35
    #   本表写入            = FC 54.90 / VM 29.57 / ash 15.53
    # 两者不吻合。原因是该 proximate 块的 ash 7.35 与其自身 ultimate 的 ash 15.53
    # 矛盾，判定为母煤数据而非残渣数据，故不采用；FCd/Vd 改由 Fortran 的
    # fc = 65.0 换算得到。所以三者**没有任何一个**有独立 proximate 锚定。
    #
    # 依据：Source1_副本.for 行 55 对所有算例用同一个 `fc = 65.0`（dmmf），
    # 且 namelist 不带逐算例水分与发热量，所以程序本身无法区分这三者。
    #
    # 结论（写论文时必须区分）：
    #   * Illinois residue 算例在 **ultimate 与发热量**上有独立锚定，
    #     其 FCd/Vd 与其余两者一样是 fc=65.0 的推导值；
    #   * Wyodak（W-*）与 Exxon 算例除 ultimate 外全部为借用值，
    #     只是同一条代码路径的一致性检查；
    #   * 三个煤种都不能声称"经过 Wen 实验的工业分析验证"——
    #     归档中的 proximate 数据要么缺失，要么自相矛盾。
    #
    # 水分说明：取论文 JSON 的 4.53。Fortran namelist 用 xmois=0.002，两者冲突；
    # 未按 Fortran 值调参，因为那会变成用参数拟合参考实现（试过：I-1 发散到 1818 K）。
    #
    # 固定碳说明：原始 JSON 的 proximate 块与 ultimate 的灰分自相矛盾
    # （proximate ash=7.35 而 ultimate ash=15.53），判定该块是母煤数据而非残渣数据。
    # 改用 Fortran 口径：FCd = 65*(100-Ad)/100，Vd = 100-FCd-Ad。
    # FCd 会被 PyrolysisService 使用（C_vol_potential = Cd - FCd），不能随手填。
    "H-coal_residue_Illinois_No6": {
        "Cd": 74.05, "Hd": 6.25, "Od": 1.32, "Nd": 0.71, "Sd": 1.77,
        "Ad": 15.53, "Vd": 29.57, "FCd": 54.90,
        "Mt": 4.53,
        "HHV_d": 29800.0,
    },
    "H-coal_residue_Wyodak": {
        "Cd": 78.37, "Hd": 5.79, "Od": 3.70, "Nd": 0.92, "Sd": 0.07,
        "Ad": 11.05, "Vd": 31.15, "FCd": 57.80,
        "Mt": 4.53,
        "HHV_d": 29800.0,
    },
    "Exxon_DSP_vacuum_tower_bottoms": {
        "Cd": 70.74, "Hd": 4.67, "Od": 3.95, "Nd": 1.18, "Sd": 2.74,
        "Ad": 16.72, "Vd": 29.18, "FCd": 54.10,
        "Mt": 4.53,
        "HHV_d": 29800.0,
    },
}

# Mapping Validation Case names to the best available coal data
# Texaco I-* 是 H-coal residue Illinois No.6，W-* 是 H-coal residue Wyodak，
# Texaco_Exxon 是 Exxon DSP 减压塔底。2026-10-31 之前这些条目全部指向
# Paper_Base_Coal（氧 9.44%、灰 7.11%），与实际进料（氧 1.32%、灰 15.53%）
# 差 7 倍，导致校验算例不可解并输出 3400 K 的假温度。
CASE_TO_COAL_MAP = {
    "Coal_Water_Slurry_Western": "ShenYou_1",
    "Coal_Water_Slurry_Eastern": "ShenYou_2",
    "LuNan_Texaco": "LuNan_Coal",
    "Illinois_No6": "Illinois_No6",
    "Texaco_I-1": "H-coal_residue_Illinois_No6",
    "Texaco_I-2": "H-coal_residue_Illinois_No6",
    "Texaco_I-3": "H-coal_residue_Illinois_No6",
    "Texaco_I-4A": "H-coal_residue_Illinois_No6",
    "Texaco_I-4B": "H-coal_residue_Illinois_No6",
    "Texaco_I-5A": "H-coal_residue_Illinois_No6",
    "Texaco_I-5B": "H-coal_residue_Illinois_No6",
    "Texaco_I-5C": "H-coal_residue_Illinois_No6",
    "Texaco_I-6": "H-coal_residue_Illinois_No6",
    "Texaco_I-7A": "H-coal_residue_Illinois_No6",
    "Texaco_I-7B": "H-coal_residue_Illinois_No6",
    "Texaco_I-8A": "H-coal_residue_Illinois_No6",
    "Texaco_I-8B": "H-coal_residue_Illinois_No6",
    "Texaco_I-8C": "H-coal_residue_Illinois_No6",
    "Texaco_I-9": "H-coal_residue_Illinois_No6",
    "Texaco_I-10": "H-coal_residue_Illinois_No6",
    "Texaco_I-11": "H-coal_residue_Illinois_No6",
    "Texaco_W-1": "H-coal_residue_Wyodak",
    "Texaco_W-2": "H-coal_residue_Wyodak",
    "Texaco_W-3A": "H-coal_residue_Wyodak",
    "Texaco_W-3B": "H-coal_residue_Wyodak",
    "Texaco_W-4": "H-coal_residue_Wyodak",
    "Texaco_W-5": "H-coal_residue_Wyodak",
    "Texaco_W-6": "H-coal_residue_Wyodak",
    "Texaco_W-7": "H-coal_residue_Wyodak",
    "Texaco_Exxon": "Exxon_DSP_vacuum_tower_bottoms",
    "texaco i-1": "H-coal_residue_Illinois_No6",
    "texaco i-2": "H-coal_residue_Illinois_No6",
    "texaco i-5c": "H-coal_residue_Illinois_No6",
    "texaco i-10": "H-coal_residue_Illinois_No6",
    "texaco exxon": "Exxon_DSP_vacuum_tower_bottoms",
    "slurry western": "ShenYou_1",
    "slurry eastern": "ShenYou_2",
    "Australia_UBE": "ShenYou_2",
    "Fluid_Coke": "Paper_Base_Coal"
}

VALIDATION_CASES = {
    "Paper_Case_6": {
        "inputs": {
            "coal": "Paper_Base_Coal",
            "FeedRate": 41670.0, # kg/h
            "Ratio_OC": 1.05, 
            "Ratio_SC": 0.08,
            "P": 4.08e6, # Pa
            "TIN": 300.0, # K
            "HeatLossPercent": 2.85,
            "SlurryConcentration": 100.0,
            "Combustion_CO2_Fraction": 0.10,
            "WGS_CatalyticFactor": 1.5,
        },
        "expected": {
            "TOUT_C": 1370.0, "YCO": 61.7, "YH2": 30.3, "YCO2": 1.3
        }
    }
    # Remaining cases are managed via CASE_TO_COAL_MAP in run_all_validation_cases.py
}
