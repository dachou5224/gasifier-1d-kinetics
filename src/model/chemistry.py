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
    }
}

# Mapping Validation Case names to the best available coal data
CASE_TO_COAL_MAP = {
    "Coal_Water_Slurry_Western": "ShenYou_1",
    "Coal_Water_Slurry_Eastern": "ShenYou_2",
    "LuNan_Texaco": "LuNan_Coal",
    "Illinois_No6": "Illinois_No6",
    "Texaco_I-1": "Paper_Base_Coal",
    "Texaco_I-2": "Paper_Base_Coal",
    "Texaco_I-3": "Paper_Base_Coal",
    "Texaco_I-4A": "Paper_Base_Coal",
    "Texaco_I-4B": "Paper_Base_Coal",
    "Texaco_I-5A": "Paper_Base_Coal",
    "Texaco_I-5B": "Paper_Base_Coal",
    "Texaco_I-5C": "Paper_Base_Coal",
    "Texaco_I-6": "Paper_Base_Coal",
    "Texaco_I-7A": "Paper_Base_Coal",
    "Texaco_I-7B": "Paper_Base_Coal",
    "Texaco_I-8A": "Paper_Base_Coal",
    "Texaco_I-8B": "Paper_Base_Coal",
    "Texaco_I-8C": "Paper_Base_Coal",
    "Texaco_I-9": "Paper_Base_Coal",
    "Texaco_I-10": "Paper_Base_Coal",
    "Texaco_I-11": "Paper_Base_Coal",
    "Texaco_W-1": "Paper_Base_Coal",
    "Texaco_W-2": "Paper_Base_Coal",
    "Texaco_W-3A": "Paper_Base_Coal",
    "Texaco_W-3B": "Paper_Base_Coal",
    "Texaco_W-4": "Paper_Base_Coal",
    "Texaco_W-5": "Paper_Base_Coal",
    "Texaco_W-6": "Paper_Base_Coal",
    "Texaco_W-7": "Paper_Base_Coal",
    "Texaco_Exxon": "Paper_Base_Coal",
    "texaco i-1": "Paper_Base_Coal",
    "texaco i-2": "Paper_Base_Coal",
    "texaco i-5c": "Paper_Base_Coal",
    "texaco i-10": "Paper_Base_Coal",
    "texaco exxon": "Paper_Base_Coal",
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
