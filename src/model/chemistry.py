# data_calibration.py
# Standardized Coal Database and Validation Cases for Comparison
#
# Feed type note:
# - Texaco = coal-water slurry (typical 40–70% solids → 30–60% water); Cell 0 has large evaporation sink.
# - Shell  = dry pulverized coal (no slurry water); no liquid water evaporation in Cell 0.
# If the paper providing expected TOUT/YCO/YH2 is for dry-feed, set SlurryConcentration=100 (no slurry water).
# Typical O/C (oxygen-to-coal mass ratio): entrained flow ~0.9–1.1; Ratio_OC 1.05–1.06 is in range; 1.22 is high.

COAL_DATABASE = {
    "ShenYou 1 (神优1)": {
        "Mt": 14.3, "Ad": 7.22, "Vd": 34.96, "FCd": 57.82,
        "Cd": 75.83, "Hd": 4.58, "Od": 10.89, "Nd": 1.19, "Sd": 0.29,
        "HHV_d": 30720.0  # kJ/kg
    },
    "Paper_Base_Coal": {
        # Normalized to C+H+O+N+S+A=100 (from original 103.39%)
        "Cd": 77.36, "Hd": 4.66, "Od": 9.42, "Nd": 0.82, "Sd": 0.40,
        "Ad": 7.35,  "Vd": 31.24, "FCd": 61.41, 
        "Mt": 0.0,   # Set to 0 for simplicity, assumed dry in some logic
        "HHV_d": 31000.0, # kJ/kg (Refined estimate)
        "Hf": -1.2e6,     # Adjusted to match HHV
    },
    "Paper_Case_2_Coal": {
        "Cd": 70.0, "Hd": 4.0, "Od": 15.0, "Nd": 1.0, "Sd": 1.0,
        "Ad": 9.0,  "Vd": 35.0, "FCd": 56.0, 
        "Mt": 10.0,
        "HHV_d": 25000.0, # kJ/kg
    },
    "LuNan_Coal": {
        "Cd": 71.5, "Hd": 4.97, "Od": 11.15, "Nd": 1.07, "Sd": 2.16,
        "Ad": 9.15, "Vd": 32.0, "FCd": 58.85,
        "Mt": 0.0,
        "HHV_d": 27800.0,  # kJ/kg, 鲁南北宿+落陵混煤
    },
}

VALIDATION_CASES = {
    "Paper_Case_6": {
        "inputs": {
            "coal": "Paper_Base_Coal",
            "FeedRate": 41670.0, # kg/h
            "Ratio_OC": 1.05,    # O2/coal mass ~0.9–1.1 typical for entrained flow
            "Ratio_SC": 0.35,    # FIXED: Steam injection is required for dry feed to match H2 and T.
            "P": 4.08e6, # Pa
            "TIN": 300.0, # K
            "HeatLossPercent": 4.5,  # High solid-radiation heat loss tuned to match T_exit
            "SlurryConcentration": 100.0, # FIXED: Target 1.3% CO2 only achievable with Dry-Feed.
            "Combustion_CO2_Fraction": 0.15, # Tuned parameter for high-T partial oxidation dominance
            "WGS_CatalyticFactor": 1.5, # Coal ash catalytic effect
        },
        "expected": {
            "TOUT_C": 1370.0, "YCO": 61.7, "YH2": 30.3, "YCO2": 1.3
        }
    },
    "Paper_Case_1": {
        "inputs": {
            "coal": "Paper_Base_Coal",
            "FeedRate": 41670.0,
            "Ratio_OC": 1.06,
            "Ratio_SC": 0.35,
            "P": 4.08e6,
            "TIN": 300.0,
            "HeatLossPercent": 4.5,
            "SlurryConcentration": 100.0, # Dry feed consistency
            "Combustion_CO2_Fraction": 0.15,
            "WGS_CatalyticFactor": 1.5,
        },
        "expected": {
            "TOUT_C": 1333.0, "YCO": 59.9, "YH2": 29.5
        }
    },
    "Paper_Case_2": {
        "inputs": {
            "coal": "Paper_Base_Coal",
            "FeedRate": 41670.0,
            "Ratio_OC": 1.22,
            "Ratio_SC": 0.35,
            "P": 4.08e6,
            "TIN": 300.0,
            "HeatLossPercent": 4.5,
            "SlurryConcentration": 100.0,
            "Combustion_CO2_Fraction": 0.15,
            "WGS_CatalyticFactor": 1.5,
        },
        "expected": {
            "TOUT_C": 1452.0, "YCO": 61.8, "YH2": 29.7
        }
    },
    "LuNan_Texaco_Slurry": {
        "description": "Real plant data from LuNan Fertilizer Plant (Texaco technology)",
        "inputs": {
            "coal": "LuNan_Coal",
            "FeedRate": 40000.0, # kg/h (Baseline)
            "Ratio_OC": 1.15,    # Typical for Texaco slurry process
            "Ratio_SC": 0.0,     # Slurry water provides all H2O
            "P": 4.0e6,
            "TIN": 300.0,
            "HeatLossPercent": 1.2, # Lower walls loss for larger slurry reactors
            "SlurryConcentration": 60.0, # 60% solid content (High water sink)
            "Combustion_CO2_Fraction": 0.15, 
            "WGS_CatalyticFactor": 1.5,
        },
        "expected": {
            "TOUT_C": 1335.0, "YCO": 51.5, "YH2": 26.5 # Approx plant data
        }
    },
}
