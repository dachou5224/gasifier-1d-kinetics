# data_calibration.py
# Standardized Coal Database and Validation Cases for Comparison

COAL_DATABASE = {
    "ShenYou 1 (神优1)": {
        "Mt": 14.3, "Ad": 7.22, "Vd": 34.96, "FCd": 57.82,
        "Cd": 75.83, "Hd": 4.58, "Od": 10.89, "Nd": 1.19, "Sd": 0.29,
        "HHV_d": 30720.0  # kJ/kg
    },
    "Paper_Base_Coal": {
        "Cd": 80.19, "Hd": 4.83, "Od": 9.76, "Nd": 0.85, "Sd": 0.41,
        "Ad": 7.35,  "Vd": 31.24, "FCd": 61.41, 
        "Mt": 4.53,
        "HHV_d": 29800.0, # kJ/kg
        "Hf": -1.0e6, # Est J/kg
    },
    "Paper_Case_2_Coal": {
        "Cd": 70.0, "Hd": 4.0, "Od": 15.0, "Nd": 1.0, "Sd": 1.0,
        "Ad": 9.0,  "Vd": 35.0, "FCd": 56.0, 
        "Mt": 10.0,
        "HHV_d": 25000.0, # kJ/kg
    },
}

VALIDATION_CASES = {
    "Paper_Case_6": {
        "inputs": {
            "coal": "Paper_Base_Coal",
            "FeedRate": 41670.0, # kg/h
            "Ratio_OC": 1.05,
            "Ratio_SC": 0.0, # CWS uses slurry water, no external steam
            "P": 4.08e6, # Pa
            "TIN": 300.0, # K
            "HeatLossPercent": 3.0,
            "SlurryConcentration": 62.0, # Texaco typical (%)
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
            "Ratio_SC": 0.0,
            "P": 4.08e6,
            "TIN": 300.0,
            "HeatLossPercent": 3.0,
            "SlurryConcentration": 62.0,
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
            "Ratio_SC": 0.0,
            "P": 4.08e6,
            "TIN": 300.0,
            "HeatLossPercent": 3.0,
            "SlurryConcentration": 60.0,
        },
        "expected": {
            "TOUT_C": 1452.0, "YCO": 61.8, "YH2": 29.7
        }
    },
}
