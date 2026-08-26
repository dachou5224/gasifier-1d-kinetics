from __future__ import annotations

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


def test_candidate_registry_names_bounds_and_design_only_handling():
    from model.alpha_candidate_identifiability import candidate_alpha_registry

    rows = candidate_alpha_registry()
    by_name = {row["alpha_name"]: row for row in rows}
    assert {"alpha_wgs", "alpha_char_oxidation", "alpha_volatile_oxidation", "alpha_char_gasification"} <= set(by_name)
    assert by_name["alpha_wgs"]["bound_low"] == 0.02
    assert by_name["alpha_char_oxidation"]["bound_high"] == 1.0
    assert by_name["alpha_volatile_oxidation"]["mapped_parameters"] == "op_conds.CO_OxidationFactor + op_conds.H2_OxidationFactor"
    assert by_name["alpha_char_gasification"]["feasibility_status"] == "design_only_no_clean_grouped_path"


def test_inject_alpha_paths_are_explicit_and_non_mutating():
    from model.alpha_candidate_identifiability import inject_alpha

    op_conds = {"WGS_CatalyticFactor": 0.3}
    wgs = inject_alpha(op_conds, "alpha_wgs", 0.4)
    char = inject_alpha(op_conds, "alpha_char_oxidation", 0.5)
    volatile = inject_alpha(op_conds, "alpha_volatile_oxidation", 0.8)
    assert op_conds == {"WGS_CatalyticFactor": 0.3}
    assert wgs["WGS_CatalyticFactor"] == 0.4
    assert char["CharCombustionRateFactor"] == 0.5
    assert volatile["CO_OxidationFactor"] == 0.8
    assert volatile["H2_OxidationFactor"] == 0.8


def test_sensitivity_similarity_and_recommendation_schema_with_fake_solver():
    from model.alpha_candidate_identifiability import (
        RECOMMENDATION_COLUMNS,
        SENSITIVITY_COLUMNS,
        SIMILARITY_COLUMNS,
        alpha_candidate_recommendation_rows,
        alpha_candidate_sensitivity_rows,
        alpha_candidate_similarity_rows,
    )

    def fake_solver(row, alpha_name, alpha_value, n_cells):
        slope = {
            "alpha_wgs": 1.0,
            "alpha_char_oxidation": -0.5,
            "alpha_volatile_oxidation": 0.25,
        }[alpha_name]
        return {
            "case_id": row["_model_case_id"],
            "phase": row["phase"],
            "split": row["_split"],
            "target_CO_volpct_dry": 60.0,
            "target_H2_volpct_dry": 30.0,
            "target_CO2_volpct_dry": 10.0,
            "target_H2_CO_ratio": 0.5,
            "CO_volpct_dry": 60.0 + slope * alpha_value,
            "H2_volpct_dry": 30.0 - slope * alpha_value,
            "CO2_volpct_dry": 10.0 + 0.5 * slope * alpha_value,
            "H2_CO_ratio": 0.5 + 0.01 * slope * alpha_value,
            "solve_status": "ok",
            "fallback_count": 0,
            "poor_convergence_count": 1,
            "poor_convergence_cell_index": "0",
            "cell0_cost": 0.01,
        }

    model_rows = [
        {"_model_case_id": "case_train", "phase": "phase1", "_split": "train"},
        {"_model_case_id": "case_validation", "phase": "phase3", "_split": "validation"},
    ]
    sens = alpha_candidate_sensitivity_rows(model_rows, solver_fn=fake_solver)
    sim = alpha_candidate_similarity_rows(sens)
    rec = alpha_candidate_recommendation_rows(sens, sim)
    assert tuple(sens[0].keys()) == SENSITIVITY_COLUMNS
    assert tuple(sim[0].keys()) == SIMILARITY_COLUMNS
    assert tuple(rec[0].keys()) == RECOMMENDATION_COLUMNS
    assert {"train", "validation"} <= {row["split"] for row in sens}
    assert "fallback_count_minus" in sens[0]
    gas = [row for row in rec if row["alpha_name"] == "alpha_char_gasification"][0]
    assert gas["recommendation"] == "design-only"
    assert math.isfinite(float([row for row in sens if row["alpha_name"] == "alpha_wgs" and row["target"] == "CO_volpct_dry"][0]["fd_gradient"]))
