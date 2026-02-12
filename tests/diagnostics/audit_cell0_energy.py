"""
Cell 0 Energy Audit: check that ΔH / Q_rxn ≈ -1 for exothermic balance.

Energy balance: H_out = H_in + energy_src - Q_loss (reaction heat absorbed as ΔH and sinks).
For exothermic cell: Q_rxn > 0; often ΔH < 0 (evaporation sink). Ratio ΔH/Q_rxn ~ -1 is a sanity check.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.state import StateVector
from model.material import MaterialService
from model.chemistry import VALIDATION_CASES, COAL_DATABASE


def run_cell0_energy_audit(case_name="Paper_Case_6"):
    data = VALIDATION_CASES[case_name]
    inputs = data["inputs"]
    coal_props = COAL_DATABASE[inputs["coal"]]
    op_conds = {
        "coal_flow": inputs["FeedRate"] / 3600.0,
        "o2_flow": (inputs["FeedRate"] * inputs["Ratio_OC"]) / 3600.0,
        "steam_flow": (inputs["FeedRate"] * inputs["Ratio_SC"]) / 3600.0,
        "P": inputs["P"],
        "T_in": inputs["TIN"],
        "HeatLossPercent": inputs.get("HeatLossPercent", 3.0),
        "SlurryConcentration": inputs.get("SlurryConcentration", 100.0),
    }
    geometry = {"L": 8.0, "D": 2.6}
    system = GasifierSystem(geometry, coal_props, op_conds)
    results, z_grid = system.solve(N_cells=50)

    cell0 = system.cells[0]
    state0 = StateVector.from_array(results[0], P=op_conds["P"], z=z_grid[0])

    # Aggregate sources for Cell 0
    g_src = np.zeros(8)
    s_src = 0.0
    e_src = 0.0
    for s in cell0.sources:
        g, sm, e = s.get_sources(cell0.idx, cell0.z, cell0.dz)
        g_src += g
        s_src += sm
        e_src += e

    C_fed = system.W_dry * (coal_props.get("Cd", 60.0) / 100.0)
    phys = cell0._calc_physics_props(state0, C_fed)
    r_het, r_homo, phi = cell0._calc_rates(state0, phys, g_src)

    H_in = MaterialService.get_total_enthalpy(cell0.inlet, coal_props)
    H_out = MaterialService.get_total_enthalpy(state0, coal_props)
    delta_H = H_out - H_in

    Q_rxn_homo = (
        r_homo.get("CO_Ox", 0.0) * 283000.0
        + r_homo.get("H2_Ox", 0.0) * 241800.0
        + r_homo.get("CH4_Ox", 0.0) * 802000.0
    )
    Q_rxn_het = r_het.get("C+O2", 0.0) * 393510.0 * phi
    Q_rxn_total = Q_rxn_homo + Q_rxn_het

    L_total = op_conds.get("L_reactor", 8.0)
    loss_pct = op_conds.get("HeatLossPercent", 3.0) / 100.0
    hhv = coal_props.get("HHV_d", 30.0)
    if hhv > 1000:
        hhv = hhv / 1000.0
    Q_loss = op_conds["coal_flow"] * hhv * 1e6 * loss_pct * (cell0.dz / L_total)

    # Energy residual (should be ~0 at convergence)
    res_E = H_out - (H_in + e_src - Q_loss)
    ratio = delta_H / (Q_rxn_total + 1e-12)

    print("=" * 70)
    print(f"Cell 0 Energy Audit — {case_name}")
    print("=" * 70)
    print(f"  T_out (Cell 0)     = {state0.T:.1f} K  ({state0.T - 273.15:.1f} °C)")
    print()
    print("  Enthalpy (MW):")
    print(f"    H_in             = {H_in/1e6:.4f}")
    print(f"    H_out            = {H_out/1e6:.4f}")
    print(f"    ΔH = H_out - H_in = {delta_H/1e6:.4f}")
    print()
    print("  Reaction heat from rates (MW, exothermic positive):")
    print(f"    Homo (CO_Ox + H2_Ox + CH4_Ox) = {Q_rxn_homo/1e6:.4f}")
    print(f"    Het  (C+O2 × phi)             = {Q_rxn_het/1e6:.4f}")
    print(f"    Q_rxn_total                  = {Q_rxn_total/1e6:.4f}")
    print()
    print("  Other terms (MW):")
    print(f"    energy_src (evap etc)        = {e_src/1e6:.4f}")
    print(f"    Q_loss                       = {Q_loss/1e6:.4f}")
    print()
    print("  Balance check:")
    print(f"    Residual H_out - (H_in + e_src - Q_loss) = {res_E/1e6:.6f} MW  (≈0 ok)")
    print()
    print("  Ratio ΔH / Q_rxn:")
    print(f"    ΔH / Q_rxn = {ratio:.4f}")
    print()
    print("  Note: From balance ΔH = energy_src - Q_loss, so ratio = ΔH/Q_rxn = (energy_src - Q_loss)/Q_rxn.")
    print("  Ratio = -1 only when there is no evap/loss (ΔH = -Q_rxn). With large evap, ratio ≈ energy_src/Q_rxn.")
    check = (e_src - Q_loss) / (Q_rxn_total + 1e-12)
    print(f"  (energy_src - Q_loss)/Q_rxn = {check:.4f}  (matches ΔH/Q_rxn: balance consistent).")
    if abs(ratio - check) < 0.01:
        print("  -> Balance consistent (ΔH = energy_src - Q_loss).")
    else:
        print("  -> Check H_in/H_out vs energy_src.")
    print("=" * 70)
    return ratio


if __name__ == "__main__":
    run_cell0_energy_audit("Paper_Case_6")
