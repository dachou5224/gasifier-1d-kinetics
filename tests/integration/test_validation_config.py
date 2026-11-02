import json
import os
import sys
import numpy as np
import logging
import unittest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem
from model.validation_loader import (
    get_validation_cases_final_path,
    iter_validation_cases_final,
    load_case_from_repo,
    normalize_final_case_to_legacy,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from model.chemistry import COAL_DATABASE

# 非 Wen & Chaung 系列的算例中，已知在当前物理设置下不收敛的。
# 这些算例使用各自的进料与几何（L=6.0 m、D=2.0 m），不属于本 PR 的修复范围；
# 显式登记有两个作用：
#   1. 未收敛不会静默通过——白名单之外的算例必须收敛；
#   2. 若其中一个开始收敛，日志里会可见，提示更新本表。
# 依据：2026-11-02 用与测试相同的求解路径逐例复算，见 PR #5 描述。
KNOWN_NONCONVERGENT_CASES = frozenset({
    "slurry western",   # ShenYou_1，cost≈1.9e+01
    "Illinois_No6",     # Illinois_No6 原煤，cost≈2.6e-04
    "Paper_Case_6",     # Paper_Base_Coal，cost≈8.3e-04
})


class TestValidationCases(unittest.TestCase):

    def _assert_physical_outlet(self, case_name: str, T_out: float, y_co: float) -> None:
        """对一个已收敛的解断言出口落在气化炉的物理区间。

        这是冒烟级检查，不是与实验值的精度验证：原料跨度大，故取宽窗口。
        """
        self.assertGreater(T_out, 800.0,
                           f"{case_name}: converged but implausibly cold ({T_out:.0f} K)")
        self.assertLess(T_out, 2500.0,
                        f"{case_name}: converged but implausibly hot ({T_out:.0f} K)")
        self.assertGreater(y_co, 10.0, f"{case_name}: implausibly low dry CO ({y_co:.1f} %)")

    @classmethod
    def setUpClass(cls):
        json_path = get_validation_cases_final_path()
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        cases = {}
        for name, case_data in iter_validation_cases_final(raw):
            cases[name] = normalize_final_case_to_legacy(name, case_data)
        cls.config = {"cases": cases}
            
        # Use Centralized Coal Database
        cls.coals = COAL_DATABASE


    def test_cases_from_json(self):
        cases = self.config.get('cases', {})
        
        for case_name, data in cases.items():
            with self.subTest(case=case_name):
                logger.info(f"Running Validation Case: {case_name}")
                
                # 1. Prepare Inputs
                inputs = data['inputs']
                coal_key = data['coal_type']
                coal_props = self.coals.get(coal_key).copy()
                
                # Convert Units
                coal_flow_kg_s = inputs['FeedRate_kg_h'] / 3600.0
                o2_flow_kg_s = coal_flow_kg_s * inputs['Ratio_OC']
                steam_flow_kg_s = coal_flow_kg_s * inputs.get('SteamRatio_SC', 0.0)
                
                op_conds = {
                    'coal_flow': coal_flow_kg_s,
                    'o2_flow': o2_flow_kg_s,
                    'steam_flow': steam_flow_kg_s,
                    'P': inputs['P_Pa'],
                    'T_in': inputs['T_in_K'],
                    'SlurryConcentration': inputs.get('SlurryConcentration', 100.0)
                }

                # Wen & Chaung (1979) 的 Fortran 算例：参考实现用的是 Fortran 式扩散、
                # WGS rat 因子与不缩放的炭氧化速率。用主线设置跑这些算例会大量不收敛，
                # 见 outputs/diagnostics/round-wen-fortran-texaco-20261031。
                if data.get('solver_compat_mode') == 'fortran':
                    op_conds.update({
                        'UseFortranDiffusion': True,
                        'WGS_RatFactor': True,
                        'CharCombustionRateFactor': 1.0,
                    })

                geometry = {
                    'L': inputs.get('L_reactor', 6.0),
                    'D': inputs.get('D_reactor', 2.0)
                }
                
                # 2. Run Model
                system = GasifierSystem(geometry, coal_props, op_conds)
                results, z = system.solve(N_cells=25)
                
                # 3. Check Results
                # Exit Temperature (Last Cell)
                # results is (N_cells, 11) array. 
                # [O2, CH4, CO, CO2, H2S, H2, N2, H2O, Ws, Xc, T]
                res_out = results[-1]
                T_out = res_out[10]
                
                # Exit Composition (CO mole fraction)
                # Gas Moles: Indices 0-7
                gas_moles_vec = res_out[0:8]
                total_gas = np.sum(gas_moles_vec)
                co_moles = res_out[2]
                if total_gas > 1e-9:
                    y_co = (co_moles / total_gas) * 100.0
                else:
                    y_co = 0.0
                
                expected = data['expected']
                
                logger.info(f"  Result T_out: {T_out:.1f} K (Exp: ~{expected.get('TOUT_K_approx')})")
                logger.info(f"  Result Y_CO:  {y_co:.1f} % (Exp: ~{expected.get('YCO_pct_approx')})")
                
                print("\n=== Axial Temperature Profile ===")
                print(f"{'Z (m)':<10} | {'T (K)':<10}")
                print("-" * 25)
                for i in range(len(z)):
                    print(f"{z[i]:<10.3f} | {results[i][10]:<10.1f}")
                print("=================================\n")
                
                print("\n=== Axial Het Reaction Rate Profile (mol/s) ===")
                print(f"{'Z (m)':<10} | {'T (K)':<8} | {'C+O2 (Comb)':<12} | {'C+H2O (Gasif)':<12}")
                print("-" * 55)
                # Need to run a pass to get rates, or extracting from stored result if possible.
                # Actually, 'results' doesn't store rates. We need to recalculate or instrument.
                # Simplest way: Recalculate rates using system.kinetics
                
                for i in range(len(z)):
                    # Reconstruct state
                    res_i = results[i]
                    # [O2, CH4, CO, CO2, H2S, H2, N2, H2O, Ws, Xc, T]
                    # To array x
                    x_Arr = res_i
                    
                    # We need Cell object or similar context.
                    # Or just call kinetics directly.
                    # Creating a dummy state vector
                    from model.state import StateVector
                    sv = StateVector.from_array(x_Arr, P=op_conds['P'])
                    
                    # We need partial pressures.
                    # Heterogeneous rates need T, P, Xc, Ws
                    # Accessing system.cells[i]
                    cell_i = system.cells[i]
                    
                    # 1. Calc Physics
                    # Need solid ref flux. Approximate with coal_flow * 0.6
                    ref_flux = cell_i.coal_flow_dry * 0.6
                    phys = cell_i._calc_physics_props(sv, ref_flux)
                    
                    # 2. Calc Rates
                    # We pass zero source terms for this diagnostic visualization
                    # This might under-estimate availability (clipping), but for major components (C, O2, H2O)
                    # availability comes from Inlet (Solid) or Gas (if abundant).
                    # Actually for Gasification, H2O comes from Inlet+Source.
                    # This diagnostic will show rates as seen by the cell logic
                    dummy_src = np.zeros(8)
                    r_het, r_homo, _ = cell_i._calc_rates(sv, phys, dummy_src)
                    
                    print(f"{z[i]:<10.3f} | {sv.T:<8.1f} | {r_het['C+O2']:<12.2f} | {r_het['C+H2O']:<12.2f}")
                    
                print("=================================\n")


                # 判收敛，再判物理合理性。2026-10-31 之前这里直接断言
                # 1000 K < T_out < 3000 K，但那时算例的煤质是错的（Paper_Base_Coal，
                # 氧 9.44% 而实际进料 1.32%），失败表现为 3400 K 的假温度。
                #
                # 2026-11-02 复审修正：早先的版本在未收敛分支只断言
                # `max_cost > 1e-4`——那正是进入该分支的条件，属同义反复；
                # 即使全部算例退化为非根，测试仍会绿灯。现在改为：
                #   * Texaco 系列（本 PR 修复的对象）必须逐例断言收敛；
                #   * 非 Texaco 的已知未收敛算例走显式白名单，白名单之外的算例
                #     若未收敛即失败；
                #   * 用 np.isfinite 全量检查，避免 Python max 在某种顺序下掩盖 NaN。
                costs = np.asarray([
                    0.5 * float(np.sum(np.asarray(
                        system.cells[j].residuals(np.asarray(results[j], dtype=float)),
                        dtype=float) ** 2))
                    for j in range(len(system.cells))
                ], dtype=float)
                self.assertTrue(
                    bool(np.all(np.isfinite(costs))),
                    f"{case_name}: non-finite residual cost {costs}",
                )
                max_cost = float(np.max(costs)) if costs.size else float("nan")
                converged = max_cost <= 1e-4

                self.assertTrue(np.isfinite(T_out), "T_out is not finite")

                if case_name.startswith("Texaco_"):
                    # 这些是 Wen & Chaung 的算例，本 PR 修正了进料与几何，
                    # 因此必须真的收敛——退化为非根要让测试失败。
                    self.assertLessEqual(
                        max_cost, 1e-4,
                        f"{case_name}: expected convergence after the feedstock fix, "
                        f"got residual cost {max_cost:.3e}",
                    )
                    self._assert_physical_outlet(case_name, T_out, y_co)
                elif case_name in KNOWN_NONCONVERGENT_CASES:
                    # 显式登记的已知未收敛算例：不在本 PR 修复范围内。
                    # 它们仍可能收敛（那是好事），所以这里只记录不断言。
                    logger.warning(
                        "  %s is on the known-nonconvergent list "
                        "(max residual cost %.3e, converged=%s)",
                        case_name, max_cost, converged,
                    )
                    if converged:
                        self._assert_physical_outlet(case_name, T_out, y_co)
                else:
                    # 白名单之外的算例：新出现的未收敛是真问题，必须失败。
                    self.assertTrue(
                        converged,
                        f"{case_name}: not on the known-nonconvergent list but did not "
                        f"converge (max residual cost {max_cost:.3e}). Either fix the case "
                        f"or register it explicitly.",
                    )
                    self._assert_physical_outlet(case_name, T_out, y_co)

                # Note: Exact match requires calibration step

    def test_normalize_final_case_infers_coal_type_for_texaco(self):
        """Texaco I-* 是 H-coal residue Illinois No.6，不是原煤 Paper_Base_Coal。

        2026-10-31 之前这里断言的是 Paper_Base_Coal，那正是 3400 K 假温度的来源：
        两者的氧含量差 7 倍（1.32% vs 9.44%）、灰分差一倍（15.53% vs 7.11%）。
        """
        texaco_i3 = self.config["cases"]["Texaco_I-3"]
        self.assertEqual(texaco_i3["coal_type"], "H-coal_residue_Illinois_No6")
        self.assertEqual(texaco_i3["inputs"]["coal"], "H-coal_residue_Illinois_No6")
        self.assertEqual(texaco_i3["solver_compat_mode"], "fortran")

    def test_texaco_wyodak_uses_its_own_coal(self):
        texaco_w1 = self.config["cases"]["Texaco_W-1"]
        self.assertEqual(texaco_w1["coal_type"], "H-coal_residue_Wyodak")

    def test_known_nonconvergent_case_reports_its_status(self):
        """已知未收敛的算例必须如实报告，而不是悄悄给出"正常"出口。

        回归复审意见（2026-11-02 P1）：原来的未收敛分支只断言
        `max_cost > 1e-4`，等于重复进入条件本身；模型是否真的报告了
        失败从未被验证。这里用白名单里的一个算例做正向验证。
        """
        data = self.config["cases"]["slurry western"]
        inputs = data["inputs"]
        coal_props = self.coals[data["coal_type"]].copy()
        coal_flow = inputs["FeedRate_kg_h"] / 3600.0
        op_conds = {
            "coal_flow": coal_flow,
            "o2_flow": coal_flow * inputs["Ratio_OC"],
            "steam_flow": coal_flow * inputs.get("SteamRatio_SC", 0.0),
            "P": inputs["P_Pa"],
            "T_in": inputs["T_in_K"],
            "SlurryConcentration": inputs.get("SlurryConcentration", 100.0),
            "SolverStrictRoot": True,
        }
        geometry = {"L": inputs["L_reactor"], "D": inputs["D_reactor"]}
        system = GasifierSystem(geometry, coal_props, op_conds)
        rows, _z = system.solve(N_cells=25)

        # 严格模式下模型必须声明无根，并给出单元序号
        self.assertEqual(system.execution_status, "no_root")
        self.assertIsNotNone(system.no_root_cell_index)
        # 结果截断到首个非根之前，出口不能当作有效值传播
        self.assertEqual(len(rows), system.no_root_cell_index)
        # 非有限成本不得被当作合格的根
        costs = np.asarray([d["residual_cost"] for d in system.cell_diagnostics], dtype=float)
        self.assertTrue(bool(np.all(np.isfinite(costs))))
        self.assertGreater(float(np.max(costs)), 1e-4)

    def test_texaco_geometry_matches_fortran_source(self):
        """几何以 reference_fortran 行 32 的 height=330 cm、dt=152.4 cm 为准。

        论文表头写 20 ft x 5 ft（6.096 x 1.524 m），与该程序自身不一致；
        原 JSON 的 6.0 x 2.0 m 两处都不符，直径偏大 31%。
        """
        for name in ("Texaco_I-1", "Texaco_I-3", "Texaco_I-6"):
            inputs = self.config["cases"][name]["inputs"]
            self.assertAlmostEqual(inputs["L_reactor"], 3.3, places=3)
            self.assertAlmostEqual(inputs["D_reactor"], 1.524, places=3)

    def test_load_case_from_repo_preserves_lowercase_pilot_case(self):
        case = load_case_from_repo("texaco i-1")
        self.assertEqual(case["inputs"]["coal"], "texaco i-1_Coal")
        self.assertEqual(case["inputs"]["FeedRate"], 76.66)
        self.assertEqual(case["inputs"]["HeatLossPercent"], 4)

    def test_load_case_from_repo_canonical_case_uses_final_dataset(self):
        case = load_case_from_repo("Texaco_I-1")
        self.assertEqual(case["inputs"]["coal"], "H-coal_residue_Illinois_No6")
        self.assertEqual(case["inputs"]["FeedRate"], 276.0)
        self.assertEqual(case["inputs"]["HeatLossPercent"], 3.0)
        self.assertEqual(case["solver_compat_mode"], "fortran")
                
if __name__ == '__main__':
    unittest.main()
