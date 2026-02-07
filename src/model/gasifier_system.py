import numpy as np
from scipy.optimize import least_squares
from .state import StateVector
from .cell import Cell
from .material import MaterialService
from .kinetics_service import KineticsService
from .pyrolysis_service import PyrolysisService
from .pyrolysis_service import PyrolysisService
from .constants import PhysicalConstants
from .grid_service import AdaptiveMeshGenerator, MeshConfig
from .source_terms import EvaporationSource, PyrolysisSource
from .solver import NewtonSolver
import logging

# Set up logging
logger = logging.getLogger(__name__)


class GasifierSystem:
    def __init__(self, geometry, coal_props, op_conds):
        # [VALIDATION] Input integrity check
        self._validate_inputs(geometry, coal_props, op_conds)
        
        self.geometry = geometry
        self.coal_props = coal_props
        self.op_conds = op_conds.copy()
        
        # Ensure Hf_coal is consistent with HHV (Critical for Energy Balance)
        if 'Hf_coal' not in self.coal_props:
            # 1. Calculate LHV_d (Lower Heating Value Dry)
            # HHV_d is given. LHV_d = HHV_d - (Latent Heat of Water from H)
            # Latent heat of water at 25C ~ 44 kJ/mol = 2442 kJ/kg_water
            Cd = self.coal_props.get('Cd', 0.0)/100.0
            Hd = self.coal_props.get('Hd', 0.0)/100.0
            Sd = self.coal_props.get('Sd', 0.0)/100.0
            
            hhv_val = self.coal_props.get('HHV_d', 30.0)
            HHV_d_kJ = hhv_val if hhv_val > 1000.0 else hhv_val * 1000.0
            
            # LHV_d = HHV_d - 9*Hd * (Hh2o_vap / Mw_h2o)
            LHV_d_kJ = HHV_d_kJ - (9.0 * Hd * 2442.0)
            self.coal_props['LHV_d_kJ'] = LHV_d_kJ

            # 2. Hf_coal (Formation Enthalpy)
            # Must be consistent with simulator enthalpy basis (H2O is Gas)
            # Hf_coal = LHV_d + Sum(Hf_products_gas)
            # Products: CO2(g), H2O(g), SO2(g)
            H_prod_gas_kJ = (Cd/0.012011)*(-393.51) + \
                            (Hd/0.001008)*0.5*(-241.83) + \
                            (Sd/0.03206)*(-296.81)
            
            Hf_coal_kJ = H_prod_gas_kJ + LHV_d_kJ
            self.coal_props['Hf_coal'] = Hf_coal_kJ * 1000.0 # J/kg
            self.coal_props['Hf_coal'] = Hf_coal_kJ * 1000.0 # J/kg
            logger.info(f"Consistent Hf_coal: {self.coal_props['Hf_coal']/1e6:.2f} MJ/kg (Basis: LHV_d={LHV_d_kJ/1000.0:.2f} MJ/kg)")
        
        # Initialize Services
        self.material = MaterialService()
        self.kinetics = KineticsService()
        self.pyrolysis = PyrolysisService()
        
        self.results = []
        self.cells = []

    def _validate_inputs(self, geometry, coal_props, op_conds):
        """
        Validates the integrity of input dictionaries.
        Raises ValueError if required parameters are missing or invalid.
        """
        # 1. Geometry
        required_geom = ['L', 'D']
        missing_geom = [k for k in required_geom if k not in geometry]
        if missing_geom:
            raise ValueError(f"Missing geometry parameters: {missing_geom}")
        
        if geometry['L'] <= 0 or geometry['D'] <= 0:
            raise ValueError("Geometry dimensions L and D must be positive.")

        # 2. Operating Conditions
        required_op = ['coal_flow', 'o2_flow', 'P', 'T_in']
        missing_op = [k for k in required_op if k not in op_conds]
        if missing_op:
            raise ValueError(f"Missing operating conditions: {missing_op}")
            
        if op_conds['coal_flow'] < 0 or op_conds['o2_flow'] < 0:
            raise ValueError("Flow rates cannot be negative.")
        if op_conds['P'] <= 0 or op_conds['T_in'] <= 0:
            raise ValueError("Pressure and Temperature must be positive.")

        # 3. Coal Properties (Basic integrity)
        if 'Cd' not in coal_props:
             raise ValueError("Coal property 'Cd' (Carbon dry basis) is required.")

    def _initialize_inlet(self):
        """
        Prepare Cell 0 Inlet State.
        Now simplified: Raw Coal + Oxidant Only.
        Moisture and Volatiles are handled via SourceTerms in Cell 0.
        """
        # 1. Physical Feed Components (Dry Coal)
        W_wet_coal = self.op_conds['coal_flow'] 
        Mt = self.coal_props.get('Mt', 0.0)
        W_dry = W_wet_coal * (1 - Mt/100.0)
        
        # 2. Moisture Calculation (For Source Term generation later)
        W_coal_moist = W_wet_coal * (Mt/100.0)
        
        # Slurry Water
        slurry_conc = self.op_conds.get('SlurryConcentration', 100.0)
        if slurry_conc < 100.0:
            W_slurry_h2o = (W_dry / (slurry_conc/100.0)) - W_dry
        else:
            W_slurry_h2o = 0.0
        
        W_steam = self.op_conds.get('steam_flow', 0.0)
        W_h2o_total = W_coal_moist + W_slurry_h2o + W_steam
        
        # Store for Source Term Factory
        self.tmp_W_h2o_total = W_h2o_total
        self.tmp_W_liq_evap = W_coal_moist + W_slurry_h2o # Only liquid needs latent heat
        
        # 3. Oxidant (Gas)
        MW_O2 = 31.998
        F_O2 = (self.op_conds['o2_flow'] / MW_O2) * 1000.0 # mol/s
        F_N2_ox = self.op_conds.get('n2_flow', 0.0) / 28.013 * 1000.0
        
        # 4. Assemble Initial Gas State (mol/s)
        # Only Oxidant enters here. Volatiles/Steam added via Source.
        gas_moles = np.zeros(8)
        gas_moles[0] = F_O2
        gas_moles[6] = F_N2_ox
        
        # 5. Solid State (Raw Dry Coal)
        # Volatiles are still IN the solid at inlet.
        W_solid_in = W_dry
        
        # Carbon Fraction in Raw Coal
        Cd_total = self.coal_props.get('Cd', 60.0) / 100.0
        # Xc matches Cd initially
        Xc_in = Cd_total
        
        inlet = StateVector(
            gas_moles=gas_moles,
            solid_mass=W_solid_in,
            carbon_fraction=Xc_in,
            T=self.op_conds['T_in'],
            P=self.op_conds['P'],
            z=0.0
        )
        
        # Store metadata
        self.W_dry = W_dry
        self.Cd_total = Cd_total
        self.char_Xc0 = Xc_in # Initial Xc is raw coal Xc
        
        # Pre-calc Pyrolysis for Source Term
        molar_yields_per_kg, W_vol_mass_per_kg = self.pyrolysis.calc_yields(self.coal_props)

        self.tmp_F_volatiles = molar_yields_per_kg * W_dry
        self.tmp_W_vol_loss = W_dry * W_vol_mass_per_kg

        
        # Update Char Xc0 just for reference? 
        # Actually, once volatiles leave, the REMAINING solid is Char.
        # But inlet is Coal. Cell 0 output will be Char.
        # We need to know what the target Char Xc is for consistency? 
        # Let's let the mass balance handle it.
        # But we need 'self.char_Xc0' for scaling residuals.
        # Let's estimate Char Xc based on assumption that all volatiles leave.
        C_in_volatiles = np.sum(self.tmp_F_volatiles[1:4]) * 0.012011 + self.tmp_F_volatiles[2]*0.012011 # Wait, indices.
        # [O2, CH4, CO, CO2...]
        # C in CH4(1), CO(2), CO2(3)
        C_in_vol_kg = (self.tmp_F_volatiles[1] + self.tmp_F_volatiles[2] + self.tmp_F_volatiles[3]) * 0.012011
        C_total = W_dry * Cd_total
        C_char = C_total - C_in_vol_kg
        W_char = W_dry - self.tmp_W_vol_loss
        self.char_Xc0 = C_char / max(W_char, 1e-9)
        
        return inlet

    def solve(self, N_cells=100, solver_method='minimize'):
        """Sequential Solver"""
        L = self.geometry['L']
        D = self.geometry['D']
        A = np.pi * (D/2)**2
        V_total = A * L
        
        # [GRID STRATEGY] Variable Mesh via Service
        mesh_cfg = MeshConfig(
            total_length=L,
            n_cells=N_cells,
            ignition_zone_length=PhysicalConstants.FIRST_CELL_LENGTH,
            ignition_zone_res=PhysicalConstants.IGNITION_ZONE_DZ,
            min_grid_size=PhysicalConstants.MIN_GRID_SIZE
        )
        generator = AdaptiveMeshGenerator(mesh_cfg)
        dz_list, self.z_positions = generator.generate()
        
        # Init with Instant Pyrolysis
        current_inlet = self._initialize_inlet()
        
        # [DEBUG] Verify Volatile Flux
        _f = current_inlet.gas_moles
        # [DEBUG] Verify Volatile Flux
        _f = current_inlet.gas_moles
        logger.debug(f"Volatile Flux (mol/s) -> CH4: {_f[1]:.4f}, CO: {_f[2]:.4f}, H2: {_f[5]:.4f}, O2: {_f[0]:.4f}")
        
        self.results = []
        self.cells = []
        
        cell_ops = self.op_conds.copy()
        cell_ops['L_reactor'] = L
        cell_ops['pyrolysis_done'] = True # Pyrolysis already accounted for in inlet
        
        logger.info(f"Starting Solver (Instant Pyrolysis Mode) with {N_cells} cells.")
        
        for i in range(N_cells):
            z_curr = self.z_positions[i]
            dz_curr = dz_list[i]
            V_curr = A * dz_curr
            
        # [SOURCE TERMS]
        # 1. Evaporation Source (Cell 0)
        # [SOURCE TERMS]
        # 1. Evaporation Source (Cell 0)
        # We are injecting H2O mass into the system from "outside" (not in Inlet State).
        # We must provide the Enthalpy of this mass (Liquid Water).
        # H_liq = H_f(liq) approx -285.8 MJ/kmol.
        # Energy Source = Flow * H_liq.
        # EvaporationSource returns -Q, so we pass Q = -H_liq (Positive Magnitude).
        # H_f_liq_mol = -285830 J/mol (NIST: -285.8 kJ/mol)
        # Total Enthalpy Input = Flow_mol * (-285830).
        # But we also should consider T_in sensible heat? 
        # Assume T_in = 298K for moisture reference.
        
        F_H2O_total_mol = (self.tmp_W_h2o_total / 18.015) * 1000.0
        H_liq_J_mol = -285830.0 # Standard Enthalpy of Liquid Water
        
        # Source Energy = Flow * H_liq (Negative value)
        # Class expects 'Q_evap' which it subtracts.
        # src = -Q_evap.
        # We want src = Flow * H_liq.
        # So -Q_evap = Flow * H_liq
        # Q_evap = -(Flow * H_liq).
        Q_water_enthalpy_mag = -(F_H2O_total_mol * H_liq_J_mol)
        logger.info(f"Evaporation Source (Cell 0): Flow={F_H2O_total_mol/1000.0:.2f} mol/s, Enthalpy_Src=-{Q_water_enthalpy_mag/1e6:.2f} MW")
        
        evap_src = EvaporationSource(F_H2O_total_mol, Q_water_enthalpy_mag, target_cell_idx=0)
        
        # 2. Pyrolysis Source (Cell 0)
        # tmp_F_volatiles is mol/s array. tmp_W_vol_loss is kg/s.
        pyro_src = PyrolysisSource(self.tmp_F_volatiles, self.tmp_W_vol_loss, target_cell_idx=0)


        
        sources = [evap_src, pyro_src]
        
        logger.info(f"Starting Solver (Source Term Mode) with {N_cells} cells.")
        
        for i in range(N_cells):
            z_curr = self.z_positions[i]
            dz_curr = dz_list[i]
            V_curr = A * dz_curr
            
            # Update ops for L reference
            cell_ops['L_reactor'] = L
            cell_ops['HeatLossPercent'] = self.op_conds.get('HeatLossPercent', PhysicalConstants.DEFAULT_HEAT_LOSS_PERCENT)
            
            # Pass sources instead of manual Q_source_term
            cell = Cell(i, self.z_positions[i], dz_curr, V_curr, A, 
                        current_inlet, self.kinetics, self.pyrolysis, self.coal_props, cell_ops,
                        sources=sources)
            
            # Pass Coal-Basis parameters for X calculation
            cell.coal_flow_dry = self.W_dry 
            cell.Cd_total = self.Cd_total
            cell.char_Xc0 = self.char_Xc0
            self.cells.append(cell)
            
            # Bounds for least_squares (11 variables)
            # [O2, CH4, CO, CO2, H2S, H2, N2, H2O, Ws, Xc, T]
            lower = np.zeros(11)
            upper = np.ones(11) * 3000.0 # Strict upper bound for gas moles (3x theoretical max)
            
            # Ws, Xc, T specific bounds
            lower[8] = 0.0; upper[8] = self.W_dry * 1.5 
            lower[9] = 0.0; upper[9] = 1.0 # Xc
            lower[10] = 300.0; upper[10] = 4000.0 # T
            
            def func(x):
                return cell.residuals(x)
            
            if i == 0:
                # Multi-Guess ignition for first Active cell
                # [IGNITION STRATEGY] Start with HIGH temperatures (2000K) to overcome evaporation
                guesses_T = [400.0, 1000.0, 1500.0, 2000.0, 3000.0]
                best_sol = None
                best_cost = 1e9
                
                for t_start in guesses_T:
                    x0 = current_inlet.to_array()
                    x0[10] = t_start 
                    
                    if t_start > 900.0:
                        # [IGNITION HELPER]
                        # Balanced Ignition Guess (Strict Atomic Conservation)
                        # Step 1: Burn CH4 to CO + 2H2 (JL Partial Oxidation)
                        n_CH4 = x0[1]
                        n_O2 = x0[0]
                        xi_1 = min(n_CH4 * 0.999, n_O2 * 0.999 / 0.5)
                        x0[1] -= xi_1          # -CH4
                        x0[0] -= 0.5 * xi_1    # -0.5 O2
                        x0[2] += xi_1          # +CO
                        x0[5] += 2.0 * xi_1    # +2 H2
                        
                        # Step 2: Burn H2 to H2O (JL H2_Ox)
                        n_H2 = x0[5]
                        n_O2 = x0[0]
                        xi_2 = min(n_H2 * 0.999, n_O2 * 0.999 / 0.5)
                        x0[5] -= xi_2          # -H2
                        x0[0] -= 0.5 * xi_2    # -0.5 O2
                        x0[7] += xi_2          # +H2O
                        
                        # Step 3: Burn CO to CO2 (Shift/Combustion proxy)
                        n_CO = x0[2]
                        n_O2 = x0[0]
                        xi_3 = min(n_CO * 0.999, n_O2 * 0.999 / 0.5)
                        x0[2] -= xi_3          # -CO
                        x0[0] -= 0.5 * xi_3    # -0.5 O2
                        x0[3] += xi_3          # +CO2

                        # [DIAGNOSTIC REQUEST] Check Heat Release at this Initial Guess (T=2000K)
                        try:
                            diag_state = StateVector.from_array(x0, P=current_inlet.P, z=0.0)
                            # Get Calc Volume
                            diag_V = self.geometry['D']**2/4 * np.pi * dz_list[0]
                            
                            # Calc Rates
                            r_homo_diag = self.kinetics.calc_homogeneous_rates(diag_state, diag_V)
                            
                            # Heats of Reaction (Approx J/mol)
                            # H2+0.5O2->H2O: -241.8 kJ
                            # CO+0.5O2->CO2: -283.0 kJ
                            # CH4+2O2->... : -802.0 kJ
                            Q_rxn_H2  = r_homo_diag['H2_Ox'] * 241800.0
                            Q_rxn_CO  = r_homo_diag['CO_Ox'] * 283000.0
                            Q_rxn_CH4 = r_homo_diag['CH4_Ox'] * 802000.0
                            Q_total_diag = Q_rxn_H2 + Q_rxn_CO + Q_rxn_CH4
                            
                            
                            # Residence Time Check
                            velocity = (diag_state.total_gas_moles * 8.314 * t_start / current_inlet.P) / A
                            residence_time = dz_list[0] / max(velocity, 0.001)
                            
                            logger.info(f"  [IGNITION DIAGNOSTIC] T_guess={t_start}K")
                            logger.info(f"    Geometry: L_cell={dz_list[0]:.3f} m, D={D:.2f} m, Vol={diag_V:.3f} m3")
                            logger.info(f"    Flow:     Vel={velocity:.2f} m/s, Tau={residence_time:.3f} s")
                            logger.info(f"    R_H2_Ox:  {r_homo_diag['H2_Ox']:.2e} mol/s -> {Q_rxn_H2/1e6:.2f} MW")
                            logger.info(f"    R_CO_Ox:  {r_homo_diag['CO_Ox']:.2e} mol/s -> {Q_rxn_CO/1e6:.2f} MW")
                            logger.info(f"    R_CH4_Ox: {r_homo_diag['CH4_Ox']:.2e} mol/s -> {Q_rxn_CH4/1e6:.2f} MW")
                            logger.info(f"    TOTAL REACT HEAT (Reacted State): {Q_total_diag/1e6:.2f} MW")

                            
                            # [DIAGNOSTIC EXTENSION] Potential Heat Release (Inlet + Pyrolysis Volatiles @ T_guess)
                            x_pot = current_inlet.to_array() # Fresh inlet (Oxidant Only)
                            
                            # MANUALLY ADD PYROLYSIS VOLATILES to the check
                            # self.tmp_F_volatiles: [O2, CH4, CO, CO2, H2S, H2, N2, H2O]
                            x_pot[:8] += self.tmp_F_volatiles
                            
                            x_pot[10] = t_start # Force High T
                            state_pot = StateVector.from_array(x_pot, P=current_inlet.P, z=0.0)
                            
                            r_pot = self.kinetics.calc_homogeneous_rates(state_pot, diag_V)
                            Q_pot_H2  = r_pot['H2_Ox'] * 241800.0
                            Q_pot_CO  = r_pot['CO_Ox'] * 283000.0
                            Q_pot_CH4 = r_pot['CH4_Ox'] * 802000.0
                            Q_total_pot = Q_pot_H2 + Q_pot_CO + Q_pot_CH4
                            
                            logger.info(f"    POTENTIAL RATE (Inlet+Volatiles @ {t_start}K):")
                            logger.info(f"      R_CH4_Ox: {r_pot['CH4_Ox']:.2e} mol/s")
                            logger.info(f"      Potential Heat Release: {Q_total_pot/1e6:.2f} MW (Target: >42 MW)")
                        except Exception as e:
                            logger.warning(f"Failed to run ignition diagnostic: {e}")
 


                    
                    logger.info(f"  > Attempting Solver with Initial T = {x0[10]:.1f} K (Method: {solver_method})")
                    
                    if solver_method == 'newton':
                        # Manual Newton-Raphson
                        ns = NewtonSolver(tol=1e-8, max_iter=100, damper=0.8) # Slightly damped
                        sol = ns.solve(func, x0, bounds=(lower, upper))
                    else:
                        # Default Scipy TRF (Newton-like)
                        sol = least_squares(func, x0, bounds=(lower, upper), method='trf', 
                                            xtol=1e-12, ftol=1e-8, max_nfev=500)
                    
                    # DEBUG Ignition
                    logger.info(f"  [Ignition Guess] T: {t_start}K -> Res: {sol.x[10]:.1f}K, Cost: {sol.cost:.2e}, Success: {sol.success}")
                    
                    if sol.success:
                        # Log detailed residuals if verified success
                        final_res = cell.residuals(sol.x)
                        logger.info(f"    Final Residuals (Max): {np.max(np.abs(final_res)):.4e}")

                        is_ignited = sol.x[10] > 900.0
                        
                        update_best = False
                        if best_sol is None:
                            update_best = True
                        else:
                            best_ignited = best_sol.x[10] > 900.0
                            if is_ignited and not best_ignited:
                                update_best = True # Always prefer ignited

                            elif is_ignited == best_ignited:
                                if sol.cost < best_cost:
                                    update_best = True
                        
                        if update_best:
                            best_sol = sol
                            best_cost = sol.cost
                            
                        # Early exit if we have a great ignited solution
                        if is_ignited and sol.cost < 1e-6:
                            break
                                
                sol = best_sol if best_sol is not None else sol
            else:
                x0 = current_inlet.to_array()
                # Downstream cells need robustness to track the profile
                sol = least_squares(func, x0, bounds=(lower, upper), method='trf', 
                                    xtol=1e-10, ftol=1e-10, max_nfev=1000)
            
            if not sol.success or sol.cost > 1e-4:
                logger.warning(f"Cell {i} convergence poor. Cost: {sol.cost:.2e}, Success: {sol.success}, T: {sol.x[10]:.1f}K")
                # [DIAGNOSTICS]
                cell.diagnose_failure(sol.x)
            
            sol_state = StateVector.from_array(sol.x, P=current_inlet.P, z=cell.z)
            self.results.append(sol_state)
            current_inlet = sol_state
            
        logger.info("Solver finished successfully.")
        return self._package_results()

    def _package_results(self):
        # Return numpy array matching old format for compatibility
        # [F0..F7, Ws, Xc, T]
        rows = [s.to_array() for s in self.results]
        return np.array(rows), self.z_positions
