import numpy as np
from scipy.optimize import least_squares
from .state import StateVector
from .cell import Cell
from .material import MaterialService
from .kinetics_service import KineticsService
from .pyrolysis_service import PyrolysisService
import logging

class GasifierSystem:
    def __init__(self, geometry, coal_props, op_conds, scaling=None):
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
            print(f"[System] Consistent Hf_coal: {self.coal_props['Hf_coal']/1e6:.2f} MJ/kg (Basis: LHV_d={LHV_d_kJ/1000.0:.2f} MJ/kg)")
        
        # Initialize Services
        # Default scaling can be overridden here
        scaling = scaling or {}
        self.material = MaterialService()
        self.kinetics = KineticsService(scaling.get('kinetics'))
        self.pyrolysis = PyrolysisService()
        
        self.results = []
        self.cells = []

    def _initialize_inlet(self):
        """
        Prepare Cell 0 Inlet State with 'Instant Pyrolysis' assumption.
        Coal enters as Char + Volatiles + Oxidant + Steam.
        """
        # 1. Physical Feed Components (Dry Coal, Moisture, Slurry Water, Steam)
        W_wet_coal = self.op_conds['coal_flow'] 
        Mt = self.coal_props.get('Mt', 0.0)
        W_dry = W_wet_coal * (1 - Mt/100.0)
        W_coal_moist = W_wet_coal * (Mt/100.0)
        
        # Coal Water Slurry (CWS) Handling
        # Concentration = W_dry / (W_dry + W_slurry_h2o)
        slurry_conc = self.op_conds.get('SlurryConcentration', 100.0) # Default 100% (Dry feed)
        if slurry_conc < 100.0:
            # W_slurry_tot = W_dry / (conc/100)
            W_slurry_h2o = (W_dry / (slurry_conc/100.0)) - W_dry
        else:
            W_slurry_h2o = 0.0
            
        W_steam = self.op_conds.get('steam_flow', 0.0)
        
        # Total moisture to be vaporized (Coal internal moisture + slurry water)
        W_liq_to_evap = W_coal_moist + W_slurry_h2o
        W_h2o_total = W_liq_to_evap + W_steam
        
        # Latent Heat of Moisture Evaporation (at 25C reference)
        # 2442 kJ/kg = 44 kJ/mol / 0.018 kg/mol
        Q_evap = W_liq_to_evap * 2442.0 * 1000.0 
        self.evap_heat_load = -Q_evap 
        
        # 3. Calculated LHV_ar (As-Received) for Reference
        LHV_d = self.coal_props.get('LHV_d_kJ', 25000.0)
        LHV_ar = LHV_d * (1 - Mt/100.0) - (2442.0 * Mt/100.0)
        
        # Heat Loss Reference Power (MW)
        # Standard: % of LHV_ar power
        self.op_conds['Q_total_MW'] = (W_wet_coal * LHV_ar) / 1000.0 
        
        # 2. Pyrolysis (Instant Decomposition)
        # Convert Dry Coal -> Volatile Gas + Char (Solid)
        molar_yields_per_kg, W_vol_mass_per_kg = self.pyrolysis.calc_yields(self.coal_props)
        
        # Volatile Fluxes (mol/s)
        F_volatiles = molar_yields_per_kg * W_dry
        
        # 3. Oxidant & Steam (Gas)
        MW_O2 = 31.998
        F_O2 = (self.op_conds['o2_flow'] / MW_O2) * 1000.0 # mol/s
        F_N2_ox = self.op_conds.get('n2_flow', 0.0) / 28.013 * 1000.0
        
        MW_H2O = 18.015
        F_H2O_steam = (W_h2o_total / MW_H2O) * 1000.0 # mol/s
        
        # 4. Assemble Initial Gas State (mol/s)
        # [O2, CH4, CO, CO2, H2S, H2, N2, H2O]
        # Includes Oxidant/Steam + ALL Volatiles
        gas_moles = F_volatiles.copy()
        
        gas_moles[0] += F_O2
        gas_moles[6] += F_N2_ox
        gas_moles[7] += F_H2O_steam
        
        # 5. Char State (Solid)
        # W_char = W_dry - W_volatiles
        W_char = W_dry * (1.0 - W_vol_mass_per_kg)
        
        # Carbon Fraction in Char (Fixed Carbon Basis)
        # FCd is the percentage of fixed carbon in dry coal.
        # carbon_fraction in char = FCd / (1 - Volatiles_mass)
        FCd = self.coal_props.get('FCd', 50.0) / 100.0
        # Check that FCd/W_char is physical. 
        # Actually, let's keep it simple: Total Carbon Mass = Cd * W_dry.
        # Volatiles also contain carbon (CH4, CO, CO2).
        # We need the REMAINING carbon in the solid.
        C_in_volatiles_kg = (F_volatiles[1] + F_volatiles[2] + F_volatiles[3]) * 0.012011 
        Cd_total = self.coal_props.get('Cd', 60.0) / 100.0
        C_total_mass = Cd_total * W_dry
        C_remaining_mass = max(C_total_mass - C_in_volatiles_kg, 1e-6)
        
        char_carbon_frac = min(C_remaining_mass / (W_char + 1e-9), 1.0)
        self.char_Xc0 = char_carbon_frac
        
        inlet = StateVector(
            gas_moles=gas_moles,
            solid_mass=W_char,
            carbon_fraction=char_carbon_frac,
            T=self.op_conds['T_in'],
            P=self.op_conds['P'],
            z=0.0
        )
        # Store for use in solve loop and cells
        self.W_dry = W_dry
        self.Cd_total = Cd_total
        
        return inlet

    def solve(self, N_cells=100):
        """Sequential Solver"""
        L = self.geometry['L']
        D = self.geometry['D']
        A = np.pi * (D/2)**2
        V_total = A * L
        
        # [GRID STRATEGY] Variable Mesh
        # 0. Ignition/Evaporation (0.4m) - Large residence time (Direct Inlet)
        # 1. Flame Front (0.1m) - Fine resolution
        # 2. Downstream - Expanding
        
        dz_list = np.zeros(N_cells)
        fixed_len = 0.0
        
        # Explicit Sizes
        if N_cells > 2:
            dz_list[0] = 0.40
            fixed_len += 0.40
            
            # Fine Zone (Cells 1 to 20, or as many as fit)
            fine_end = min(20, N_cells)
            for k in range(1, fine_end):
                dz_list[k] = 0.10
                fixed_len += 0.10
                
            # Remaining length distributed
            rem_cells = N_cells - fine_end
            if rem_cells > 0:
                rem_len = max(L - fixed_len, 0.1) # Safety
                dz_avg = rem_len / rem_cells
                for k in range(fine_end, N_cells):
                    dz_list[k] = dz_avg
        else:
            # Fallback for small N
            dz_list[:] = L / N_cells

        # Compute Z positions (Centers)
        self.z_positions = np.zeros(N_cells)
        current_z = 0.0
        for k in range(N_cells):
            self.z_positions[k] = current_z + dz_list[k]/2.0
            current_z += dz_list[k]
        
        # Init with Instant Pyrolysis
        current_inlet = self._initialize_inlet()
        
        # [DEBUG] Verify Volatile Flux
        _f = current_inlet.gas_moles
        print(f"[Inlet Check] Volatile Flux (mol/s) -> CH4: {_f[1]:.4f}, CO: {_f[2]:.4f}, H2: {_f[5]:.4f}, O2: {_f[0]:.4f}")
        
        self.results = []
        self.cells = []
        
        cell_ops = self.op_conds.copy()
        cell_ops['L_reactor'] = L
        cell_ops['pyrolysis_done'] = True # Pyrolysis already accounted for in inlet
        
        print(f"[System] Starting Solver (Instant Pyrolysis Mode) with {N_cells} cells.")
        
        for i in range(N_cells):
            z_curr = self.z_positions[i]
            dz_curr = dz_list[i]
            V_curr = A * dz_curr
            
            # 1. Distributed Source Terms (Evaporation and Heat Loss)
            # Evaporation is now instantaneous in Cell 0.
            cell_ops['L_reactor'] = L
            cell_ops['HeatLossPercent'] = self.op_conds.get('HeatLossPercent', 3.0)
            
            if i == 0:
                # Apply entire evaporation heat load to cell 0 (Ignition Zone)
                cell_ops['Q_source_term'] = self.evap_heat_load
            else:
                cell_ops['Q_source_term'] = 0.0
            
            cell_ops['pyrolysis_done'] = True # Pyrolysis already accounted for in inlet
            cell = Cell(i, self.z_positions[i], dz_curr, V_curr, A, 
                        current_inlet, self.kinetics, self.pyrolysis, self.coal_props, cell_ops)
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
                guesses_T = [2000.0, 2500.0, 3000.0, 1500.0, 1000.0]
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
 


                    
                    sol = least_squares(func, x0, bounds=(lower, upper), method='trf', 
                                        xtol=1e-12, ftol=1e-8, max_nfev=500)
                    
                    # DEBUG Ignition
                    print(f"  [Ignition Debug] Guess: {t_start}K -> Res: {sol.x[10]:.1f}K, Cost: {sol.cost:.2e}, Success: {sol.success}")
                    
                    if sol.success and sol.cost < best_cost:
                        best_sol = sol
                        best_cost = sol.cost
                        # Accept if cost is low and T maintained high (Ignition success)
                        if sol.cost < 1e-6 and sol.x[10] > 1000.0:
                            break
                                
                sol = best_sol if best_sol is not None else sol
            else:
                x0 = current_inlet.to_array()
                # Downstream cells need robustness to track the profile
                sol = least_squares(func, x0, bounds=(lower, upper), method='trf', 
                                    xtol=1e-10, ftol=1e-10, max_nfev=1000)
            
            if not sol.success or sol.cost > 1e-4:
                print(f"  [Warn] Cell {i} convergence poor. Cost: {sol.cost:.2e}, Success: {sol.success}, T: {sol.x[10]:.1f}K")
            
            sol_state = StateVector.from_array(sol.x, P=current_inlet.P, z=cell.z)
            self.results.append(sol_state)
            current_inlet = sol_state
            
        print("[System] Solved.")
        return self._package_results()

    def _package_results(self):
        # Return numpy array matching old format for compatibility
        # [F0..F7, Ws, Xc, T]
        rows = [s.to_array() for s in self.results]
        return np.array(rows), self.z_positions
