import numpy as np
import logging
from model.state import StateVector
from model.material import MaterialService, SPECIES_NAMES
from model.kinetics_service import KineticsService

# Set up logging
logger = logging.getLogger(__name__)

class Cell:
    """
    Represents a single Control Volume (CV) in the 1D Gasifier.
    Solves for Outlet State given Inlet State.
    """
    def __init__(self, cell_index, z, dz, V, A, inlet_state, kinetics, pyrolysis, coal_props, op_conds):
        self.idx = cell_index
        self.z = z
        self.dz = dz
        self.V = V
        self.A = A
        self.inlet = inlet_state
        self.kinetics = kinetics
        self.pyrolysis = pyrolysis
        self.coal_props = coal_props
        self.op_conds = op_conds
        
        # Determine L_dev (Devolatilization Length) - only used if distributed pyrolysis
        # Default: L/2? Or user input?
        L_total = self.op_conds.get('L_reactor', 10.0)
        self.L_dev = self.op_conds.get('L_dev', 0.5) # Default 0.5m
        
    def get_snapshot(self, state: StateVector):
        """
        Returns detailed physics snapshot for the given state.
        Useful for diagnostics and reporting.
        """
        # 1. Setup
        d_p = self.op_conds.get('particle_diameter', 100e-6)
        rho_p = 1500.0
        
        # 2. Pyrolysis info - Removed (Handled at Inlet)
        rate_src = np.zeros(8)
        W_pyro_loss = 0.0

        # 3. Kinetics
        # Variable Dp (Shrinking Core with Ash Skeleton)
        d_p0 = self.op_conds.get('particle_diameter', 100e-6)
        
        # Original Coal Properties
        Ash_d = self.coal_props.get('Ashd', 6.0) / 100.0
        Cd_total = self.coal_props.get('Cd', 60.0) / 100.0
        f_ash_coal = Ash_d / (Cd_total + Ash_d + 1e-6) # Mass ratio in Coal
        
        # Carbon Conversion X_total (Coal basis)
        W_dry_total = getattr(self, 'coal_flow_dry', 1.0)
        C_fed_total = W_dry_total * Cd_total
        C_local = state.solid_mass * state.carbon_fraction
        X_total = 1.0 - (C_local / (C_fed_total + 1e-9))
        X_total = np.clip(X_total, 0.0, 1.0)
        
        # d_p = d_p0 * (f_ash_coal + (1-f_ash_coal)*(1-X_total))^(1/3)
        d_p = d_p0 * (f_ash_coal + (1.0 - f_ash_coal) * (1.0 - X_total))**(1/3.0)
        
        # Calculate Surface Area S_total based on Residence Time
        M_LIST = [31.998, 16.04, 28.01, 44.01, 34.08, 2.016, 28.013, 18.015]
        W_gas = sum(state.gas_moles[i] * M_LIST[i] for i in range(8)) * 1e-3 # kg/s
        rho_gas = (state.P * (sum(state.gas_moles[i] * M_LIST[i] for i in range(8))/max(state.total_gas_moles,1e-9)) * 1e-3) / (8.314 * state.T)
        v_g = W_gas / (rho_gas * self.A + 1e-9)
        
        tau = self.dz / max(v_g, 0.1)
        m_hold = state.solid_mass * tau # kg_holdup
        S_total = (6 * m_hold) / (1500.0 * d_p + 1e-9)
        
        # Calculate Re and Sc for mass transfer accuracy
        from model.physics import calculate_gas_viscosity, calculate_diffusion_coefficient
        mu_g = calculate_gas_viscosity(state.T)
        v_slip = 0.1 # Minimum turbulence slip for Sh calculation
        Re_p = (rho_gas * v_slip * d_p) / (mu_g + 1e-9)
        D_ref = calculate_diffusion_coefficient(state.T, state.P, 'O2')
        Sc_p = mu_g / (max(rho_gas, 1e-3) * D_ref + 1e-12)
        
        r_het = self.kinetics.calc_heterogeneous_rates(
            state, d_p, S_total, X_total=X_total, Re=Re_p, Sc=Sc_p
        )
        phi = r_het.pop('phi', 1.0)
        
        r_homo = self.kinetics.calc_homogeneous_rates(state, self.V)
        
        # 4. Detailed Production/Consumption
        # Just return the raw rates, user can derive balance.
        
        return {
            'z': self.z,
            'T': state.T,
            'P': state.P,
            'v_g': v_g,
            'Re_p': Re_p,
            'Sh': 2.0 + 0.6 * (Re_p**0.5) * (Sc_p**(1/3.0)),
            'rho_gas': rho_gas,
            'GasMoles': state.gas_moles.tolist(),
            'SolidMass': state.solid_mass,
            'CarbonFrac': state.carbon_fraction,
            'Rates_Het': r_het, # Dict
            'Rates_Homo': r_homo, # Dict
            'Src_Pyrolysis': rate_src.tolist(),
            'Phi': phi,
            'S_total': S_total,
            'd_p': d_p
        }
    def residuals(self, x_flat):
        """
        Calculates function residuals F(x) = 0.
        vars_arr: [O2, CH4, CO, CO2, H2S, H2, N2, H2O, W_s, X_c, T]
        """
        # 1. Unpack to StateVector
        P_curr = self.inlet.P 
        current = StateVector.from_array(x_flat, P=P_curr, z=self.z)
        
        # 2. Source Terms (Pyrolysis & Evaporation)
        rate_src = np.zeros(8)
        W_pyro_loss = 0.0
        W_pyro_loss_C = 0.0
        
        # [FIX] Mass Balance Consistency: 
        # Moisture/Slurry water is already added to Inlet Gas in GasifierSystem._initialize_inlet.
        # Here we set rate_evap_H2O to 0 to avoid double counting.
        rate_evap_H2O = 0.0 

        # 3. Reaction Rates
        # 3.1 Heterogeneous (Variable Dp)
        d_p0 = self.op_conds.get('particle_diameter', 100e-6)
        
        # Coal Basis Definition
        Ash_d = self.coal_props.get('Ashd', 6.0) / 100.0
        Cd_total = self.coal_props.get('Cd', 60.0) / 100.0
        f_ash_coal = Ash_d / (Cd_total + Ash_d + 1e-6)
        
        W_dry_total = getattr(self, 'coal_flow_dry', 1.0)
        C_fed_total = W_dry_total * Cd_total
        C_local = current.solid_mass * current.carbon_fraction
        
        # Conversion relative to Original Coal
        X_total = 1.0 - (C_local / (C_fed_total + 1e-9))
        X_total = np.clip(X_total, 0.0, 1.0)
        
        # d_p shrinks with conversion from coal basis
        d_p = d_p0 * (f_ash_coal + (1.0 - f_ash_coal) * (1.0 - X_total))**(1/3.0)
        d_p = max(d_p, 1e-6)
        
        # Updated Solid Density 
        rho_p = 1500.0 
        
        # Calculate Surface Area S_total based on Residence Time
        M_LIST = [31.998, 16.04, 28.01, 44.01, 34.08, 2.016, 28.013, 18.015]
        W_gas = sum(current.gas_moles[i] * M_LIST[i] for i in range(8)) * 1e-3 # kg/s
        rho_gas = (P_curr * (sum(current.gas_moles[i] * M_LIST[i] for i in range(8))/max(current.total_gas_moles,1e-9)) * 1e-3) / (8.314 * current.T)
        v_g = W_gas / (rho_gas * self.A + 1e-9)
        
        tau = self.dz / max(v_g, 0.1)
        m_hold = current.solid_mass * tau # kg_holdup
        S_total = (6 * m_hold) / (rho_p * d_p + 1e-9)
        
        # Calculate Re and Sc for mass transfer accuracy
        from model.physics import calculate_gas_viscosity, calculate_diffusion_coefficient
        mu_g = calculate_gas_viscosity(current.T)
        v_slip = 0.1 # Minimum turbulence slip
        Re_p = (rho_gas * v_slip * d_p) / (mu_g + 1e-9)
        D_ref = calculate_diffusion_coefficient(current.T, P_curr, 'O2')
        Sc_p = mu_g / (max(rho_gas, 1e-3) * D_ref + 1e-12)
        
        r_het = self.kinetics.calc_heterogeneous_rates(
            current, d_p, S_total, X_total=X_total, Re=Re_p, Sc=Sc_p
        )
        phi = r_het.pop('phi', 1.0) 
        
        # 3.2 Homogeneous (Table 2-5)
        # CO_Ox, H2_Ox, WGS, RWGS, CH4_Ox, MSR
        r_homo = self.kinetics.calc_homogeneous_rates(current, self.V)
        
        # 3.3 Ash-Catalytic WGS (Reaction 11)
        # R11 = 0.5 * X_ash * k11 * (C_CO * C_H2O - C_CO2 * C_H2 / Keq)
        # Approximate as a small fast term if not detailed.
        # Literature k11 = 1.1e10? No, let's look for a standard.
        # Simplified: Use WGS from homo but scaled by ash presence? 
        # For now, we use the 6 homo reactions which already include WGS/RWGS.
        r11 = 0.0 # Catalyst term for future tuning
        
        # 3.4 Rate Damping / Clipping (Safety to prevent over-consumption)
        # Calculate available quantities (mol/s)
        avail_O2 = max(self.inlet.gas_moles[0] + rate_src[0], 0.0)
        avail_H2O = max(self.inlet.gas_moles[7] + rate_src[7] + rate_evap_H2O, 0.0)
        avail_CO2 = max(self.inlet.gas_moles[3] + rate_src[3], 0.0)
        avail_CO = max(self.inlet.gas_moles[2] + rate_src[2], 0.0)
        avail_H2 = max(self.inlet.gas_moles[5] + rate_src[5], 0.0)
        avail_CH4 = max(self.inlet.gas_moles[1] + rate_src[1], 0.0)
        avail_C = max(self.inlet.solid_mass * self.inlet.carbon_fraction / 0.012011, 0.0)
        
        # Clip Het Rates
        rCOmb = min(r_het['C+O2'], avail_O2 * phi * 0.99)
        rH2Og = min(r_het['C+H2O'], avail_H2O * 0.99)
        rCO2g = min(r_het['C+CO2'], avail_CO2 * 0.99)
        rH2g  = min(r_het['C+H2'], avail_H2 * 0.5 * 0.99)
        
        # Clip Homo Rates (simplified budget check)
        r1, r2, r3, r4, r5, r6 = r_homo['CO_Ox'], r_homo['H2_Ox'], r_homo['WGS'], r_homo['RWGS'], r_homo['CH4_Ox'], r_homo['MSR']
        
        # 4. Assemble Component Balances (mol/s)
        
        # 4.1 O2
        # Stoichiometry check (Table 2-5 and Reaction 7)
        # rCOmb uses (1/phi) O2
        cons_O2 = (rCOmb/phi) + 0.5*r1 + 0.5*r2 + 2.0*r5
        res_O2 = current.gas_moles[0] - (self.inlet.gas_moles[0] + rate_src[0] - cons_O2)
        
        # 4.2 CH4
        prod_CH4 = rH2g + rate_src[1]
        cons_CH4 = r5 + r6
        res_CH4 = current.gas_moles[1] - (self.inlet.gas_moles[1] + prod_CH4 - cons_CH4)
        
        # 4.3 CO
        # C + (1/phi)O2 -> (2 - 2/phi)CO + (2/phi - 1)CO2
        prod_CO = (2.0 - 2.0/phi)*rCOmb + rH2Og + 2.0*rCO2g + r4 + r6 + rate_src[2]
        cons_CO = r1 + r3
        res_CO = current.gas_moles[2] - (self.inlet.gas_moles[2] + prod_CO - cons_CO)
        
        # 4.4 CO2
        prod_CO2 = (2.0/phi - 1.0)*rCOmb + r1 + r3 + r5 + rate_src[3]
        cons_CO2 = rCO2g + r4
        res_CO2 = current.gas_moles[3] - (self.inlet.gas_moles[3] + prod_CO2 - cons_CO2)
        
        # 4.5 H2S
        res_H2S = current.gas_moles[4] - (self.inlet.gas_moles[4] + rate_src[4])
        
        # 4.6 H2
        prod_H2 = rH2Og + r3 + 3.0*r6 + rate_src[5]
        cons_H2 = 2.0*rH2g + r2 + r4
        res_H2 = current.gas_moles[5] - (self.inlet.gas_moles[5] + prod_H2 - cons_H2)
        
        # 4.7 N2
        res_N2 = current.gas_moles[6] - (self.inlet.gas_moles[6] + rate_src[6])
        
        # 4.8 H2O Balance
        prod_H2O = r2 + r4 + 2.0*r5 + rate_src[7] + rate_evap_H2O
        cons_H2O = rH2Og + r3 + r6
        res_H2O = current.gas_moles[7] - (self.inlet.gas_moles[7] + prod_H2O - cons_H2O)
        
        # 5. Solid Mass Balance
        C_gasified = rCOmb + rH2Og + rCO2g + rH2g
        W_surf_loss = C_gasified * 0.012011 
        res_Ws = current.solid_mass - (self.inlet.solid_mass - W_surf_loss - W_pyro_loss)

        # 6. Char Fraction Balance
        C_fixed_in = self.inlet.solid_mass * self.inlet.carbon_fraction
        C_fixed_out = current.solid_mass * current.carbon_fraction
        res_Xc = C_fixed_out - (C_fixed_in - W_surf_loss - W_pyro_loss_C)
        
        # 7. Energy Balance
        H_in = MaterialService.get_total_enthalpy(self.inlet, self.coal_props)
        H_out = MaterialService.get_total_enthalpy(current, self.coal_props)
        
        L_total = self.op_conds.get('L_reactor', 8.0)
        loss_pct = self.op_conds.get('HeatLossPercent', 0.1) 
        Q_total_MW = self.op_conds.get('Q_total_MW', 0.0)
        
        if Q_total_MW > 0:
            Q_ref_MW = Q_total_MW
        else:
            Q_ref_MW = (self.op_conds['coal_flow'] * self.coal_props.get('HHV_d', 25.0) * 1000.0)
            
        Q_loss = (loss_pct/100.0) * (Q_ref_MW * 1e6) * (self.dz / L_total)
        Q_source = self.op_conds.get('Q_source_term', 0.0) 
        
        # Energy Residual
        res_E = H_out - (H_in - Q_loss + Q_source)
        
        # 7. Atomic Conservation (Verification Residuals)
        # Total Atoms In/Out (mol_atoms/s)
        # Species: [O2, CH4, CO, CO2, H2S, H2, N2, H2O]
        def calc_atoms(gas, solid, x_c):
            # C: CH4(1), CO(1), CO2(1), Solid(C/0.012011)
            C = gas[1] + gas[2] + gas[3] + (solid * x_c / 0.012011)
            # H: CH4(4), H2S(2), H2(2), H2O(2), Solid(H approx nil in char)
            H = 4*gas[1] + 2*gas[4] + 2*gas[5] + 2*gas[7] 
            # O: O2(2), CO(1), CO2(2), H2O(1)
            O = 2*gas[0] + gas[2] + 2*gas[3] + gas[7]
            # N: N2(2)
            N = 2*gas[6]
            return np.array([C, H, O, N])
            
        atoms_in = calc_atoms(self.inlet.gas_moles, self.inlet.solid_mass, self.inlet.carbon_fraction)
        atoms_out = calc_atoms(current.gas_moles, current.solid_mass, current.carbon_fraction)
        res_atoms = atoms_out - atoms_in
        
        # Scaling logic refined
        
        # Global reference for atomic residuals
        mols_ref = np.sum(calc_atoms(self.inlet.gas_moles + [0,0,0,0,0,0,0,rate_evap_H2O], self.inlet.solid_mass, self.inlet.carbon_fraction)) / 4.0
        sc_atom = max(mols_ref, 1.0)

        # Normalize by species flow to ensure relative accuracy for minor species
        def get_scale(idx):
            ref = self.inlet.gas_moles[idx]
            if idx < 8:
                ref += rate_src[idx]
            # Use max(ref, 1.0) to avoid division by zero and over-weighting trace species
            return max(ref, 1.0)

        residuals = np.array([
            res_O2 / get_scale(0),
            res_CH4 / get_scale(1),
            res_CO / get_scale(2),
            res_CO2 / get_scale(3),
            res_H2S / get_scale(4),
            res_H2 / get_scale(5),
            res_N2 / get_scale(6),
            res_H2O / get_scale(7),
            res_Ws / max(self.inlet.solid_mass, 0.1),
            res_Xc / max(self.char_Xc0, 0.1),
            res_E / 1.0e6, # MW basis
            # Atomic Residuals (Heavy penalty for mass violation)
            # Scaling atoms by total flow is fine for global check
            res_atoms[0] / sc_atom,
            res_atoms[1] / sc_atom,
            res_atoms[2] / sc_atom,
            res_atoms[3] / sc_atom
        ])
        
        return residuals
