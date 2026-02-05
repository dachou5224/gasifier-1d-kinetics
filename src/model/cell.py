import numpy as np
import logging
from model.state import StateVector
from model.material import MaterialService, SPECIES_NAMES
from model.kinetics_service import KineticsService
from model.constants import PhysicalConstants

# Set up logging
logger = logging.getLogger(__name__)

class Cell:
    """
    Represents a single Control Volume (CV) in the 1D Gasifier.
    Solves for Outlet State given Inlet State and Source Terms.
    """
    MOLAR_MASSES = np.array([31.998, 16.04, 28.01, 44.01, 34.08, 2.016, 28.013, 18.015])

    def __init__(self, cell_index, z, dz, V, A, inlet_state, kinetics, pyrolysis, coal_props, op_conds, sources=None):
        self.idx = cell_index
        self.z = z
        self.dz = dz
        self.V = V
        self.A = A
        self.inlet = inlet_state
        self.kinetics = kinetics
        self.coal_props = coal_props
        self.op_conds = op_conds
        self.sources = sources if sources is not None else []
        
        # Determine Reference Scales for normalization
        self.ref_flow = max(self.inlet.total_gas_moles, 1.0)
        self.ref_solid = max(self.inlet.solid_mass, 1e-3)
        self.coal_flow_dry = getattr(self, 'coal_flow_dry', 1.0)
        self.char_Xc0 = self.inlet.carbon_fraction

    def _calc_physics_props(self, current, C_fed_total):
        """Calculates physical properties and transport coefficients."""
        # 1. Conversion relative to Original Coal
        C_local = current.solid_mass * current.carbon_fraction
        if C_fed_total < 1e-9: C_fed_total = 1e-9
        X_total = np.clip(1.0 - (C_local / C_fed_total), 0.0, 1.0)
        
        # 2. Particle Diameter (Shrinking Core vs Ash skeleton)
        d_p0 = self.op_conds.get('particle_diameter', 100e-6)
        Ash_d = self.coal_props.get('Ashd', 6.0) / 100.0
        Cd_total = self.coal_props.get('Cd', 60.0) / 100.0
        f_ash_coal = Ash_d / (Cd_total + Ash_d + 1e-9)
        d_p = d_p0 * (f_ash_coal + (1.0 - f_ash_coal) * (1.0 - X_total))**(1/3.0)
        d_p = max(d_p, 1e-6)
        
        # 3. Gas Dynamics & Transport (Based on User Formula: u_g = F_total * R * T / (P * A * epsilon))
        # Where F_total is local molar flow (mol/s)
        F_total = max(current.total_gas_moles, 1e-9)
        T_local = current.T
        P_local = current.P
        epsilon = self.op_conds.get('epsilon', 1.0) # Entrained flow voidage approx 1.0
        
        # u_g [m/s] = (mol/s * J/mol.K * K) / (Pa * m2) = m/s
        v_g = (F_total * 8.314 * T_local) / (P_local * self.A * epsilon + 1e-9)
        
        # Residence Time (Local approach: tau = dz / u_g)
        tau = self.dz / max(v_g, PhysicalConstants.MIN_SLIP_VELOCITY)
        m_hold = current.solid_mass * tau
        S_total = (6 * m_hold) / (PhysicalConstants.PARTICLE_DENSITY * d_p + 1e-9)
        
        # 4. Transport Properties
        from model.physics import calculate_gas_viscosity, calculate_diffusion_coefficient
        rho_gas = (P_local * (np.sum(current.gas_moles * self.MOLAR_MASSES)/F_total) * 1e-3) / (8.314 * T_local)
        mu_g = calculate_gas_viscosity(T_local)
        v_slip = PhysicalConstants.MIN_SLIP_VELOCITY
        Re_p = (rho_gas * v_slip * d_p) / (mu_g + 1e-9)
        D_ref = calculate_diffusion_coefficient(T_local, P_local, 'O2')
        Sc_p = mu_g / (max(rho_gas, 1e-3) * D_ref + 1e-12)
        
        return {
            'X_total': X_total, 'd_p': d_p, 'S_total': S_total, 
            'Re_p': Re_p, 'Sc_p': Sc_p, 'v_g': v_g
        }

    def _calc_rates(self, current, phys, gas_src):
        """Calculates Heterogeneous and Homogeneous reaction rates."""
        # 1. Base Rates
        r_het = self.kinetics.calc_heterogeneous_rates(
            current, phys['d_p'], phys['S_total'], 
            X_total=phys['X_total'], Re=phys['Re_p'], Sc=phys['Sc_p']
        )
        phi = r_het.pop('phi', 1.0)
        r_homo = self.kinetics.calc_homogeneous_rates(current, self.V)
        
        # 2. Availability Clipping (Prevents non-physical consumption)
        # Total molecules in cell = Inlet + Source
        avail = [max(self.inlet.gas_moles[i] + gas_src[i], 0.0) for i in range(8)]
        
        # Het Clipping
        r_het['C+O2'] = min(r_het['C+O2'], avail[0] * phi * 0.99)
        r_het['C+H2O'] = min(r_het['C+H2O'], avail[7] * 0.99)
        r_het['C+CO2'] = min(r_het['C+CO2'], avail[3] * 0.99)
        r_het['C+H2'] = min(r_het['C+H2'], avail[5] * 0.5 * 0.99)

        return r_het, r_homo, phi

    def _calc_balances(self, current, r_het, r_homo, phi, gas_src, solid_src):
        """Computes component mass residuals."""
        r1, r2, r3 = r_homo['CO_Ox'], r_homo['H2_Ox'], r_homo['WGS']
        r4, r5, r6 = r_homo['RWGS'], r_homo['CH4_Ox'], r_homo['MSR']
        rCOmb, rH2Og, rCO2g, rH2g = r_het['C+O2'], r_het['C+H2O'], r_het['C+CO2'], r_het['C+H2']
        
        # Gas Balances
        res_O2 = current.gas_moles[0] - (self.inlet.gas_moles[0] + gas_src[0] - (rCOmb/phi + 0.5*r1 + 0.5*r2 + 2.0*r5))
        res_CH4 = current.gas_moles[1] - (self.inlet.gas_moles[1] + gas_src[1] + rH2g - (r5 + r6))
        res_CO = current.gas_moles[2] - (self.inlet.gas_moles[2] + gas_src[2] + (2.0-2.0/phi)*rCOmb + rH2Og + 2*rCO2g + r4 + r6 - (r1 + r3))
        res_CO2 = current.gas_moles[3] - (self.inlet.gas_moles[3] + gas_src[3] + (2.0/phi-1.0)*rCOmb + r1 + r3 + r5 - (rCO2g + r4))
        res_H2S = current.gas_moles[4] - (self.inlet.gas_moles[4] + gas_src[4])
        res_H2 = current.gas_moles[5] - (self.inlet.gas_moles[5] + gas_src[5] + rH2Og + r3 + 3*r6 - (2*rH2g + r2 + r4))
        res_N2 = current.gas_moles[6] - (self.inlet.gas_moles[6] + gas_src[6])
        res_H2O = current.gas_moles[7] - (self.inlet.gas_moles[7] + gas_src[7] + r2 + r4 + 2*r5 - (rH2Og + r3 + r6))
        
        # Solid Mass Balance (CRITICAL: include solid_src)
        W_surf_loss = (rCOmb + rH2Og + rCO2g + rH2g) * 0.012011
        res_Ws = current.solid_mass - (self.inlet.solid_mass + solid_src - W_surf_loss)
        
        # Carbon Balance for Xc
        C_loss_pyro_mols = gas_src[1] + gas_src[2] + gas_src[3] # CH4, CO, CO2 from pyro
        C_loss_pyro_kg = C_loss_pyro_mols * 0.012011
        C_in_solid = self.inlet.solid_mass * self.inlet.carbon_fraction
        C_out_solid = current.solid_mass * current.carbon_fraction
        res_Xc = C_out_solid - (C_in_solid - C_loss_pyro_kg - W_surf_loss)
        
        gas_res = [res_O2, res_CH4, res_CO, res_CO2, res_H2S, res_H2, res_N2, res_H2O]
        return gas_res, res_Ws, res_Xc

    def _calc_energy_balance(self, current, energy_src):
        """Computes enthalpy residual."""
        H_in = MaterialService.get_total_enthalpy(self.inlet, self.coal_props)
        H_out = MaterialService.get_total_enthalpy(current, self.coal_props)
        
        # Heat Loss
        L_total = self.op_conds.get('L_reactor', 6.0)
        loss_pct = self.op_conds.get('HeatLossPercent', 1.0)
        Q_total_MW = (self.op_conds['coal_flow'] * self.coal_props.get('HHV_d', 30.0))
        Q_loss = (loss_pct/100.0) * (Q_total_MW * 1e6) * (self.dz / L_total)
        
        # 0 = H_in + Source - H_out - Loss -> Residual = H_out - (H_in + Source - Loss)
        return H_out - (H_in + energy_src - Q_loss)

    def residuals(self, x_flat):
        """Standard Solver Interface."""
        current = StateVector.from_array(x_flat, P=self.inlet.P, z=self.z)
        
        # Aggregate Sources
        g_src, s_src, e_src = np.zeros(8), 0.0, 0.0
        for s in self.sources:
            g, s_m, e = s.get_sources(self.idx, self.z, self.dz)
            g_src += g
            s_src += s_m
            e_src += e
            
        phys = self._calc_physics_props(current, self.coal_flow_dry * (self.coal_props.get('Cd', 60.0)/100.0))
        r_het, r_homo, phi = self._calc_rates(current, phys, g_src)
        res_gas, res_Ws, res_Xc = self._calc_balances(current, r_het, r_homo, phi, g_src, s_src)
        res_E = self._calc_energy_balance(current, e_src)
        
        # Scaling
        res_gas_sc = [r / self.ref_flow for r in res_gas]
        res_Ws_sc = res_Ws / self.ref_solid
        res_Xc_sc = res_Xc / max(self.char_Xc0, 0.1)
        res_E_sc = res_E / 1.0e6
        
        return np.concatenate([res_gas_sc, [res_Ws_sc, res_Xc_sc, res_E_sc]])

    def diagnose_failure(self, x_flat):
        """Detailed print logging for failures."""
        current = StateVector.from_array(x_flat, P=self.inlet.P, z=self.z)
        g_src, s_src, e_src = np.zeros(8), 0.0, 0.0
        for s in self.sources:
            g, sm, e = s.get_sources(self.idx, self.z, self.dz)
            g_src += g; s_src += sm; e_src += e
            
        phys = self._calc_physics_props(current, self.coal_flow_dry * (self.coal_props.get('Cd', 60.0)/100.0))
        r_het, r_homo, phi = self._calc_rates(current, phys, g_src)
        
        # Recalculate H for log
        H_in = MaterialService.get_total_enthalpy(self.inlet, self.coal_props)
        H_out = MaterialService.get_total_enthalpy(current, self.coal_props)
        
        v_g = phys['v_g']
        tau = self.dz / max(v_g, 1e-3)
        
        logger.error(f"--- Cell {self.idx} Diagnostic (T={current.T:.1f} K) ---")
        logger.error(f"  Fluid Dynamics: u_g={v_g:.2f} m/s, tau={tau:.4f} s, dz={self.dz:.3f} m")
        logger.error(f"  Enthalpy Balance: H_in={H_in/1e6:.2f}, H_out={H_out/1e6:.2f}, Src={e_src/1e6:.2f} MW")
        logger.error(f"  Residual: {(H_out - (H_in + e_src))/1e6:.2f} MW")
        logger.error(f"  Reaction Heat Potential (CH4_Ox): {r_homo['CH4_Ox']*802340.0/1e6:.2f} MW")
        for i, sp in enumerate(SPECIES_NAMES):
            if abs(current.gas_moles[i]) > 1e-3 or abs(g_src[i]) > 1e-3:
                logger.error(f"    {sp:4}: {self.inlet.gas_moles[i]:8.1f} -> {current.gas_moles[i]:8.1f} (Src: {g_src[i]:8.1f})")
        logger.error(f"  Solid: {self.inlet.solid_mass:8.3f} -> {current.solid_mass:8.3f} (Src: {s_src:8.3f})")
        logger.error(f"------------------------------------------------")
