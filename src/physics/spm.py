import torch
import torch.nn as nn
import numpy as np

class AdvancedSingleParticleModel(nn.Module):
    def __init__(self, D_s, R_s, F, T, R_gas, a_s, M_sei=None, rho_sei=None, kappa_sei=None):
        """
        Advanced Single Particle Model (SPM) Physics Constraints with Degradation and Thermal coupling.
        Based on the comprehensive review of lithium-ion mathematical models.
        
        Parameters:
        D_s: Solid-phase diffusion coefficient [m^2/s]
        R_s: Particle radius [m]
        F: Faraday's constant (96485.33 C/mol)
        T: Temperature [K]
        R_gas: Universal gas constant (8.314 J/(mol K))
        a_s: Specific interfacial area of electrode [m^2/m^3]
        M_sei: Molar mass of SEI [kg/mol]
        rho_sei: Density of SEI [kg/m^3]
        kappa_sei: Ionic conductivity of SEI [S/m]
        """
        super().__init__()
        self.D_s = D_s
        self.R_s = R_s
        self.F = F
        self.T = T
        self.R_gas = R_gas
        self.a_s = a_s
        
        # SEI Parameters
        self.M_sei = M_sei if M_sei else 0.162  # Example default
        self.rho_sei = rho_sei if rho_sei else 1690.0
        self.kappa_sei = kappa_sei if kappa_sei else 5e-6
        self.alpha_c = 0.5  # Cathodic transfer coefficient
        self.U_sei = 0.4    # Equilibrium potential of SEI [V] vs Li/Li+
        self.U_lp = 0.0     # Equilibrium potential of Li Plating [V] vs Li/Li+

    def ficks_second_law(self, c, r, t):
        """
        Computes the residual for Fick's second law in spherical coordinates.
        dc/dt - D_s * (d^2c/dr^2 + 2/r * dc/dr) = 0
        """
        dc_dt = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True, allow_unused=True)[0]
        if dc_dt is None: dc_dt = torch.zeros_like(c)
            
        dc_dr = torch.autograd.grad(c, r, grad_outputs=torch.ones_like(c), create_graph=True, allow_unused=True)[0]
        if dc_dr is None: dc_dr = torch.zeros_like(c)
            
        d2c_dr2 = torch.autograd.grad(dc_dr, r, grad_outputs=torch.ones_like(dc_dr), create_graph=True, allow_unused=True)[0]
        if d2c_dr2 is None: d2c_dr2 = torch.zeros_like(c)
        
        epsilon = 1e-8
        residual = dc_dt - self.D_s * (d2c_dr2 + (2.0 / (r + epsilon)) * dc_dr)
        return residual

    def thermal_energy_balance(self, T_core, V_cell, U_ocp, dU_dT, I_val, T_amb, h_conv, C_p, M_cell):
        """
        Lumped energy balance equation capturing reversible and irreversible heat.
        Q_total = Q_irreversible (polarization/ohmic) + Q_reversible (entropic)
        
        Parameters:
        T_core: Predicted internal cell temperature [K]
        V_cell: Predicted/Actual cell terminal voltage [V]
        U_ocp: Open Circuit Potential (OCP) at current SoC [V]
        dU_dT: Entropic heat coefficient [V/K]
        I_val: Applied current [A] (Positive for discharge)
        T_amb: Ambient temperature [K]
        h_conv: Convective heat transfer coefficient [W/(m^2 K)]
        C_p: Specific heat capacity of the cell [J/(kg K)]
        M_cell: Mass of the cell [kg]
        """
        # Irreversible heat (Joule heating + overpotential)
        Q_irr = I_val * (U_ocp - V_cell)
        
        # Reversible heat (Entropic)
        Q_rev = I_val * T_core * dU_dT
        
        # Total heat generation [W]
        Q_gen = Q_irr + Q_rev
        
        # Convective cooling [W]
        # Assuming a surface area A_cell
        A_cell = 0.015 # m^2 (example for 18650 cell)
        Q_cool = h_conv * A_cell * (T_core - T_amb)
        
        # dT/dt using autograd. (Assuming T_core is predicted over time t)
        # However, since this is a residual function, we return the thermal residual:
        # M_cell * C_p * dT/dt = Q_gen - Q_cool
        # Here we just return the net heat rate derivative matching:
        dT_dt_expected = (Q_gen - Q_cool) / (M_cell * C_p)
        
        return dT_dt_expected

    def sei_degradation_kinetics(self, phi_s, phi_e, j_tot, R_sei, j_0_sei, delta_sei_t):
        """
        Models the formation of Solid Electrolyte Interphase (SEI) leading to capacity fade.
        Based on Tafel kinetics for SEI formation (Eq 11-13).
        
        Parameters:
        phi_s: Solid phase potential [V]
        phi_e: Electrolyte phase potential [V]
        j_tot: Total molar flux density (intercalation + side reactions) [mol/(m^2 s)]
        R_sei: Predicted SEI film resistance [Ohm m^2]
        j_0_sei: Exchange current density for SEI reaction
        delta_sei_t: Predicted SEI thickness over time
        """
        # Overpotential for SEI reaction (Eq 12)
        eta_sei = phi_s - phi_e - self.U_sei - (j_tot * self.F * R_sei) / self.a_s
        
        # SEI reaction molar flux density (Tafel kinetics) (Eq 11)
        j_sei = -j_0_sei * self.a_s * torch.exp(- (self.alpha_c * self.F) / (self.R_gas * self.T) * eta_sei)
        
        # Rate of change of SEI thickness (Eq 13)
        # d(delta_sei)/dt = - (j_sei * M_sei) / (a_s * rho_sei)
        d_delta_dt_expected = - (j_sei * self.M_sei) / (self.a_s * self.rho_sei)
        
        # Expected SEI Resistance relationship
        # R_sei = delta_sei / kappa_sei
        R_sei_expected = delta_sei_t / self.kappa_sei
        
        # Return the residuals for thickness growth and resistance consistency
        return d_delta_dt_expected, R_sei_expected

    def lithium_plating_overpotential(self, phi_s, phi_e, j_lp, R_lp, j_0_lp):
        """
        Models the overpotential and reaction flux for Lithium Plating.
        This phenomenon occurs primarily during fast charging at low temperatures.
        (Eq 20-22)
        
        Parameters:
        j_lp: Molar flux density for lithium plating
        R_lp: Resistance of the plated lithium layer
        j_0_lp: Exchange molar flux density for Li plating
        """
        # Overpotential for Li Plating (Eq 22)
        # Note: U_lp is typically 0V vs Li/Li+
        eta_lp = phi_s - phi_e - self.U_lp - (R_lp * j_lp)
        
        # Tafel kinetics for Li Plating (Eq 21)
        j_lp_expected = -j_0_lp * torch.exp(- (self.alpha_c * self.F) / (self.R_gas * self.T) * eta_lp)
        
        # Safe Plating Flag (If eta_lp < 0, plating occurs)
        plating_risk_flag = torch.where(eta_lp < 0.0, torch.tensor(1.0).to(eta_lp.device), torch.tensor(0.0).to(eta_lp.device))
        
        return eta_lp, j_lp_expected, plating_risk_flag

class SPMAnsatz:
    """
    Hard-constrained PINN approach: 
    Using an Ansatz to strictly enforce Boundary Conditions mathematically, 
    so the neural network doesn't need to learn them via loss penalties.
    """
    @staticmethod
    def apply_boundary_ansatz(nn_output, r, R_s, I_val, D_s, F, A):
        """
        Ansatz for solid diffusion boundary conditions:
        At r = 0: dc/dr = 0 (Symmetry)
        At r = R_s: dc/dr = -I / (D_s * F * A) = J (Flux proportional to current)
        """
        J = -I_val / (D_s * F * A) 
        
        # Base function satisfying BCs
        g_r = (J / (2 * R_s)) * (r ** 2)
        
        # Smoothing function
        modifier = (r**3 / 3.0) - (R_s * r**2 / 2.0)
        
        c_approx = g_r + modifier * nn_output
        return c_approx
