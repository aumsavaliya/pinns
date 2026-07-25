import torch
import torch.nn as nn
from src.physics.spm import SPMAnsatz

class PINNNetwork(nn.Module):
    def __init__(self, hidden_sizes=[64, 64, 64], R_s=1e-5, D_s=1e-14, F=96485.33, A=0.015):
        """
        Physics-Informed Neural Network (PINN) for Battery Modeling.
        Takes (t, I, r) as input and outputs (V, T, c).
        The concentration 'c' is hard-constrained using the SPM Ansatz.
        
        Args:
            hidden_sizes (list): Sizes of hidden layers.
            R_s (float): Particle radius [m].
            D_s (float): Solid-phase diffusion coefficient [m^2/s].
            F (float): Faraday's constant [C/mol].
            A (float): Electrode specific area [m^2/m^3] or surface area.
        """
        super().__init__()
        self.R_s = R_s
        self.D_s = D_s
        self.F = F
        self.A = A
        
        # Inputs: t (time), I (current), r (spatial coordinate)
        input_size = 3
        # Outputs: V (Voltage), T (Temperature), c_raw (unconstrained concentration)
        output_size = 3
        
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())  # Tanh is smooth and differentiable for PINNs
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, t, I, r):
        """
        Forward pass for the PINN.
        
        Args:
            t (torch.Tensor): Time tensor of shape [batch_size, 1].
            I (torch.Tensor): Applied current tensor of shape [batch_size, 1].
            r (torch.Tensor): Spatial coordinate tensor of shape [batch_size, 1].
            
        Returns:
            V (torch.Tensor): Predicted voltage.
            T (torch.Tensor): Predicted temperature.
            c_constrained (torch.Tensor): Predicted solid concentration (with hard constraints).
        """
        # Concatenate inputs along the last dimension
        x = torch.cat([t, I, r], dim=-1)
        
        out = self.network(x)
        
        V = out[..., 0:1]
        T = out[..., 1:2]
        c_raw = out[..., 2:3]
        
        # Apply the Hard Constraint Ansatz for Concentration 'c'
        # c_approx(r, t) = g(r) + (r^2 - R_s^2) * r * NN(r,t)
        # using the provided SPMAnsatz logic.
        c_constrained = SPMAnsatz.apply_boundary_ansatz(
            nn_output=c_raw, 
            r=r, 
            R_s=self.R_s, 
            I_val=I, 
            D_s=self.D_s, 
            F=self.F, 
            A=self.A
        )
        
        return V, T, c_constrained
