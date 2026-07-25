import torch
import torch.nn as nn
from src.physics.spm import SPMAnsatz

class PhysicsInformedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Physics-Informed LSTM (PI-LSTM)
        Outputs the target variable (Voltage) AND the latent physics variable (Concentration 'c')
        """
        super(PhysicsInformedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # We output Voltage/SoH + Latent concentration 'c'
        self.fc = nn.Linear(hidden_size, output_size + 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        
        # Split output into main prediction (e.g., Voltage) and latent physics state (c)
        main_pred = out[:, :, :-1]
        c_latent = out[:, :, -1:]
        
        return main_pred, c_latent


class HardConstrainedPINN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, R_s, D_s, F, A):
        """
        Feedforward Neural Network where Boundary Conditions are mathematically
        enforced via an Ansatz. Excellent for robust BMS deployment.
        """
        super(HardConstrainedPINN, self).__init__()
        self.R_s = R_s
        self.D_s = D_s
        self.F = F
        self.A = A
        
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh()) # Tanh is preferred for PINNs for smooth 2nd derivatives
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size + 1)) # Output + latent c
        self.network = nn.Sequential(*layers)
        
    def forward(self, x_features, r_colloc, I_val):
        """
        x_features: Standard ML inputs (t, I, T, etc.)
        r_colloc: Spatial collocation points
        I_val: Current applied (needed for Boundary Condition flux)
        """
        # Concatenate x_features and r_colloc if we are evaluating the spatiotemporal field
        # For simplicity, assuming x_features already includes 't' and 'r' for the PDE evaluation
        out = self.network(x_features)
        
        main_pred = out[:, :-1]
        c_raw = out[:, -1:]
        
        # Apply the Hard Constraint Ansatz for Concentration 'c'
        # c_approx(r, t) = g(r) + (r^2 - R_s^2) * r * NN(r,t)
        c_constrained = SPMAnsatz.apply_boundary_ansatz(
            nn_output=c_raw, 
            r=r_colloc, 
            R_s=self.R_s, 
            I_val=I_val, 
            D_s=self.D_s, 
            F=self.F, 
            A=self.A
        )
        
        return main_pred, c_constrained
