import torch
import torch.nn as nn
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PINNTrainer:
    def __init__(self, model, spm_physics, optimizer, lambda_phys=1.0, adaptive_lambda=False):
        """
        Trainer for the PINN model combining data loss and physics loss.
        
        Args:
            model (nn.Module): The PINN network.
            spm_physics (AdvancedSingleParticleModel): Physics model for residual computation.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            lambda_phys (float): Weight for the physics loss.
            adaptive_lambda (bool): Whether to dynamically balance losses.
        """
        self.model = model
        self.spm_physics = spm_physics
        self.optimizer = optimizer
        self.mse_loss = nn.MSELoss()
        
        self.adaptive_lambda = adaptive_lambda
        self.lambda_phys = lambda_phys
        self.alpha = 0.1 # Moving average for adaptive lambda
        
        # Default cell parameters for thermal model (can be injected via config in a full system)
        self.T_amb = 298.15
        self.h_conv = 10.0
        self.C_p = 800.0
        self.M_cell = 0.05
        
    def _compute_physics_loss(self, t_colloc, I_colloc, r_colloc):
        """
        Computes the physics residuals based on the AdvancedSingleParticleModel.
        """
        t_colloc.requires_grad_(True)
        r_colloc.requires_grad_(True)
        
        # Forward pass for collocation points
        V_colloc, T_colloc, c_colloc = self.model(t_colloc, I_colloc, r_colloc)
        
        # 1. Fick's Second Law (Concentration)
        res_c = self.spm_physics.ficks_second_law(c_colloc, r_colloc, t_colloc)
        loss_ficks = torch.mean(res_c ** 2)
        
        # 2. Thermal Energy Balance
        # For demonstration, assuming simplified U_ocp and dU_dT.
        # In a complete model, these would depend on SOC (derived from volume-averaged c).
        U_ocp = 4.0 
        dU_dT = -0.0003 
        
        dT_dt_expected = self.spm_physics.thermal_energy_balance(
            T_core=T_colloc, 
            V_cell=V_colloc, 
            U_ocp=U_ocp, 
            dU_dT=dU_dT, 
            I_val=I_colloc, 
            T_amb=self.T_amb, 
            h_conv=self.h_conv, 
            C_p=self.C_p, 
            M_cell=self.M_cell
        )
        
        # Calculate actual dT/dt from the neural network output
        dT_dt = torch.autograd.grad(
            T_colloc, t_colloc, 
            grad_outputs=torch.ones_like(T_colloc), 
            create_graph=True, 
            allow_unused=True
        )[0]
        
        if dT_dt is None:
            dT_dt = torch.zeros_like(T_colloc)
            
        res_T = dT_dt - dT_dt_expected
        loss_thermal = torch.mean(res_T ** 2)
        
        return loss_ficks + loss_thermal

    def train_step(self, t_data, I_data, V_data, T_data, t_colloc, I_colloc, r_colloc):
        self.optimizer.zero_grad()
        
        # --- 1. Data Loss ---
        # Evaluate model at data points. For surface measurements, r = R_s.
        r_data = torch.full_like(t_data, self.model.R_s)
        V_pred, T_pred, _ = self.model(t_data, I_data, r_data)
        
        loss_data = self.mse_loss(V_pred, V_data) + self.mse_loss(T_pred, T_data)
        
        # --- 2. Physics Loss ---
        loss_physics = self._compute_physics_loss(t_colloc, I_colloc, r_colloc)
        
        # --- 3. Total Loss ---
        loss_total = loss_data + self.lambda_phys * loss_physics
        
        # --- 4. Backpropagation ---
        loss_total.backward(retain_graph=self.adaptive_lambda)
        
        if self.adaptive_lambda:
            # Placeholder for adaptive lambda logic based on gradient norms
            pass 
            
        self.optimizer.step()
        
        return loss_data.item(), loss_physics.item(), loss_total.item()

    def train(self, dataloader, epochs, num_colloc=1000, early_stopping_patience=5):
        """
        Main training loop.
        
        Args:
            dataloader: Yields (t, I, V, T).
            epochs: Number of training epochs.
            num_colloc: Number of collocation points generated per batch.
            early_stopping_patience: Number of epochs to wait for improvement before stopping.
        """
        self.model.train()
        history = {'data_loss': [], 'physics_loss': [], 'total_loss': [], 'lambda': []}
        device = next(self.model.parameters()).device
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_data_loss, epoch_phys_loss, epoch_tot_loss = 0.0, 0.0, 0.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (t_batch, I_batch, V_batch, T_batch) in enumerate(pbar):
                t_batch, I_batch = t_batch.to(device), I_batch.to(device)
                V_batch, T_batch = V_batch.to(device), T_batch.to(device)
                
                # Dynamic Collocation Points Generation
                # Sample t uniformly within the batch's time range
                t_colloc = torch.rand(num_colloc, 1, device=device) * (t_batch.max() - t_batch.min()) + t_batch.min()
                # Sample r uniformly from 0 to R_s
                r_colloc = torch.rand(num_colloc, 1, device=device) * self.model.R_s
                # Use mean current of the batch for simplicity, or resample from realistic current profiles
                I_colloc = I_batch.mean() * torch.ones_like(t_colloc)
                
                l_data, l_phys, l_total = self.train_step(
                    t_batch, I_batch, V_batch, T_batch, 
                    t_colloc, I_colloc, r_colloc
                )
                
                epoch_data_loss += l_data
                epoch_phys_loss += l_phys
                epoch_tot_loss += l_total
                
                pbar.set_postfix({
                    'L_data': f"{l_data:.4e}", 
                    'L_phys': f"{l_phys:.4e}", 
                    'Lam': f"{self.lambda_phys:.2f}"
                })
                
            num_batches = len(dataloader)
            avg_data_loss = epoch_data_loss / num_batches
            avg_phys_loss = epoch_phys_loss / num_batches
            avg_tot_loss = epoch_tot_loss / num_batches
            
            history['data_loss'].append(avg_data_loss)
            history['physics_loss'].append(avg_phys_loss)
            history['total_loss'].append(avg_tot_loss)
            history['lambda'].append(self.lambda_phys)
            
            # Step the LR scheduler
            scheduler.step(avg_tot_loss)
            
            # Dynamic Lambda: increase slowly if adaptive_lambda is true
            if self.adaptive_lambda and self.lambda_phys < 10.0:
                self.lambda_phys *= 1.1
            
            # Early Stopping
            if avg_tot_loss < best_loss:
                best_loss = avg_tot_loss
                patience_counter = 0
                # Optionally save best model weights here
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
        return history
