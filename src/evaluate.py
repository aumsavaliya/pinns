import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    max_error = np.max(np.abs(y_true - y_pred))
    return rmse, mae, max_error

def evaluate_pinn(model, dataloader, scaler_y=None):
    """
    Evaluates the PINN model on V and T predictions.
    dataloader should yield (t_batch, I_batch, V_batch, T_batch).
    """
    model.eval()
    
    device = next(model.parameters()).device
    
    V_preds, V_targets = [], []
    T_preds, T_targets = [], []
    
    with torch.no_grad():
        for t_batch, I_batch, V_batch, T_batch in dataloader:
            t_batch = t_batch.to(device)
            I_batch = I_batch.to(device)
            
            # Predict at particle surface
            r_batch = torch.full_like(t_batch, model.R_s)
            
            # PINN Network output: V, T, c
            V_pred, T_pred, _ = model(t_batch, I_batch, r_batch)
            
            V_preds.append(V_pred.cpu().numpy())
            V_targets.append(V_batch.cpu().numpy())
            
            T_preds.append(T_pred.cpu().numpy())
            T_targets.append(T_batch.cpu().numpy())
            
    V_preds = np.vstack(V_preds)
    V_targets = np.vstack(V_targets)
    T_preds = np.vstack(T_preds)
    T_targets = np.vstack(T_targets)
    
    # Optionally inverse transform if scaled
    if scaler_y is not None:
        # Assuming scaler_y can transform V and T concatenated or similar
        # Since this is a specialized PINN evaluation, we might need separate scalers
        # For simplicity, we just leave it if not provided or handle it based on setup.
        pass
        
    V_rmse, V_mae, V_max_err = compute_metrics(V_targets, V_preds)
    T_rmse, T_mae, T_max_err = compute_metrics(T_targets, T_preds)
    
    logging.info(f"Voltage Metrics -> RMSE: {V_rmse:.4f}, MAE: {V_mae:.4f}, Max Error: {V_max_err:.4f}")
    logging.info(f"Temperature Metrics -> RMSE: {T_rmse:.4f}, MAE: {T_mae:.4f}, Max Error: {T_max_err:.4f}")
    
    return {
        'V_rmse': V_rmse, 'V_mae': V_mae, 'V_max_error': V_max_err,
        'T_rmse': T_rmse, 'T_mae': T_mae, 'T_max_error': T_max_err
    }
