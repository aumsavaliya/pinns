import torch
import torch.optim as optim
import numpy as np
import logging

from src.data_pipeline import get_dataloaders
from src.models.pinn_network import PINNNetwork
from src.physics.spm import AdvancedSingleParticleModel
from src.train import PINNTrainer
from src.evaluate import evaluate_pinn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Initializing Data Pipeline...")
    raw_dir = r"data/raw/NASA"
    dataloader = get_dataloaders(raw_dir, batch_size=32, add_noise=True, noise_std=0.02)
    
    logging.info("Initializing Models...")
    # Initialize physics model
    spm = AdvancedSingleParticleModel(
        D_s=1e-14, R_s=1e-5, F=96485.33, T=298.15, R_gas=8.314, a_s=0.015
    )
    
    # Initialize neural network
    model = PINNNetwork(
        hidden_sizes=[128, 128, 128, 128],
        R_s=1e-5, D_s=1e-14, F=96485.33, A=0.015
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    logging.info("Initializing Trainer...")
    trainer = PINNTrainer(
        model=model, 
        spm_physics=spm, 
        optimizer=optimizer, 
        lambda_phys=1.0, 
        adaptive_lambda=True
    )
    
    logging.info("Starting Training...")
    # Train for 20 epochs with early stopping
    history = trainer.train(dataloader, epochs=20, num_colloc=100, early_stopping_patience=5)
    
    logging.info("Evaluating Model...")
    # Evaluate model
    metrics = evaluate_pinn(model, dataloader)
    
    logging.info("Exporting Model to ONNX for IoT Edge Deployment...")
    # Dummy inputs for ONNX export: (t, I, r)
    dummy_t = torch.randn(1, 1)
    dummy_I = torch.randn(1, 1)
    dummy_r = torch.randn(1, 1)
    torch.onnx.export(
        model, 
        (dummy_t, dummy_I, dummy_r), 
        "pinn_bms.onnx", 
        input_names=['time', 'current', 'radius'], 
        output_names=['voltage', 'temperature', 'concentration'], 
        dynamic_axes={
            'time': {0: 'batch_size'},
            'current': {0: 'batch_size'},
            'radius': {0: 'batch_size'},
            'voltage': {0: 'batch_size'},
            'temperature': {0: 'batch_size'},
            'concentration': {0: 'batch_size'}
        }
    )
    logging.info("Model exported to pinn_bms.onnx successfully!")
    
    logging.info("Pipeline executed successfully!")

if __name__ == "__main__":
    main()
