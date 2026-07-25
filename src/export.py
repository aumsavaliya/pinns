import torch
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def export_to_onnx(model, dummy_input, save_path="models/onnx/pinn_bms.onnx"):
    """
    Export the trained PyTorch model to ONNX format for deployment
    on BMS embedded systems (e.g., STM32 microcontrollers).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    
    # Exporting
    logging.info(f"Exporting model to ONNX format at {save_path}...")
    torch.onnx.export(
        model, 
        dummy_input,               # Model input (or a tuple for multiple inputs)
        save_path, 
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=11,          # ONNX version to export to (11 is widely supported)
        do_constant_folding=True,  # Constant folding for optimization
        input_names=['input', 'r_colloc', 'I_val'],   # The model's input names
        output_names=['pred_voltage', 'pred_c'], # The model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # Variable length axes
            'r_colloc': {0: 'batch_size'},
            'pred_voltage': {0: 'batch_size'},
            'pred_c': {0: 'batch_size'}
        }
    )
    logging.info("ONNX Export completed successfully. Ready for C/C++ inference engine.")

if __name__ == "__main__":
    # Example usage hook
    pass
