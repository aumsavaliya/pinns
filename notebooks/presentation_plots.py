import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set premium aesthetics for presentation
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk", font_scale=1.1)

def plot_voltage_curves(time, v_actual, v_pred_baseline, v_pred_pinn, cycle_num):
    """
    Plots the predicted vs actual voltage curves comparing the Pure ML Baseline
    to the PINN. Highlights how PINNs stay physically bounded.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, v_actual, 'k--', label='True Voltage', linewidth=2)
    plt.plot(time, v_pred_baseline, 'r-', label='Baseline LSTM', alpha=0.7)
    plt.plot(time, v_pred_pinn, 'g-', label='PINN (Hard-Constrained)', linewidth=2)
    plt.title(f"Voltage Discharge Curve (Cycle {cycle_num})", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend(frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f"voltage_curve_cycle_{cycle_num}.png", dpi=300)
    plt.close()

def plot_pde_residual_heatmap(time_grid, r_grid, residuals):
    """
    Spatiotemporal heatmap of the PDE residuals.
    Proves to the committee that the network is actually obeying Fick's Second Law.
    """
    plt.figure(figsize=(8, 6))
    # residuals shape: (len(time_grid), len(r_grid))
    heatmap = plt.contourf(r_grid, time_grid, residuals, levels=50, cmap='inferno')
    plt.colorbar(heatmap, label='PDE Residual Absolute Error')
    plt.title("Spatiotemporal PDE Physics Enforcement", fontweight='bold')
    plt.xlabel("Particle Radius ($r$)")
    plt.ylabel("Discharge Time ($t$)")
    plt.tight_layout()
    plt.savefig("pde_residual_heatmap.png", dpi=300)
    plt.close()

def plot_soh_degradation(cycles, soh_actual, soh_pred):
    """
    Tracks capacity fade over hundreds of cycles.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(cycles, soh_actual, c='black', label='Actual SoH', alpha=0.6, s=15)
    plt.plot(cycles, soh_pred, 'b-', label='Predicted SoH', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', label='End of Life (80%)')
    plt.title("State of Health (SoH) Degradation Over Time", fontweight='bold')
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity (Ah)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("soh_degradation.png", dpi=300)
    plt.close()

def plot_spider_comparison(metrics, baseline_scores, pinn_scores):
    """
    Spider/Radar chart to compare PINN vs Baseline on multiple BMS metrics:
    e.g., [RMSE, MAE, Max Error, Inference Time, Robustness to Noise]
    * Note: Scores should be normalized so higher is better for visualization.
    """
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    baseline_scores += baseline_scores[:1]
    pinn_scores += pinn_scores[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.plot(angles, baseline_scores, color='red', linewidth=2, label='Baseline ML')
    ax.fill(angles, baseline_scores, color='red', alpha=0.25)
    
    ax.plot(angles, pinn_scores, color='green', linewidth=2, label='PINN')
    ax.fill(angles, pinn_scores, color='green', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    
    plt.title("Model Architecture Comparison", size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("spider_comparison.png", dpi=300)
    plt.close()
