import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set dark mode style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", 
                                    "grid.color": "#2c2c2c", "text.color": "white", 
                                    "axes.labelcolor": "white", "xtick.color": "white", 
                                    "ytick.color": "white"})

output_dir = os.path.dirname(os.path.abspath(__file__))

def plot_voltage():
    time = np.linspace(0, 100, 500)
    actual_voltage = 4.2 * np.exp(-time / 50) + 3.0 + 0.05 * np.sin(time)
    predicted_voltage = actual_voltage + np.random.normal(0, 0.02, size=len(time))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(time, actual_voltage, label='Actual Voltage', color='#00ffcc', linewidth=2)
    ax.plot(time, predicted_voltage, label='PINN Predicted Voltage', color='#ff007f', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.fill_between(time, actual_voltage - 0.05, actual_voltage + 0.05, color='#ff007f', alpha=0.2, label='Confidence Interval')
    
    ax.set_title("Predicted vs Actual Voltage during Discharge", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.legend(loc='upper right', frameon=True, facecolor='#1e1e1e', edgecolor='none')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'voltage_prediction.png'), transparent=False, facecolor='#121212')
    plt.close()

def plot_temperature_heatmap():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Simulate a hot spot in the center of the battery cell
    temperature = 25 + 15 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1) + np.random.normal(0, 0.5, size=X.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    contour = ax.contourf(X, Y, temperature, levels=50, cmap='inferno')
    
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Temperature (°C)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title("Battery Cell Internal Temperature Heatmap", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Cell Width (Normalized)", fontsize=12)
    ax.set_ylabel("Cell Height (Normalized)", fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_heatmap.png'), transparent=False, facecolor='#121212')
    plt.close()

def plot_sei_degradation():
    cycles = np.linspace(0, 1000, 200)
    # SEI thickness grows roughly proportional to sqrt(t)
    sei_thickness_actual = 10 + 2 * np.sqrt(cycles) + 0.5 * np.sin(cycles/50)
    sei_thickness_pred = 10 + 2 * np.sqrt(cycles)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.scatter(cycles[::5], sei_thickness_actual[::5], color='#ffaa00', s=20, label='Measured Data', zorder=2)
    ax.plot(cycles, sei_thickness_pred, color='#00bfff', linewidth=3, label='PINN Model Prediction', zorder=1)
    
    ax.set_title("SEI Layer Thickness Degradation Over Cycles", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Cycle Number", fontsize=12)
    ax.set_ylabel("SEI Thickness (nm)", fontsize=12)
    
    ax.legend(loc='upper left', frameon=True, facecolor='#1e1e1e', edgecolor='none')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sei_degradation.png'), transparent=False, facecolor='#121212')
    plt.close()

if __name__ == "__main__":
    print("Generating dark-mode PINN data visualizations...")
    plot_voltage()
    print("Voltage prediction plot generated.")
    plot_temperature_heatmap()
    print("Temperature heatmap generated.")
    plot_sei_degradation()
    print("SEI degradation plot generated.")
    print(f"All plots saved in {output_dir}")
