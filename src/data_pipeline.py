import os
import glob
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader

def load_nasa_battery_data(file_path):
    """
    Dummy/Basic parser for NASA battery .mat files.
    The actual NASA battery dataset structure can be deeply nested.
    This provides a simplified parsing structure for V, I, T.
    """
    try:
        mat = scipy.io.loadmat(file_path)
        # Extract features (dummy logic depending on actual .mat structure)
        # In a real implementation, we'd navigate the nested matlab structs.
        length = 1000  # Dummy length
        t = np.linspace(0, 3600, length)
        V = np.linspace(4.2, 2.5, length)
        I = np.ones(length) * -2.0
        T = np.linspace(24, 35, length)
        return t, V, I, T
    except Exception as e:
        print(f"Error parsing {file_path} or file not found, generating dummy data.")
        length = 1000
        t = np.linspace(0, 3600, length)
        V = np.linspace(4.2, 2.5, length)
        I = np.ones(length) * -2.0
        T = np.linspace(24, 35, length)
        return t, V, I, T

def normalize_features(t, V, I, T):
    """
    Normalize Time (t), Voltage (V), Current (I), and Temperature (T).
    """
    def z_score(arr):
        std = np.std(arr)
        if std == 0:
            return arr - np.mean(arr)
        return (arr - np.mean(arr)) / std
        
    def min_max(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

    t_norm = min_max(t)
    V_norm = z_score(V)
    I_norm = z_score(I)
    T_norm = z_score(T)
    return t_norm, V_norm, I_norm, T_norm

def inject_gaussian_noise(data_array, mean=0.0, std=0.01):
    """
    Add Gaussian noise to the data to simulate real BMS sensor noise.
    """
    noise = np.random.normal(mean, std, data_array.shape)
    return data_array + noise

class BatteryDataset(Dataset):
    def __init__(self, t, V, I, T, add_noise=False, noise_std=0.01):
        self.t = t
        self.V = V
        self.I = I
        self.T = T
        
        if add_noise:
            self.V = inject_gaussian_noise(self.V, std=noise_std)
            self.I = inject_gaussian_noise(self.I, std=noise_std)
            self.T = inject_gaussian_noise(self.T, std=noise_std)
            
        # Normalizing after adding noise
        self.t, self.V, self.I, self.T = normalize_features(self.t, self.V, self.I, self.T)
        
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, idx):
        t_val = torch.tensor([self.t[idx]], dtype=torch.float32)
        I_val = torch.tensor([self.I[idx]], dtype=torch.float32)
        V_val = torch.tensor([self.V[idx]], dtype=torch.float32)
        T_val = torch.tensor([self.T[idx]], dtype=torch.float32)
        return t_val, I_val, V_val, T_val

def get_dataloaders(data_dir, batch_size=32, add_noise=True, noise_std=0.05):
    """
    Reads data from raw directory and returns PyTorch DataLoader.
    """
    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    
    all_t, all_V, all_I, all_T = [], [], [], []
    
    if not mat_files:
        print(f"No .mat files found in {data_dir}. Generating a single dummy trajectory.")
        t, V, I, T = load_nasa_battery_data("dummy.mat")
        all_t.extend(t)
        all_V.extend(V)
        all_I.extend(I)
        all_T.extend(T)
    else:
        for f in mat_files:
            t, V, I, T = load_nasa_battery_data(f)
            all_t.extend(t)
            all_V.extend(V)
            all_I.extend(I)
            all_T.extend(T)
            
    dataset = BatteryDataset(np.array(all_t), np.array(all_V), np.array(all_I), np.array(all_T), 
                             add_noise=add_noise, noise_std=noise_std)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    # Test the pipeline
    raw_dir = r"../data/raw/NASA"
    dataloader = get_dataloaders(raw_dir, add_noise=True, noise_std=0.02)
    
    for batch_x, batch_y in dataloader:
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Batch Y shape: {batch_y.shape}")
        print(f"Sample X (V, I, T): {batch_x[0]}")
        break
