import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline

class TrajectoryDataset1D(Dataset):
    """
    Synthetic 1D Trajectory Dataset.
    Generates smooth 1D position trajectories over sequence length T=64.
    Trajectories start at x_0 ~ N(0, 0.1) and end near target condition goals.
    
    Conditions:
      0: Fast linear trajectory to goal
      1: Oscillatory/S-curve trajectory to goal
      2: Parabolic overshoot/return trajectory to goal
    """
    def __init__(self, num_samples=5000, seq_len=64, num_conditions=3, noise_std=0.02):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_conditions = num_conditions
        self.noise_std = noise_std
        
        self.data, self.conditions, self.goals = self._generate_data()

    def _generate_data(self):
        t = np.linspace(0, 1, self.seq_len)
        data = []
        conditions = []
        goals = []

        for _ in range(self.num_samples):
            cond = np.random.randint(0, self.num_conditions)
            start_pos = np.random.normal(0.0, 0.1)
            target_goal = np.random.uniform(-1.0, 1.0)
            
            if cond == 0:
                # Direct smooth transition
                traj = start_pos + (target_goal - start_pos) * t
            elif cond == 1:
                # S-curve / Oscillatory motion
                traj = start_pos + (target_goal - start_pos) * (t**2 * (3 - 2 * t)) + 0.2 * np.sin(2 * np.pi * t)
            else:
                # Parabolic peak / arc path
                traj = start_pos + (target_goal - start_pos) * t + 0.5 * np.sin(np.pi * t)

            # Add minor sensor/physical noise
            traj += np.random.normal(0, self.noise_std, size=self.seq_len)
            
            data.append(traj[np.newaxis, :]) # Shape: (1, seq_len)
            conditions.append(cond)
            goals.append(target_goal)

        data = torch.tensor(np.array(data), dtype=torch.float32)
        conditions = torch.tensor(np.array(conditions), dtype=torch.long)
        goals = torch.tensor(np.array(goals), dtype=torch.float32).unsqueeze(1) # Shape: (B, 1)

        return data, conditions, goals

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'trajectory': self.data[idx],     # (1, T)
            'condition': self.conditions[idx], # Class label (0, 1, or 2)
            'goal': self.goals[idx]           # Continuous scalar target goal
        }

if __name__ == "__main__":
    dataset = TrajectoryDataset1D(num_samples=100)
    sample = dataset[0]
    print(f"Trajectory shape: {sample['trajectory'].shape}")
    print(f"Condition: {sample['condition']}, Goal: {sample['goal'].item():.4f}")
