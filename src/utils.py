import os
import matplotlib.pyplot as plt
import torch

def plot_trajectories(trajectories, conditions=None, goals=None, save_path=None, title="Generated 1D Trajectories"):
    """
    Plots 1D trajectory curves (position vs time step).
    trajectories: Tensor or array of shape (B, 1, T) or (B, T)
    """
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()
    if len(trajectories.shape) == 3:
        trajectories = trajectories.squeeze(1) # (B, T)

    fig, ax = plt.subplots(figsize=(9, 5))
    seq_len = trajectories.shape[1]
    time_steps = list(range(seq_len))

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    styles = ['Linear-like', 'S-Curve Oscillatory', 'Parabolic Arc']

    for i in range(len(trajectories)):
        c_idx = conditions[i].item() if conditions is not None else 0
        label = styles[c_idx] if (conditions is not None and i < 3) else None
        
        ax.plot(time_steps, trajectories[i], color=colors[c_idx % len(colors)], 
                alpha=0.7, linewidth=2, label=label)

        if goals is not None:
            g = goals[i].item() if isinstance(goals[i], torch.Tensor) else goals[i]
            ax.plot(seq_len - 1, g, 'ro', markersize=6, label='Goal' if i == 0 else "")

    ax.set_xlabel("Time Step (t)", fontsize=12)
    ax.set_ylabel("1D Position (x)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Avoid duplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return fig
