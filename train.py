import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TrajectoryDataset1D
from src.model import Unet1D
from src.diffusion import GaussianDiffusion1D
from src.utils import plot_trajectories

def train(
    epochs=15,
    batch_size=64,
    lr=1e-3,
    num_samples=3000,
    seq_len=64,
    timesteps=500,
    save_dir="checkpoints"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    dataset = TrajectoryDataset1D(num_samples=num_samples, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Model & Diffusion Engine
    model = Unet1D(in_dim=1, base_channels=64, channel_mults=(1, 2, 4)).to(device)
    diffusion = GaussianDiffusion1D(model, timesteps=timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)

    # 3. Training Loop
    print("Starting Training...")
    model.train()
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        total_loss = 0.0

        for batch in pbar:
            x_0 = batch['trajectory'].to(device) # (B, 1, T)
            cond = batch['condition'].to(device) # (B,)
            goal = batch['goal'].to(device)      # (B, 1)

            # Randomly drop condition for Classifier-Free Guidance (15% drop rate)
            if torch.rand(1).item() < 0.15:
                cond = None
                goal = None

            t = torch.randint(0, timesteps, (x_0.shape[0],), device=device).long()
            
            loss = diffusion.p_losses(x_0, t, cond=cond, goal=goal)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Complete - Average Loss: {avg_loss:.4f}")

    # 4. Save Checkpoint
    checkpoint_path = os.path.join(save_dir, "guided_diffusion_1d.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved successfully to {checkpoint_path}")

    # 5. Generate Validation Trajectories Plot
    print("Generating validation samples...")
    val_conds = torch.tensor([0, 1, 2, 0, 1, 2], device=device)
    val_goals = torch.tensor([[-0.8], [0.5], [0.9], [-0.5], [0.2], [-0.9]], device=device)
    samples = diffusion.sample(shape=(6, 1, seq_len), cond=val_conds, goal=val_goals, cfg_scale=3.0, ddim=True, ddim_steps=50)
    
    plot_trajectories(samples, conditions=val_conds, goals=val_goals, save_path="val_trajectories.png", title="Guided Diffusion 1D Trajectories")
    print("Validation plot saved to val_trajectories.png")

if __name__ == "__main__":
    train(epochs=10, batch_size=64, num_samples=2000)
