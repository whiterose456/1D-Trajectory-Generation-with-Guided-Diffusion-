import argparse
import os
import torch
import matplotlib.pyplot as plt

from src.model import Unet1D
from src.diffusion import GaussianDiffusion1D
from src.utils import plot_trajectories

def main():
    parser = argparse.ArgumentParser(description="Sample 1D Trajectories using Guided Diffusion")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of trajectories to sample")
    parser.add_argument("--condition", type=int, default=1, choices=[0, 1, 2], help="Style condition (0: Linear, 1: S-Curve, 2: Parabolic)")
    parser.add_argument("--goal", type=float, default=0.75, help="Target 1D goal position (-1.0 to 1.0)")
    parser.add_argument("--cfg_scale", type=float, default=3.5, help="Classifier-free guidance scale")
    parser.add_argument("--ddim", action="store_true", help="Use DDIM fast sampling")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/guided_diffusion_1d.pt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="generated_sample.png", help="Output plot filepath")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Model & Diffusion
    model = Unet1D(in_dim=1, base_channels=64, channel_mults=(1, 2, 4)).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with un-trained weights for demonstration.")

    diffusion = GaussianDiffusion1D(model, timesteps=500, device=device)

    # Condition Tensors
    conds = torch.full((args.num_samples,), args.condition, dtype=torch.long, device=device)
    goals = torch.full((args.num_samples, 1), args.goal, dtype=torch.float32, device=device)

    print(f"Sampling {args.num_samples} trajectories with Condition={args.condition}, Goal={args.goal}, CFG Scale={args.cfg_scale}...")
    trajectories = diffusion.sample(
        shape=(args.num_samples, 1, 64),
        cond=conds,
        goal=goals,
        cfg_scale=args.cfg_scale,
        ddim=args.ddim,
        ddim_steps=50
    )

    plot_trajectories(
        trajectories,
        conditions=conds,
        goals=goals,
        save_path=args.output,
        title=f"1D Trajectory Generation (CFG Scale={args.cfg_scale}, Goal={args.goal:.2f})"
    )
    print(f"Saved generated plot to {args.output}")

if __name__ == "__main__":
    main()
