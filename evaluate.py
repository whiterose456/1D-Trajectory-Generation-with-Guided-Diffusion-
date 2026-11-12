"""Quantitative benchmark for the 1D guided-diffusion trajectory model.

Run after training:
    python evaluate.py --checkpoint checkpoints/guided_diffusion_1d.pt

The benchmark reports endpoint error, acceleration/jerk smoothness, and
between-sample diversity for each supported motion style.
"""
import argparse
import csv
import os

import numpy as np
import torch

from src.diffusion import GaussianDiffusion1D
from src.model import Unet1D


STYLE_NAMES = {
    0: "linear",
    1: "s_curve",
    2: "parabolic",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate guided 1D diffusion trajectory generation."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/guided_diffusion_1d.pt",
        help="Path to a trained model state_dict.",
    )
    parser.add_argument("--samples", type=int, default=32,
                        help="Number of trajectories per style.")
    parser.add_argument("--goal", type=float, default=0.75,
                        help="Shared target endpoint in [-1, 1].")
    parser.add_argument("--cfg-scale", type=float, default=3.5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="evaluation_results.csv",
                        help="Destination CSV path.")
    return parser.parse_args()


def trajectory_metrics(trajectories, goal):
    """Compute per-batch metrics for trajectories shaped (N, 1, T)."""
    positions = trajectories.squeeze(1)
    endpoint_error = (positions[:, -1] - goal).abs()

    velocity = positions[:, 1:] - positions[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]

    # Mean temporal standard deviation measures variation between samples.
    diversity = positions.std(dim=0, unbiased=False).mean()

    return {
        "endpoint_mae": endpoint_error.mean().item(),
        "endpoint_error_std": endpoint_error.std(unbiased=False).item(),
        "acceleration_rms": acceleration.square().mean().sqrt().item(),
        "jerk_rms": jerk.square().mean().sqrt().item(),
        "mean_abs_velocity": velocity.abs().mean().item(),
        "diversity": diversity.item(),
        "finite_fraction": torch.isfinite(trajectories).float().mean().item(),
    }


def main():
    args = parse_args()
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2 to measure diversity.")
    if not -1.0 <= args.goal <= 1.0:
        raise SystemExit("--goal must be between -1.0 and 1.0.")
    if args.ddim_steps < 1:
        raise SystemExit("--ddim-steps must be positive.")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train the model first with: python train.py"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet1D(
        in_dim=1, base_channels=64, channel_mults=(1, 2, 4)
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    diffusion = GaussianDiffusion1D(model, timesteps=500, device=device)
    rows = []

    with torch.inference_mode():
        for condition, style_name in STYLE_NAMES.items():
            conditions = torch.full(
                (args.samples,), condition, dtype=torch.long, device=device
            )
            goals = torch.full(
                (args.samples, 1), args.goal, dtype=torch.float32, device=device
            )
            generated = diffusion.sample(
                shape=(args.samples, 1, 64),
                cond=conditions,
                goal=goals,
                cfg_scale=args.cfg_scale,
                ddim=True,
                ddim_steps=min(args.ddim_steps, diffusion.timesteps),
            )

            metrics = trajectory_metrics(generated, args.goal)
            rows.append({
                "style": style_name,
                "condition_id": condition,
                "samples": args.samples,
                "goal": args.goal,
                "cfg_scale": args.cfg_scale,
                "sampler": "DDIM",
                "ddim_steps": min(args.ddim_steps, diffusion.timesteps),
                **metrics,
            })

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluation complete on {device}.")
    print(f"Results saved to: {args.output}")
    print(
        f"{'Style':<12} {'Endpoint MAE':>14} {'Accel RMS':>12} "
        f"{'Jerk RMS':>12} {'Diversity':>12}"
    )
    for row in rows:
        print(
            f"{row['style']:<12} {row['endpoint_mae']:>14.5f} "
            f"{row['acceleration_rms']:>12.5f} "
            f"{row['jerk_rms']:>12.5f} {row['diversity']:>12.5f}"
        )


if __name__ == "__main__":
    main()
