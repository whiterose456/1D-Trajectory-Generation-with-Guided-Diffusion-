import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusion1D(nn.Module):
    """
    1D Trajectory Gaussian Diffusion Engine (DDPM & DDIM Sampling + Guided Diffusion).
    """
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Add noise to trajectory data x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, cond=None, goal=None, noise=None):
        """Compute training MSE loss between predicted and true noise."""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, cond=cond, goal=goal)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, shape, cond=None, goal=None, cfg_scale=3.0, ddim=False, ddim_steps=50):
        """
        Generates trajectories using Classifier-Free Guidance (CFG).
        Supports DDPM and fast DDIM sampling.
        """
        self.model.eval()
        batch_size = shape[0]
        device = self.device

        # Start from pure Gaussian noise
        img = torch.randn(shape, device=device)

        if ddim:
            # DDIM Fast Sampler
            times = torch.linspace(self.timesteps - 1, 0, ddim_steps, dtype=torch.long, device=device)
            for i in range(len(times)):
                t = times[i]
                prev_t = times[i + 1] if i < len(times) - 1 else -1

                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # CFG Guided Prediction
                if cfg_scale > 0.0:
                    pred_noise_cond = self.model(img, t_batch, cond=cond, goal=goal)
                    pred_noise_uncond = self.model(img, t_batch, cond=None, goal=None)
                    pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
                else:
                    pred_noise = self.model(img, t_batch, cond=cond, goal=goal)

                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)

                pred_x0 = (img - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                dir_xt = torch.sqrt(1.0 - alpha_prev) * pred_noise
                img = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
        else:
            # Standard DDPM Sampler
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

                if cfg_scale > 0.0:
                    pred_noise_cond = self.model(img, t_batch, cond=cond, goal=goal)
                    pred_noise_uncond = self.model(img, t_batch, cond=None, goal=None)
                    pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
                else:
                    pred_noise = self.model(img, t_batch, cond=cond, goal=goal)

                beta_t = self.betas[t]
                sqrt_alpha_t = torch.sqrt(self.alphas[t])
                sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

                mean = (1.0 / sqrt_alpha_t) * (img - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise)

                if t > 0:
                    noise = torch.randn_like(img)
                    variance = torch.sqrt(self.posterior_variance[t])
                    img = mean + variance * noise
                else:
                    img = mean

        return img
