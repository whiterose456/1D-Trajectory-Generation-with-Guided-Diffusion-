import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Timestep Embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock1D(nn.Module):
    """1D Temporal Residual Block with Timestep and Condition Conditioning."""
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.gn1 = nn.GroupNorm(8, out_channels)
        self.gn2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, emb):
        # x: (B, C_in, T)
        # emb: (B, emb_dim)
        h = F.silu(self.gn1(self.conv1(x)))
        # Add conditioning embedding along temporal dimension
        h = h + self.emb_proj(emb).unsqueeze(-1)
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)

class Unet1D(nn.Module):
    """
    1D Temporal U-Net for Guided Trajectory Generation.
    Takes noisy trajectory (B, 1, T), timestep t, and optional goal & style condition.
    """
    def __init__(self, in_dim=1, base_channels=64, channel_mults=(1, 2, 4), num_conditions=3, cond_dim=32):
        super().__init__()
        self.base_channels = base_channels
        emb_dim = base_channels * 4

        # Timestep Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Class / Style Condition Embedding
        self.cond_emb = nn.Embedding(num_conditions + 1, cond_dim) # Extra index for unconditioned (CFG)
        self.goal_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.cond_proj = nn.Linear(cond_dim * 2, emb_dim)

        # U-Net Down / Mid / Up
        channels = [base_channels * m for m in channel_mults]
        
        self.init_conv = nn.Conv1d(in_dim, base_channels, kernel_size=3, padding=1)
        
        self.down1 = ResidualBlock1D(base_channels, channels[0], emb_dim)
        self.downsample1 = nn.Conv1d(channels[0], channels[0], kernel_size=3, stride=2, padding=1)
        
        self.down2 = ResidualBlock1D(channels[0], channels[1], emb_dim)
        self.downsample2 = nn.Conv1d(channels[1], channels[1], kernel_size=3, stride=2, padding=1)

        self.mid = ResidualBlock1D(channels[1], channels[2], emb_dim)

        self.upsample2 = nn.ConvTranspose1d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock1D(channels[1] * 2, channels[1], emb_dim)

        self.upsample1 = nn.ConvTranspose1d(channels[1], channels[0], kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock1D(channels[0] * 2, channels[0], emb_dim)

        self.final_conv = nn.Sequential(
            nn.Conv1d(channels[0], base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_dim, kernel_size=1)
        )

    def forward(self, x, t, cond=None, goal=None):
        # x: (B, 1, T)
        # t: (B,)
        # cond: (B,) class label or None
        # goal: (B, 1) continuous target goal position or None
        
        t_emb = self.time_mlp(t)
        
        # Build condition embedding
        B = x.shape[0]
        device = x.device

        if cond is None:
            cond = torch.full((B,), fill_value=3, dtype=torch.long, device=device) # Unconditioned index
        if goal is None:
            goal = torch.zeros((B, 1), dtype=torch.float32, device=device)

        c_emb = self.cond_emb(cond)
        g_emb = self.goal_mlp(goal)
        full_cond_emb = self.cond_proj(torch.cat([c_emb, g_emb], dim=-1))

        total_emb = t_emb + full_cond_emb

        # Forward U-Net
        x0 = self.init_conv(x)
        x1 = self.down1(x0, total_emb)
        x2 = self.down2(self.downsample1(x1), total_emb)
        
        mid = self.mid(self.downsample2(x2), total_emb)

        up2 = self.upsample2(mid)
        up2 = torch.cat([up2, x2], dim=1)
        up2 = self.up2(up2, total_emb)

        up1 = self.upsample1(up2)
        up1 = torch.cat([up1, x1], dim=1)
        up1 = self.up1(up1, total_emb)

        out = self.final_conv(up1)
        return out

if __name__ == "__main__":
    model = Unet1D()
    x = torch.randn(4, 1, 64)
    t = torch.randint(0, 1000, (4,))
    cond = torch.tensor([0, 1, 2, 0])
    goal = torch.randn(4, 1)
    out = model(x, t, cond, goal)
    print(f"Output shape: {out.shape}")
