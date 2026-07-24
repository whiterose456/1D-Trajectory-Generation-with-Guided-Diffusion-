import os
import torch
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.model import Unet1D
from src.diffusion import GaussianDiffusion1D
from src.utils import plot_trajectories

st.set_page_config(
    page_title="1D Trajectory Generation with Guided Diffusion",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 1D Trajectory Generation with Guided Diffusion")
st.markdown("""
Interactive Portfolio Demonstration of **Classifier-Free Guided Diffusion** applied to 1D Motion Trajectory Planning.
Select desired motion profile styles, target goals, and Classifier-Free Guidance (CFG) scales below.
""")

# Sidebar Controls
st.sidebar.header("Configuration & Parameters")
ckpt_path = "checkpoints/guided_diffusion_1d.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_diffusion_model():
    model = Unet1D(in_dim=1, base_channels=64, channel_mults=(1, 2, 4)).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    diffusion = GaussianDiffusion1D(model, timesteps=500, device=device)
    return diffusion

diffusion = load_diffusion_model()

# User Inputs
style_mapping = {
    "Direct / Linear Smooth": 0,
    "S-Curve / Oscillatory": 1,
    "Parabolic Arc / Overshoot": 2
}
selected_style_name = st.sidebar.selectbox("Motion Trajectory Style", list(style_mapping.keys()))
condition_val = style_mapping[selected_style_name]

goal_val = st.sidebar.slider("Target Destination Goal (x_end)", min_value=-1.0, max_value=1.0, value=0.6, step=0.05)
cfg_scale = st.sidebar.slider("Classifier-Free Guidance (CFG) Scale (w)", min_value=0.0, max_value=7.0, value=3.5, step=0.5)
num_samples = st.sidebar.slider("Number of Trajectories", min_value=1, max_value=10, value=4, step=1)
use_ddim = st.sidebar.checkbox("Use Fast DDIM Sampler (50 steps)", value=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Generation")
    if st.button("✨ Generate Trajectories", type="primary"):
        with st.spinner("Diffusion model sampling trajectories..."):
            conds = torch.full((num_samples,), condition_val, dtype=torch.long, device=device)
            goals = torch.full((num_samples, 1), goal_val, dtype=torch.float32, device=device)

            trajectories = diffusion.sample(
                shape=(num_samples, 1, 64),
                cond=conds,
                goal=goals,
                cfg_scale=cfg_scale,
                ddim=use_ddim,
                ddim_steps=50
            )

            fig = plot_trajectories(
                trajectories,
                conditions=conds,
                goals=goals,
                title=f"Style: {selected_style_name} | Goal: {goal_val:.2f} | CFG: {cfg_scale}"
            )
            st.pyplot(fig)

with col2:
    st.subheader("Architecture Details")
    st.info("""
    - **Backbone**: 1D Temporal Residual U-Net
    - **Conditioning**: Joint Timestep + Style Embedding + Scalar Goal Projection
    - **Sampling**: DDPM / DDIM Fast Sampling (50 steps)
    - **CFG Rule**: $\\hat{\\epsilon} = \\epsilon_{uncond} + w \\cdot (\\epsilon_{cond} - \\epsilon_{uncond})$
    """)
