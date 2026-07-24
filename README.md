# Guided Diffusion for 1D Trajectory Generation & Motion Planning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An applied deep learning framework implementing **Conditional Temporal Sequence Generation via Guided Diffusion** for 1D robotic motion planning and trajectory control.

This repository implements applied methodologies grounded in seminal generative diffusion research:
- **Diffusion-based Planning**: *Diffuser: Planning with Diffusion for Sequential Decision Making* (Janner et al., ICML 2022)
- **Classifier-Free Guidance (CFG)**: *Classifier-Free Diffusion Guidance* (Ho & Salimans, NeurIPS Workshop 2021)
- **Fast Sampling**: *Denoising Diffusion Implicit Models (DDIM)* (Song et al., ICLR 2021)

---

## 🔬 Overview & Applied Methodology

Traditional robotic motion planning relies on trajectory optimization algorithms (e.g., CHOMP, TrajOpt) or RRT variants, which can suffer from high computational complexity or local minima in constrained environments. 

This project formulates **1D trajectory synthesis as a conditional generative process**. By modeling trajectory dynamics $\mathbf{x}_0 \in \mathbb{R}^{1 \times T}$ as a temporal denoising task, we utilize a **1D Temporal Residual U-Net** to generate smooth, physically constrained kinematic trajectories conditioned on style dynamics and continuous goal destinations.

```
                    Noise Injection (Forward Process)
  Trajectory (x_0) -----------------------------------> Noisy Trajectory (x_t)
                                                                |
                                                                v
  Goal / Style Condition (c, g) ----> 1D U-Net Denoiser <-------+
                                           |
                                           v
                              Denoised Trajectory (x_0)
```

---

## 🌟 Key Components & Research Implementation

### 1. 1D Temporal Denoising Backbone (`src/model.py`)
- **Residual 1D U-Net Architecture**: Adapted from 2D image diffusion models to 1D temporal signals ($B, C, T$), preserving temporal locality via 1D convolutions and Group Normalization.
- **Joint Condition Embedding**: Incorporates sinusoidal timestep projections, discrete style embeddings (e.g., linear, oscillatory, parabolic profile), and continuous goal state MLP projections.

### 2. Guided Diffusion Engine (`src/diffusion.py`)
- **DDPM & DDIM Dynamics**: Standard Gaussian noise schedule with linear beta schedules. Supports 50-step deterministic DDIM sampling for fast inference.
- **Classifier-Free Guidance (CFG)**: Evaluates both conditioned $\mathbf{\epsilon}_\theta(\mathbf{x}_t, t, c, g)$ and unconditioned $\mathbf{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset, \emptyset)$ passes to steer generation without requiring an explicit classifier network.

### 3. Synthetic Trajectory Benchmark Dataset (`src/dataset.py`)
- Simulates realistic 1D kinematic trajectories with varying boundary constraints, endpoint goals $x_{\text{end}} \in [-1.0, 1.0]$, and noise disturbances.

### 4. Interactive Research Dashboard (`app.py`)
- Web interface allowing real-time trajectory visualization, dynamic target endpoint manipulation, and empirical study of Classifier-Free Guidance scale parameter $w$.

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/whiterose456/1D-Trajectory-Generation-with-Guided-Diffusion-.git
cd 1D-Trajectory-Generation-with-Guided-Diffusion-
pip install -r requirements.txt
```

### 2. Model Training

Train the temporal diffusion model using synthetic kinematic trajectory distributions:

```bash
python train.py
```

### 3. Conditional Trajectory Sampling

Sample trajectories conditioned on target endpoint $x_{\text{end}} = 0.75$ using DDIM sampling:

```bash
python sample.py --condition 1 --goal 0.75 --cfg_scale 3.5 --ddim
```

### 4. Interactive Dashboard

Launch the interactive research UI:

```bash
streamlit run app.py
```

---

## 📐 Mathematical Formulation

### 1. Forward Diffusion Process
Following Ho et al. (2020), noise is added across timesteps $t \in [1, T]$ according to variance schedule $\beta_t$:
$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}\right), \quad \text{where } \alpha_t = 1 - \beta_t, \ \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

### 2. Classifier-Free Guidance Score
The score function estimate $\mathbf{s}_\theta(\mathbf{x}_t, c, g) \approx \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \mid c, g)$ is extrapolated using guidance scale $w$:
$$\hat{\mathbf{\epsilon}}_\theta(\mathbf{x}_t, t, c, g) = (1 + w) \mathbf{\epsilon}_\theta(\mathbf{x}_t, t, c, g) - w \mathbf{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset, \emptyset)$$

---

## 📚 References & Citation

1. Janner, M., Du, Y., Tenenbaum, J. B., & Levine, S. (2022). *Planning with Diffusion for Sequential Decision Making*. International Conference on Machine Learning (ICML).
2. Ho, J., & Salimans, T. (2021). *Classifier-Free Diffusion Guidance*. NeurIPS Workshop on Generative AI.
3. Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. International Conference on Learning Representations (ICLR).
4. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. Advances in Neural Information Processing Systems (NeurIPS).

---

## 📄 License
Distributed under the MIT License.