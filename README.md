# Variational Autoencoder (VAE) for MNIST

A comprehensive implementation of Variational Autoencoders (VAEs) for unsupervised representation learning and generative modeling on the MNIST dataset. This project includes detailed experiments comparing VAEs with deterministic autoencoders, ablation studies on latent dimensions and β-parameters, and extensive visualizations of the learned latent space.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Experiments](#experiments)
- [Documentation](#documentation)
- [License](#license)

## Overview

This project implements a Variational Autoencoder (VAE) using TensorFlow/Keras to learn a probabilistic latent representation of MNIST handwritten digits. The VAE framework enables:

- **Unsupervised representation learning**: Learn meaningful latent codes without labels
- **Generative modeling**: Sample new digits from the learned prior distribution
- **Latent space analysis**: Visualize and interpret the learned representations
- **Smooth interpolation**: Generate intermediate digits between two examples

The implementation follows the standard VAE formulation with an encoder-decoder architecture, reparameterization trick, and ELBO optimization objective.

## Features

- **Full VAE Implementation**: Custom VAE model with encoder, decoder, and sampling layer
- **Baseline Comparison**: Deterministic autoencoder for comparison
- **Ablation Studies**: 
  - Latent dimension experiments (d ∈ {2, 8, 16})
  - β-VAE experiments (β ∈ {0.5, 1.0, 2.0, 4.0})
- **Comprehensive Visualizations**:
  - Reconstruction comparisons
  - Generated samples from prior
  - 2D latent space visualization (with PCA for d>2)
  - Latent interpolation sequences
  - Training curves
- **Reproducible**: Fixed random seeds and documented hyperparameters

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MariamSh03/VariationalAutoEncoder.git
cd VariationalAutoEncoder
```

2. Install required packages:
```bash
pip install tensorflow numpy matplotlib
```

## Usage

### Basic VAE Training (d=2)

Run the main experiment script:

```bash
python code,py
```

This will:
- Train a VAE with latent dimension d=2
- Generate reconstructions, samples, and latent visualizations
- Train a baseline autoencoder for comparison

### Ablation Study (d=16)

Run the ablation experiment:

```bash
python exp1.py
```

This trains a VAE with latent dimension d=16 and includes PCA-based 2D visualization of the latent space.

### Custom Configuration

Modify the hyperparameters in the scripts:

```python
LATENT_DIM = 2      # Latent dimension: 2, 8, or 16
BETA = 1.0          # KL weight: 0.5, 1.0, 2.0, or 4.0
EPOCHS = 25         # Number of training epochs
LR = 1e-3           # Learning rate
BATCH_SIZE = 128    # Batch size
```

## Project Structure

```
VariationalAutoEncoder/
│
├── code,py                 # Main VAE implementation (d=2)
├── exp1.py                 # Ablation study (d=16)
│
├── Documentation/
│   ├── problemFormulation.md    # Problem statement and data structure
│   ├── Method_Selection.md      # Theoretical justification for VAE
│   ├── architecture.md          # Model architecture details
│   ├── step5.md                 # Experimental setup and evaluation plan
│   ├── results.md               # Experimental results and analysis
│   ├── 2vs6.md                  # Latent dimension ablation analysis
│   └── final_Presentation.pdf   # Project presentation
│
├── results/                # Generated figures and visualizations
│   ├── Figure_1.png
│   ├── Figure_2.png
│   └── ...
│
├── 16/                     # Results for d=16 experiments
│   └── Figure_*.png
│
├── vae_diagram.png         # Architecture diagram
└── README.md              # This file
```

## Key Results

### Reconstruction Quality

- **VAE (d=2)**: Slightly blurry but recognizable reconstructions
- **Baseline AE**: Sharper reconstructions (test loss: 0.1855 BCE)
- **VAE (d=16)**: Improved reconstruction fidelity (test loss: ~75 BCE)

### Latent Space Structure

- **d=2**: Forms structured clusters when visualized, with overlaps for similar digits
- **d=16**: Higher capacity allows better separation and reconstruction quality
- Latent interpolation produces smooth transitions between digits

### Generative Sampling

- Successfully generates plausible MNIST digits from N(0,I) prior
- Sample quality improves with higher latent dimensions
- Demonstrates VAE's advantage over deterministic autoencoders for generation

## Architecture

### Encoder
- Input: 28×28×1 grayscale image
- Architecture: 
  - Conv2D(32, 3×3, stride=2) → 14×14
  - Conv2D(64, 3×3, stride=2) → 7×7
  - Flatten → Dense(128)
  - Two parallel heads: μ(x) and log σ²(x)

### Decoder
- Input: Latent code z ∈ ℝᵈ
- Architecture:
  - Dense(7×7×64) → Reshape(7,7,64)
  - Conv2DTranspose(64, 3×3, stride=2) → 14×14
  - Conv2DTranspose(32, 3×3, stride=2) → 28×28
  - Conv2DTranspose(1, 3×3) → 28×28×1 (sigmoid)

### Loss Function

The VAE optimizes the Evidence Lower Bound (ELBO):

```
L_VAE = E[-log p(x|z)] + β · D_KL(q(z|x) || p(z))
```

- **Reconstruction loss**: Binary cross-entropy between input and reconstruction
- **KL divergence**: Regularizes latent distribution toward N(0,I) prior
- **β parameter**: Controls trade-off between reconstruction and regularization

## Experiments

### Experiment A: Reconstruction Quality
- Visual comparison: Original vs reconstructed images
- Quantitative: Test reconstruction loss (BCE)
- Comparison: VAE vs deterministic autoencoder

### Experiment B: Latent Space Structure
- 2D visualization of latent means μ(x)
- Color-coded by digit labels (for interpretation only)
- PCA projection for higher-dimensional latent spaces

### Experiment C: Generative Sampling
- Sample from prior: z ~ N(0,I)
- Generate 64-digit grids
- Latent interpolation between pairs of digits

### Ablation Studies
1. **Latent Dimension**: d ∈ {2, 8, 16}
2. **β-VAE**: β ∈ {0.5, 1.0, 2.0, 4.0}

## Documentation

Detailed documentation is available in the markdown files:

- **Problem Formulation**: Data structure and modeling challenges
- **Method Selection**: Theoretical justification for VAE approach
- **Architecture**: Detailed model architecture and training procedure
- **Experimental Plan**: Setup, metrics, and evaluation protocol
- **Results**: Comprehensive analysis of experimental findings

## Technical Details

- **Framework**: TensorFlow 2.x / Keras
- **Dataset**: MNIST (60,000 training, 10,000 test images)
- **Optimizer**: Adam (learning rate: 1e-3)
- **Batch Size**: 128
- **Epochs**: 25 (configurable)
- **Reproducibility**: Fixed random seeds (SEED=42)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{variationalautoencoder2024,
  title={Variational Autoencoder for MNIST},
  author={MariamSh03},
  year={2024},
  url={https://github.com/MariamSh03/VariationalAutoEncoder}
}
```

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- MNIST dataset by Yann LeCun and collaborators
- TensorFlow/Keras team for the deep learning framework
- VAE paper: "Auto-Encoding Variational Bayes" by Kingma & Welling (2014)

---

**Note**: This project was developed as part of a course on probabilistic generative models and unsupervised learning.

