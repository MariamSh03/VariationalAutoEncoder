## Step 5: Experimental Demonstration and Evaluation Plan

This step defines the experiments, baselines, metrics, and artifacts (figures/tables) that will be included to demonstrate the correctness and insight of the VAE approach on MNIST.

### 5.1 Experimental Setup
**Data:** MNIST (28×28 grayscale), normalized to [0,1], reshaped to (28, 28, 1). Labels are not used for training (unsupervised).

**Models:**
1. **VAE (primary model)** with latent dimension \(d \in \{2, 8, 16\}\)
2. **Baseline Autoencoder (AE)** with the same encoder/decoder capacity but without KL regularization and without stochastic sampling.

**Training:**
- Optimizer: Adam
- Batch size: 128
- Epochs: 20–50
- Fixed seeds and recorded hyperparameters for reproducibility.

---

### 5.2 Core Experiments

#### Experiment A — Reconstruction Quality (Qualitative + Quantitative)
**Goal:** Verify the model learns a meaningful latent representation that preserves input information.

**What to show:**
- A grid of original vs reconstructed images (e.g., 16–32 samples).
- Plot training curves:
  - Total VAE loss
  - Reconstruction loss
  - KL divergence

**Metrics:**
- Mean reconstruction loss (BCE) on test set.

**Expected observation:**
- Reconstructions are recognizable but slightly blurry, especially for smaller latent dimensions.

---

#### Experiment B — Latent Space Structure (Representation Learning)
**Goal:** Evaluate whether the latent space is structured and smooth.

**Protocol:**
- Train with \(d=2\) to enable direct visualization.

**What to show:**
- Scatter plot of 2D latent means \(\mu(x)\) for test samples.
- (Optional for visualization only) color points by digit label to interpret clustering.

**Expected observation:**
- Samples form clusters/continuous manifolds for digits; overlaps reflect ambiguous handwriting.

---

#### Experiment C — Generative Sampling (Data Generation)
**Goal:** Demonstrate that sampling from the prior produces realistic MNIST-like digits.

**Protocol:**
- Sample \(z \sim \mathcal{N}(0, I)\), decode to images.

**What to show:**
- A grid of generated samples (e.g., 8×8).
- (Optional) Latent interpolation:
  - Choose two test images, encode to \(z_1, z_2\), interpolate linearly and decode.

**Expected observation:**
- Generated digits look plausible; interpolation changes shape smoothly.

---

### 5.3 Baseline and Ablations (Required for insight)

#### Baseline — Deterministic Autoencoder (AE)
**Why:** Shows what the KL term changes (structure and smoothness vs raw reconstruction).

**Comparison:**
- AE typically reconstructs slightly sharper but has an unstructured latent space and poor sampling.

#### Ablation 1 — Latent Dimension \(d\)
Train VAE with \(d \in \{2, 8, 16\}\) and compare:
- Reconstruction loss (test)
- Visual quality of generated samples
- Latent visualization (only for d=2)

#### Ablation 2 — β-VAE (KL Weighting)
Modify objective:
\[
\mathcal{L} = \text{Recon} + \beta \cdot KL
\]
Test \(\beta \in \{0.5, 1.0, 2.0, 4.0\}\).

**Expected trade-off:**
- Higher β → more regularized latent space, better sampling smoothness, worse reconstruction.

---

### 5.4 Deliverable Artifacts (What your report/slides will include)
Minimum set (easy and strong):
1. Original vs reconstructed image grid
2. Generated sample grid
3. Loss curves (total, recon, KL)
4. 2D latent plot (d=2)
5. A small table:
   - d, β, test recon loss, mean KL

---

### 5.5 Reproducibility Checklist
- Fixed seeds (NumPy + TensorFlow)
- Document: batch size, epochs, learning rate, d, β, architecture
- Save model weights and generated figures
- Keep a single script/notebook that reproduces all plots
