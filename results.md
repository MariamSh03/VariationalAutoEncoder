## Step 6: Results and Analysis

This section reports the experimental results of the VAE on MNIST and compares them to a deterministic Autoencoder (AE) baseline. Unless otherwise stated, results are reported for latent dimension \(d=2\) and \(\beta=1.0\).

### 6.1 Training Dynamics
The VAE training curves show a rapid decrease in total loss during the first few epochs followed by gradual convergence. The reconstruction loss decreases steadily, while the KL term increases early and then stabilizes, indicating that the encoder learns to balance reconstruction accuracy with matching the prior \(p(z)=\mathcal{N}(0,I)\). This behavior is consistent with ELBO optimization: the model first learns to reconstruct, then adjusts the posterior to satisfy the regularization constraint.

### 6.2 Reconstruction Quality (VAE vs AE)
Qualitatively, VAE reconstructions preserve digit identity but appear slightly blurry compared to the deterministic AE. This is expected: the KL term restricts the latent space to remain close to the Gaussian prior, which reduces the model’s ability to memorize fine details. In contrast, the AE is optimized purely for reconstruction and therefore can achieve sharper reconstructions.

Quantitatively, the AE test reconstruction loss is:
- **AE test reconstruction loss (BCE): 0.1855**

This lower loss supports the qualitative observation that the AE prioritizes reconstruction fidelity.

### 6.3 Generative Sampling
The VAE generates plausible MNIST digits by sampling \(z \sim \mathcal{N}(0,I)\) and decoding. The generated samples show recognizable digits with some ambiguous or blurry cases, which is expected for a low-dimensional latent space (\(d=2\)) and a simple decoder distribution.

This experiment demonstrates an advantage of VAEs over deterministic autoencoders: because the VAE enforces a known latent prior, sampling from the prior produces meaningful outputs. In contrast, the AE latent space is not constrained to match a known distribution, so random latent sampling is not guaranteed to produce valid digits.

### 6.4 Latent Space Structure (2D Visualization)
For \(d=2\), plotting the latent means \(\mu(x)\) for test images reveals a structured embedding space. When colored by digit label (used only for interpretation, not training), digits form partially separated clusters with overlapping regions. Overlaps are expected because some handwritten digits are visually similar (e.g., 4 vs 9, 3 vs 5), and the VAE learns a continuous manifold rather than perfectly separable class clusters.

### 6.5 Latent Interpolation
Interpolation between two encoded test images produces a smooth transformation of digit appearance. This provides evidence that the latent space learned by the VAE is continuous and semantically meaningful: small changes in \(z\) correspond to gradual changes in the decoded image, which is a key goal of probabilistic latent-variable modeling.

### 6.6 Summary of Key Findings
- The VAE successfully learns a structured latent space on MNIST and supports sampling and interpolation.
- The AE achieves better reconstruction loss (**0.1855 BCE**) and sharper reconstructions, but lacks a principled latent prior.
- The VAE’s KL regularization trades reconstruction quality for latent structure and generative capability.

