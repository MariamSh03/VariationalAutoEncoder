## Step 4: Model Architecture and Training Procedure

### Implementation Framework
This project is implemented in **TensorFlow (Keras)** using the built-in MNIST loader `tf.keras.datasets.mnist.load_data()`. Images are normalized to \([0,1]\) and reshaped to include a channel dimension, resulting in tensors of shape \((28, 28, 1)\). Although MNIST provides labels, they are **not used for training**, since the VAE is trained in an unsupervised manner.

### Model Architecture

#### Encoder \(q_\phi(z|x)\)
The encoder maps an input image \(x \in \mathbb{R}^{28 \times 28 \times 1}\) to a Gaussian distribution in latent space. It outputs:
- A mean vector \(\mu(x) \in \mathbb{R}^d\)
- A log-variance vector \(\log \sigma^2(x) \in \mathbb{R}^d\)

A simple convolutional encoder is used:
- Conv2D layers with stride 2 for downsampling
- Flatten
- Dense layer(s)
- Two parallel Dense heads for \(\mu\) and \(\log \sigma^2\)

#### Reparameterization Trick
To enable backpropagation through stochastic sampling, the latent variable is computed as:
\[
z = \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
\]
where \(\sigma = \exp\left(\frac{1}{2}\log \sigma^2\right)\).

#### Decoder \(p_\theta(x|z)\)
The decoder maps latent codes \(z \in \mathbb{R}^d\) back to the image space and outputs \(\hat{x}\). A convolutional transpose (deconvolution) decoder is used:
- Dense layer to expand from \(z\) to a low-resolution feature map
- Reshape
- Conv2DTranspose layers to upsample back to \(28 \times 28\)
- Output layer with **sigmoid activation** to model pixel values in \([0,1]\)

### Objective Function (ELBO)
The VAE is trained by minimizing the negative Evidence Lower Bound (ELBO):
\[
\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[-\log p_\theta(x|z)]}_{\text{reconstruction loss}}
+ \underbrace{D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{KL regularization}}
\]
- The reconstruction term is implemented using **binary cross-entropy (BCE)** between \(x\) and \(\hat{x}\).
- The KL term is computed analytically for diagonal Gaussians with prior \(p(z)=\mathcal{N}(0,I)\).

### Training Setup
- Optimizer: **Adam**
- Batch size: 128
- Number of epochs: 20–50 (depending on convergence)
- Latent dimensions tested: \(d \in \{2, 8, 16\}\)
- Reproducibility: fixed random seeds for NumPy/TensorFlow where possible, and all hyperparameters recorded.

### Training Procedure Summary
1. Load MNIST using `tf.keras.datasets.mnist.load_data()`
2. Normalize and reshape images to \((28, 28, 1)\)
3. Build encoder, sampling layer, and decoder
4. Train by minimizing reconstruction loss + KL divergence
5. Save the trained model and generate:
   - reconstructions
   - samples from prior \(z \sim \mathcal{N}(0,I)\)
   - latent embeddings for analysis
