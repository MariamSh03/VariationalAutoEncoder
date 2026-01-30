## Problem Formulation

### Input Space
The input data consists of grayscale images from the MNIST dataset. Each image is of size \(28 \times 28\) pixels and is flattened into a 784-dimensional real-valued vector. Pixel intensities are normalized to the range \([0,1]\). The data is unlabeled during training, as the task is fully unsupervised.

### Output Space
The model outputs a reconstructed image \(\hat{x} \in \mathbb{R}^{784}\) for each input image \(x\). In addition, the encoder produces parameters of a latent distribution, namely a mean vector \(\mu(x)\) and a diagonal covariance vector \(\sigma^2(x)\), defining a latent variable \(z \in \mathbb{R}^d\).

### Data Structure
MNIST images exhibit structured, low-dimensional variation despite their high-dimensional representation. Variability arises from stroke thickness, digit shape, orientation, and writing style, making the dataset suitable for latent-variable modeling.

### Core Modeling Challenge
The central challenge is to learn a compact and continuous latent representation that captures meaningful variations in the data while ensuring that the latent space follows a known prior distribution. This requires balancing reconstruction fidelity against latent-space regularization through the KL divergence term in the VAE objective.
