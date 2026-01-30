import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# Data (MNIST via TensorFlow)
# ----------------------------
(x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]  # (N,28,28,1)
x_test = x_test[..., None]

BATCH_SIZE = 128

# VAE training dataset: only x is needed
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000, seed=SEED).batch(BATCH_SIZE)

# Keep labels ONLY for plotting latent space; not used for training
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# AE training dataset: MUST be (x, x)
ae_train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train)) \
    .shuffle(60000, seed=SEED) \
    .batch(BATCH_SIZE)

ae_test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(BATCH_SIZE)

# ----------------------------
# Model builders
# ----------------------------
def build_encoder(latent_dim: int):
    inp = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)   # 14x14
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)     # 7x7
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)
    return keras.Model(inp, [z_mean, z_logvar], name="encoder")

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

def build_decoder(latent_dim: int):
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(inp)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)  # 14x14
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)  # 28x28
    out = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inp, out, name="decoder")

# ----------------------------
# VAE Model (custom train_step)
# ----------------------------
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.sampling = Sampling()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x, training=True)
            z = self.sampling([z_mean, z_logvar])
            x_hat = self.decoder(z, training=True)

            # Reconstruction loss (BCE per-pixel)
            bce = keras.losses.binary_crossentropy(x, x_hat)
            recon_loss = tf.reduce_mean(tf.reduce_sum(bce, axis=(1, 2)))  # sum over H,W

            # KL divergence to N(0,I)
            kl = -0.5 * (1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl, axis=1))

            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x, training=False):
        z_mean, z_logvar = self.encoder(x, training=training)
        z = self.sampling([z_mean, z_logvar])
        return self.decoder(z, training=training)

# ----------------------------
# Baseline Autoencoder (deterministic)
# ----------------------------
def build_autoencoder(latent_dim: int):
    inp = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, activation="linear", name="z")(x)

    x = layers.Dense(7 * 7 * 64, activation="relu")(z)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)

    return keras.Model(inp, out, name="autoencoder")

# ----------------------------
# Training helpers
# ----------------------------
def plot_history(history, title="Training Curves"):
    plt.figure()
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def show_reconstructions(model, x, n=16, title="Original vs Reconstruction"):
    x = x[:n]
    x_hat = model.predict(x, verbose=0)
    plt.figure(figsize=(8, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, n, n + i + 1)
        plt.imshow(x_hat[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

def show_samples(decoder, latent_dim, n=64, title="Samples from N(0,I)"):
    z = tf.random.normal((n, latent_dim))
    x_gen = decoder.predict(z, verbose=0)
    side = int(np.sqrt(n))
    plt.figure(figsize=(6, 6))
    for i in range(n):
        plt.subplot(side, side, i + 1)
        plt.imshow(x_gen[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

def plot_latent_2d(encoder, x, y, max_points=5000, title="2D Latent Means (colored by label)"):
    x = x[:max_points]
    y = y[:max_points]
    z_mean, _ = encoder.predict(x, verbose=0)
    plt.figure()
    plt.scatter(z_mean[:, 0], z_mean[:, 1], s=2, c=y)
    plt.title(title)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()

def latent_interpolation(encoder, decoder, x_a, x_b, steps=10, title="Latent interpolation"):
    z_a, _ = encoder.predict(x_a[None, ...], verbose=0)
    z_b, _ = encoder.predict(x_b[None, ...], verbose=0)
    z_a = z_a[0]
    z_b = z_b[0]
    alphas = np.linspace(0.0, 1.0, steps)
    zs = np.stack([(1 - a) * z_a + a * z_b for a in alphas], axis=0)
    imgs = decoder.predict(zs, verbose=0)

    plt.figure(figsize=(steps, 2))
    for i in range(steps):
        plt.subplot(1, steps, i + 1)
        plt.imshow(imgs[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# ----------------------------
# NEW: d=16 latent plot (PCA 2D) with legend on the RIGHT
# ----------------------------
def plot_latent_d16_pca_legend_right(encoder, x, y, max_points=10000,
                                    title="Latent d=16 → PCA 2D (colored by digit)"):
    x = x[:max_points]
    y = y[:max_points]

    # z_mean: (N,16)
    z_mean, _ = encoder.predict(x, verbose=0)

    # PCA with numpy SVD -> 2D
    Z = z_mean - z_mean.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    Z2 = Z @ Vt[:2].T  # (N,2)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10", 10)

    for d in range(10):
        idx = (y == d)
        ax.scatter(Z2[idx, 0], Z2[idx, 1], s=6, alpha=0.7, color=cmap(d), label=str(d))

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # legend on right
    ax.legend(title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.show()

# ============================================================
# Option A Ablation: Train + Evaluate VAE with LATENT_DIM = 16
# ============================================================
LATENT_DIM_ABL = 16
BETA_ABL = 1.0
EPOCHS_ABL = 25
LR_ABL = 1e-3

np.random.seed(SEED)
tf.random.set_seed(SEED)

vae_train_ds_abl = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000, seed=SEED).batch(BATCH_SIZE)

def evaluate_vae(vae_model, encoder_model, decoder_model, x_test, beta=1.0, batch_size=128):
    n = x_test.shape[0]
    recon_losses, kl_losses, total_losses = [], [], []

    for i in range(0, n, batch_size):
        x = x_test[i:i+batch_size]
        z_mean, z_logvar = encoder_model(x, training=False)

        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_logvar) * eps
        x_hat = decoder_model(z, training=False)

        bce = keras.losses.binary_crossentropy(x, x_hat)   # (B,28,28)
        recon = tf.reduce_sum(bce, axis=(1, 2))            # (B,)
        recon = tf.reduce_mean(recon).numpy()

        kl = -0.5 * (1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
        kl = tf.reduce_sum(kl, axis=1)
        kl = tf.reduce_mean(kl).numpy()

        total = recon + beta * kl

        recon_losses.append(recon)
        kl_losses.append(kl)
        total_losses.append(total)

    return float(np.mean(recon_losses)), float(np.mean(kl_losses)), float(np.mean(total_losses))

# Train VAE for d=16
encoder_16 = build_encoder(LATENT_DIM_ABL)
decoder_16 = build_decoder(LATENT_DIM_ABL)

vae_16 = VAE(encoder_16, decoder_16, beta=BETA_ABL)
vae_16.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_ABL))

hist_vae_16 = vae_16.fit(vae_train_ds_abl, epochs=EPOCHS_ABL, verbose=1)

# Curves
plot_history(hist_vae_16, title=f"VAE Training Curves (d={LATENT_DIM_ABL}, beta={BETA_ABL})")

# Recon + sampling + interpolation
show_reconstructions(vae_16, x_test, n=16, title=f"VAE d={LATENT_DIM_ABL}: Original (top) vs Reconstruction (bottom)")
show_samples(decoder_16, LATENT_DIM_ABL, n=64, title=f"VAE d={LATENT_DIM_ABL}: Samples from N(0, I)")
latent_interpolation(encoder_16, decoder_16, x_test[0], x_test[1], steps=12,
                     title=f"VAE d={LATENT_DIM_ABL}: Latent Interpolation")

# NEW: plot d=16 latent space (PCA 2D) + legend on right
plot_latent_d16_pca_legend_right(
    encoder=encoder_16,
    x=x_test,
    y=y_test,
    max_points=10000,
    title=f"VAE Latent (d={LATENT_DIM_ABL}) → PCA 2D with legend"
)

# Quantitative evaluation
vae16_recon, vae16_kl, vae16_total = evaluate_vae(
    vae_model=vae_16,
    encoder_model=encoder_16,
    decoder_model=decoder_16,
    x_test=x_test,
    beta=BETA_ABL,
    batch_size=BATCH_SIZE
)

print("\n" + "="*60)
print("VAE ABLATION RESULTS (TEST SET)")
print(f"Latent dim (d):       {LATENT_DIM_ABL}")
print(f"Beta (β):             {BETA_ABL}")
print(f"Epochs:               {EPOCHS_ABL}")
print(f"Learning rate:        {LR_ABL}")
print("-"*60)
print(f"Test recon loss (BCE, sum over pixels): {vae16_recon:.4f}")
print(f"Test KL divergence (per sample):        {vae16_kl:.4f}")
print(f"Test total loss (recon + β*KL):         {vae16_total:.4f}")
print("="*60 + "\n")

# Optional: save weights so you don't retrain next time
# vae_16.save_weights("vae_d16.keras")
# encoder_16.save_weights("encoder_d16.keras")
# decoder_16.save_weights("decoder_d16.keras")
