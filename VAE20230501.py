import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import utils


class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_layers = [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=-1)
        return mean, logvar


# 解碼器
class Decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder_layers = [
            tf.keras.layers.Dense(units=7 * 7 * 32, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME",
                                            activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME",
                                            activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(x)
        return x
#損失函數
def compute_loss(encoder, decoder, x):
    mean, logvar = encoder(x)
    eps = tf.random.normal(shape=mean.shape)
    z = eps * tf.exp(logvar * 0.5) + mean
    x_recon = decoder(z)
    recon_loss = tf.reduce_sum(tf.square(x - x_recon), axis=[1, 2, 3])
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
    loss = tf.reduce_mean(recon_loss + kl_loss)
    return loss


# 梯度下降優化器
def train_step(encoder, decoder, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(encoder, decoder, x)
    gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return loss


d = utils.DataLoader()
sketchdata, realdata = d.load_data(metrics='[-1,1]')  # metrics是可以改成'[-1,1]'，代表圖片標準化後的值區間
test_sketchdata, test_realdata = d.load_testing_data(metrics='[-1,1]')




def generate_data(decoder, n_samples, latent_dim):
    z = tf.random.normal(shape=(n_samples, latent_dim))
    x_generated = decoder(z)
    return x_generated

# 創建VAE
latent_dim = 2
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


num_epochs = 10000
batch_size = 128
num_batches = sketchdata.shape[0] // batch_size

loss = None
for epoch in range(num_epochs):
    for batch_idx in range(num_batches):

        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        x_batch = sketchdata[start_idx:end_idx]
        loss = train_step(encoder, decoder, x_batch, optimizer);print(loss)
    print("Epoch: {}, Loss: {}".format(epoch+1, loss))

x_generated = generate_data(decoder, 16, latent_dim)

# 顯示生成的數據
fig, axs = plt.subplots(4, 4, figsize=(6, 6))
axs = axs.flatten()
for i in range(16):
    axs[i].imshow(x_generated[i, :, :, 0], cmap="gray")
    axs[i].axis("off")
plt.tight_layout()
plt.show()