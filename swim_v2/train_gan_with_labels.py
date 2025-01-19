import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Ensure eager execution
tf.config.run_functions_eagerly(True)

# Parameters
INPUT_DIM = 100
WIN_LEN = 180
FEATURES = 8

# Load dataset
def load_gan_dataset(gan_dataset_dir):
    data = []
    for style_dir in os.listdir(gan_dataset_dir):
        style_path = os.path.join(gan_dataset_dir, style_dir)
        if os.path.isdir(style_path):
            for file in os.listdir(style_path):
                file_path = os.path.join(style_path, file)
                if file.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    data.append(df.values)
    return np.array(data)

# Build generator
def build_generator(input_dim, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='relu', input_dim=input_dim),
        tf.keras.layers.Reshape((output_shape[0], output_shape[1])),
        tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(output_shape[1], kernel_size=3, strides=1, padding='same', activation='tanh'),
    ])
    return model

# Build discriminator
def build_discriminator(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(128, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv1D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    return model

# Training loop
def train_gan(generator, discriminator, gan, data, epochs, batch_size):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, INPUT_DIM))

        with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
            fake_data = generator(noise, training=True)

            # Discriminator losses
            real_labels = tf.ones((batch_size, 1), dtype=tf.float32)
            fake_labels = tf.zeros((batch_size, 1), dtype=tf.float32)
            real_preds = discriminator(real_data, training=True)
            fake_preds = discriminator(fake_data, training=True)
            d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_preds)
            d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_preds)
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

            # Generator loss
            g_loss = tf.keras.losses.binary_crossentropy(real_labels, fake_preds)
            g_loss = tf.reduce_mean(g_loss)

        # Apply gradients
        disc_gradients = tape_disc.gradient(d_loss, discriminator.trainable_variables)
        gen_gradients = tape_gen.gradient(g_loss, generator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss Real = {tf.reduce_mean(d_loss_real)}, D Loss Fake = {tf.reduce_mean(d_loss_fake)}, G Loss = {g_loss}")

# Main process
if __name__ == "__main__":
    GAN_DATASET_DIR = "gan_dataset"
    gan_data = load_gan_dataset(GAN_DATASET_DIR)
    gan_data_min = gan_data.min(axis=(0, 1), keepdims=True)
    gan_data_max = gan_data.max(axis=(0, 1), keepdims=True)
    gan_data = (gan_data - gan_data_min) / (gan_data_max - gan_data_min)

    generator = build_generator(INPUT_DIM, gan_data.shape[1:])
    discriminator = build_discriminator(gan_data.shape[1:])

    train_gan(generator, discriminator, None, gan_data, epochs=5000, batch_size=32)
