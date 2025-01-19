import tensorflow as tf
import numpy as np
import os
import pickle
from gan_vanilla3 import build_generator, build_discriminator, get_default_generator_parameters, get_default_discriminator_parameters
import learning_data
import utils
import datetime

# Define paths
data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified_users'
save_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_gan'

# Ensure save path exists
os.makedirs(save_path, exist_ok=True)

# A list of user names which are loaded
users_all = utils.folders_in_path(data_path)
users = [u for u in users_all]  # Load all users
users.sort()

# List of users we want to train a model for
users_test = users

# Hyper-parameters for loading data
data_parameters = {
    'users': users,  # Users whose data is loaded
    'labels': [0, 1, 2, 3, 4, 5, 6],  # Labels we want to use
    'combine_labels': {0: [0, 5]},  # Labels we want to combine
    'data_columns': ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],
    'stroke_labels': [0, 1],
    'time_scale_factors': [0.9, 1.1],
    'win_len': 180,
    'slide_len': 30,
    'window_normalization': 'tanh_scaled',
    'label_type': 'majority',
    'majority_thresh': 0.75,
    'validation_set': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
}

# Data is loaded and stored in this object
swimming_data = learning_data.LearningData()

# Load recordings from data_path
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=data_parameters['users'],
                        labels=data_parameters['labels'],
                        stroke_labels=data_parameters['stroke_labels'])

# Combine labels if specified
# print("Combining labels...")
# for new_label, old_labels in data_parameters['combine_labels'].items():
#     swimming_data.combine_labels(labels=old_labels, new_label=new_label)
# print("Labels combined.")

# Augment recordings
swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

# Compute sliding window locations
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile windows
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])

# GAN training parameters
gan_training_parameters = {
    'lr_generator': 0.0005,
    'lr_discriminator': 0.0001,
    'beta_1': 0.5,
    'batch_size': 64,
    'max_epochs': 100,
    'latent_dim': 100,
    'steps_per_epoch': 100,
    'noise_std': 0.01,
    'mirror_prob': 0.5,
    'random_rot_deg': 30
}

# Define GAN components
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']) + 2)  # Include labels + stroke_labels(+ 2) in input
generator_parameters = get_default_generator_parameters()
discriminator_parameters = get_default_discriminator_parameters()

generator = build_generator(gan_training_parameters['latent_dim'], input_shape, generator_parameters)
discriminator = build_discriminator(input_shape, discriminator_parameters)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(gan_training_parameters['lr_generator'], beta_1=gan_training_parameters['beta_1'])
discriminator_optimizer = tf.keras.optimizers.Adam(gan_training_parameters['lr_discriminator'], beta_1=gan_training_parameters['beta_1'])

# Draw users for each class
train_dict, val_dict = swimming_data.draw_train_val_dicts(users_test, users_per_class=data_parameters['validation_set'])
print("Validation dictionary: %s" % val_dict)

# Prepare validation data
x_val, y_val_cat, val_sample_weights = swimming_data.get_windows_dict(val_dict, return_weights=True)

try:
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
except Exception as e:
    print("Failed to reshape x_val:", e)

# Create training data generator
gen = swimming_data.batch_generator_dicts_3D(train_dict=train_dict,
                                          batch_size=gan_training_parameters['batch_size'],
                                          noise_std=gan_training_parameters['noise_std'],
                                          mirror_prob=gan_training_parameters['mirror_prob'],
                                          random_rot_deg=gan_training_parameters['random_rot_deg'])

# TensorBoard setup
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# Create a summary writer for TensorBoard
summary_writer = tf.summary.create_file_writer(log_dir)

# Training loop
def train_gan(generator, discriminator, data_generator, epochs, steps_per_epoch, x_val):
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # Get a batch of real data
            real_data, _ = next(data_generator)
            noise = np.random.normal(0, 1, (gan_training_parameters['batch_size'], gan_training_parameters['latent_dim']))

            # Train Discriminator
            with tf.GradientTape() as tape_disc:
                fake_data = generator(noise, training=True)

                # Discriminator losses with label smoothing
                real_labels = tf.ones((gan_training_parameters['batch_size'], 1), dtype=tf.float32) * 0.9
                fake_labels = tf.zeros((gan_training_parameters['batch_size'], 1), dtype=tf.float32)
                
                # Add noise to discriminator inputs
                real_data_noisy = real_data + 0.05 * tf.random.normal(tf.shape(real_data))
                fake_data_noisy = fake_data + 0.05 * tf.random.normal(tf.shape(fake_data))

                real_preds = discriminator(real_data_noisy, training=True)
                fake_preds = discriminator(fake_data_noisy, training=True)
                d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_preds)
                d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_preds)
                d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

            disc_gradients = tape_disc.gradient(d_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # Train Generator more frequently
            for _ in range(2):  # Train generator two times for each discriminator update
                with tf.GradientTape() as tape_gen:
                    fake_data = generator(noise, training=True)
                    fake_preds = discriminator(fake_data, training=True)
                    g_loss = tf.keras.losses.binary_crossentropy(real_labels, fake_preds)
                    g_loss = tf.reduce_mean(g_loss)

                gen_gradients = tape_gen.gradient(g_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

            # Log metrics to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('d_loss_real', tf.reduce_mean(d_loss_real), step=epoch * steps_per_epoch + step)
                tf.summary.scalar('d_loss_fake', tf.reduce_mean(d_loss_fake), step=epoch * steps_per_epoch + step)
                tf.summary.scalar('g_loss', g_loss, step=epoch * steps_per_epoch + step)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: D Loss Real = {tf.reduce_mean(d_loss_real)}, D Loss Fake = {tf.reduce_mean(d_loss_fake)}, G Loss = {g_loss}")

# Train the GAN
train_gan(generator, discriminator, data_generator=gen, epochs=gan_training_parameters['max_epochs'], steps_per_epoch=gan_training_parameters['steps_per_epoch'], x_val=x_val)

# Save models
generator.save(os.path.join(save_path, 'generator.h5'))
discriminator.save(os.path.join(save_path, 'discriminator.h5'))

# Saving the history and parameters
with open(os.path.join(save_path, 'train_val_dicts.pkl'), 'wb') as f:
    pickle.dump([train_dict, val_dict], f)

with open(os.path.join(save_path, 'data_parameters.pkl'), 'wb') as f:
    pickle.dump([data_parameters], f)
with open(os.path.join(save_path, 'gan_training_parameters.pkl'), 'wb') as f:
    pickle.dump([gan_training_parameters], f)
with open(os.path.join(save_path, 'generator_parameters.pkl'), 'wb') as f:
    pickle.dump([generator_parameters], f)
with open(os.path.join(save_path, 'discriminator_parameters.pkl'), 'wb') as f:
    pickle.dump([discriminator_parameters], f)
