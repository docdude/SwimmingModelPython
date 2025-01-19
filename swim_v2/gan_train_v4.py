import tensorflow as tf
import numpy as np
import os
import pickle
from gan_vanilla4 import build_generator, build_discriminator, get_default_generator_parameters, get_default_discriminator_parameters
import learning_data
import utils
import datetime
import matplotlib.pyplot as plt


# Define paths
data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified_users'
save_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_gan'

# Ensure save path exists
os.makedirs(save_path, exist_ok=True)

# A list of user names which are loaded
users_all = utils.folders_in_path(data_path)
users = [u for u in users_all]  # Load all users
users.sort(key=int)

# List of users we want to train a model for
users_test = users

# Hyper-parameters for loading data
data_parameters = {
    'users': users,  # Users whose data is loaded
    'labels': [0, 1, 2, 3, 4, 5, 6],  # Labels for swim styles and transitions
    'combine_labels': {0: [0, 5]},  # Combine '0' and '5' as 'transition' for swim style transitions
    'data_columns': ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],  # Sensor data columns
    'stroke_labels': ['stroke_labels'],  # Binary stroke labels: 0 for no stroke, 1 for stroke
    'time_scale_factors': [0.9, 1.1],  # Scale timestamps by 10% faster/slower
    'win_len': 180,  # Window length in time steps
    'slide_len': 30,  # Slide length for overlapping windows
    'window_normalization': 'tanh_scaled',  # Normalization method for windowed data
    'label_type': 'majority',  # Labeling strategy for overlapping windows
    'majority_thresh': 0.75,  # Threshold for majority labeling
    'validation_set': {
        0: 1,  # Null
        1: 1,  # Freestyle
        2: 1,  # Backstroke
        3: 1,  # Breaststroke
        4: 1,  # Butterfly
        5: 1,  # Turn
        6: 1,  # Kick 
        'stroke_labels': ['stroke_labels'],  # Distinct entry for stroke_labels
    },
    # Optional debug keys for easier control and testing:
    'debug': {
        'enable_time_scaling': True,
        'enable_window_normalization': True,
        'use_majority_labeling': True,
    },
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
    'max_epochs': 200,
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
generator.summary()
discriminator.summary()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(gan_training_parameters['lr_generator'], beta_1=gan_training_parameters['beta_1'])
discriminator_optimizer = tf.keras.optimizers.Adam(gan_training_parameters['lr_discriminator'], beta_1=gan_training_parameters['beta_1'])

# Draw users for each class
train_dict, val_dict = swimming_data.draw_train_val_dicts(users_test, users_per_class=data_parameters['validation_set'])
print("Validation dictionary: %s" % val_dict)

# Prepare validation data
x_val, y_val_cat, val_sample_weights = swimming_data.get_windows_dict(val_dict, return_weights=True)
"""
try:
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
except Exception as e:
    print("Failed to reshape x_val:", e)
"""
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

def train_gan(generator, discriminator, data_generator, epochs, steps_per_epoch, x_val):
    """
    Training loop for GAN with proper dimension handling
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for step in range(steps_per_epoch):
            # Get a batch of real data - already augmented from generator
            #real_data, real_labels = next(data_generator)
            for real_data, label_dict in gen:
                # Use only the sensor and stroke label data for GAN training
                # Combine sensor data and stroke labels
                real_sensor_data = real_data  # This is batch_data
                real_labels = label_dict['style_output']  # Swim styles
                real_stroke_labels = label_dict['stroke_output']  # Stroke labels
            current_batch_size = real_data.shape[0]
            
            # Generate random noise
            noise = np.random.normal(0, 1, (current_batch_size, gan_training_parameters['latent_dim']))
            
            # Train Discriminator
            with tf.GradientTape() as tape_disc:
                # Generate fake data
                fake_data = generator(noise, training=True)
                
                # Get predictions
                real_preds = discriminator(real_data, training=True)
                fake_preds = discriminator(fake_data, training=True)
                
                # Use label smoothing for real labels (0.9 instead of 1.0)
                real_labels_smooth = tf.ones_like(real_preds) * 0.9
                fake_labels = tf.zeros_like(fake_preds)
                
                # Calculate discriminator losses
                d_loss_real = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(real_labels_smooth, real_preds))
                d_loss_fake = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(fake_labels, fake_preds))
                d_loss = d_loss_real + d_loss_fake
            
            # Apply discriminator gradients
            disc_gradients = tape_disc.gradient(d_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
            
            # Train Generator (twice per discriminator update for better training)
            for _ in range(2):
                with tf.GradientTape() as tape_gen:
                    # Generate new fake data
                    fake_data = generator(noise, training=True)
                    fake_preds = discriminator(fake_data, training=True)
                    
                    # Generator tries to fool discriminator
                    g_loss = tf.reduce_mean(
                        tf.keras.losses.binary_crossentropy(tf.ones_like(fake_preds), fake_preds))
                
                # Apply generator gradients
                gen_gradients = tape_gen.gradient(g_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: d_loss_real = {d_loss_real:.4f}, "
                      f"d_loss_fake = {d_loss_fake:.4f}, g_loss = {g_loss:.4f}")
            # Log metrics to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('d_loss_real', tf.reduce_mean(d_loss_real), step=epoch * steps_per_epoch + step)
                tf.summary.scalar('d_loss_fake', tf.reduce_mean(d_loss_fake), step=epoch * steps_per_epoch + step)
                tf.summary.scalar('g_loss', g_loss, step=epoch * steps_per_epoch + step)

        # Optional: Generate and save sample data at end of each epoch
        if epoch % 10 == 0:
            model_save_path = "models"

            # Ensure the directories exist
            os.makedirs(os.path.join(save_path, model_save_path, "generator"), exist_ok=True)
            os.makedirs(os.path.join(save_path, model_save_path, "discriminator"), exist_ok=True)

            # Save the generator and discriminator models
            generator.save(os.path.join(save_path, model_save_path, "generator", f'generator_epoch_{epoch}.h5'))
            discriminator.save(os.path.join(save_path, model_save_path, "discriminator", f'discriminator_epoch_{epoch}.h5'))

            # Save the generated sample data
            sample_noise = np.random.normal(0, 1, (5, gan_training_parameters['latent_dim']))
            generated_samples = generator(sample_noise, training=False)

            # Save generated samples for each user
            for user in users_test:
                # Ensure user directory exists
                user_save_path = os.path.join(save_path, user)
                os.makedirs(user_save_path, exist_ok=True)

                # Save generated data as a .npy file
                np.save(os.path.join(user_save_path, f'generated_data_epoch_{epoch}.npy'), generated_samples)

                """
                import pandas as pd

                # Loop for saving generated samples
                for user in users_test:
                    os.makedirs(os.path.join(save_path, user), exist_ok=True)
                    
                    # Convert generated samples to DataFrame
                    generated_df = pd.DataFrame(generated_samples, columns=['ACC_0', 'ACC_1', 'ACC_2', 
                                                                            'GYRO_0', 'GYRO_1', 'GYRO_2'])
                    
                    # Add placeholder columns for timestamp, label, stroke_labels
                    generated_df['timestamp'] = np.arange(len(generated_df))  # Sequential timestamps
                    generated_df['label'] = 'generated'  # Default label
                    generated_df['stroke_labels'] = 0  # Default stroke count
                    
                    # Save to CSV
                    csv_path = os.path.join(save_path, user, f'generated_data_epoch_{epoch}.csv')
                    generated_df.to_csv(csv_path, index=False)
                """


        # Optional: Evaluate on validation data
        if x_val is not None and epoch % 10 == 0:
            # Generate noise with the same batch size as x_val
            val_noise = np.random.normal(0, 1, (len(x_val), gan_training_parameters['latent_dim']))
            
            # Generate fake data
            val_fake_data = generator(val_noise, training=False)  # Shape: (N, win_len, num_features)
            print(f"x_val shape: {x_val.shape}, val_fake_data shape: {val_fake_data.shape}")
            
            # Ensure shapes match
           # if val_fake_data.shape != x_val.shape:
            #    raise ValueError(
             #       f"Shape mismatch: x_val {x_val.shape}, val_fake_data {val_fake_data.shape}. "
              #      "Ensure generator outputs data matching x_val dimensions."
               # )
            
            # Compare generated data to real data
           # compare_generated_to_real(x_val, val_fake_data)


def compare_generated_to_real(real_data, generated_data, num_samples=5):
    """
    Compare generated data to real data by visualizing samples.

    Args:
        real_data (np.ndarray): Real data samples.
        generated_data (np.ndarray): Generated data samples.
        num_samples (int): Number of samples to visualize.
    """
    # Ensure generated_data and real_data have the same shape for plotting
    if generated_data.shape[1:] != real_data.shape[1:]:
        generated_data = np.reshape(generated_data, real_data.shape)

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.plot(real_data[i], label='Real')
        plt.title('Real Data')
        plt.axis('off')

        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.plot(generated_data[i], label='Generated')
        plt.title('Generated Data')
        plt.axis('off')

    plt.tight_layout()
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Sensor Values")

    plt.show()



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
