import tensorflow as tf
import numpy as np
import os
import pickle
from ac_gan_vanilla2 import build_conditional_generator, build_conditional_discriminator, get_default_generator_parameters, get_default_discriminator_parameters
import learning_data
import utils
import datetime
import matplotlib.pyplot as plt
import io


# Define paths
data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified_users'
save_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_gan2'

# Ensure save path exists
os.makedirs(save_path, exist_ok=True)

# A list of user names which are loaded
users_all = utils.folders_in_path(data_path)
users = [u for u in users_all]  # Load all users
users.sort(key=int)

# Keeping it simple. Comment this out and use the code above if you want to load everybody
users = ['2','6','7','11']

# List of users we want to train a model for
users_test = users

# Hyper-parameters for loading data
data_parameters = {
    'users': users,  # Users whose data is loaded
    'labels': [0, 1, 2, 3, 4, 5],  # Labels for swim styles and transitions
    'combine_labels': {0: [0, 5]},  # Combine '0' and '5' as 'transition' for swim style transitions
    'data_columns': ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],  # Sensor data columns
    'stroke_labels': ['stroke_labels'],  # Binary stroke labels: 0 for no stroke, 1 for stroke
    'time_scale_factors': [0.9, 1.1],  # Scale timestamps by 10% faster/slower
    'stroke_range':         6,       # Augments stroke labels in the dataset to include a range around detected peaks
    'win_len': 180,  # Window length in time steps
    'slide_len': 30,  # Slide length for overlapping windows
    'window_normalization': 'tanh_scaled',  # Normalization method for windowed data
    'label_type': 'sparse',  # Labeling strategy for overlapping windows
    'majority_thresh': 0.75,  # Threshold for majority labeling
    'validation_set': {
        0: 1,  # Null
        1: 1,  # Freestyle
        2: 1,  # Backstroke
        3: 1,  # Breaststroke
        4: 1,  # Butterfly
        5: 1  # Turn
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

# Augments stroke labels in the dataset to include a range around detected peaks
swimming_data.augment_stroke_labels(stroke_range=data_parameters['stroke_range'])

# Compute sliding window locations
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile windows
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])

# GAN training parameters
gan_training_parameters = {
    'lr_generator': 0.0002,
    'lr_discriminator': 0.0001,
    'beta_1': 0.5,
    'batch_size': 64,
    'max_epochs': 200,
    'latent_dim': 100,
    'steps_per_epoch': 100,
    'noise_std': 0.01,
    'mirror_prob': 0.5,
    'random_rot_deg': 30,
    'stroke_mask':     False,    # Whether to use a mask for stroke labels
    'stroke_label_output':       True,
    'swim_style_output':         True
}

# Define GAN components
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']) + 2)  # Include labels + stroke_labels(+ 2) in input

generator_parameters = get_default_generator_parameters()
discriminator_parameters = get_default_discriminator_parameters()

generator = build_conditional_generator(
    gan_training_parameters['latent_dim'],
    num_styles=len(data_parameters['labels']),
    output_shape=input_shape,
    generator_parameters=generator_parameters
)

discriminator = build_conditional_discriminator(
    input_shape=input_shape,
    num_styles=len(data_parameters['labels']),
    discriminator_parameters=discriminator_parameters
)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(
    gan_training_parameters['lr_generator'], beta_1=gan_training_parameters['beta_1']
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    gan_training_parameters['lr_discriminator'], beta_1=gan_training_parameters['beta_1']
)


generator.summary()
discriminator.summary()

# Users whose data we use for training
users_train = [u for u in users if u != users_test]

# Draw users for each class
train_dict, val_dict = swimming_data.draw_train_val_dicts(users_test, users_per_class=data_parameters['validation_set'])
print("Training dictionary: %s" % train_dict)
print("Validation dictionary: %s" % val_dict)


# Get the validation data with weights and mask
x_val, y_val_sparse, y_val_cat, y_stroke_val, val_sample_weights, val_stroke_mask, y_val_raw = swimming_data.get_windows_dict(
    val_dict, return_weights=True, return_mask=True, transition_label=0, return_raw_labels=True
)
y_val_raw_3d = np.expand_dims(y_val_raw, axis=-1)
x_val_combined = np.concatenate((x_val, y_val_raw_3d, y_stroke_val), axis=2)

# The generator used to draw mini-batches
train_gen = swimming_data.batch_generator_dicts(train_dict=train_dict,
                                            batch_size=gan_training_parameters['batch_size'],
                                            noise_std=gan_training_parameters['noise_std'],
                                            mirror_prob=gan_training_parameters['mirror_prob'],
                                            random_rot_deg=gan_training_parameters['random_rot_deg'],
                                            use_4D=False,
                                            swim_style_output=gan_training_parameters['swim_style_output'], 
                                            stroke_label_output=gan_training_parameters['stroke_label_output'],
                                            return_stroke_mask=gan_training_parameters['stroke_mask'],
                                            return_raw_labels=True)

# TensorBoard setup
log_dir = os.path.join("logs", "gan", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# Create a summary writer for TensorBoard
summary_writer = tf.summary.create_file_writer(log_dir)


# ------------------------------------------------------
# (2) Define a @tf.function-based train_step
# ------------------------------------------------------
@tf.function
def train_step(real_sensors, real_styles, real_strokes,
               generator, discriminator,
               gen_optimizer, disc_optimizer,
               batch_size, latent_dim, num_styles):
    """
    Performs a single training step for both discriminator and generator.
    Args:
      real_sensors:   (batch_size, 180, 8) real input signals
      real_styles:    (batch_size,) integer swim style labels
      real_strokes:   (batch_size,) 0/1 stroke labels
      generator, discriminator:  Keras models
      gen_optimizer, disc_optimizer:  tf.optimizers
      batch_size, latent_dim, num_styles: integers
    Returns:
      d_loss: Discriminator loss
      g_loss: Generator loss
    """
    # 1) Train Discriminator
    noise = tf.random.normal([batch_size, latent_dim])
    
    # Generate proper stroke labels for discriminator
    fake_styles = tf.random.uniform([batch_size], minval=0, maxval=num_styles, dtype=tf.int32)
    """
    #use if want more realism
    fake_strokes_disc = tf.repeat(
        tf.random.uniform([batch_size, 1], minval=0, maxval=2, dtype=tf.int32),
        repeats=180,
        axis=1
    )
    fake_strokes_disc = tf.cast(fake_strokes_disc, tf.float32)[..., tf.newaxis]  # (B, 180, 1)
    """
    shared_strokes = tf.repeat(
        tf.random.uniform([batch_size, 1], minval=0, maxval=2, dtype=tf.int32),
        repeats=180,
        axis=1
    )
    shared_strokes = tf.cast(shared_strokes, tf.float32)[..., tf.newaxis]  # (B, 180, 1)

    with tf.GradientTape() as disc_tape:
        # Generate fake data with proper stroke shape
        fake_sensors = generator([noise, fake_styles, shared_strokes], training=True)
        
        # Forward pass real
        real_output = discriminator(real_sensors, training=True)
        (real_fake_real, style_real, stroke_real) = real_output

        # Forward pass fake
        fake_output = discriminator(fake_sensors, training=True)
        (real_fake_fake, style_fake, stroke_fake) = fake_output

        # Discriminator losses on real
        d_loss_real = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_fake_real)*0.9, real_fake_real  # Label smoothing
            )
        )
        style_loss_real = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(real_styles, style_real)
        )
        stroke_loss_real = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(real_strokes, stroke_real)
        )
        
        # Discriminator losses on fake
        d_loss_fake = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.zeros_like(real_fake_fake)*0.1, real_fake_fake
            )
        )
        style_loss_fake = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(fake_styles, style_fake)
        )
        stroke_loss_fake = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(shared_strokes, stroke_fake)
        )

        total_disc_loss = d_loss_real + style_loss_real + stroke_loss_real + d_loss_fake + style_loss_fake + stroke_loss_fake

    disc_grads = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
    disc_grads = [tf.clip_by_value(g, -1., 1.) for g in disc_grads]
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # 2) Train Generator
    noise = tf.random.normal([batch_size, latent_dim])
    fake_styles = tf.random.uniform([batch_size], minval=0, maxval=num_styles, dtype=tf.int32)
    """
    # Generate proper stroke labels for generator
    fake_strokes_gen = tf.repeat(
        tf.random.uniform([batch_size, 1], minval=0, maxval=2, dtype=tf.int32),
        repeats=180,
        axis=1
    )
    fake_strokes_gen = tf.cast(fake_strokes_gen, tf.float32)[..., tf.newaxis]  # (B, 180, 1) use different strokes to avoid collapsing increase realism
    """
    with tf.GradientTape() as gen_tape:
        fake_sensors = generator([noise, fake_styles, shared_strokes], training=True)
        fake_output = discriminator(fake_sensors, training=True)
        (real_fake_fake, style_fake, stroke_fake) = fake_output

        # Generator losses
        g_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.ones_like(real_fake_fake), real_fake_fake)
        )
        style_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(fake_styles, style_fake)
        )
        stroke_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(shared_strokes, stroke_fake) #use fake_strokes_gen if using in generator
        )

        total_gen_loss = g_loss + style_loss + stroke_loss

    gen_grads = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    return total_disc_loss, d_loss_real, d_loss_fake, total_gen_loss, style_loss, stroke_loss


# ------------------------------------------------------
# (3) Refactor train_gan to use train_step
# ------------------------------------------------------
def train_gan(generator, discriminator, data_generator, epochs, steps_per_epoch, x_val):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step in range(steps_per_epoch):
            # Pull a batch from the data generator
            real_data, label_dict = next(data_generator)

            # real_data is shape (batch_size, 180, 6) or (batch_size, 180, 8), depending on your config
            # label_dict has e.g. label_dict['swim_style_output'] and label_dict['stroke_label_output']
            real_sensors = real_data                       # shape (B, 180, 8)
            real_styles  = label_dict['swim_style_output'] # shape (B,)
            raw_labels = label_dict['raw_labels']
            real_strokes = label_dict['stroke_label_output']  # shape (B,)
            raw_labels_3d = np.expand_dims(raw_labels, axis=-1)
            real_data_combined = np.concatenate((real_sensors, raw_labels_3d, real_strokes), axis=2)
            # Make sure your batch_size is correct
            current_batch_size = real_sensors.shape[0]
            if current_batch_size < gan_training_parameters['batch_size']:
                # If you're at the end of an epoch and don't have a full batch, break or adjust your steps
                break

            # One train step
            total_d_loss, d_loss_real, d_loss_fake, total_g_loss, style_loss, stroke_loss = train_step(
                real_data_combined, real_styles, real_strokes,
                generator, discriminator,
                generator_optimizer, discriminator_optimizer,
                gan_training_parameters['batch_size'],
                gan_training_parameters['latent_dim'],
                num_styles=len(data_parameters['labels'])
            )

            # Logging
            if step % 10 == 0:
                print(f"Step {step:03d} | D_loss = {total_d_loss:.4f} | G_loss = {total_g_loss:.4f}")
                
            # Optionally log to TensorBoard
            step_global = epoch * steps_per_epoch + step
            with summary_writer.as_default():
                # Main losses
                tf.summary.scalar('d_loss/total', total_d_loss, step=step_global)
                tf.summary.scalar('g_loss/total', total_g_loss, step=step_global)
                tf.summary.scalar('g_loss/style', style_loss, step=step_global)
                tf.summary.scalar('g_loss/stroke', stroke_loss, step=step_global)
                # Discriminator components
                tf.summary.scalar('d_loss/real', d_loss_real, step=step_global)
                tf.summary.scalar('d_loss/fake', d_loss_fake, step=step_global)
                tf.summary.scalar('gen_learning_rate', generator_optimizer.learning_rate, step=step_global)
                tf.summary.scalar('disc_learning_rate', discriminator_optimizer.learning_rate, step=step_global)

                # Histograms (only log these periodically to reduce overhead)
                if step % 50 == 0:

                    # Log variable histograms
                    for var in discriminator.trainable_variables:
                        tf.summary.histogram(f"D_vars/{var.name}", var, step=step_global)
                    for var in generator.trainable_variables:
                        tf.summary.histogram(f"G_vars/{var.name}", var, step=step_global)

        if epoch % 10 == 0:
            # Save models
            model_save_path = os.path.join(save_path, "models")
            gen_save_dir = os.path.join(model_save_path, "generator")
            disc_save_dir = os.path.join(model_save_path, "discriminator")
            os.makedirs(gen_save_dir, exist_ok=True)
            os.makedirs(disc_save_dir, exist_ok=True)
            
            generator.save(os.path.join(gen_save_dir, f'gen_epoch_{epoch}.keras'))
            discriminator.save(os.path.join(disc_save_dir, f'disc_epoch_{epoch}.keras'))

            # Generate sample data with proper stroke shape (5, 180, 1)
            # Create interpolated noise between two random vectors
            z1 = tf.random.normal([1, gan_training_parameters['latent_dim']])  # Shape (1, 100)
            z2 = tf.random.normal([1, gan_training_parameters['latent_dim']])  # Shape (1, 100)
            ratios = tf.reshape(tf.linspace(0., 1., 5), (-1, 1))  # Shape (5, 1)
            sample_noise = z1 + (z2 - z1) * ratios  # Shape (5, 100)

            #sample_noise = tf.random.normal([5, gan_training_parameters['latent_dim']])
            sample_styles = tf.random.uniform([5], minval=0, maxval=len(data_parameters['labels']), dtype=tf.int32)
            sample_strokes = tf.random.uniform([5, 180, 1], minval=0, maxval=2, dtype=tf.int32)
            sample_strokes = tf.cast(sample_strokes, tf.float32)  # Cast to float32 for generator input


            generated_samples = generator([sample_noise, sample_styles, sample_strokes], training=False)
            print(f"Sample shapes - Noise: {sample_noise.shape}, Styles: {sample_styles.shape}, " 
                f"Strokes: {sample_strokes.shape}, Output: {generated_samples.shape}")

        # Validation evaluation (every 10 epochs)
        if x_val is not None and epoch % 10 == 0:
            # Generate validation fake data with matching temporal dimensions
            val_noise = tf.random.normal([x_val.shape[0], gan_training_parameters['latent_dim']])
            val_styles = tf.random.uniform([x_val.shape[0]], minval=0, maxval=len(data_parameters['labels']), dtype=tf.int32)
            val_strokes = tf.random.uniform([x_val.shape[0], 180, 1], minval=0, maxval=2, dtype=tf.int32)

            val_strokes = tf.cast(val_strokes, tf.float32)

            # Generate validation fake data with proper channel dimensions
            val_fake_data = generator([val_noise, val_styles, val_strokes], training=False)            
            # Compare real vs fake data shapes (should both be [batch, 180, 8])
            print(f"\nValidation Stats:")
            print(f"Real data shape: {x_val.shape} (sensors + labels)")
            print(f"Generated shape: {val_fake_data.shape}")
            real_mean = np.mean(x_val[..., :6])
            real_min = np.min(x_val[..., :6])
            real_max = np.max(x_val[..., :6])
            fake_mean = tf.reduce_mean(val_fake_data[..., :6]).numpy()
            fake_min = tf.reduce_min(val_fake_data[..., :6]).numpy()
            fake_max = tf.reduce_max(val_fake_data[..., :6]).numpy()
            # Optional: Calculate basic distribution stats
            #print(f"Real data range: [{x_val.min():.2f}, {x_val.max():.2f}]")
            #print(f"Fake data range: [{tf.reduce_min(val_fake_data).numpy():.2f}, {tf.reduce_max(val_fake_data).numpy():.2f}]")
            # Check value distribution
            #real_mean = np.mean(x_val[..., :6])  # Sensor channels only
            #fake_mean = tf.reduce_mean(val_fake_data[..., :6]).numpy()  # TF-native + convert
            print(f"Real stats | Mean: {real_mean:.2f}, Min: {real_min:.2f}, Max: {real_max:.2f}")
            print(f"Fake stats | Mean: {fake_mean:.2f}, Min: {fake_min:.2f}, Max: {fake_max:.2f}")
            if abs(real_mean - fake_mean) > 0.5:
                print(f"Significant mean discrepancy: Real {real_mean:.2f} vs Fake {fake_mean:.2f}")

        # After epoch completes
        with summary_writer.as_default():
            if epoch % 10 == 0:
                tf.summary.histogram('real_samples', x_val, step=step_global)
                tf.summary.histogram('fake_samples', val_fake_data, step=step_global)
            # Log sample images every 25 epochs
            if epoch % 25 == 0:
                # Get sample real and fake data
                num_samples = 5
                real_samples = x_val[:num_samples]  # First 5 real samples
                
                # Generate matching fake samples
                # Create interpolated noise between two random vectors
                z1 = tf.random.normal([1, gan_training_parameters['latent_dim']])  # Shape (1, 100)
                z2 = tf.random.normal([1, gan_training_parameters['latent_dim']])  # Shape (1, 100)
                ratios = tf.reshape(tf.linspace(0., 1., num_samples), (-1, 1))  # Shape (5, 1)
                noise = z1 + (z2 - z1) * ratios  # Shape (5, 100)
                #noise = tf.random.normal([num_samples, gan_training_parameters['latent_dim']])
                styles = tf.random.uniform([num_samples], 
                                        minval=0, 
                                        maxval=len(data_parameters['labels']), 
                                        dtype=tf.int32)
                strokes = tf.random.uniform([num_samples, 180, 1], 
                                        minval=0, 
                                        maxval=2, 
                                        dtype=tf.int32)
                
                fake_samples = generator([noise, styles, strokes], training=False)
                
                # Create and log figure
                fig = plot_samples(real_samples, fake_samples.numpy(), num_samples=num_samples)
                tf.summary.image("Sample Comparison", 
                            plot_to_image(fig), 
                            step=epoch)

def log_and_show(fig, epoch):
    """Handle both TensorBoard logging and interactive display"""
    # Log to TensorBoard
    tf.summary.image("Sample Comparison", fig_to_array(fig), step=epoch)
    
    # Conditional display (only if in interactive environment)
    if sys.stdout.isatty():  # Checks if running in terminal/console
        plt.show()
    else:
        plt.close(fig)

def plot_to_image(fig):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(fig)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def fig_to_array(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape((*reversed(fig.canvas.get_width_height()), 4))
    img = img.astype(float) / 255.0
    alpha = img[..., 3:]
    img = img[..., :3] / alpha  # Divide RGB by alpha
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    plt.close(fig)
    return img

def plot_samples(real_samples, fake_samples, num_samples=5):
    """Plot all 6 sensor channels with stroke peak indicators"""
    plt.figure(figsize=(20, 10), facecolor='white')
    
    for i in range(num_samples):
        # Real Sample
        plt.subplot(2, num_samples, i+1)
        
        # Plot sensor channels 0-5 with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        for ch in range(6):  # Now plotting all 6 sensor channels
            sensor_data = real_samples[i, :, ch]
            plt.plot(sensor_data, 
                    color=colors[ch], 
                    alpha=0.7,
                    label=f'Sensor {ch+1}')
            
            # Add stroke peak indicators as scatter points (channel 7 is stroke label)
            if ch == 0:  # Only add to first sensor to avoid duplicate points
                stroke_indices = np.where(real_samples[i, :, 7] > 0.5)[0]
                plt.scatter(stroke_indices, sensor_data[stroke_indices],
                          color='red', marker='o', s=10,
                          edgecolor='black', zorder=5,
                          label='Stroke Peaks (Real)')
            
        plt.title(f'Real Sample {i+1}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Fake Sample
        plt.subplot(2, num_samples, num_samples+i+1)
        for ch in range(6):  # Now plotting all 6 sensor channels
            sensor_data = fake_samples[i, :, ch]
            plt.plot(sensor_data,
                    color=colors[ch],
                    alpha=0.7)
            
            # Add generated stroke indicators
            if ch == 0:  # Only add to first sensor
                fake_strokes = np.where(fake_samples[i, :, 7] > 0.5)[0]
                plt.scatter(fake_strokes, sensor_data[fake_strokes],
                          color='blue', marker='o', s=10,
                          edgecolor='black', zorder=5,
                          label='Stroke Peaks (Generated)')
            
        plt.title(f'Generated Sample {i+1}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return plt.gcf()




train_gan(
    generator, 
    discriminator, 
    data_generator=train_gen, 
    epochs=gan_training_parameters['max_epochs'],
    steps_per_epoch=gan_training_parameters['steps_per_epoch'],
    x_val=x_val_combined
)

# After training completes, save final models and parameters
generator.save(os.path.join(save_path, 'final_generator.keras'))
discriminator.save(os.path.join(save_path, 'final_discriminator.keras'))

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

def train_gan_old(generator, discriminator, data_generator, epochs, steps_per_epoch, x_val):
    """
    Training loop for GAN with proper dimension handling
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for step in range(steps_per_epoch):
            # Get a batch of real data - already augmented from generator
            real_data, label_dict = next(data_generator)
           # for real_data, label_dict in gen:
                # Use only the sensor and stroke label data for GAN training
                # Combine sensor data and stroke labels
            real_sensor_data = real_data  # This is batch_data
            real_labels = label_dict['swim_style_output']  # Swim styles
            real_stroke_labels = label_dict['stroke_label_output']  # Stroke labels
            current_batch_size = real_data.shape[0]
            real_data_combined = np.concatenate((real_sensor_data, real_labels, real_stroke_labels), axis=2)

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
