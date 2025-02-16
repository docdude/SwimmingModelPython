import tensorflow as tf
import numpy as np
import os
import pickle
from ac_gan_vanilla4 import build_conditional_generator, build_conditional_discriminator, get_default_generator_parameters, get_default_discriminator_parameters
import learning_data
import utils
import datetime
import matplotlib.pyplot as plt
import io
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

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
#users = ['2','6','7','11']

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
    'stroke_range':         None,       # Augments stroke labels in the dataset to include a range around detected peaks
    'win_len': 180,  # Window length in time steps
    'slide_len': 0,  # Slide length for overlapping windows
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
#swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

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
    'max_epochs': 300,
    'latent_dim': 100,
    'steps_per_epoch': 100,
    'noise_std': None,
    'mirror_prob': None,
    'random_rot_deg': None,
    'stroke_mask':     False,    # Whether to use a mask for stroke labels
    'stroke_label_output':       True,
    'swim_style_output':         True,
    'output_bias': None
}

# Define GAN components
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']) + 2)  # Include labels + stroke_labels(+ 2) in input

# Users whose data we use for training
users_train = [u for u in users if u != users_test]

# Draw users for each class
train_dict, val_dict = swimming_data.draw_train_val_dicts(users_test, users_per_class=data_parameters['validation_set'])
print("Training dictionary: %s" % train_dict)
print("Validation dictionary: %s" % val_dict)

# Calculate stroke label distribution for training set (excluding label 0)
training_probabilities, training_mean, training_bias, training_class_weights = utils.calculate_stroke_label_distribution(
    label_user_dict=train_dict,
    swimming_data=swimming_data,
    data_type="training",
    exclude_label=None
)
# Set training bias for generator
gan_training_parameters['output_bias'] = training_bias

# Calculate stroke label distribution for validation set (excluding label 0)
validation_probabilities, validation_mean, validation_bias, validation_class_weights = utils.calculate_stroke_label_distribution(
    label_user_dict=val_dict,
    swimming_data=swimming_data,
    data_type="validation",
    exclude_label=None
)

generator_parameters = get_default_generator_parameters()
discriminator_parameters = get_default_discriminator_parameters()

generator = build_conditional_generator(
    gan_training_parameters['latent_dim'],
    num_styles=len(data_parameters['labels']),
    generator_parameters=generator_parameters,
    output_bias=gan_training_parameters['output_bias']
)

discriminator = build_conditional_discriminator(
    input_shape=input_shape,
    num_styles=len(data_parameters['labels']),
    discriminator_parameters=discriminator_parameters,
    output_bias=gan_training_parameters['output_bias']

)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(
    gan_training_parameters['lr_generator'], beta_1=gan_training_parameters['beta_1']
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    gan_training_parameters['lr_discriminator'], beta_1=gan_training_parameters['beta_1']
)
"""
# Apply exponential decay
generator_optimizer = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0002, decay_steps=1000, decay_rate=0.96
)
discriminator_optimizer = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.96
)
"""

generator.summary()
discriminator.summary()



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

def compute_batch_class_weights(batch_swim_styles):
    batch_swim_styles_flat = batch_swim_styles.flatten().tolist()
    unique_classes = np.unique(batch_swim_styles_flat)

    # Compute weights per class
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=batch_swim_styles_flat)

    # Create a dictionary mapping the unique class labels to their computed weights
    class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    # Convert the dictionary into a tensor, ensuring it can be indexed correctly
    max_class = max(unique_classes)  # Ensure the tensor can be indexed safely
    class_weights_tensor = tf.constant(
        [class_weight_dict.get(i, 1.0) for i in range(max_class + 1)],  # Default weight = 1.0 for missing classes
        dtype=tf.float32
    )
    return class_weights_tensor

def compute_batch_sample_weights(batch_labels):
    batch_labels_flat = batch_labels.flatten()
    sample_weights = compute_sample_weight('balanced', batch_labels_flat)
    # Reshape sample weights to match the original batch label shape
    sample_weights = sample_weights.reshape(batch_labels.shape[:-1])
    #sample_weights = sample_weights.squeeze(axis=-1)
    # Convert to a TensorFlow tensor with the desired shape
    sample_weights_tensor = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
    
    # Ensure the tensor has the correct shape (64, 180)
    #sample_weights_tensor = tf.squeeze(sample_weights_tensor, axis=-1)#tf.reshape(sample_weights_tensor, batch_labels.shape)
    return sample_weights_tensor

@tf.function
def train_step(real_data, real_styles, real_strokes, 
                generator, discriminator, 
                gen_optimizer, disc_optimizer, 
                batch_size, latent_dim, style_weights, stroke_weights):

    noise_gen = tf.random.normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
    noise_disc = tf.random.normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
    real_styles = tf.convert_to_tensor(real_styles, dtype=tf.int64)
    real_strokes = tf.convert_to_tensor(real_strokes, dtype=tf.int64)
    # Generator input: fake_styles as (batch_size,) for embedding layer
    #fake_styles = tf.random.uniform([batch_size], minval=0, maxval=num_styles, dtype=tf.int32)
    #fake_styles = tf.gather(real_styles, tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int64))
   # tf.print(fake_styles)
    #fake_strokes_disc = tf.gather(real_strokes, tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int64))
    #tf.print(fake_strokes_disc)
    # Stroke probabilities derived from empirical dataset
    #fake_styles = real_styles
    #fake_strokes_disc = real_strokes

    # Ensure style_weights is indexed correctly
    #style_weights = tf.gather(class_weights, tf.squeeze(real_styles, axis=-1))  # Shape: (64, 180)

    # Expand style_weights to match (64, 180, 1) for proper broadcasting
    #style_weights = tf.expand_dims(style_weights, axis=-1)  # Shape: (64, 180, 1)


    # ---- Discriminator Training ----
    with tf.GradientTape(persistent=True) as disc_tape:
        fake_data = generator([noise_disc, real_styles, real_strokes], training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)

        real_fake_real, style_real, stroke_real = real_output
        real_fake_fake, style_fake, stroke_fake = fake_output

        #real_labels = tf.random.uniform(tf.shape(real_fake_real), minval=0.8, maxval=1.0)
        #fake_labels = tf.random.uniform(tf.shape(real_fake_fake), minval=0.0, maxval=0.2)
        real_labels = tf.ones_like(real_fake_real)
        fake_labels = tf.zeros_like(real_fake_fake)


        # Compute individual losses
        d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, real_fake_real))
        style_loss_real = tf.reduce_mean(style_weights * tf.keras.losses.mean_squared_error(real_styles, style_real))
        stroke_loss_real = tf.reduce_mean(stroke_weights * tf.keras.losses.binary_crossentropy(real_strokes, stroke_real))

        d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, real_fake_fake))
        style_loss_fake = tf.reduce_mean(style_weights * tf.keras.losses.mean_squared_error(real_styles, style_fake))
        stroke_loss_fake = tf.reduce_mean(stroke_weights * tf.keras.losses.binary_crossentropy(real_strokes, stroke_fake))

        # Total discriminator loss
       # total_disc_loss = d_loss_real + style_loss_real + stroke_loss_real + d_loss_fake + style_loss_fake  + stroke_loss_fake # + (0 * gradient_penalty))
        total_disc_loss = (
            1.0 * (d_loss_real + d_loss_fake) + 
            1.0 * (style_loss_real + style_loss_fake) + 
            1.0 * (stroke_loss_real + stroke_loss_fake)
        )
    # Compute separate gradients
    d_grads_real = disc_tape.gradient(d_loss_real, discriminator.trainable_variables)
    d_grads_style = disc_tape.gradient(style_loss_real, discriminator.trainable_variables)
    d_grads_stroke = disc_tape.gradient(stroke_loss_real, discriminator.trainable_variables)
    d_grads_fake = disc_tape.gradient(d_loss_fake, discriminator.trainable_variables)
    d_grads_style_fake = disc_tape.gradient(style_loss_fake, discriminator.trainable_variables)
    d_grads_stroke_fake = disc_tape.gradient(stroke_loss_fake, discriminator.trainable_variables)

    # Aggregate gradients for discriminator
    def normalize_grads(grad, variables):
        return [(g / (tf.norm(g) + 1e-8)) if g is not None else tf.zeros_like(v) for g, v in zip(grad, variables)]


    d_grads = [g1 + g2 + g3 + g4 + g5 + g6 for g1, g2, g3, g4, g5, g6 in zip(
        normalize_grads(d_grads_real, discriminator.trainable_variables), 
        normalize_grads(d_grads_style, discriminator.trainable_variables), 
        normalize_grads(d_grads_stroke, discriminator.trainable_variables), 
        normalize_grads(d_grads_fake, discriminator.trainable_variables), 
        normalize_grads(d_grads_style_fake, discriminator.trainable_variables), 
        normalize_grads(d_grads_stroke_fake, discriminator.trainable_variables)
    )]

    disc_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    del disc_tape  # Free memory

    # ---- Generator Training ----
    with tf.GradientTape(persistent=True) as gen_tape:
        fake_data = generator([noise_gen, real_styles, real_strokes], training=True)
        fake_output = discriminator(fake_data, training=True)
        real_fake_fake, style_fake, stroke_fake = fake_output

        g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_fake_fake), real_fake_fake))
        style_loss = tf.reduce_mean(style_weights * tf.keras.losses.mean_squared_error(real_styles, style_fake))
        stroke_loss = tf.reduce_mean(stroke_weights * tf.keras.losses.binary_crossentropy(real_strokes, stroke_fake))

        # Total generator loss
        total_gen_loss = (
            1.0 * g_loss + 
            1.0 * style_loss + 
            1.0 * stroke_loss)

    # Compute generator gradients per loss
    g_grads_g = gen_tape.gradient(g_loss, generator.trainable_variables)
    g_grads_style = gen_tape.gradient(style_loss, generator.trainable_variables)
    g_grads_stroke = gen_tape.gradient(stroke_loss, generator.trainable_variables)

    # Normalize and aggregate generator gradients safely
    g_grads_g = normalize_grads(g_grads_g, generator.trainable_variables)
    g_grads_style = normalize_grads(g_grads_style, generator.trainable_variables)
    g_grads_stroke = normalize_grads(g_grads_stroke, generator.trainable_variables)

    # Summing normalized generator gradients
    g_grads = [g1 + g2 + g3 for g1, g2, g3 in zip(
        g_grads_g, g_grads_style, g_grads_stroke
    )]


    gen_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))


    return total_disc_loss, d_loss_real, d_loss_fake, total_gen_loss, style_loss, stroke_loss

@tf.function
def train_step2(real_data, real_styles, real_strokes,
               generator, discriminator,
               gen_optimizer, disc_optimizer,
               batch_size, latent_dim, style_weights, stroke_weights):

    λ_style = 1.0
    λ_stroke = 1.0
    λ_pred = 1.0
    noise_gen = tf.random.normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
    noise_disc = tf.random.normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
    real_styles = tf.convert_to_tensor(real_styles, dtype=tf.int64)
    real_strokes = tf.convert_to_tensor(real_strokes, dtype=tf.int64)
    # Generator input: fake_styles as (batch_size,) for embedding layer
    #fake_styles = tf.random.uniform([batch_size], minval=0, maxval=num_styles, dtype=tf.int32)
    #fake_styles = tf.gather(real_styles, tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int64))
   # tf.print(fake_styles)
    #fake_strokes_disc = tf.gather(real_strokes, tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int64))
    #tf.print(fake_strokes_disc)
    # Stroke probabilities derived from empirical dataset
    fake_styles = real_styles
    fake_strokes_disc = real_strokes
    """
    stroke_probs = {0: 0.0, 1: 0.0156, 2: 0.0137, 3: 0.0138, 4: 0.0189, 5: 0.0}  
    #tf.print(fake_styles)
    # Convert dictionary to tensor
    stroke_probs_tensor = tf.convert_to_tensor(list(stroke_probs.values()))

    # Gather stroke probability for each sample's swim style
    stroke_prob_selected = tf.gather(stroke_probs_tensor, fake_styles)  # Works for 1D fake_styles
    #tf.print(stroke_prob_selected)
    # Expand dimensions to match expected shape (batch_size, 180, 1)

    stroke_prob_selected = tf.expand_dims(stroke_prob_selected, axis=-1)  # (batch_size, 1)
    stroke_prob_selected = tf.expand_dims(stroke_prob_selected, axis=-1)  # (batch_size, 1, 1)
    stroke_prob_selected = tf.tile(stroke_prob_selected, [1, 180, 1])  # (batch_size, 180, 1)
    #tf.print(stroke_prob_selected)
    # Sample strokes based on probability for discriminator
    fake_strokes_disc = tf.where(
        tf.random.uniform([batch_size, 180, 1]) < stroke_prob_selected,
        tf.ones([batch_size, 180, 1]), 
        tf.zeros([batch_size, 180, 1])
    )
    #tf.print(fake_strokes_disc)
    # Sample strokes separately for generator (ensuring variation)
    fake_strokes_gen = tf.where(
        tf.random.uniform([batch_size, 180, 1]) < stroke_prob_selected,
        tf.ones([batch_size, 180, 1]), 
        tf.zeros([batch_size, 180, 1])
    )
    """
    with tf.GradientTape() as disc_tape:

        fake_data = generator([noise_disc, fake_styles, fake_strokes_disc], training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)

        real_fake_real, style_real, stroke_real = real_output
        real_fake_fake, style_fake, stroke_fake = fake_output

        # Apply random label smoothing
        #real_labels = tf.random.uniform(tf.shape(real_fake_real), minval=0.8, maxval=1.0)
        #fake_labels = tf.random.uniform(tf.shape(real_fake_fake), minval=0.0, maxval=0.2)
        # Random label smoothing
        #real_labels = tf.random.uniform(tf.shape(real_fake_real), minval=0.9, maxval=1.0)
        #fake_labels = tf.random.uniform(tf.shape(real_fake_fake), minval=0.1, maxval=0.3)
        real_labels = tf.ones_like(real_fake_real)
        fake_labels = tf.zeros_like(real_fake_fake)
        # Discriminator loss
        d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, real_fake_real))
        style_loss_real = tf.reduce_mean(style_weights * tf.keras.losses.mean_squared_error(real_styles, style_real))
        stroke_loss_real = tf.reduce_mean(stroke_weights * tf.keras.losses.binary_crossentropy(real_strokes, stroke_real))

        d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, real_fake_fake))
        style_loss_fake = tf.reduce_mean(style_weights * tf.keras.losses.mean_squared_error(fake_styles, style_fake))
        stroke_loss_fake = tf.reduce_mean(stroke_weights * tf.keras.losses.binary_crossentropy(fake_strokes_disc, stroke_fake))
        gradient_penalty = compute_gradient_penalty(real_data, fake_data)


        #total_disc_loss = (d_loss_real + style_loss_real + stroke_loss_real + d_loss_fake + style_loss_fake  + stroke_loss_fake + (0 * gradient_penalty))
       # total_disc_loss = (
        #    0.5 * (d_loss_real + d_loss_fake) + 
         #   0.25 * (style_loss_real + style_loss_fake) + 
          #  0.25 * (stroke_loss_real + stroke_loss_fake)
        #)
        total_disc_loss = (
            d_loss_real +  style_loss_real + stroke_loss_real +
            d_loss_fake +  style_loss_fake + stroke_loss_fake
        )


    disc_grads = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
    disc_grads = [tf.clip_by_value(g, -1., 1.) for g in disc_grads]

    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # Generator Training
    with tf.GradientTape() as gen_tape:
        fake_data = generator([noise_gen, fake_styles, fake_strokes_disc], training=True)
        fake_output = discriminator(fake_data, training=True)
        real_fake_fake, style_fake, stroke_fake = fake_output

        g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_fake_fake), real_fake_fake))
        style_loss = tf.reduce_mean(style_weights * tf.keras.losses.mean_squared_error(fake_styles, style_fake))
        stroke_loss = tf.reduce_mean(stroke_weights * tf.keras.losses.binary_crossentropy(fake_strokes_disc, stroke_fake))  # Use fake_strokes_gen

      #  total_gen_loss = g_loss + style_loss + stroke_loss
        total_gen_loss = g_loss + style_loss + stroke_loss

    gen_grads = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    return total_disc_loss, d_loss_real, d_loss_fake, total_gen_loss, style_loss, stroke_loss


def progressive_label_smoothing(epoch, real_labels, min_smooth=0.8, max_smooth=0.95):
    """Progressive label smoothing that increases with epochs"""
    smoothing = min_smooth + (max_smooth - min_smooth) * (1 - np.exp(-epoch/50))
    return real_labels * smoothing + (1 - smoothing) * 0.5

def adaptive_learning_rate(base_lr, epoch, decay_factor=0.95, min_lr=1e-6):
    """Adaptive learning rate based on training progress"""
    return max(base_lr * (decay_factor ** (epoch // 10)), min_lr)

@tf.function
def train_step3(real_data, real_styles, real_strokes,
               generator, discriminator,
               gen_optimizer, disc_optimizer,
               batch_size, latent_dim, style_weights, stroke_weights, epoch):
    
    noise_gen = tf.random.normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
    noise_disc = tf.random.normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
    
    # Convert inputs to appropriate types
    real_styles = tf.convert_to_tensor(real_styles, dtype=tf.int64)
    real_strokes = tf.convert_to_tensor(real_strokes, dtype=tf.int64)
    
    # Extract real sensor data (first 6 channels)
    real_sensors = real_data[..., :6]
    
    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        # Generate fake data
        fake_data = generator([noise_disc, real_styles, real_strokes], training=True)
        fake_sensors = fake_data[..., :6]
        
        # Get discriminator outputs for real and fake data
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)
        
        # Unpack discriminator outputs
        real_fake_real, style_real, stroke_real = real_output
        real_fake_fake, style_fake, stroke_fake = fake_output
        
        # Progressive label smoothing
        real_labels = progressive_label_smoothing(epoch, tf.ones_like(real_fake_real))
        fake_labels = tf.zeros_like(real_fake_fake)
        
        # Compute various losses
        # Adversarial losses
        d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, real_fake_real))
        d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, real_fake_fake))
        
        # Style classification losses
        style_loss_real = tf.reduce_mean(style_weights * tf.keras.losses.mean_squared_error(real_styles, style_real))
        style_loss_fake = tf.reduce_mean(style_weights * tf.keras.losses.mean_squared_error(real_styles, style_fake))
        
        # Stroke detection losses
        stroke_loss_real = tf.reduce_mean(stroke_weights * tf.keras.losses.binary_crossentropy(real_strokes, stroke_real))
        stroke_loss_fake = tf.reduce_mean(stroke_weights * tf.keras.losses.binary_crossentropy(real_strokes, stroke_fake))
        
        # Additional losses for generator
        spectral_loss_term = spectral_loss(real_sensors, fake_sensors)
        temporal_loss_term = temporal_consistency_loss(fake_sensors)
        distribution_loss_term = sensor_distribution_loss(real_sensors, fake_sensors)
        
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(real_data, fake_data, discriminator)
        
        # Compute total losses with dynamic weighting
        disc_loss_weights = tf.nn.softmax(tf.stack([
            d_loss_real, d_loss_fake, 
            style_loss_real, style_loss_fake,
            stroke_loss_real, stroke_loss_fake,
            gradient_penalty
        ]))
        
        total_disc_loss = (
            disc_loss_weights[0] * d_loss_real +
            disc_loss_weights[1] * d_loss_fake +
            disc_loss_weights[2] * style_loss_real +
            disc_loss_weights[3] * style_loss_fake +
            disc_loss_weights[4] * stroke_loss_real +
            disc_loss_weights[5] * stroke_loss_fake +
            0.1 * disc_loss_weights[6] * gradient_penalty
        )
        
        # Generator loss with dynamic weighting
        gen_loss_weights = tf.nn.softmax(tf.stack([
            tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_fake_fake), real_fake_fake)),
            style_loss_fake,
            stroke_loss_fake,
            spectral_loss_term,
            temporal_loss_term,
            distribution_loss_term
        ]))
        
        total_gen_loss = (
            gen_loss_weights[0] * tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_fake_fake), real_fake_fake)) +
            gen_loss_weights[1] * style_loss_fake +
            gen_loss_weights[2] * stroke_loss_fake +
            0.1 * gen_loss_weights[3] * spectral_loss_term +
            0.05 * gen_loss_weights[4] * temporal_loss_term +
            0.1 * gen_loss_weights[5] * distribution_loss_term
        )
    
    # Compute and apply gradients with clipping
    disc_grads = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
    gen_grads = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    
    # Clip gradients
    disc_grads = [tf.clip_by_norm(g, 1.0) for g in disc_grads]
    gen_grads = [tf.clip_by_norm(g, 1.0) for g in gen_grads]
    
    # Apply gradients
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    
    return (total_disc_loss, d_loss_real, d_loss_fake, total_gen_loss, 
            style_loss_fake, stroke_loss_fake, spectral_loss_term, 
            distribution_loss_term)

def train_gan(generator, discriminator, data_generator, epochs, steps_per_epoch, x_val, y_val_sparse, y_stroke_val, y_val_raw):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step in range(steps_per_epoch):
            # Pull a batch from the data generator
            real_sensor_data, label_dict = next(data_generator)

            # real_data is shape (batch_size, 180, 6) or (batch_size, 180, 8), depending on your config
            # label_dict has e.g. label_dict['swim_style_output'] and label_dict['stroke_label_output']
            real_sensors = real_sensor_data                     # shape (B, 180, 6)
            real_styles  = label_dict['swim_style_output'] # shape (B,)
            raw_labels = label_dict['raw_labels'] # shape (B, 180)
            real_strokes = label_dict['stroke_label_output']  # shape (B, 180, 1)
            raw_labels_3d = np.expand_dims(raw_labels, axis=-1)
            # Expand dimensions to (B, 1, 1) to prepare for broadcasting
            real_styles_expanded = np.expand_dims(real_styles, axis=(1, 2))  # Shape: (B, 1, 1)

            # Broadcast to (B, 180, 1) to match time steps
            real_styles_3d = np.tile(real_styles_expanded, (1, 180, 1))  # Shape: (B, 180, 1)

            real_data_combined = np.concatenate((real_sensors, raw_labels_3d, real_strokes), axis=2)
            # Make sure your batch_size is correct
            current_batch_size = real_data_combined.shape[0]
            if current_batch_size < gan_training_parameters['batch_size']:
                # If you're at the end of an epoch and don't have a full batch, break or adjust your steps
                break

            style_weights = compute_batch_sample_weights(raw_labels_3d)
            stroke_weights = compute_batch_sample_weights(real_strokes)
            # One train step
            total_d_loss, d_loss_real, d_loss_fake, total_g_loss, style_loss, stroke_loss = train_step2(
                real_data_combined, raw_labels_3d, real_strokes,
                generator, discriminator,
                generator_optimizer, discriminator_optimizer,
                gan_training_parameters['batch_size'],
                gan_training_parameters['latent_dim'], 
                style_weights=style_weights,
                stroke_weights=stroke_weights
               # num_styles=len(data_parameters['labels'])
            )

            # Logging
            if step % 10 == 0:
                print(f"Step {step:03d} | D_loss = {total_d_loss:.4f} | G_loss = {total_g_loss:.4f}")
            # set to swim styles and strokes to tensors for use in logging
            real_styles = tf.convert_to_tensor(real_styles, dtype=tf.int64)
            real_strokes = tf.convert_to_tensor(real_strokes, dtype=tf.int64)                
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


            # Generator input: fake_styles as (batch_size,) for embedding layer
            #fake_styles = tf.random.uniform([batch_size], minval=0, maxval=num_styles, dtype=tf.int32)
            sample_styles = tf.gather(raw_labels_3d, tf.random.uniform([5], 0, 5, dtype=tf.int64))
        # tf.print(fake_styles)
            sample_strokes = tf.gather(real_strokes, tf.random.uniform([5], 0, 5, dtype=tf.int64))
            #sample_noise = tf.random.normal([5, gan_training_parameters['latent_dim']])
           # sample_styles = tf.random.uniform([5], minval=0, maxval=len(data_parameters['labels']), dtype=tf.int32)
           # sample_strokes = tf.random.uniform([5, 180, 1], minval=0, maxval=2, dtype=tf.int32)
           # sample_strokes = tf.cast(sample_strokes, tf.float32)  # Cast to float32 for generator input


            generated_samples = generator([sample_noise, sample_styles, sample_strokes], training=False)
            print(f"Sample shapes - Noise: {sample_noise.shape}, Styles: {sample_styles.shape}, " 
                f"Strokes: {sample_strokes.shape}, Output: {generated_samples.shape}")

        # Validation evaluation (every 10 epochs)
        if x_val is not None and epoch % 10 == 0:
            # Generate validation fake data with matching temporal dimensions
            val_noise = tf.random.normal(shape=(x_val.shape[0], gan_training_parameters['latent_dim']), mean=0.0, stddev=1.0)
          #  val_styles = tf.random.uniform([x_val.shape[0]], minval=0, maxval=len(data_parameters['labels']), dtype=tf.int32)
           # val_strokes = tf.random.uniform([x_val.shape[0], 180, 1], minval=0, maxval=2, dtype=tf.int32)
            y_val_raw_3d = np.expand_dims(y_val_raw, axis=-1)
            val_styles = tf.convert_to_tensor(y_val_raw_3d, dtype=tf.int64)
            val_strokes = tf.convert_to_tensor(y_stroke_val, dtype=tf.int64)          
            #val_strokes = tf.cast(val_strokes, tf.float32)
            #real_styles = tf.convert_to_tensor(real_styles, dtype=tf.int64)
            #real_strokes = tf.convert_to_tensor(real_strokes, dtype=tf.int64)
            # Generator input: fake_styles as (batch_size,) for embedding layer
            #fake_styles = tf.random.uniform([batch_size], minval=0, maxval=num_styles, dtype=tf.int32)
            #val_styles = tf.gather(real_styles, tf.random.uniform([y_val_sparse.shape[0]], 0, y_val_sparse.shape[0], dtype=tf.int64))
        # tf.print(fake_styles)
            
            #val_strokes = tf.gather(real_strokes, tf.random.uniform([y_stroke_val.shape[0]], 0, y_stroke_val.shape[0], dtype=tf.int64))
            val_stroke_count = tf.reduce_sum(val_strokes).numpy()
            print("Total validation stroke count:", val_stroke_count)
            # Generate validation fake data with proper channel dimensions
            val_fake_data = generator([val_noise, val_styles, val_strokes], training=False) 
            fake_strokes_float = val_fake_data[..., 7]
            fake_styles_float = val_fake_data[..., 6]

            tf.print("Total stroke max value:", tf.reduce_max(fake_strokes_float))
            tf.print("Total stroke min value:", tf.reduce_min(fake_strokes_float))
            # Round the stroke outputs at a threshold, e.g. 0.5
            fake_strokes_bin = tf.where(fake_strokes_float > 0.5, 1, 0)
            #fake_styles_float = tf.round(fake_styles_float)
            print("Real raw style outputs:\n", y_val_raw)
            print("Fake style outputs:\n", fake_styles_float.numpy())
            # Sum them up across the batch and/or timesteps
            # For example, summing all strokes in the batch:
            stroke_count = tf.reduce_sum(fake_strokes_bin)

            print("Rounded fake stroke outputs:\n", fake_strokes_bin.numpy())
            print("Total fake stroke count:", stroke_count.numpy())
            real_stroke_count = np.sum(y_stroke_val)
            fake_stroke_count = stroke_count#tf.reduce_sum(val_fake_data[..., 7]).numpy()

            print(f"Real Stroke Count: {real_stroke_count} | Fake Stroke Count: {fake_stroke_count}")
           
            # Compare real vs fake data shapes (should both be [batch, 180, 8])
            print(f"\nValidation Stats:")
            print(f"Real data shape: {x_val.shape} (sensors + labels)")
            print(f"Generated shape: {val_fake_data.shape}")
            real_mean = np.mean(x_val[..., :6])
            real_min = np.min(x_val[..., :6])
            real_max = np.max(x_val[..., :6])
            real_std = np.std(x_val[..., :6])
            fake_mean = tf.reduce_mean(val_fake_data[..., :6]).numpy()
            fake_min = tf.reduce_min(val_fake_data[..., :6]).numpy()
            fake_max = tf.reduce_max(val_fake_data[..., :6]).numpy()
            fake_std = tf.math.reduce_std(val_fake_data[..., :6]).numpy()

            print(f"Real stats | Mean: {real_mean:.2f}, Std: {real_std:.2f}, Min: {real_min:.2f}, Max: {real_max:.2f}")
            print(f"Fake stats | Mean: {fake_mean:.2f}, Std: {fake_std:.2f}, Min: {fake_min:.2f}, Max: {fake_max:.2f}")
            if abs(real_mean - fake_mean) > 0.5:
                print(f"Significant mean discrepancy: Real {real_mean:.2f} vs Fake {fake_mean:.2f}")



        # After epoch completes
        with summary_writer.as_default():
            if epoch % 10 == 0:
                tf.summary.histogram('real_samples', x_val, step=step_global)
                tf.summary.histogram('fake_samples', val_fake_data, step=step_global)
                tf.summary.scalar('d_accuracy_real', tf.reduce_mean(tf.cast(d_loss_real < 0.5, tf.float32)), step=step_global)
                tf.summary.scalar('d_accuracy_fake', tf.reduce_mean(tf.cast(d_loss_fake > 0.5, tf.float32)), step=step_global)

                tf.summary.scalar('g_fake_std', tf.math.reduce_std(val_fake_data), step=step_global)
                tf.summary.scalar('real_std', real_std, step=step_global)
                tf.summary.scalar('fake_std', fake_std, step=step_global)
                #real_stroke_count = np.sum(x_val[..., 7])  # Count real stroke labels
                real_stroke_count = np.sum(y_stroke_val)
                fake_stroke_count = stroke_count#tf.reduce_sum(val_fake_data[..., 7]).numpy()

                print(f"Real Stroke Count: {real_stroke_count} | Fake Stroke Count: {fake_stroke_count}")
                tf.summary.scalar('real_stroke_count', real_stroke_count, step=step_global)
                tf.summary.scalar('fake_stroke_count', fake_stroke_count, step=step_global)
                fake_variance = tf.math.reduce_variance(val_fake_data, axis=0).numpy()
                print(f"Fake sample variance (mean across features): {np.mean(fake_variance):.4f}")
                tf.summary.scalar('fake_variance', np.mean(fake_variance), step=step_global)
            # Log sample images every 25 epochs
            if epoch % 25 == 0:
                # Get sample real and fake data
                start_index = 100
                num_samples = 5

                real_samples = x_val[start_index : start_index + num_samples]  # Get 5 samples starting at row 250

                # Generate matching fake samples
                # Create interpolated noise between two random vectors
                z1 = tf.random.normal([1, gan_training_parameters['latent_dim']])  # Shape (1, 100)
                z2 = tf.random.normal([1, gan_training_parameters['latent_dim']])  # Shape (1, 100)
                ratios = tf.reshape(tf.linspace(0., 1., num_samples), (-1, 1))  # Shape (5, 1)
                noise = z1 + (z2 - z1) * ratios  # Shape (5, 100)
                noise = tf.random.normal(shape=(num_samples, gan_training_parameters['latent_dim']), mean=0.0, stddev=1.0)
                """
                styles = tf.random.uniform([num_samples], 
                                        minval=0, 
                                        maxval=len(data_parameters['labels']), 
                                        dtype=tf.int32)
                strokes = tf.random.uniform([num_samples, 180, 1], 
                                        minval=0, 
                                        maxval=2, 
                                        dtype=tf.int32)
                """
                #real_styles = tf.convert_to_tensor(y_val_sparse, dtype=tf.int64)
                #real_strokes = tf.convert_to_tensor(y_stroke_val, dtype=tf.int64)   
               # y_val_raw_3d = np.expand_dims(y_val_raw, axis=-1)

                #real_styles = tf.convert_to_tensor(real_styles, dtype=tf.int64)
                #real_strokes = tf.convert_to_tensor(real_strokes, dtype=tf.int64)
                # Generator input: fake_styles as (batch_size,) for embedding layer
                #fake_styles = tf.random.uniform([batch_size], minval=0, maxval=num_styles, dtype=tf.int32)
                #styles = tf.gather(real_styles, tf.random.uniform([num_samples], 0, num_samples, dtype=tf.int64))
            # tf.print(fake_styles)
                #strokes = tf.gather(real_strokes, tf.random.uniform([num_samples], 0, num_samples, dtype=tf.int64))
                y_val_raw_3d = np.expand_dims(y_val_raw, axis=-1)

                styles = tf.convert_to_tensor(y_val_raw_3d[start_index : start_index + num_samples], dtype=tf.int64)
                strokes =  tf.convert_to_tensor(y_stroke_val[start_index : start_index + num_samples], dtype=tf.int64)
                fake_samples = generator([noise, styles, strokes], training=False)
                
                # Create and log figure
                fig = plot_samples(real_samples, fake_samples.numpy(), num_samples=num_samples)
                tf.summary.image("Sample Comparison", 
                            plot_to_image(fig), 
                            step=epoch)


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


def plot_samples(real_samples, fake_samples, num_samples=5):
    """Plot all 6 sensor channels with stroke peak indicators"""
    plt.figure(figsize=(20, 10), facecolor='white')
    
    for i in range(num_samples):
        # Real Sample
        plt.subplot(2, num_samples, i+1)
        
        # Plot sensor channels 0-5 with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        for ch in range(3):  # Now plotting all 6 sensor channels
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
        for ch in range(3):  # Now plotting all 6 sensor channels
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

# Add to discriminator loss calculation:
def compute_gradient_penalty(real_samples, fake_samples):
    # Cast to float32 to resolve dtype mismatch
    real_samples = tf.cast(real_samples, tf.float32)
    fake_samples = tf.cast(fake_samples, tf.float32)
    
    # Compute interpolated samples
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1], 0., 1., dtype=tf.float32)
    interpolated = alpha * real_samples + (1. - alpha) * fake_samples
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    
    gradients = gp_tape.gradient(pred, interpolated)
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    return tf.reduce_mean((gradients_norm - 1.0) ** 2)
def gradient_penalty(real_samples, fake_samples):
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0., 1.)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred = discriminator(interpolates)
    grads = tape.gradient(pred, [interpolates])[0]
    penalty = tf.reduce_mean((tf.norm(grads, axis=[1,2]) - 1.0) ** 2)
    return penalty



train_gan(
    generator, 
    discriminator, 
    data_generator=train_gen, 
    epochs=gan_training_parameters['max_epochs'],
    steps_per_epoch=gan_training_parameters['steps_per_epoch'],
    x_val=x_val_combined,
    y_val_sparse=y_val_sparse,
    y_stroke_val=y_stroke_val,
    y_val_raw=y_val_raw 
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
