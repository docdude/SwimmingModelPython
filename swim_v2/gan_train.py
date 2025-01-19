# A basic tutorial in how I load data and train a model
import cnn_vanilla
import utils
import learning_data
import os
import random as rn
import tensorflow as tf
import numpy as np
import pickle
import datetime
from tensorboard.plugins.hparams import api as hp
# GAN Model: Import `gan_vanilla.py`
from gan_vanilla import build_generator, build_discriminator

# A path to re-sampled recordings which are organized into folders by user name.
data_path = 'data_modified_users'

# Path to where we want to save the training results
save_path = 'tutorial_save_path_gan'

# A list of user names which are loaded.

users_all = utils.folders_in_path(data_path)
users = [u for u in users_all] #if u not in users_ignore]
users.sort()

# Keeping it simple. Comment this out and use the code above if you want to load everybody
#users = ['2','6','7','11']

# List of users we want to train a model for
#users_test = ['2','6','7','11']

users_test = users
# Hyper-parameters for loading data.
data_parameters = {'users':                users,   # Users whose data is loaded
                   'labels':               [0, 1, 2, 3, 4, 5],  # Labels we want to use
                   'combine_labels':       {0: [0, 5]},     # Labels we want to combine. Here I am combining NULL and
                                                            # TURN into NULL
                   'data_columns':         ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],#, 'MAG_0',
                                      #      'MAG_1', 'MAG_2', 'PRESS', 'LIGHT'],    # The sensor data we want to load
                                       #     'MAG_1', 'MAG_2'],    # The sensor data we want to load
                   'time_scale_factors':   [0.9, 1.1],  # time-scaling factors we want to use. A copy is made of each
                                                        # recording with these factors.
                   'win_len':              180,     # The length of the segmentation window in number of samples
                   'slide_len':            30,      # The slide length used for segmentation
                   'window_normalization': 'statistical',   # How we want to normalize the windows. Statistical means
                                                            # zero-mean and unit variance for each signal
                   'label_type':           'majority',  # How we label windows.
                   'majority_thresh':      0.75,    # Proportion of samples in a window that have to have the same label
                   'validation_set':       {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},  # The number of users that represent each
                                                                            # class in the validation set
                   }

# Data is loaded and stored in this object
swimming_data = learning_data.LearningData()

# Load recordings from data_path. Recordings are stored under
# swimming_data.data_dict['original][user_name][recording_name] which is a Pandas DataFrame
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=data_parameters['users'],
                        labels=data_parameters['labels'])

# Combine labels
if data_parameters['combine_labels'] is not None:
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)

# Data augmentation for recordings. This is only for time-scaling. Other data augmentations happen during the learning
# Stored under swimming_data['time_scaled_1.1'][user_name]...
swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

# Compute the locations of the sliding windows in each recording
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile the windows. Stored under swimming_data.data_windows[group][label][user]['data' or 'label']
# Recordings are still stored under swimming_data.data_dict so a lot of memory might be needed
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])

# Parameters for the generator model
generator_parameters = {'filters':    [64, 64, 64, 64],
                    'kernel_sizes':   [3, 3, 3, 3],
                    'strides':        [None, None, None, None],
                    'max_pooling':    [3, 3, 3, 3],
                    'units':          [128],
                    'activation':     ['relu', 'relu', 'relu', 'relu', 'relu'],
                    'batch_norm':     [False, False, False, False, False],
                    'drop_out':       [0.5, 0.75, 0.25, 0.1, 0.25],
                    'max_norm':       [0.1, 0.1, None, 4.0, 4.0],
                    'l2_reg':         [None, None, None, None, None],
                    'labels':         swimming_data.labels
                    }

# Parameters for the generator model
discriminator_parameters = {'filters':    [64, 64, 64, 64],
                    'kernel_sizes':   [3, 3, 3, 3],
                    'strides':        [2, 2, 2, 2],
                    'max_pooling':    [3, 3, 3, 3],
                    'units':          [128],
                    'activation':     ['relu', 'relu', 'relu', 'relu', 'relu'],
                    'batch_norm':     [False, False, False, False, False],
                    'drop_out':       [0.5, 0.75, 0.25, 0.1, 0.25],
                    'max_norm':       [0.1, 0.1, None, 4.0, 4.0],
                    'l2_reg':         [None, None, None, None, None],
                    'labels':         swimming_data.labels
                    }

# Define GAN training parameters
gan_training_parameters = {
                'lr':              0.0002,
                'beta_1':          0.9, # ?0.5 
                'beta_2':          0.999,
                'batch_size':      64,
                'max_epochs':      100,
                'latent_dim':      100,
                'steps_per_epoch': 100,      # Keeping small for quick testing
                'noise_std':       0.01,    # Noise standard deviation for data augmentation
                'mirror_prob':     0.5,     # Probability of reversing a window for data augmentation
                'random_rot_deg':  30,      # [-30, 30] is the range of rotation degrees we sample for each
                                            # window in the mini-batch
                'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                'labels':          swimming_data.labels
}


# The input shape of the GAN
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']))

# Train all models
for (i, user_test) in enumerate(users_test):
    # Random seed stuff. Maybe this is overkill
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1337)
    tf.random.set_seed(1337)

    # Path for saving results
    print("Running experiment: %s" % user_test)
    experiment_save_path = os.path.join(save_path, user_test)

    # A path to log directory for Tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"run-{user_test}"

    if os.path.exists(experiment_save_path):
        print("Skipping: %s" % user_test)
        continue
    else:
        os.mkdir(experiment_save_path)
   

    # Users whose data we use for training
    users_train = [u for u in users if u != user_test]

    # Draw users for each class. train_dict and val_dict are dictionaries whose keys are labels and they contain
    # lists of names for each label
    train_dict, val_dict = swimming_data.draw_train_val_dicts(users_train,
                                                              users_per_class=data_parameters['validation_set'])

    print("Validation dictionary: %s" % val_dict)

    # The generator used to draw mini-batches
    gen = swimming_data.batch_generator_dicts(train_dict=train_dict,
                                              batch_size=gan_training_parameters['batch_size'],
                                              noise_std=gan_training_parameters['noise_std'],
                                              mirror_prob=gan_training_parameters['mirror_prob'],
                                              random_rot_deg=gan_training_parameters['random_rot_deg'])

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=gan_training_parameters['lr'], beta_1=gan_training_parameters['beta_1'],
                                      beta_2=gan_training_parameters['beta_2'])

    # Path to the "best" weights w.r.t. the validation accuracy
    best_path = os.path.join(experiment_save_path, 'model_best.weights.h5')

    # Which model is the best model and where we save it
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(best_path, monitor='val_weighted_acc', verbose=1,
                                                 save_best_only=True, save_weights_only=True, mode='max')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Get the validation data
    x_val, y_val_cat, val_sample_weights = swimming_data.get_windows_dict(val_dict, return_weights=True)

    try:
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
    except Exception as e:
        print("Failed to reshape x_val:", e)

    # Initialize GAN model
    generator = build_generator(gan_training_parameters['latent_dim'])
    discriminator = build_discriminator()

    # Compile GAN
    gan = GAN(generator, discriminator, gan_training_parameters['learning_rate'], gan_training_parameters['beta_1'])
    gan.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'], weighted_metrics=['acc'])

    # Train the model
    history = gan.fit(gen, validation_data=(x_val, y_val_cat, val_sample_weights),
                                  epochs=gan_training_parameters['max_epochs'],
                                  steps_per_epoch=gan_training_parameters['steps_per_epoch'],
                                  callbacks=[checkpoint_callback, tensorboard_callback])
    # Save model in h5 format
    generator_h5_path = os.path.join(experiment_save_path, 'generator.h5')
    generator.save(generator_h5_path)
        # Save model in h5 format
    discriminator_h5_path = os.path.join(experiment_save_path, 'discriminator.h5')
    discriminator.save(discriminator_h5_path)

    # Saving the history and parameters
    with open(os.path.join(experiment_save_path, 'train_val_dicts.pkl'), 'wb') as f:
        pickle.dump([train_dict, val_dict], f)
    with open(os.path.join(experiment_save_path, 'history.pkl'), 'wb') as f:
        pickle.dump([history.history], f)
    with open(os.path.join(experiment_save_path, 'data_parameters.pkl'), 'wb') as f:
        pickle.dump([data_parameters], f)
    with open(os.path.join(experiment_save_path, 'gan_training_parameters.pkl'), 'wb') as f:
        pickle.dump([gan_training_parameters], f)
    with open(os.path.join(experiment_save_path, 'generator_parameters.pkl'), 'wb') as f:
        pickle.dump([generator_parameters], f)
    with open(os.path.join(experiment_save_path, 'discriminator_parameters.pkl'), 'wb') as f:
        pickle.dump([discriminator_parameters], f)
