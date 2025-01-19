# A basic tutorial in how I load data and train a model
import cnn_vanilla1 #LSTM
import cnn_vanilla2 #TimeDistibuted layer wrapping dense
import cnn_vanilla3 #2 dense layers after main backbone
import cnn_vanilla4 #single dense layer after main backbone
import cnn_vanilla5 #convtranspose layer to upsample no dense conv1d sigmoid final
import cnn_vanilla6 #single dense layer using swim_style_branch final connected dense
import cnn_vanilla7 # parallel branches
import cnn_vanilla8 # parallel branches with weighted loss
import test_lstm
import utils
import plot_utils
import learning_data
import os
import random as rn
import tensorflow as tf
import numpy as np
import pickle
import datetime
from tensorboard.plugins.hparams import api as hp
from sklearn.utils import class_weight

# A path to re-sampled recordings which are organized into folders by user name.
data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified_users'

# Path to where we want to save the training results
save_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch_Parallel5_weighted'
#save_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epochLSTM3_weighted'
run_name = 'Parallel5_weighted'
# A list of user names which are loaded.

users_all = utils.folders_in_path(data_path)
users = [u for u in users_all] #if u not in users_ignore]
users.sort(key=int)

# Keeping it simple. Comment this out and use the code above if you want to load everybody
users = ['2','6','7','11']
users_test = users

# List of users we want to train a model for
#users_test = ['2','6','7','11']

# Hyper-parameters for loading data.
data_parameters = {'users':                users,   # Users whose data is loaded
                   'labels':               [0, 1, 2, 3, 4, 5],  # Labels we want to use
                   'combine_labels':       {0: [0, 5]},     # Labels we want to combine. Here I am combining NULL and
                                                            # TURN into NULL
                   'data_columns':         ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2'],#, 'MAG_0',
                                      #      'MAG_1', 'MAG_2', 'PRESS', 'LIGHT'],    # The sensor data we want to load
                                       #     'MAG_1', 'MAG_2'],    # The sensor data we want to load
                   'stroke_labels': ['stroke_labels'],  # Binary stroke labels: 0 for no stroke, 1 for stroke
                   'time_scale_factors':   [0.9, 1.1],  # time-scaling factors we want to use. A copy is made of each
                                                        # recording with these factors.
                   'win_len':              180,     # The length of the segmentation window in number of samples
                   'slide_len':            30,      # The slide length used for segmentation
                   'window_normalization': 'statistical',   # How we want to normalize the windows. Statistical means
                                                            # zero-mean and unit variance for each signal
                   'label_type':           'majority',  # How we label windows.
                   'majority_thresh':      0.75,    # Proportion of samples in a window that have to have the same label
                   'validation_set': {
                        0: 1,  # Transition (NULL + TURN combined as 0)
                        1: 1,  # Freestyle
                        2: 1,  # Backstroke
                        3: 1,  # Breaststroke
                        4: 1,  # Butterfly
                        'stroke_labels': ['stroke_labels'],  # Track stroke labels separately
                   },
                   'debug':               False,
                   'debug_plot':           False
                   }

# Data is loaded and stored in this object
swimming_data = learning_data.LearningData()

# Load recordings from data_path. Recordings are stored under
# swimming_data.data_dict['original][user_name][recording_name] which is a Pandas DataFrame
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=data_parameters['users'],
                        labels=data_parameters['labels'],
                        stroke_labels=data_parameters['stroke_labels'])
# Combine labels
if data_parameters['combine_labels'] is not None:
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)

# Data augmentation for recordings. This is only for time-scaling. Other data augmentations happen during the learning
# Stored under swimming_data['time_scaled_1.1'][user_name]...
swimming_data.augment_recordings(time_scale_factors=data_parameters['time_scale_factors'])

# Debug visualization time auementation of sensor data and stroke data
if data_parameters['debug_plot']:
    plot_utils.plot_style_specific_axes(swimming_data)


# Compute the locations of the sliding windows in each recording
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

# Compile the windows. Stored under swimming_data.data_windows[group][label][user]['data' or 'label' or 'stroke_label']
# Recordings are still stored under swimming_data.data_dict so a lot of memory might be needed
swimming_data.compile_windows(norm_type=data_parameters['window_normalization'],
                              label_type=data_parameters['label_type'],
                              majority_thresh=data_parameters['majority_thresh'])

# Parameters for the CNN model
model_parameters = {'filters':        [64, 64, 64, 64],
                    'kernel_sizes':   [3, 3, 3, 3],
                    'strides':        [None, None, None, None],
                    'max_pooling':    [3, 3, 3, 3],
                    'units':          [128],
                    'activation':     ['elu', 'elu', 'elu', 'elu', 'elu'],
                    'batch_norm':     [False, False, False, False, False],
                    'drop_out':       [0.5, 0.75, 0.25, 0.1, 0.25],
                    'max_norm':       [0.1, 0.1, None, 4.0, 4.0],
                    'l2_reg':         [None, None, None, None, None],
                    'labels':         swimming_data.labels,
                    'stroke_labels':  swimming_data.stroke_labels
                    }
swim_model_parameters = cnn_vanilla8.get_default_swim_model_parameters()
stroke_model_parameters = cnn_vanilla8.get_default_stroke_model_parameters()

# Parameters for training the CNN model
training_parameters = {'lr':              0.0005,
                       'beta_1':          0.9,
                       'beta_2':          0.999,
                       'batch_size':      256,
                       'max_epochs':      48,      # Keeping small for quick testing
                       'steps_per_epoch': 100,      # Keeping small for quick testing
                       'noise_std':       0.01,    # Noise standard deviation for data augmentation
                       'mirror_prob':     0.5,     # Probability of reversing a window for data augmentation
                       'random_rot_deg':  30,      # [-30, 30] is the range of rotation degrees we sample for each
                                                   # window in the mini-batch
                       'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                       'labels':          swimming_data.labels,
                       'stroke_labels':   swimming_data.stroke_labels
                       }

# The input shape of the CNN
#input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']) + len(data_parameters['stroke_labels']), 1)
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']), 1)


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
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"run-{user_test} + -{run_name}"

    if os.path.exists(experiment_save_path):
        print(f"Skipping: {user_test}, already processed.")
        continue
    else:
        os.makedirs(experiment_save_path)  # Creates all parent directories if they don't exist

    # Users whose data we use for training
    users_train = [u for u in users if u != user_test]

    # Draw users for each class. train_dict and val_dict are dictionaries whose keys are labels and they contain
    # lists of names for each label
    train_dict, val_dict = swimming_data.draw_train_val_dicts(users_train,
                                                              users_per_class=data_parameters['validation_set'])
    if data_parameters['debug']:
        print("Training dictionary: %s" % train_dict)   
        print("Validation dictionary: %s" % val_dict)


    # Calculate stroke label distribution for training set (excluding label 0)
    training_probabilities, training_mean, training_bias = utils.calculate_stroke_label_distribution(
        label_user_dict=train_dict,
        swimming_data=swimming_data,
        data_type="training",
        exclude_label=0
    )

    # Calculate stroke label distribution for validation set (excluding label 0)
    validation_probabilities, validation_mean, validation_bias = utils.calculate_stroke_label_distribution(
        label_user_dict=val_dict,
        swimming_data=swimming_data,
        data_type="validation",
        exclude_label=0
    )

    # Optional: Visualize the distributions
    if data_parameters['debug_plot']:
        plot_utils.plot_label_distribution(training_probabilities, validation_probabilities)    
        feature_indices = [0, 1, 2, 3, 4, 5]  # Example: indices of x, y, z axes for accelerometer

        plot_utils.analyze_feature_distribution(train_dict, swimming_data, feature_indices, data_type="training", exclude_label=0)
        plot_utils.analyze_feature_distribution(val_dict, swimming_data, feature_indices, data_type="validation", exclude_label=0)

    # The generator used to draw mini-batches
    gen = swimming_data.batch_generator_dicts(train_dict=train_dict,
                                              batch_size=training_parameters['batch_size'],
                                              noise_std=training_parameters['noise_std'],
                                              mirror_prob=training_parameters['mirror_prob'],
                                              random_rot_deg=training_parameters['random_rot_deg'],
                                              use_4D=True)

    if data_parameters['debug']:
        # Retrieve and print a batch for inspection
        batch_data_combined, batch_outputs, batch_sample_weights = next(gen)

        # Debugging combined sensor and stroke data
        print(f"Batch data (combined) shape: {batch_data_combined.shape}")

        # Debugging swim style and stroke label outputs
        print(f"Swim Style Output Shape: {batch_outputs['swim_style_output'].shape}")
        print(f"Stroke Label Output Shape: {batch_outputs['stroke_label_output'].shape}")
        print(f"Stroke Label Output Unique Values: {np.unique(batch_outputs['stroke_label_output'])}")

        # Debugging stroke mask
        print(f"Stroke Mask Shape: {batch_sample_weights['stroke_label_output'].shape}")
        print(f"Stroke Mask Unique Values: {np.unique(batch_sample_weights['stroke_label_output'])}")

        # Compare combined stroke data with stroke labels and mask
        example_index = 0  # Choose a batch sample index to debug
        print("\nSample Debug:")
        print(f"Batch Data (Sensor Features + Stroke Labels): {batch_data_combined[example_index, :, :]}")
        print(f"Stroke Labels (from Combined): {batch_outputs['stroke_label_output'][example_index, :, :]}")
        print(f"Stroke Mask (for Combined): {batch_sample_weights['stroke_label_output'][example_index, :, :]}")

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_parameters['lr'], beta_1=training_parameters['beta_1'],
                                      beta_2=training_parameters['beta_2'])

    # Path to the "best" weights w.r.t. the validation accuracy
    best_style_path = os.path.join(experiment_save_path, f'model_best_swim_style_wa_{user_test}.keras')
    best_stroke_path = os.path.join(experiment_save_path, f'model_best_stroke_label_wa_{user_test}.keras')

    
    # Callback to save the best model for swim style accuracy
    checkpoint_swim_style = tf.keras.callbacks.ModelCheckpoint(
        best_style_path,
        monitor='val_swim_style_output_weighted_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # Callback to save the best model for stroke label accuracy
    checkpoint_stroke_label = tf.keras.callbacks.ModelCheckpoint(
        best_stroke_path,
        monitor='val_stroke_label_output_weighted_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # The cnn_vanilla module contains contains everything to generate the CNN model
    model = cnn_vanilla8.cnn_model(input_shape, swim_model_parameters, stroke_model_parameters, use_seed=True, output_bias=training_bias)

    model.summary()

    def weighted_binary_crossentropy(weight_zero=1.0, weight_one=15.0):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            # Reshape y_true to match y_pred if needed
            y_true = tf.reshape(y_true, tf.shape(y_pred))
            
            # Calculate weights maintaining the shape
            weights = y_true * weight_one + (1 - y_true) * weight_zero
            
            # Calculate binary crossentropy
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            
            # Ensure shapes match for multiplication
            weights = tf.squeeze(weights, axis=-1)
            
            # Calculate weighted loss
            weighted_loss = weights * bce
            return tf.reduce_mean(weighted_loss)
        return loss
    def weighted_binary_crossentropy_smooth(weight_zero=1.0, weight_one=8.0):  # Reduced from 15.0
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_true = tf.reshape(y_true, tf.shape(y_pred))
            
            # Add label smoothing
            epsilon = 0.1
            y_true = y_true * (1 - epsilon) + epsilon/2
            
            # Original weighting
            weights = y_true * weight_one + (1 - y_true) * weight_zero
            weights = tf.squeeze(weights, axis=-1)
            
            # BCE with label smoothing
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            
            return tf.reduce_mean(weights * bce)
        return loss   
    model.compile(optimizer=optimizer, loss={'swim_style_output'     : 'categorical_crossentropy',
                                             'stroke_label_output'   :  weighted_binary_crossentropy_smooth(weight_zero=1.0, weight_one=8.0)},
                                       metrics={'swim_style_output'  : ['accuracy'],
                                                'stroke_label_output': ['accuracy']},
                              weighted_metrics={'swim_style_output'  : ['accuracy'],
                                                'stroke_label_output': ['accuracy']})
    # Get the validation data with weights and mask
    x_val, y_val_sparse, y_val_cat, y_stroke_val, y_val_combined, val_sample_weights, val_stroke_mask = swimming_data.get_windows_dict(
        val_dict, return_weights=True, return_mask=True, transition_label=0
    )
    # Flatten y_stroke_val to a 1D array
    #y_stroke_flat = y_val_combined.flatten()  
    #weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_stroke_flat), y=y_stroke_flat)
    #class_weights = dict(enumerate(weights))

    # Print validation data shapes
    if data_parameters['debug']:
        print(f"x_val shape: {x_val.shape}")
        print(f"y_val_cat shape: {y_val_cat.shape}")
        print(f"y_val_combined shape: {y_val_combined.shape}")
        print(f"val_sample_weights shape: {val_sample_weights.shape}")
        print(f"val_stroke_mask shape: {val_stroke_mask.shape}")
        # Check unique values in stroke mask
        print(f"Stroke mask unique values: {np.unique(val_stroke_mask)}")

    # Pass to model.fit
    history = model.fit(
        gen,  # Updated batch generator
        validation_data=(
            x_val,
            {'swim_style_output': y_val_cat, 'stroke_label_output': y_stroke_val},
            {'swim_style_output': val_sample_weights, 'stroke_label_output': val_stroke_mask}
        ),
        epochs=training_parameters['max_epochs'],
        steps_per_epoch=training_parameters['steps_per_epoch'],
        callbacks=[checkpoint_swim_style, checkpoint_stroke_label, tensorboard_callback]
    )

    # Save model in h5 format
    model_h5_path = os.path.join(experiment_save_path, f'model_{user_test}.h5')
    model.save(model_h5_path)
    # Save model in native Keras format
    model_keras_path = os.path.join(experiment_save_path, f'model_{user_test}.keras')
    model.save(model_keras_path)
    # Saving the history and parameters
    with open(os.path.join(experiment_save_path, 'train_val_dicts.pkl'), 'wb') as f:
        pickle.dump([train_dict, val_dict], f)
    with open(os.path.join(experiment_save_path, 'history.pkl'), 'wb') as f:
        pickle.dump([history.history], f)
    with open(os.path.join(experiment_save_path, 'data_parameters.pkl'), 'wb') as f:
        pickle.dump([data_parameters], f)
    with open(os.path.join(experiment_save_path, 'model_parameters.pkl'), 'wb') as f:
        pickle.dump([model_parameters], f)
    with open(os.path.join(experiment_save_path, 'training_parameters.pkl'), 'wb') as f:
        pickle.dump([training_parameters], f)

