# Load models and predict on recordings
# Save traces to a file for easy post-processing implementations
import os
import tensorflow as tf
import cnn_vanilla
import utils 

data_path = 'data/processed_30Hz_relabeled'
results_path = 'tutorial_save_path/'
save_path = 'tutorial_save_path/'


#users = ['2','6','7','11']
users_all = utils.folders_in_path(data_path)
users = [u for u in users_all] #if u not in users_ignore]
users.sort()
#users = ['2','6','7','11']
# Hyper-parameters for loading data.
data_parameters = {'users':                users,   # Users whose data is loaded
                   'labels':               [0, 1, 2, 3, 4, 5],  # Labels we want to use
                   'combine_labels':       {0: [0, 5]},     # Labels we want to combine. Here I am combining NULL and
                                                            # TURN into NULL
                   'data_columns':         ['ACC_0', 'ACC_1', 'ACC_2', 'GYRO_0', 'GYRO_1', 'GYRO_2', 'MAG_0',
                                            'MAG_1', 'MAG_2', 'PRESS', 'LIGHT'],    # The sensor data we want to load
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
                    'labels':         [0, 1, 2, 3, 4]
                    }

# The input shape of the CNN
input_shape = (data_parameters['win_len'], len(data_parameters['data_columns']), 1)

for (i, user) in enumerate(users):
    print("Restoring on %s. %d of %d" % (user, i+1, len(users)))
        # The cnn_vanilla module contains contains everything to generate the CNN model
    model = cnn_vanilla.cnn_model(input_shape, model_parameters)
    #model.summary()
    model.load_weights(os.path.join(results_path, user, 'model_best.weights.h5'))
    model_h5_path = os.path.join(save_path, user, 'model_best.h5')
    model_path = os.path.join(save_path, user, 'model_best.keras')
    # Save model in h5 format
    model.save(model_h5_path)
    # Save model in native Keras format
    model.save(model_path)
    