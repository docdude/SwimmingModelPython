import keras


def get_default_model_parameters():
    """
    Get a default set of parameters used to define a cnn model
    :return: A dictionary containing parameter names and values
    """
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
    return model_parameters


def get_default_training_parameters():
    """
    Get a default set of parameters used to train a cnn model
    :return: A dictionary containing parameter names and values
    """
    training_parameters = {'lr':              0.0005,
                           'beta_1':          0.9,
                           'beta_2':          0.999,
                           'batch_size':      64,
                           'max_epochs':      10,
                           'steps_per_epoch': 10,
                           'noise_std':       0.01,
                           'mirror_prob':     0.5,
                           'random_rot_deg':  30,
                           'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                           'labels':          [0, 1, 2, 3, 4]
                           }
    return training_parameters


def cnn_model(input_shape, model_parameters, use_seed=True):
    """
    Returns a cnn model based on an input shape and a set of model parameters
    :param input_shape: A tuple with 3 fields: (x, y, 1). Where x is the window length and y the slide length
    :param model_parameters: A dictionary of model parameters
    :param use_seed: A boolean value indicating whether to use a fixed random seed or not.
    :return: A keras sequential model
    """
    num_cl = len(model_parameters['filters'])
    num_fcl = len(model_parameters['units'])
    cnt_layer = 0
    model = keras.Sequential()
    if use_seed:
        seed = 1337
    else:
        seed = None
    # Convolutional layers
    for i in range(num_cl):
        if model_parameters['max_norm'][cnt_layer] is None:
            kernel_constraint = None
        else:
            kernel_constraint = keras.constraints.max_norm(model_parameters['max_norm'][cnt_layer])
        if model_parameters['l2_reg'][cnt_layer] is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = keras.regularizers.l2(model_parameters['l2_reg'][cnt_layer])
        if 'strides' in model_parameters.keys():
            if model_parameters['strides'][i] is None:
                strides = 1
            else:
                strides = (model_parameters['strides'][i], 1)
        else:
            strides = 1
        if i == 0:
            model.add(keras.layers.Conv2D(input_shape=input_shape,
                                          filters=model_parameters['filters'][i],
                                          kernel_size=(model_parameters['kernel_sizes'][i], 1),
                                          strides=strides,
                                          kernel_constraint=kernel_constraint,
                                          kernel_regularizer=kernel_regularizer,
                                          kernel_initializer=keras.initializers.glorot_normal(seed=seed),
                                          bias_initializer='zeros'))
        else:
            model.add(keras.layers.Conv2D(filters=model_parameters['filters'][i],
                                          kernel_size=(model_parameters['kernel_sizes'][i], 1),
                                          strides=strides,
                                          kernel_constraint=kernel_constraint,
                                          kernel_regularizer=kernel_regularizer,
                                          kernel_initializer=keras.initializers.glorot_normal(seed=seed),
                                          bias_initializer='zeros'))
        if model_parameters['batch_norm'][cnt_layer]:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(model_parameters['activation'][cnt_layer]))
        if model_parameters['max_pooling'][i] is not None:
            model.add(keras.layers.MaxPooling2D((model_parameters['max_pooling'][i], 1)))
        if model_parameters['drop_out'][cnt_layer] is not None:
            model.add(keras.layers.Dropout(model_parameters['drop_out'][cnt_layer], seed=seed))
        cnt_layer = cnt_layer + 1
    model.add(keras.layers.Flatten())
    # Fully connected layers
    for i in range(num_fcl):
        if model_parameters['max_norm'][cnt_layer] is None:
            kernel_constraint = None
        else:
            kernel_constraint = keras.constraints.max_norm(model_parameters['max_norm'][cnt_layer])
        if model_parameters['l2_reg'][cnt_layer] is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = keras.regularizers.l2(model_parameters['l2_reg'][cnt_layer])
        model.add(keras.layers.Dense(units=model_parameters['units'][i],
                                     kernel_constraint=kernel_constraint,
                                     kernel_regularizer=kernel_regularizer,
                                     kernel_initializer=keras.initializers.he_uniform(seed=seed),
                                     bias_initializer='zeros'))
        if model_parameters['batch_norm'][cnt_layer]:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(model_parameters['activation'][cnt_layer]))
        if model_parameters['drop_out'][cnt_layer] is not None:
            model.add(keras.layers.Dropout(model_parameters['drop_out'][cnt_layer], seed=seed))
        cnt_layer = cnt_layer + 1
    # Final soft-max layer
    model.add(keras.layers.Dense(len(model_parameters['labels']),
                                 activation='softmax',
                                 kernel_initializer=keras.initializers.he_uniform(seed=seed)))
    return model


def main():
    input_shape = (180, 6, 1)
    model_parameters = get_default_model_parameters()
    model = cnn_model(input_shape, model_parameters)
    print(model.summary())


if __name__ == '__main__':
    main()


