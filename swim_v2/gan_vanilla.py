import tensorflow as tf

def get_default_generator_parameters():
    """
    Get a default set of parameters used to define a gan model
    :return: A dictionary containing parameter names and values
    """
    generator_parameters = {'filters':    [128, 64, 64, 64],
                        'kernel_sizes':   [3, 3, 3, 3],
                        'strides':        [None, None, None, None],
                        'max_pooling':    [3, 3, 3, 3],
                        'units':          [128],
                        'activation':     ['relu', 'relu', 'relu', 'relu', 'tanh'],
                        'batch_norm':     [False, False, False, False, False],
                        'drop_out':       [0.5, 0.75, 0.25, 0.1, 0.25],
                        'max_norm':       [0.1, 0.1, None, 4.0, 4.0],
                        'l2_reg':         [None, None, None, None, None],
                        'labels':         [0, 1, 2, 3, 4]
                        }
    return generator_parameters

def get_default_discriminator_parameters():
    """
    Get a default set of parameters used to define a gan model
    :return: A dictionary containing parameter names and values
    """
    discriminator_parameters = {'filters':[128, 64, 64, 64],
                        'kernel_sizes':   [3, 3, 3, 3],
                        'strides':        [2, 2, 2, 2],
                        'max_pooling':    [3, 3, 3, 3],
                        'units':          [128],
                        'activation':     ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu'],
                        'batch_norm':     [False, False, False, False, False],
                        'drop_out':       [0.5, 0.75, 0.25, 0.1, 0.25],
                        'max_norm':       [0.1, 0.1, None, 4.0, 4.0],
                        'l2_reg':         [None, None, None, None, None],
                        'labels':         [0, 1, 2, 3, 4]
                        }
    return discriminator_parameters


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


def build_generator(latent_dim, output_shape, generator_parameters, use_seed=True):
    """
    Returns a gan model based on an input shape and a set of model parameters
    :param input_shape: A double with 2 fields: (x, y). Where x is the window length and y the slide length
    :param generator_parameters: A dictionary of discriminator parameters
    :param use_seed: A boolean value indicating whether to use a fixed random seed or not.
    :return: A keras sequential model
    """
    num_cl = len(generator_parameters['filters'])
    num_fcl = len(generator_parameters['units'])
    cnt_layer = 0
    model = tf.keras.Sequential(name="Generator")
    if use_seed:
        seed = 1337
    else:
        seed = None
    # Convolutional layers
    for i in range(num_cl):
        if generator_parameters['max_norm'][cnt_layer] is None:
            kernel_constraint = None
        else:
            kernel_constraint = tf.keras.constraints.max_norm(generator_parameters['max_norm'][cnt_layer])
        if generator_parameters['l2_reg'][cnt_layer] is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = tf.keras.regularizers.l2(generator_parameters['l2_reg'][cnt_layer])
        if 'strides' in generator_parameters.keys():
            if generator_parameters['strides'][i] is None:
                strides = 1
            else:
                strides = (generator_parameters['strides'][i])
        else:
            strides = 1
        if i == 0:
            model.add(tf.keras.layers.Input(shape=(latent_dim,)))
            model.add(tf.keras.layers.Dense(output_shape[0] * output_shape[1],
                                    kernel_constraint=kernel_constraint,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=tf.keras.initializers.he_uniform(seed=seed),
                                    bias_initializer='zeros'))
            model.add(tf.keras.layers.Reshape((output_shape[0], output_shape[1])))
            model.add(tf.keras.layers.Conv1D(
                                          filters=generator_parameters['filters'][i],
                                          kernel_size=(generator_parameters['kernel_sizes'][i]),
                                          strides=strides,
                                          kernel_constraint=kernel_constraint,
                                          kernel_regularizer=kernel_regularizer,
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
                                          bias_initializer='zeros'))
        else:
            model.add(tf.keras.layers.Conv1D(filters=generator_parameters['filters'][i],
                                          kernel_size=(generator_parameters['kernel_sizes'][i]),
                                          strides=strides,
                                          kernel_constraint=kernel_constraint,
                                          kernel_regularizer=kernel_regularizer,
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
                                          bias_initializer='zeros'))

        if generator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(generator_parameters['activation'][cnt_layer]))
        if generator_parameters['max_pooling'][i] is not None:
            model.add(tf.keras.layers.MaxPooling1D((generator_parameters['max_pooling'][i])))
        if generator_parameters['drop_out'][cnt_layer] is not None:
            model.add(tf.keras.layers.Dropout(generator_parameters['drop_out'][cnt_layer], seed=seed))
        cnt_layer = cnt_layer + 1

    return model

def build_discriminator(input_shape, discriminator_parameters, use_seed=True):
    """
    Returns a gan model based on an input shape and a set of model parameters
    :param input_shape: A tuple with 3 fields: (x, y, 1). Where x is the window length and y the slide length
    :param discriminator_parameters: A dictionary of discriminator parameters
    :param use_seed: A boolean value indicating whether to use a fixed random seed or not.
    :return: A keras sequential model
    """
    num_cl = len(discriminator_parameters['filters'])
    num_fcl = len(discriminator_parameters['units'])
    cnt_layer = 0
    model = tf.keras.Sequential(name="Discriminator")
    if use_seed:
        seed = 1337
    else:
        seed = None
    # Convolutional layers
    for i in range(num_cl):
        if discriminator_parameters['max_norm'][cnt_layer] is None:
            kernel_constraint = None
        else:
            kernel_constraint = tf.keras.constraints.max_norm(discriminator_parameters['max_norm'][cnt_layer])
        if discriminator_parameters['l2_reg'][cnt_layer] is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = tf.keras.regularizers.l2(discriminator_parameters['l2_reg'][cnt_layer])
        if 'strides' in discriminator_parameters.keys():
            if discriminator_parameters['strides'][i] is None:
                strides = 1
            else:
                strides = (discriminator_parameters['strides'][i])
        else:
            strides = 1
        if i == 0:
            model.add(tf.keras.layers.Input(shape=input_shape))

            model.add(tf.keras.layers.Conv1D(
                                          filters=discriminator_parameters['filters'][i],
                                          kernel_size=(discriminator_parameters['kernel_sizes'][i]),
                                          strides=strides,
                                          padding='same',  # Use 'same' padding to avoid dimension reduction
                                          kernel_constraint=kernel_constraint,
                                          kernel_regularizer=kernel_regularizer,
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
                                          bias_initializer='zeros'))
        else:
            model.add(tf.keras.layers.Conv1D(filters=discriminator_parameters['filters'][i],
                                          kernel_size=(discriminator_parameters['kernel_sizes'][i]),
                                          strides=strides,
                                          padding='same',  # Use 'same' padding to avoid dimension reduction
                                          kernel_constraint=kernel_constraint,
                                          kernel_regularizer=kernel_regularizer,
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
                                          bias_initializer='zeros'))
        if discriminator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer]))
        if discriminator_parameters['max_pooling'][i] is not None:
            # Ensure the pooling size is appropriate for the current input dimension
            current_length = model.layers[-1].output_shape[1]
            pooling_size = min(discriminator_parameters['max_pooling'][i], current_length)
            model.add(tf.keras.layers.MaxPooling1D(pool_size=pooling_size))
        if discriminator_parameters['drop_out'][cnt_layer] is not None:
            model.add(tf.keras.layers.Dropout(discriminator_parameters['drop_out'][cnt_layer], seed=seed))
        cnt_layer = cnt_layer + 1
    model.add(tf.keras.layers.Flatten())
    # Fully connected layers
    for i in range(num_fcl):
        if discriminator_parameters['max_norm'][cnt_layer] is None:
            kernel_constraint = None
        else:
            kernel_constraint = tf.keras.constraints.max_norm(discriminator_parameters['max_norm'][cnt_layer])
        if discriminator_parameters['l2_reg'][cnt_layer] is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = tf.keras.regularizers.l2(discriminator_parameters['l2_reg'][cnt_layer])
        model.add(tf.keras.layers.Dense(units=discriminator_parameters['units'][i],
                                     kernel_constraint=kernel_constraint,
                                     kernel_regularizer=kernel_regularizer,
                                     kernel_initializer=tf.keras.initializers.he_uniform(seed=seed),
                                     bias_initializer='zeros'))
        if discriminator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer]))
        if discriminator_parameters['drop_out'][cnt_layer] is not None:
            model.add(tf.keras.layers.Dropout(discriminator_parameters['drop_out'][cnt_layer], seed=seed))
        cnt_layer = cnt_layer + 1
    # Final layer
    model.add(tf.keras.layers.Dense(1,
                                 activation='sigmoid',
                                 kernel_initializer=tf.keras.initializers.he_uniform(seed=seed)))
    return model


def main():
    input_shape = (180, 8) # Load sensor data, accel, gyro, label, stroke_labels
    latent_dim = 100

    generator_parameters = get_default_generator_parameters()
    discriminator_parameters = get_default_discriminator_parameters()

    generator = build_generator(latent_dim, input_shape, generator_parameters)
    discriminator = build_discriminator(input_shape, discriminator_parameters)
    generator.summary()
    discriminator.summary()

if __name__ == '__main__':
    main()

