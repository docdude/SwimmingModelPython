import tensorflow as tf

def get_default_generator_parameters():
    """
    Get a default set of parameters used to define a GAN generator model
    :return: A dictionary containing parameter names and values
    """

    generator_parameters = {
        'dense_units': 128 * 45,
        'reshape_dim': (45, 128),
        'filters':      [64, 32, 8],
        'kernel_sizes': [5, 5, 5],
        'strides':      [2, 2, 2],
        #'max_pooling':  [3, 3, 3],
        'max_pooling':  [None, None, None],
        'units': [128 * 45],
        'activation':  [None, 'relu', 'relu', 'tanh'],

        'batch_norm':   [True, True, True],
        'drop_out':     [0.5, 0.5, 0.5],
        #'max_norm': [0.1, 0.1, None, 4.0, 4.0],
        'max_norm': [None, None, None, None, None],
        'l2_reg': [None, None, None, None, None]

    }

    return generator_parameters

def get_default_discriminator_parameters():
    """
    Get a default set of parameters used to define a GAN discriminator model
    :return: A dictionary containing parameter names and values
    """

    discriminator_parameters = {
        'filters':      [64, 128],
        'kernel_sizes': [5, 5],
        'strides':      [2, 2],
        #'max_pooling':  [3, 3],
        'max_pooling':  [None, None, None],
        'activation':  ['leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid'],
        'dense_units': 128,
        'units':        [128, 1],
        'batch_norm':   [True, True, True, True],
        'drop_out':     [0.3, 0.3, None, None],
        #'max_norm': [0.1, 0.1, None, 4.0, 4.0],
        'max_norm': [None, None, None, None, None],
        'l2_reg': [None, None, None, None, None],
    }    
    return discriminator_parameters

def build_generator(latent_dim, output_shape, generator_parameters, use_seed=True):
    """
    Build the Generator model with configurable parameters.

    Args:
        latent_dim (int): Dimension of the latent space.
        output_shape (tuple): Desired output shape (sequence length, feature count).
        generator_parameters (dict): Dictionary containing parameters for the generator layers.

    Returns:
        tf.keras.Model: The Generator model.
    """
    num_cl = len(generator_parameters['filters'])
    num_fcl = len(generator_parameters['units'])
    cnt_layer = 0

    if use_seed:
        seed = 1337
    else:
        seed = None
        
    model = tf.keras.Sequential(name="Generator_v2")

    #model.add(tf.keras.layers.Dense(generator_parameters['dense_units'], activation="relu"))
    #model.add(tf.keras.layers.Reshape(generator_parameters['reshape_dim']))

    for i in range(num_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(generator_parameters['max_norm'][cnt_layer])
            if generator_parameters['max_norm'][cnt_layer]
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(generator_parameters['l2_reg'][cnt_layer])
            if generator_parameters['l2_reg'][cnt_layer]
            else None
        )

        strides = generator_parameters['strides'][i] if generator_parameters['strides'][i] else 1
        if i == 0:
            model.add(tf.keras.layers.Input(shape=(latent_dim,)))
            model.add(tf.keras.layers.Dense(units=generator_parameters['units'][i],
                                        kernel_constraint=kernel_constraint,
                                        kernel_regularizer=kernel_regularizer,
                                        kernel_initializer=tf.keras.initializers.he_uniform(seed=seed),
                                        bias_initializer='zeros'))

            model.add(tf.keras.layers.Reshape(generator_parameters['reshape_dim']))

            model.add(tf.keras.layers.Conv1DTranspose(            
                filters=generator_parameters['filters'][i],
                kernel_size=generator_parameters['kernel_sizes'][i],
                strides=strides,
                padding="same",
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed)
            ))
        else: 

            model.add(tf.keras.layers.Conv1DTranspose(            
                filters=generator_parameters['filters'][i],
                kernel_size=generator_parameters['kernel_sizes'][i],
                strides=strides,
                padding="same",
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed)
            ))
        if generator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
        if generator_parameters['activation'][i] is not None:
            model.add(tf.keras.layers.Activation(generator_parameters['activation'][cnt_layer]))
        if generator_parameters['max_pooling'][i] is not None:
            current_length = model.layers[-1].output_shape[1]
            pooling_size = min(generator_parameters['max_pooling'][i], current_length)
            if pooling_size > 0:  # Ensure pooling size is valid
                model.add(tf.keras.layers.MaxPooling1D(pool_size=pooling_size))
            else:
                raise ValueError(f"Invalid pooling size {pooling_size} at layer {i}")

        if i < len(generator_parameters['drop_out']) and generator_parameters['drop_out'][i] is not None:
            model.add(tf.keras.layers.Dropout(generator_parameters['drop_out'][i]))

        cnt_layer += 1

    current_length = model.layers[-1].output_shape[1]
    cropping_amount = current_length - output_shape[0]
    model.add(tf.keras.layers.Cropping1D(cropping=(0, cropping_amount)))
    
    return model


def build_discriminator(input_shape, discriminator_parameters, use_seed=True):
    """
    Build the Discriminator model with configurable parameters.

    Args:
        input_shape (tuple): Shape of the input data (sequence length, feature count).
        discriminator_parameters (dict): Dictionary containing parameters for the discriminator layers.

    Returns:
        tf.keras.Model: The Discriminator model.
    """
    num_cl = len(discriminator_parameters['filters'])
    num_fcl = len(discriminator_parameters['units'])
    cnt_layer = 0

    if use_seed:
        seed = 1337
    else:
        seed = None

    model = tf.keras.Sequential(name="Discriminator_v2")
    model.add(tf.keras.layers.Input(shape=input_shape))

    for i in range(num_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(discriminator_parameters['max_norm'][cnt_layer])
            if discriminator_parameters['max_norm'][cnt_layer]
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(discriminator_parameters['l2_reg'][cnt_layer])
            if discriminator_parameters['l2_reg'][cnt_layer]
            else None
        )
        strides = discriminator_parameters['strides'][i] if discriminator_parameters['strides'][i] else 1
        if i == 0:
            model.add(tf.keras.layers.Conv1D(
                filters=discriminator_parameters['filters'][i],
                kernel_size=discriminator_parameters['kernel_sizes'][i],
                strides=strides,
                padding="same",
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed)
            ))
        else: 
            model.add(tf.keras.layers.Conv1D(
                filters=discriminator_parameters['filters'][i],
                kernel_size=discriminator_parameters['kernel_sizes'][i],
                strides=strides,
                padding="same",
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed)
            ))
            
        if discriminator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
        if discriminator_parameters['activation'][i] is not None:
            model.add(tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer]))
        if discriminator_parameters['max_pooling'][i] is not None:
            current_length = model.layers[-1].output_shape[1]
            pooling_size = min(discriminator_parameters['max_pooling'][i], current_length)
            if pooling_size > 0:  # Ensure pooling size is valid
                model.add(tf.keras.layers.MaxPooling1D(pool_size=pooling_size))
            else:
                raise ValueError(f"Invalid pooling size {pooling_size} at layer {i}")

        if i < len(discriminator_parameters['drop_out']) and discriminator_parameters['drop_out'][i] is not None:
            model.add(tf.keras.layers.Dropout(discriminator_parameters['drop_out'][i]))

        cnt_layer += 1

    model.add(tf.keras.layers.Flatten())

    # Fully connected layers
    for i in range(num_fcl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(discriminator_parameters['max_norm'][cnt_layer])
            if discriminator_parameters['max_norm'][cnt_layer]
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(discriminator_parameters['l2_reg'][cnt_layer])
            if discriminator_parameters['l2_reg'][cnt_layer]
            else None
        )
        model.add(tf.keras.layers.Dense(units=discriminator_parameters['units'][i],
                                     kernel_constraint=kernel_constraint,
                                     kernel_regularizer=kernel_regularizer,
                                     kernel_initializer=tf.keras.initializers.he_uniform(seed=seed),
                                     bias_initializer='zeros'))
        if discriminator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
        if discriminator_parameters['activation'][i] is not None:
            model.add(tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer]))
        if discriminator_parameters['drop_out'][cnt_layer]:
            model.add(tf.keras.layers.Dropout(discriminator_parameters['drop_out'][cnt_layer]))

        cnt_layer += 1

    #model.add(tf.keras.layers.Dense(discriminator_parameters['dense_units'], activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    #model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    
    return model

def main():
    input_shape = (180, 8)  # Adjust based on your data, including labels, stroke_labels
    latent_dim = 100
    generator_parameters = get_default_generator_parameters()
    discriminator_parameters = get_default_discriminator_parameters()

    generator = build_generator(latent_dim, input_shape, generator_parameters, use_seed=False)
    discriminator = build_discriminator(input_shape, discriminator_parameters, use_seed=False)

    generator.summary()
    discriminator.summary()

if __name__ == '__main__':
    main()
