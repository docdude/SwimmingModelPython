import tensorflow as tf

def get_default_generator_parameters():
    """
    Get a default set of parameters used to define a GAN model.
    """
    generator_parameters = {
        'filters': [128, 64, 32, 16],  # Decreasing number of filters
        'kernel_sizes': [5, 5, 5, 5],
        'strides': [1, 1, 1, 1],
        'max_pooling': [None, None, None, None],
        'units': [128],
        'activation': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'tanh'],
        'batch_norm': [True, True, True, False],
        'drop_out': [0.3, 0.3, 0.3, 0],
        'max_norm': [None, None, None, None],
        'l2_reg': [None, None, None, None],
        'labels': [0, 1, 2, 3, 4],
    }
    return generator_parameters

def get_default_discriminator_parameters():
    """
    Get a default set of parameters used to define a GAN model.
    """
    discriminator_parameters = {
        'filters': [32, 64, 128, 256],  # Increasing number of filters
        'kernel_sizes': [5, 5, 5, 5],
        'strides': [2, 2, 2, 2],
        'max_pooling': [None, None, None, None],
        'units': [128],
        'activation': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu'],
        'batch_norm': [True, True, True, True, True],
        'drop_out': [0.3, 0.3, 0.3, 0.3, 0.3],
        'max_norm': [None, None, None, None, None],
        'l2_reg': [None, None, None, None, None],
        'labels': [0, 1, 2, 3, 4],
    }
    return discriminator_parameters

def build_generator(latent_dim, output_shape, generator_parameters, use_seed=True):
    """
    Build the Generator model ensuring correct output dimensions.
    """
    num_cl = len(generator_parameters['filters'])
    model = tf.keras.Sequential(name="Generator_v4")

    if use_seed:
        seed = 1337
    else:
        seed = None

    # Calculate initial reshape dimensions based on target output shape
    initial_dim = output_shape[0] // (2 ** (num_cl - 1))  # Account for subsequent upsampling
    initial_filters = generator_parameters['filters'][0]

    # Initial dense layer to reshape noise
    model.add(tf.keras.layers.Dense(
        initial_dim * initial_filters,
        input_dim=latent_dim,
        kernel_initializer=tf.keras.initializers.he_uniform(seed=seed)
    ))
    model.add(tf.keras.layers.Reshape((initial_dim, initial_filters)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))

    # Upsampling layers
    current_dim = initial_dim
    for i in range(num_cl):
        if current_dim < output_shape[0]:
            model.add(tf.keras.layers.UpSampling1D(2))
            current_dim *= 2
            
        model.add(tf.keras.layers.Conv1D(
            filters=generator_parameters['filters'][i],
            kernel_size=generator_parameters['kernel_sizes'][i],
            strides=generator_parameters['strides'][i],
            padding='same',
            kernel_initializer=tf.keras.initializers.he_uniform(seed=seed)
        ))
        
        if generator_parameters['batch_norm'][i]:
            model.add(tf.keras.layers.BatchNormalization())
            
        if generator_parameters['activation'][i] == 'leaky_relu':
            model.add(tf.keras.layers.LeakyReLU(0.2))
        else:
            model.add(tf.keras.layers.Activation(generator_parameters['activation'][i]))
            
        if generator_parameters['drop_out'][i]:
            model.add(tf.keras.layers.Dropout(generator_parameters['drop_out'][i]))

    # Final layer to match target shape
    model.add(tf.keras.layers.Conv1D(
        filters=output_shape[1],  # Number of features in target
        kernel_size=1,
        activation='tanh',
        padding='same'
    ))

    # Ensure output length matches target length
    if current_dim != output_shape[0]:
        if current_dim > output_shape[0]:
            crop_size = (current_dim - output_shape[0]) // 2
            model.add(tf.keras.layers.Cropping1D(cropping=(crop_size, crop_size)))
        else:
            model.add(tf.keras.layers.ZeroPadding1D(padding=(0, output_shape[0] - current_dim)))

    return model

def build_discriminator(input_shape, discriminator_parameters, use_seed=True):
    """
    Build the Discriminator model.
    """
    num_cl = len(discriminator_parameters['filters'])
    num_fcl = len(discriminator_parameters['units'])
    cnt_layer = 0
    model = tf.keras.Sequential(name="Discriminator_v4")

    if use_seed:
        seed = 1337
    else:
        seed = None

    # Add input layer with correct shape
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # Convolutional layers
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

        model.add(tf.keras.layers.Conv1D(
            filters=discriminator_parameters['filters'][i],
            kernel_size=discriminator_parameters['kernel_sizes'][i],
            strides=strides,
            padding="same",
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        ))
        
        if discriminator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
            
        if discriminator_parameters['activation'][cnt_layer] == 'leaky_relu':
            model.add(tf.keras.layers.LeakyReLU(0.2))
        else:
            model.add(tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer]))

        if discriminator_parameters['max_pooling'][i] is not None:
            current_length = model.layers[-1].output_shape[1]
            pooling_size = min(discriminator_parameters['max_pooling'][i], current_length)
            if pooling_size > 0:
                model.add(tf.keras.layers.MaxPooling1D(pool_size=pooling_size))

        if discriminator_parameters['drop_out'][cnt_layer]:
            model.add(tf.keras.layers.Dropout(discriminator_parameters['drop_out'][cnt_layer]))

        cnt_layer += 1

    model.add(tf.keras.layers.Flatten())

    # Fully connected layers
    for i in range(num_fcl):
        model.add(tf.keras.layers.Dense(
            units=discriminator_parameters['units'][i],
            kernel_initializer=tf.keras.initializers.he_uniform(seed=seed),
        ))
        if discriminator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
            
        if discriminator_parameters['activation'][cnt_layer] == 'leaky_relu':
            model.add(tf.keras.layers.LeakyReLU(0.2))
        else:
            model.add(tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer]))
            
        if discriminator_parameters['drop_out'][cnt_layer]:
            model.add(tf.keras.layers.Dropout(discriminator_parameters['drop_out'][cnt_layer]))
        
        cnt_layer += 1

    # Final output layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model

def main():
    input_shape = (180, 8)  # Includes label and stroke_labels
    latent_dim = 100

    generator_parameters = get_default_generator_parameters()
    discriminator_parameters = get_default_discriminator_parameters()

    generator = build_generator(latent_dim, input_shape, generator_parameters)
    discriminator = build_discriminator(input_shape, discriminator_parameters)

    # Print model summaries
    print("Generator Summary:")
    generator.summary()
    print("\nDiscriminator Summary:")
    discriminator.summary()

    # Test generator output shape
    test_noise = tf.random.normal([1, latent_dim])
    test_output = generator(test_noise)
    print(f"\nGenerator test output shape: {test_output.shape}")

if __name__ == "__main__":
    main()
