import tensorflow as tf


def get_default_generator_parameters():
    """
    Get a default set of parameters used to define a GAN model.
    """
    generator_parameters = {
        'filters': [128, 64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [None, None, None, None],
        'max_pooling': [3, 3, 3, 3],
        'units': [128],
        'activation': ['relu', 'relu', 'relu', 'tanh'],
        'batch_norm': [False, False, False, False, False],
        'drop_out': [0.5, 0.75, 0.25, 0.1, 0.25],
        'max_norm': [0.1, 0.1, None, 4.0, 4.0],
        'l2_reg': [None, None, None, None, None],
        'labels': [0, 1, 2, 3, 4],
    }
    return generator_parameters


def get_default_discriminator_parameters():
    """
    Get a default set of parameters used to define a GAN model.
    """
    discriminator_parameters = {
        'filters': [128, 64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [2, 2, 2, 2],
        'max_pooling': [3, 3, 3, 3],
        'units': [128],
        'activation': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu'],
        'batch_norm': [False, False, False, False, False],
        'drop_out': [0.5, 0.75, 0.25, 0.1, 0.25],
        'max_norm': [0.1, 0.1, None, 4.0, 4.0],
        'l2_reg': [None, None, None, None, None],
        'labels': [0, 1, 2, 3, 4],
    }
    return discriminator_parameters


def build_generator(latent_dim, output_shape, generator_parameters, use_seed=True):
    """
    Build the Generator model.
    """
    num_cl = len(generator_parameters['filters'])
    num_fcl = len(generator_parameters['units'])
    cnt_layer = 0
    model = tf.keras.Sequential(name="Generator")

    if use_seed:
        seed = 1337
    else:
        seed = None

    # Initial dense layer and reshape
    model.add(tf.keras.layers.Input(shape=(latent_dim,)))
    model.add(tf.keras.layers.Dense(
        output_shape[0] * output_shape[1],
        kernel_initializer=tf.keras.initializers.he_uniform(seed=seed),
    ))
    model.add(tf.keras.layers.Reshape((output_shape[0], output_shape[1])))

    # Add convolutional layers
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

        model.add(tf.keras.layers.Conv1D(
            filters=generator_parameters['filters'][i],
            kernel_size=generator_parameters['kernel_sizes'][i],
            strides=strides,
            padding="same",
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        ))
        if generator_parameters['batch_norm'][cnt_layer]:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(generator_parameters['activation'][cnt_layer]))
        if generator_parameters['max_pooling'][i]:
            model.add(tf.keras.layers.MaxPooling1D(generator_parameters['max_pooling'][i]))
        if generator_parameters['drop_out'][cnt_layer]:
            model.add(tf.keras.layers.Dropout(generator_parameters['drop_out'][cnt_layer]))

        cnt_layer += 1

    return model


def build_discriminator(input_shape, discriminator_parameters, use_seed=True):
    """
    Build the Discriminator model.
    """
    num_cl = len(discriminator_parameters['filters'])
    num_fcl = len(discriminator_parameters['units'])
    cnt_layer = 0
    model = tf.keras.Sequential(name="Discriminator")

    if use_seed:
        seed = 1337
    else:
        seed = None

    # Add input layer separately before the loop
    model.add(tf.keras.layers.Input(shape=(input_shape[0], input_shape[1])))  # Explicit shape specification

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
        model.add(tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer]))

        if discriminator_parameters['max_pooling'][i] is not None:
            current_length = model.layers[-1].output_shape[1]
            pooling_size = min(discriminator_parameters['max_pooling'][i], current_length)
            if pooling_size > 0:  # Ensure pooling size is valid
                model.add(tf.keras.layers.MaxPooling1D(pool_size=pooling_size))
            else:
                raise ValueError(f"Invalid pooling size {pooling_size} at layer {i}")

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

    generator.summary()
    discriminator.summary()


if __name__ == "__main__":
    main()
