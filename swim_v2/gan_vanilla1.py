import tensorflow as tf

def get_default_generator_parameters():
    """
    Get a default set of parameters used to define a GAN generator model
    :return: A dictionary containing parameter names and values
    """
    generator_parameters = {
        'filters':      [128, 64],
        'kernel_sizes': [3, 3],
        'strides':      [1, 1],
        'activation':   ['relu', 'tanh'],
        'batch_norm':   [True, True],
        'drop_out':     [None, None]
    }
    return generator_parameters

def get_default_discriminator_parameters():
    """
    Get a default set of parameters used to define a GAN discriminator model
    :return: A dictionary containing parameter names and values
    """
    discriminator_parameters = {
        'filters':      [128, 64],
        'kernel_sizes': [3, 3],
        'strides':      [2, 2],
        'activation':   ['leaky_relu', 'leaky_relu'],
        'batch_norm':   [False, False],
        'drop_out':     [0.3, 0.3]
    }
  
    return discriminator_parameters

def build_generator(latent_dim, output_shape, generator_parameters):
    model = tf.keras.Sequential(name="Generator_v1")
    model.add(tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='relu', input_dim=latent_dim))
    model.add(tf.keras.layers.Reshape((output_shape[0], output_shape[1])))
    for i in range(len(generator_parameters['filters'])):
        model.add(tf.keras.layers.Conv1D(
            filters=generator_parameters['filters'][i],
            kernel_size=generator_parameters['kernel_sizes'][i],
            strides=generator_parameters['strides'][i],
            padding='same',
            activation=generator_parameters['activation'][i]
        ))
        if generator_parameters['drop_out'][i] is not None:
            model.add(tf.keras.layers.Dropout(generator_parameters['drop_out'][i]))
        if generator_parameters['batch_norm'][i]:
            model.add(tf.keras.layers.BatchNormalization())
    # Final layer to ensure output shape matches (180, 8)
    model.add(tf.keras.layers.Conv1D(
        filters=output_shape[1],  # Ensure it matches the number of features
        kernel_size=1,
        strides=1,
        padding='same',
        activation='tanh'  # or another suitable activation
    ))
    return model


def build_discriminator(input_shape, discriminator_parameters):
    model = tf.keras.Sequential(name="Discriminator_v1")
    model.add(tf.keras.layers.Input(shape=input_shape))
    for i in range(len(discriminator_parameters['filters'])):
        model.add(tf.keras.layers.Conv1D(
            filters=discriminator_parameters['filters'][i],
            kernel_size=discriminator_parameters['kernel_sizes'][i],
            strides=discriminator_parameters['strides'][i],
            padding='same',
            activation=discriminator_parameters['activation'][i]
        ))
        if discriminator_parameters['drop_out'][i] is not None:
            model.add(tf.keras.layers.Dropout(discriminator_parameters['drop_out'][i]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def main():
    input_shape = (180, 8)  # Adjust based on your data, including labels, stroke_labels
    latent_dim = 100
    generator_parameters = get_default_generator_parameters()
    discriminator_parameters = get_default_discriminator_parameters()

    generator = build_generator(latent_dim, input_shape, generator_parameters)
    discriminator = build_discriminator(input_shape, discriminator_parameters)

    generator.summary()
    discriminator.summary()

if __name__ == '__main__':
    main()
