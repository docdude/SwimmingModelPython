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

def build_conditional_generator(latent_dim, num_styles,  output_shape, generator_parameters, use_seed=True, use_stroke=True,):
    """
    Build the Generator model ensuring correct output dimensions.
    """
    num_cl = len(generator_parameters['filters'])
    model = tf.keras.Sequential(name="Generator_v4")

    if use_seed:
        seed = 1337
    else:
        seed = None

    # Inputs
    noise_input = tf.keras.Input(shape=(latent_dim,), name="noise")  # (B, 100)
    swim_style_input = tf.keras.Input(shape=(1,), name="swim_style_label")  # (B, 1)
    if use_stroke:
        stroke_label_input = tf.keras.layers.Input(shape=(180,1), name="stroke_label")  # Per-sample (B, 1)
    
    # Embeddings
    swim_style_embedding = tf.keras.layers.Embedding(
        input_dim=num_styles, 
        output_dim=10,
        embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        name=f'gen_swim_style_embed'
    )(swim_style_input)

    # Remove the unnecessary dimension before RepeatVector
    swim_style_embedding = tf.keras.layers.Reshape((10,), name='gen_swim_reshape')(swim_style_embedding)  # Now shape: (B, 10)
    # Expand to match temporal dimension
    swim_style_embedding = tf.keras.layers.RepeatVector(180, name='gen_swim_repeat_vector')(swim_style_embedding)  # (B, 180, 10)

    noise_proj = tf.keras.layers.Dense(
        180 * 5,
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        name='gen_noise_dense'
    )(noise_input)  # (B, 100) => (B, 900)
    
    noise_reshaped = tf.keras.layers.Reshape((180, 5),name='gen_noise_reshape')(noise_proj)  # (B, 180, 5)
    # Embed stroke labels (binary 0/1) at each timestep
    if use_stroke:
        # Expand stroke label to match timesteps
        stroke_label_embedding = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Embedding(
                input_dim=2, 
                output_dim=5, 
                embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                name='gen_stroke_embed'),  # Embed each timestep's label
            name='gen_time_dist'
        )(stroke_label_input)  # (B, 180, 5)

        # Remove the extra dimension added by TimeDistributed+Embedding
        stroke_label_embedding = tf.keras.layers.Reshape(
            (180, 5),  # Target shape: (B, 180, 5)
            name='gen_stroke_reshape'
        )(stroke_label_embedding)       
        merged = tf.keras.layers.Concatenate(name=f'gen_swim_stroke_concat')([noise_reshaped, swim_style_embedding, stroke_label_embedding])
    else:
        merged = tf.keras.layers.Concatenate(name=f'gen_swim_concat')([noise_reshaped, swim_style_embedding])

    # Temporal processing
    x = merged

    for i in range(num_cl):
            
        x = tf.keras.layers.Conv1D(
            filters=generator_parameters['filters'][i],
            kernel_size=generator_parameters['kernel_sizes'][i],
            strides=generator_parameters['strides'][i],
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
            name=f'gen_conv_{i}'
        )(x)
        
        if generator_parameters['batch_norm'][i]:
            x = tf.keras.layers.BatchNormalization(name=f'gen_bn_{i}')(x)
            
        if generator_parameters['activation'][i] == 'leaky_relu':
            x = tf.keras.layers.LeakyReLU(0.2, name=f'gen_activation_{i}')(x)
        else:
            x = tf.keras.layers.Activation(generator_parameters['activation'][i], name=f'gen_activation_{i}')(x)
            
        if generator_parameters['drop_out'][i]:
            x = tf.keras.layers.Dropout(generator_parameters['drop_out'][i], name=f'gen_dropout_{i}')(x)

    # Final layer to match target shape
    x = tf.keras.layers.Conv1D(
        filters=output_shape[-1],  # Number of features in target
        kernel_size=1,
        activation='tanh',
        padding='same',
        kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        name=f'gen_final_conv'
    )(x)

    model = tf.keras.Model([noise_input, swim_style_input, stroke_label_input] if use_stroke 
                           else [noise_input, swim_style_input], x, name="Conditional_Generator")
    return model

def build_conditional_discriminator(input_shape, num_styles, discriminator_parameters, use_seed=True, use_stroke=True):
    """
    A discriminator that returns (real_fake_output, style_output, stroke_output).
    num_styles: the number of classes for swim_style.
    use_stroke: if True, also tries to classify stroke in the output.
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
    input = tf.keras.Input(shape=input_shape)
    x = input
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

        x = tf.keras.layers.Conv1D(
            filters=discriminator_parameters['filters'][i],
            kernel_size=discriminator_parameters['kernel_sizes'][i],
            strides=strides,
            padding="same",
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
            name=f'disc_conv_{i}'
        )(x)
        
        if discriminator_parameters['batch_norm'][cnt_layer]:
            x = tf.keras.layers.BatchNormalization(name=f'disc_bn_{i}')(x)
            
        if discriminator_parameters['activation'][cnt_layer] == 'leaky_relu':
            x = tf.keras.layers.LeakyReLU(0.2,name=f'disc_activation_{i}')(x)
        else:
            x = tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer], name=f'disc_activation_{i}')(x)

        if discriminator_parameters['max_pooling'][i] is not None:
            current_length = model.layers[-1].output_shape[1]
            pooling_size = min(discriminator_parameters['max_pooling'][i], current_length)
            if pooling_size > 0:
                x = tf.keras.layers.MaxPooling1D(pool_size=pooling_size,name=f'disc_pool_{i}')(x)

        if discriminator_parameters['drop_out'][cnt_layer]:
            x = tf.keras.layers.Dropout(discriminator_parameters['drop_out'][cnt_layer],name=f'disc_dropout_{i}')(x)

        cnt_layer += 1

    swim_style_branch = tf.keras.layers.Flatten(name=f'disc_flatten')(x)

    # Fully connected layers
    for i in range(num_fcl):
        swim_style_branch = tf.keras.layers.Dense(
            units=discriminator_parameters['units'][i],
            kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
            name=f'disc_dense_{i}'
        )(swim_style_branch)
        if discriminator_parameters['batch_norm'][cnt_layer]:
            swim_style_branch = tf.keras.layers.BatchNormalization(name=f'disc_dense_bn_{i}')(swim_style_branch)
            
        if discriminator_parameters['activation'][cnt_layer] == 'leaky_relu':
            swim_style_branch = tf.keras.layers.LeakyReLU(0.2,name=f'disc_dense_activation_{i}')(swim_style_branch)
        else:
            swim_style_branch = tf.keras.layers.Activation(discriminator_parameters['activation'][cnt_layer], name=f'disc_dense_activation_{i}')(swim_style_branch)
            
        if discriminator_parameters['drop_out'][cnt_layer]:
            swim_style_branch = tf.keras.layers.Dropout(discriminator_parameters['drop_out'][cnt_layer], name=f'disc_dense_dropout_{i}')(swim_style_branch)
        
        cnt_layer += 1

    # Output A: Real vs. Fake
    real_fake_out = tf.keras.layers.Dense(1, activation='sigmoid', name='real_fake_output')(swim_style_branch)

    # Output B: style labeling
    swim_style_output = tf.keras.layers.Dense(num_styles, activation='softmax', name='swim_style_output')(swim_style_branch)

    if use_stroke:
        # Output C: stroke label
        # TimeDistributed Dense for Per-Timestep Predictions
        """
        stroke_label_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units=1,  # Single output per timestep (stroke/no stroke)
                activation="sigmoid",  # Binary classification
                #bias_initializer=output_bias,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=1337),
                name='disc_stroke_dense'
            ),
            name="stroke_label_output"
        )(x)
        """
        stroke_branch = tf.keras.layers.Conv1D(
            filters=32, 
            kernel_size=3, 
            padding='same', 
            kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
            name=f'disc_stroke_conv_0'
        )(input)  # No striding
        stroke_branch = tf.keras.layers.LeakyReLU(0.2, name=f'disc_stroke_activation_0')(stroke_branch)  # Shape: (180, 32)
        stroke_branch = tf.keras.layers.Conv1D(
            filters=64, 
            kernel_size=3, 
            padding='same', 
            kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
            name=f'disc_stroke_conv_1'
        )(stroke_branch)
        stroke_branch = tf.keras.layers.LeakyReLU(0.2, name=f'disc_stroke_activation_1')(stroke_branch)  # Shape: (180, 64)

        stroke_label_output = tf.keras.layers.Conv1D(
            filters=1,  # Output shape: (batch_size, 180, 1)
            kernel_size=3,
            padding='same',
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
            name='stroke_label_output'
        )(stroke_branch)
        #stroke_label_output = tf.keras.layers.Dense(1, activation='sigmoid', name='stroke_label_output')(x)
        model = tf.keras.Model(input, outputs=[real_fake_out, swim_style_output, stroke_label_output], name="Conditional_Discriminator")
    else:
        model = tf.keras.Model(input, outputs=[real_fake_out, swim_style_output], name="Conditional_Discriminator")

    return model


def main():
    # Define shapes
    input_shape = (180, 8)  # shape for discriminator input
    output_shape = (180, 8) # shape for generator output (sensor + style_label + stroke_label)
    latent_dim = 100

    # Load or define default parameters (assuming these exist in your script or imported)
    generator_parameters = get_default_generator_parameters()
    discriminator_parameters = get_default_discriminator_parameters()

    # Build conditional generator and discriminator
    generator = build_conditional_generator(
        latent_dim=latent_dim,
        num_styles=6,          # e.g., if your style labels are 0..5
        use_stroke=True,
        output_shape=output_shape,
        generator_parameters=generator_parameters,
        use_seed=True
    )
    discriminator = build_conditional_discriminator(
        input_shape=input_shape,
        num_styles=6,
        use_stroke=True,
        discriminator_parameters=discriminator_parameters,
        use_seed=True
    )

    # Print model summaries
    print("Generator Summary:")
    generator.summary()
    print("\nDiscriminator Summary:")
    discriminator.summary()

    # Test generator output shape
    # Conditional generator expects a list of inputs: [noise, style_label, stroke_label]
    test_noise = tf.random.normal([5, latent_dim])  # shape (1, 100)
    test_styles = tf.random.uniform([5], minval=0, maxval=6, dtype=tf.int32)
    test_stroke = tf.random.uniform([5, 180, 1], minval=0, maxval=2, dtype=tf.int32)  # (1, 180, 1)

    test_output = generator([test_noise, test_styles, test_stroke])
    print(f"\nGenerator test output shape: {test_output.shape}")

    # Test discriminator output shapes
    # Discriminator expects sensor data shaped (None, 180, 8)
    disc_output = discriminator(test_output)
    if isinstance(disc_output, list):
        print("Discriminator outputs:")
        for out in disc_output:
            print(out.shape)
    else:
        print(f"Discriminator single output shape: {disc_output.shape}")

if __name__ == "__main__":
    main()

