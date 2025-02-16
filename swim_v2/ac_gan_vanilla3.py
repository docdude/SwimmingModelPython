import tensorflow as tf

def get_default_generator_parameters():
    """
    Get a default set of parameters used to define a GAN model.
    """
    generator_parameters = {
        'filters': [64, 48, 32, 16],  # Decreasing number of filters
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [1, 1, 1, 1],
        'max_pooling': [None, None, None, None],
        'units': [128],
        'activation': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu'],
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
        'filters': [16, 32, 64, 128],  # Increasing number of filters
        'kernel_sizes': [5, 5, 3, 3],
        'strides': [1, 1, 1, 1],
        'max_pooling': [None, None, None, None],
        'units': [128],
        'activation': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu'],
        'batch_norm': [True, False, False, False],
        'drop_out': [0.3, 0.3, 0.3, 0.3, 0],
        'max_norm': [None, None, None, None, None],
        'l2_reg': [None, None, None, None, None],
        'labels': [0, 1, 2, 3, 4],
    }
    return discriminator_parameters

def build_conditional_generator(latent_dim, num_styles, generator_parameters, use_seed=True, output_bias=None):
    """
    Build the Generator model ensuring correct output dimensions.
    """
    num_cl = len(generator_parameters['filters'])
    model = tf.keras.Sequential(name="Generator_v4")

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    if use_seed:
        seed = 1337
    else:
        seed = None

    # Inputs
    noise_input = tf.keras.Input(shape=(latent_dim,), name="noise")  # (B, 100)
    
    noise_proj = tf.keras.layers.Dense(
        180 * 6,
        activation="tanh",  # NEW! This keeps noise values in the same range as real data.
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name='gen_noise_dense'
    )(noise_input)  # (B, 100) => (B, 900)

    noise_reshaped = tf.keras.layers.Reshape((180, 6),name='gen_noise_reshape')(noise_proj)  # (B, 180, 6)
    #noise_reshaped = tf.keras.layers.LayerNormalization(center=True, scale=True, name='gen_noise_norm')(noise_reshaped)
   # noise_reshaped = tf.keras.layers.GaussianNoise(0.05, name='gen_noise_gaussian')(noise_reshaped)

    swim_style_input = tf.keras.Input(shape=(180, 1), name="swim_style_label")  # (B, 1)

 # Remove the unnecessary dimension before RepeatVector
    #swim_style_embedding = tf.keras.layers.Reshape((10,), name='gen_swim_reshape')(swim_style_embedding)  # Now shape: (B, 10)
    # Expand to match temporal dimension
    #swim_style_embedding = tf.keras.layers.RepeatVector(180, name='gen_swim_repeat_vector')(swim_style_embedding)  # (B, 180, 10)   
    swim_style_embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Embedding(
            input_dim=num_styles, 
            output_dim=10, 
            embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            #   embeddings_constraint=tf.keras.constraints.MaxNorm(1),

            name='gen_swim_style_embed'), # Embed each timestep's label
        name='gen_swim_embed_time_dist'
    )(swim_style_input)  # (B, 180, 5)
    # Remove the extra dimension added by TimeDistributed+Embedding
    swim_style_embedding = tf.keras.layers.Reshape(
        (180, 10),  # Target shape: (B, 180, 5)
        name='gen_swim_reshape'
    )(swim_style_embedding)    

    # **Swim Style Embedding Transformation**
    swim_style_embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            10, 
            activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)
        ),
        name="gen_swim_style_dense_transform"
    )(swim_style_embedding)

    # Embed stroke labels (binary 0/1) at each timestep
    stroke_label_input = tf.keras.layers.Input(shape=(180,1), name="stroke_label")  # Per-sample (B, 180, 1)
    # Expand stroke label to match timesteps
    stroke_label_embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Embedding(
            input_dim=2, 
            output_dim=5, 
            embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            #   embeddings_constraint=tf.keras.constraints.MaxNorm(1),

            name='gen_stroke_embed'),  # Embed each timestep's label
        name='gen_stroke_embed_time_dist'
    )(stroke_label_input)  # (B, 180, 5)

    # Remove the extra dimension added by TimeDistributed+Embedding
    stroke_label_embedding = tf.keras.layers.Reshape(
        (180, 5),  # Target shape: (B, 180, 5)
        name='gen_stroke_reshape'
    )(stroke_label_embedding)       

    # **Stroke Label Embedding Transformation**
    stroke_label_embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            5, 
            activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)
        ),
        name="gen_stroke_dense_transform"
    )(stroke_label_embedding)
        # Center Swim Style and Stroke Labels Around Zero
        #swim_style_embedding = swim_style_embedding - tf.reduce_mean(swim_style_embedding, axis=1, keepdims=True)
        #stroke_label_embedding = stroke_label_embedding - tf.reduce_mean(stroke_label_embedding, axis=1, keepdims=True)
       # noise_reshaped = noise_reshaped - tf.reduce_mean(noise_reshaped, axis=1, keepdims=True)
        # Apply Small Gaussian Noise to Swim Style and Stroke Labels
       # swim_style_embedding = swim_style_embedding + tf.random.normal(tf.shape(swim_style_embedding), mean=0.0, stddev=0.05)
        #stroke_label_embedding = stroke_label_embedding + tf.random.normal(tf.shape(stroke_label_embedding), mean=0.0, stddev=0.05)

    merged = tf.keras.layers.Concatenate(name=f'gen_swim_stroke_concat')([noise_reshaped, swim_style_embedding, stroke_label_embedding])


    merged = tf.keras.layers.Dense(
        units=32,  # Reduce dimensionality before first conv layer
        activation="relu",  # Helps mix features before convolutions
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        name="gen_fusion_dense"
    )(merged)
    # Temporal processing
    x = merged

    for i in range(num_cl):
            
        x = tf.keras.layers.Conv1D(
            filters=generator_parameters['filters'][i],
            kernel_size=generator_parameters['kernel_sizes'][i],
            strides=generator_parameters['strides'][i],
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            name=f'gen_conv_{i}'
        )(x)
        
        if generator_parameters['batch_norm'][i]:
            x = tf.keras.layers.BatchNormalization(name=f'gen_bn_{i}')(x)
            
        if generator_parameters['activation'][i] == 'leaky_relu':
            x = tf.keras.layers.LeakyReLU(0.2, name=f'gen_leaky_activation_{i}')(x)
        else:
            x = tf.keras.layers.Activation(generator_parameters['activation'][i], name=f'gen_activation_{i}')(x)
            
        if generator_parameters['drop_out'][i]:
            x = tf.keras.layers.Dropout(generator_parameters['drop_out'][i], name=f'gen_dropout_{i}')(x)
      #  print(f"Mean of Conv1D layer_{i}:", tf.reduce_mean(x))

    # Extra Convolution Before Sensor Output
    sensors_output = tf.keras.layers.Conv1D(
        filters=6,  # Enforces coherence across all six channels
        kernel_size=3, 
        padding="same",
       # activation="tanh",  # Matches final range of sensor values
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name="gen_sensor_conv"
    )(x)
    sensors_output = sensors_output - tf.reduce_mean(sensors_output, axis=1, keepdims=True)
    # Separate heads for the generator
    sensors_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=6,  # For 6 sensor channels
            activation="tanh",  # Sensor output
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            name="gen_sensor_dense"
        ), 
        name='gen_sensor_time_dist'
    )(sensors_output)


    stroke_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=1,  # For stroke label (0 or 1)
            activation="sigmoid",  # Stroke occurrence
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1337),
            bias_initializer=output_bias,
            name="gen_stroke_dense"
        ),
        name='gen_stroke_time_dist'
    )(x)
    """
    swim_style_output = tf.keras.layers.GlobalAveragePooling1D()(x)
    swim_style_output = tf.keras.layers.Dense(
        units=1, 
        activation="linear", 
        kernel_initializer=tf.keras.initializers.glorot_normal(seed=1337),
        name="gen_style_dense"
    )(swim_style_output)
    # Repeat swim style across time dimension
    swim_style_output = tf.keras.layers.Lambda(
        lambda x: tf.repeat(x[:, tf.newaxis, :], repeats=180, axis=1)
    )(swim_style_output)
    """
    # Generate swim style as a single categorical label per row
    swim_style_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=1,  # Single label per row (not one-hot)
            activation="relu",  # Continuous representation
            kernel_initializer=tf.keras.initializers.he_normal(seed=1337),
            name="gen_style_dense"
        ),
        name="gen_style_time_dist"
    )(x)
    # Combine all outputs
    final_output = tf.keras.layers.Concatenate(axis=-1)([sensors_output, swim_style_output, stroke_output])

    model = tf.keras.Model([noise_input, swim_style_input, stroke_label_input], final_output, name="Conditional_Generator")

    return model

def build_conditional_discriminator(input_shape, num_styles, discriminator_parameters, use_seed=True, output_bias=None):
    """
    A discriminator that returns (real_fake_output, style_output, stroke_output).
    num_styles: the number of classes for swim_style.
    use_stroke: if True, also tries to classify stroke in the output.
    """
    num_cl = len(discriminator_parameters['filters'])
    num_fcl = len(discriminator_parameters['units'])
    cnt_layer = 0
    model = tf.keras.Sequential(name="Discriminator_v4")

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

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
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            name=f'disc_conv_{i}'
        )(x)
        
        if discriminator_parameters['batch_norm'][cnt_layer]:
            x = tf.keras.layers.BatchNormalization(name=f'disc_bn_{i}')(x)
            
        if discriminator_parameters['activation'][cnt_layer] == 'leaky_relu':
            x = tf.keras.layers.LeakyReLU(0.2,name=f'disc_leaky_activation_{i}')(x)
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

    # **Real/Fake Classification (Single Output)**
    real_fake_output = tf.keras.layers.GlobalAveragePooling1D(name='real_fake_avg_pool')(x)
    real_fake_output = tf.keras.layers.Dense(
        1, 
        activation='sigmoid', 
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name='real_fake_output'
    )(real_fake_output)


    # **Swim Style Classification (Per-Timestep, Preserves Shape)**
    swim_style_output = tf.keras.layers.Conv1D(
        filters=64, 
        kernel_size=3, 
        activation='relu',
        padding='same', 
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        name=f'disc_swim_style_conv'
    )(x)
    #swim_style_output = tf.keras.layers.LeakyReLU(0.2, name=f'disc_swim_style_activation')(swim_style_output)
    swim_style_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=1,  # Output per row
            activation="linear",  # Multi-class classification per timestep
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1337),
            name='disc_swim_style_dense'
        ),
        name="swim_style_output"
    )(swim_style_output)  # Shape (B, 180, num_styles)

    # **Stroke Detection (Per-Timestep, Preserves Shape)**
    stroke_label_output = tf.keras.layers.Conv1D(
        filters=32, 
        kernel_size=3, 
        activation='relu',
        padding='same', 
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        name=f'disc_stroke_conv'
    )(x)
    #stroke_label_output = tf.keras.layers.LeakyReLU(0.2, name=f'disc_stroke_activation')(stroke_label_output)
    stroke_label_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=1,  # Stroke (0 or 1) per row
            activation="sigmoid",  # Binary classification
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1337),
            bias_initializer=output_bias,
            name='disc_stroke_dense'
        ),
        name="stroke_label_output"
    )(stroke_label_output)  # Shape (B, 180, 1)

    model = tf.keras.Model(input, outputs=[real_fake_output, swim_style_output, stroke_label_output], name="Conditional_Discriminator")

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
        generator_parameters=generator_parameters,
        use_seed=True
    )
    discriminator = build_conditional_discriminator(
        input_shape=input_shape,
        num_styles=6,
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
    test_noise = tf.random.normal([5, latent_dim], mean=0.0, stddev=1.0)  # shape (1, 100)
    test_styles = tf.random.uniform([5, 180, 1], minval=0, maxval=6, dtype=tf.int32)
    test_stroke = tf.random.uniform([5, 180, 1], minval=0, maxval=2, dtype=tf.int32)  # (1, 180, 1)

    test_output = generator([test_noise, test_styles, test_stroke])
    print(f"\nGenerator test output shape: {test_output.shape}")
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,0]).numpy())  # ✅ Now works!

    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,1]).numpy())  # ✅ Now works!
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,2]).numpy())  # ✅ Now works!
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,3]).numpy())  # ✅ Now works!
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,4]).numpy())  # ✅ Now works!
    print("Mean of generated sensor output:", tf.reduce_mean(test_output[...,5]).numpy())  # ✅ Now works!

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

