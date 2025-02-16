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

# Modified Residual Block for Generator
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.shortcut = tf.keras.layers.Conv1D(filters, 1, padding='same')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        shortcut = self.shortcut(inputs)
        return tf.keras.layers.Add(name='res_block_add')([x, shortcut])

# Modified Generator build function with residual blocks and improved architecture
def build_conditional_generator(latent_dim, num_styles, generator_parameters, use_seed=True, output_bias=None):
    if use_seed:
        seed = 1337
    else:
        seed = None

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Inputs
    noise_input = tf.keras.Input(shape=(latent_dim,), name="noise")
    swim_style_input = tf.keras.Input(shape=(180, 1), name="swim_style_label")
    stroke_label_input = tf.keras.layers.Input(shape=(180,1), name="stroke_label")

    # Improved noise processing
    noise_proj = tf.keras.layers.Dense(
        180 * 8,  # Increased capacity
        activation="tanh",
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name='gen_noise_dense'
    )(noise_input)
    
    noise_reshaped = tf.keras.layers.Reshape((180, 8), name='gen_noise_reshape')(noise_proj)
    noise_reshaped = tf.keras.layers.LayerNormalization(name='gen_noise_norm')(noise_reshaped)
    
    # Improved style embedding
    swim_style_embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Embedding(
            input_dim=num_styles, 
            output_dim=16,  # Increased embedding dimension
            embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            embeddings_constraint=tf.keras.constraints.MaxNorm(1),
            name='gen_swim_style_embed'
        ),
        name='gen_swim_embed_time_dist'
    )(swim_style_input)
    
    swim_style_embedding = tf.keras.layers.Reshape((180, 16), name='gen_swim_reshape')(swim_style_embedding)
    
    # Improved stroke embedding
    stroke_embedding = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Embedding(
            input_dim=2, 
            output_dim=8,  # Increased embedding dimension
            embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            embeddings_constraint=tf.keras.constraints.MaxNorm(1),
            name='gen_stroke_embed'
        ),
        name='gen_stroke_embed_time_dist'
    )(stroke_label_input)
    
    stroke_embedding = tf.keras.layers.Reshape((180, 8), name='gen_stroke_reshape')(stroke_embedding)

    # Merge all inputs
    merged = tf.keras.layers.Concatenate(name='gen_merge')([
        noise_reshaped, 
        swim_style_embedding, 
        stroke_embedding
    ])
    
    # Initial projection
    x = tf.keras.layers.Dense(
        64,
        activation="leaky_relu",
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        name="gen_initial_dense"
    )(merged)

    # Residual blocks
    for i, filters in enumerate(generator_parameters['filters']):
        x = ResidualBlock(
            filters=filters,
            kernel_size=generator_parameters['kernel_sizes'][i],
            name=f'gen_residual_block_{i}'
        )(x)
        
        if generator_parameters['batch_norm'][i]:
            x = tf.keras.layers.BatchNormalization(name=f'gen_bn_{i}')(x)
        
        if generator_parameters['drop_out'][i]:
            x = tf.keras.layers.Dropout(generator_parameters['drop_out'][i], name=f'gen_dropout_{i}')(x)

    # Separate heads with attention
    # Sensor output head
    sensor_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=32, name='gen_sensor_attention'
    )(x, x)
    sensor_output = tf.keras.layers.Add(name='gen_sensor_add')([x, sensor_attention])
    sensor_output = tf.keras.layers.LayerNormalization()(sensor_output)
    
    # Extra Convolution Before Sensor Output
    sensor_output = tf.keras.layers.Conv1D(
        filters=32,  # Enforces coherence across all channels
        kernel_size=3, 
        padding="same",
        activation='tanh',  # Using tanh for bounded output
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name="gen_sensor_conv"
    )(sensor_output)

    # Center the sensor outputs around zero
    sensor_output = sensor_output - tf.reduce_mean(sensor_output, axis=1, keepdims=True)

    # Final sensor output layer
    sensor_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=6,  # For 6 sensor channels
            activation="tanh",  # Sensor output
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            name="gen_sensor_dense"
        ), 
        name='gen_sensor_time_dist'
    )(sensor_output)

    # Additional centering after final dense layer
    #sensor_output = sensor_output - tf.reduce_mean(sensor_output, axis=1, keepdims=True)

    # Style output head
    style_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=32, name='gen_style_attention'
    )(x, x)
    style_output = tf.keras.layers.Add(name='gen_style_add')([x, style_attention])
    style_output = tf.keras.layers.LayerNormalization()(style_output)
    
    style_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            1,
            activation="relu",
            kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
            name="gen_style_dense"
        ),
        name='gen_style_time_dist'
    )(style_output)

    # Stroke output head
    stroke_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=32, name='gen_stroke_attention'
    )(x, x)
    stroke_output = tf.keras.layers.Add(name='gen_stroke_add')([x, stroke_attention])
    stroke_output = tf.keras.layers.LayerNormalization()(stroke_output)
    
    stroke_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            bias_initializer=output_bias,
            name="gen_stroke_dense"
        ),
        name='gen_stroke_time_dist'
    )(stroke_output)

    # Combine outputs
    final_output = tf.keras.layers.Concatenate(axis=-1)([
        sensor_output, 
        style_output, 
        stroke_output
    ])

    model = tf.keras.Model(
        [noise_input, swim_style_input, stroke_label_input], 
        final_output, 
        name="Conditional_Generator"
    )

    return model


def build_conditional_discriminator(input_shape, num_styles, discriminator_parameters, use_seed=True, output_bias=None):
    """Enhanced discriminator with attention and improved architecture"""
    if use_seed:
        seed = 1337
    else:
        seed = None

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Input layer
    input_layer = tf.keras.Input(shape=input_shape)
    
    # Initial feature extraction
    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name='disc_initial_conv'
    )(input_layer)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    # Self-attention for temporal dependencies
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=32,
        name='disc_self_attention'
    )(x, x)
    x = tf.keras.layers.Add(name=f'disc_initial_add')([x, attention_output])
    x = tf.keras.layers.LayerNormalization()(x)

    # Main convolutional backbone
    for i, filters in enumerate(discriminator_parameters['filters']):
        # Residual block
        res = x
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=discriminator_parameters['kernel_sizes'][i],
            strides=discriminator_parameters['strides'][i],
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            kernel_constraint=tf.keras.constraints.MaxNorm(2),
            name=f'disc_conv_{i}'
        )(x)
        
        if discriminator_parameters['batch_norm'][i]:
            x = tf.keras.layers.BatchNormalization(name=f'disc_bn_{i}')(x)
        
        x = tf.keras.layers.LeakyReLU(0.2, name=f'disc_leaky_{i}')(x)

        # Project residual if needed
        if res.shape[-1] != filters:
            res = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=1,
                padding='same',
                name=f'disc_res_proj_{i}'
            )(res)
    
        x = tf.keras.layers.Add(name=f'disc_res_add_{i}')([res, x])
        
        if discriminator_parameters['drop_out'][i]:
            x = tf.keras.layers.Dropout(
                discriminator_parameters['drop_out'][i],
                name=f'disc_dropout_{i}'
            )(x)

    # Real/Fake output branch
    real_fake = tf.keras.layers.GlobalAveragePooling1D(name='disc_global_pool')(x)
    real_fake = tf.keras.layers.Dense(
        128, 
        activation='leaky_relu',
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name='disc_dense_1'
    )(real_fake)
    real_fake_output = tf.keras.layers.Dense(
        1, 
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name='real_fake_output'
    )(real_fake)

    # Swim style classification branch
    style_features = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name='disc_style_conv'
    )(x)
    
    # Style attention
    style_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=2,
        key_dim=32,
        name='disc_style_attention'
    )(style_features, style_features)
    style_features = tf.keras.layers.Add(name=f'disc_style_add')([style_features, style_attention])
    style_features = tf.keras.layers.LayerNormalization()(style_features)
    
    swim_style_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            1,
            activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            name='disc_style_dense'
        ),
        name='swim_style_output'
    )(style_features)

    # Stroke detection branch
    stroke_features = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        name='disc_stroke_conv'
    )(x)
    
    # Stroke attention
    stroke_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=2,
        key_dim=32,
        name='disc_stroke_attention'
    )(stroke_features, stroke_features)
    stroke_features = tf.keras.layers.Add(name=f'disc_stroke_add')([stroke_features, stroke_attention])
    stroke_features = tf.keras.layers.LayerNormalization()(stroke_features)
    
    stroke_label_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
            bias_initializer=output_bias,
            name='disc_stroke_dense'
        ),
        name='stroke_label_output'
    )(stroke_features)

    # Create model
    model = tf.keras.Model(
        inputs=input_layer,
        outputs=[real_fake_output, swim_style_output, stroke_label_output],
        name="Conditional_Discriminator"
    )

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

