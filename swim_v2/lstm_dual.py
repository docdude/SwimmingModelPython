import tensorflow as tf

def get_default_training_parameters():
    """
    Get a default set of parameters used to train a cnn model
    :return: A dictionary containing parameter names and values
    """
    training_parameters = {'swim_style_lr': 0.0005,  # Constant for swim style
                        'stroke_lr': {
                                'initial_lr': 0.0005,
                                'decay_steps': 1000,
                                'decay_rate': 0.9
                            },
                        'beta_1':          0.9,
                        'beta_2':          0.999,
                        'batch_size':      64,
                        'max_epochs':      48,      # Keeping small for quick testing
                        'steps_per_epoch': 100,      # Keeping small for quick testing
                        'noise_std':       0.01,    # Noise standard deviation for data augmentation
                        'mirror_prob':     0.5,     # Probability of reversing a window for data augmentation
                        'random_rot_deg':  30,      # [-30, 30] is the range of rotation degrees we sample for each
                                                    # window in the mini-batch
                        'group_probs':     {'original': 0.7, 'time_scaled_0.9': 0.15, 'time_scaled_1.1': 0.15},
                        'labels':          [0, 1, 2, 3, 4],
                        'stroke_labels': ['stroke_labels'],  # Labels for stroke predictions
                        'stroke_label_output':      True,
                        'swim_style_output':        False,
                        'output_bias':              None
                        }
    return training_parameters

def common_bilstm_model(inputs, use_seed=True):
    """
    Creates a BiLSTM model for swimming style and stroke count classification.

    :param input_shape: Shape of the input data (timesteps, features).
    :return: Common backbone BiLSTM model.
    """

    # First BiLSTM layer
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=128,  # First BiLSTM hidden units
            return_sequences=True,  # Keep sequences for the next layer
            dropout=0.5,  # Dropout rate
            recurrent_dropout=0.5,  # Recurrent dropout rate
            recurrent_initializer='orthogonal',  # Good for LSTM
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),  # Add LSTM kernel init
            activation="tanh",  # Tanh activation
            name="lstm_1"
        ),
        name="bilstm_1",
        merge_mode="concat",  # Concatenate outputs from both directions
    )(inputs)

    # Second BiLSTM layer
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=64,  # Second BiLSTM hidden units
            return_sequences=True,  # Output a single vector
            dropout=0.2,  # Dropout rate
            recurrent_dropout=0.2,  # Recurrent dropout rate
            recurrent_initializer='orthogonal',  # Good for LSTM
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),  # Add LSTM kernel init
            activation="tanh",  # Tanh activation
            name="lstm_2"
        ),
        name="bilstm_2",
        merge_mode="concat",
    )(x)

    return x

def swim_style_model(common_model=None, use_seed=True, output_bias=None):
    # Swim Style Branch
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    swim_branch = common_model
    swim_branch = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337),
        name="swim_dense_1",
    )(swim_branch)

    swim_branch = tf.keras.layers.Dropout(0.5, name="swim_dropout_1")(swim_branch)

    swim_output = tf.keras.layers.Dense(
        units=5,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
        name="swim_style_output"
    )(swim_branch)

    return swim_output

def stroke_model(common_model=None, use_seed=True, output_bias=None):
    # Stroke Detection Branch
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    stroke_branch = common_model
    # Reshape to (batch, time, features)
 #   temporal_dim = stroke_branch.shape[1]
  #  stroke_branch = tf.keras.layers.Reshape(
   #     (temporal_dim, -1),
    #    name='stroke_reshape'
   # )(stroke_branch)

    stroke_branch = tf.keras.layers.Dropout(0.5, name="stroke_dropout_1")(stroke_branch)
    # Layer normalization
    stroke_branch = tf.keras.layers.LayerNormalization(
        name='stroke_layer_norm'
    )(stroke_branch)

    # TimeDistributed Dense for Per-Timestep Predictions
    stroke_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=1,  # Single output per timestep (stroke/no stroke)
            activation="sigmoid",  # Binary classification
            bias_initializer=output_bias,

            kernel_initializer=tf.keras.initializers.glorot_normal(seed=1337),
            name='stroke_dense'
        ),
        name="stroke_label_output"
    )(stroke_branch)


    return stroke_output


def create_bilstm_model(input_shape, swim_model_parameters=None, stroke_model_parameters=None, use_seed=True, training_parameters=None):

    inputs = tf.keras.Input(shape=input_shape)

    common_model = common_bilstm_model(inputs)

    # Build swim style branch if swim_model_parameters are provided
    swim_style_output = None
    if training_parameters['swim_style_output']:
        swim_style_output = swim_style_model(common_model, use_seed, output_bias=None)

    # Build stroke branch if stroke_model_parameters are provided
    stroke_label_output = None
    if training_parameters['stroke_label_output']:
        stroke_label_output = stroke_model(common_model, use_seed, output_bias=training_parameters['output_bias'])

    # Combine outputs based on the branches enabled
    if swim_style_output is not None and stroke_label_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output, stroke_label_output])
    elif swim_style_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=swim_style_output)
    elif stroke_label_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=stroke_label_output)
    else:
        raise ValueError("No outputs selected for the model.")

    return model

def main():
    input_shape = (180, 6) # Load sensor data, accel, gyro
   # stroke_model_parameters = get_default_stroke_model_parameters()
    training_parameters = get_default_training_parameters()
    model = create_bilstm_model(input_shape, use_seed=True,  training_parameters=training_parameters)
    model.summary()

if __name__ == '__main__':
    main()

