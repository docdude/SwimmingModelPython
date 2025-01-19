import tensorflow as tf


def get_default_model_parameters():
    """
    Get a default set of parameters used to define a cnn model
    :return: A dictionary containing parameter names and values
    """
    model_parameters = {'filters':        [64, 64, 64, 64],
                        'kernel_sizes':   [3, 3, 3, 3],
                        'strides':        [None, None, None, None],
                        'max_pooling':    [3,3,3,3],
                        'units':          [180],
                        'activation':     ['elu', 'elu', 'elu', 'elu', 'elu'],
                        'batch_norm':     [False, False, False, False, False],
                        'drop_out':       [0.5, 0.75, 0.25, 0.1, 0.25],
                        'max_norm':       [0.1, 0.1, None, 4.0, 4.0],
                        'l2_reg':         [None, None, None, None, None],
                        'labels':         [0, 1, 2, 3, 4],
                        'stroke_labels': ['stroke_labels']  # Labels for stroke predictions
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
                           'labels':          [0, 1, 2, 3, 4],
                           'stroke_labels': ['stroke_labels']  # Labels for stroke predictions
                           }
    return training_parameters

def cnn_model(input_shape, model_parameters, use_seed=True, output_bias=None):
    """
    Returns a CNN model for swim style and stroke label predictions.
    """
    num_cl = len(model_parameters['filters'])
    num_fcl = len(model_parameters['units'])
    cnt_layer = 0

    inputs = tf.keras.Input(shape=input_shape)
    layer = inputs
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    # Convolutional Layers
    for i in range(num_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(model_parameters['max_norm'][cnt_layer])
            if model_parameters['max_norm'][cnt_layer]
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(model_parameters['l2_reg'][cnt_layer])
            if model_parameters['l2_reg'][cnt_layer]
            else None
        )
        strides = 1 if model_parameters['strides'][i] is None else (model_parameters['strides'][i], 1)

        layer = tf.keras.layers.Conv2D(
            filters=model_parameters['filters'][i],
            kernel_size=(model_parameters['kernel_sizes'][i], 1),
            strides=strides,
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),
            bias_initializer="zeros",
        )(layer)

        if model_parameters['batch_norm'][cnt_layer]:
            layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(model_parameters['activation'][cnt_layer])(layer)
        if model_parameters['max_pooling'][i] is not None:
            layer = tf.keras.layers.MaxPooling2D(1,(model_parameters['max_pooling'][i], 1))(layer)
        if model_parameters['drop_out'][cnt_layer] is not None:
            layer = tf.keras.layers.Dropout(model_parameters['drop_out'][cnt_layer], seed=use_seed and 1337)(layer)
        cnt_layer += 1

    # Flatten and Reshape Layers
    print("Shape before flatten:", layer.shape)
    shared_features = layer
    swim_style_branch = tf.keras.layers.Flatten()(shared_features)
    print("Shape of flat_features:", swim_style_branch.shape)
    # Fully connected layers
    for i in range(num_fcl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(model_parameters['max_norm'][cnt_layer])
            if model_parameters['max_norm'][cnt_layer]
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(model_parameters['l2_reg'][cnt_layer])
            if model_parameters['l2_reg'][cnt_layer]
            else None
        )
        swim_style_branch = tf.keras.layers.Dense(units=model_parameters['units'][i],
                                     kernel_constraint=kernel_constraint,
                                     kernel_regularizer=kernel_regularizer,
                                     kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
                                     bias_initializer='zeros')(swim_style_branch)
        if model_parameters['batch_norm'][cnt_layer]:
            swim_style_branch = tf.keras.layers.BatchNormalization()(swim_style_branch)
        swim_style_branch = tf.keras.layers.Activation(model_parameters['activation'][cnt_layer])(swim_style_branch)
        if model_parameters['drop_out'][cnt_layer] is not None:
            swim_style_branch = tf.keras.layers.Dropout(model_parameters['drop_out'][cnt_layer], seed=use_seed and 1337)(swim_style_branch)
        cnt_layer += 1
    

    #flat_features = layer
    # Swim Style Output
    swim_style_output = tf.keras.layers.Dense(
        len(model_parameters['labels']),
        activation="softmax",
        kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
        name="swim_style_output",
    )(swim_style_branch)
    #temporal_dim = input_shape[0]  # Assuming input_shape is (180, 7, 1)
    #stroke_label_features = tf.keras.layers.Reshape((temporal_dim, -1))(layer)

    # Replace dynamic reshape with static reshape
    #stroke_label_features = tf.reshape(layer, (-1,layer.shape[-1], 1))#temporal_dim, feature_dim))
    #stroke_label_features = tf.keras.layers.Reshape((-1, layer.shape[-1]))(layer)#temporal_dim, feature_dim))

    #print("Shape of stroke_label_features after reshape: ", tf.shape(stroke_label_features))

    # Stroke Label Output
   # stroke_label_output = tf.keras.layers.TimeDistributed(
    #    tf.keras.layers.Dense(1, activation="sigmoid"),
     #   name="stroke_label_output",
    #)(stroke_label_features)
    # Reshape shared_features from (None, 1, 7, 64) -> (None, 7, 64)
    stroke_label_branch = tf.keras.layers.Reshape((-1, shared_features.shape[2] * shared_features.shape[3]))(shared_features)

    # Upsample to restore temporal dimension to 180
    stroke_label_branch = tf.keras.layers.UpSampling1D(size=90)(stroke_label_branch)  # Adjust size based on downsampling

    # Additional Conv1D layers for feature refinement
    stroke_label_branch = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(stroke_label_branch)
    stroke_label_branch = tf.keras.layers.Dropout(0.3)(stroke_label_branch)

    stroke_label_branch = tf.keras.layers.Conv1D(32, kernel_size=2, activation='relu', padding='same')(stroke_label_branch)
    stroke_label_branch = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=32)(stroke_label_branch, stroke_label_branch)

    # LSTM for temporal dependencies
    stroke_label_branch = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(stroke_label_branch)

    # Final Dense layer for binary stroke classification
    stroke_label_output = tf.keras.layers.Dense(
        1, 
        activation="sigmoid",
        bias_initializer=output_bias,
        name="stroke_label_output"
    )(stroke_label_branch)



    # Create Model
    model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output, stroke_label_output])

    return model


def main():
    input_shape = (180, 7, 1) # Load sensor data, accel, gyro, stroke_labels
    model_parameters = get_default_model_parameters()
    model = cnn_model(input_shape, model_parameters)
    model.summary()

if __name__ == '__main__':
    main()


