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
                        'stroke_labels': ['stroke_labels'] # Labels for stroke predictions
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
    num_cl = len(model_parameters['filters'])
    num_fcl = len(model_parameters['units'])
    cnt_layer = 0

    inputs = tf.keras.Input(shape=input_shape)
    layer = inputs
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    # First store a copy of the input for the stroke branch
    stroke_branch = inputs

    # Main convolutional layers (for swim style)
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
            #padding='same',  # Use 'same' padding
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),
            bias_initializer="zeros",
        )(layer)

        if model_parameters['batch_norm'][cnt_layer]:
            layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(model_parameters['activation'][cnt_layer])(layer)
        if model_parameters['max_pooling'][i] is not None:
            layer = tf.keras.layers.MaxPooling2D(1,(model_parameters['max_pooling'][i], 1))(layer)  # Pooling along time axis, same pooling size for all layers
        if model_parameters['drop_out'][cnt_layer] is not None:
            layer = tf.keras.layers.Dropout(model_parameters['drop_out'][cnt_layer], 
                                          seed=use_seed and 1337)(layer)
        cnt_layer += 1

    # Swim Style Branch
    swim_style_branch = tf.keras.layers.Flatten()(layer)
    
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
        swim_style_branch = tf.keras.layers.Dense(
            units=model_parameters['units'][i],
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
            bias_initializer='zeros'
        )(swim_style_branch)
        
        if model_parameters['batch_norm'][cnt_layer]:
            swim_style_branch = tf.keras.layers.BatchNormalization()(swim_style_branch)
        swim_style_branch = tf.keras.layers.Activation(model_parameters['activation'][cnt_layer])(swim_style_branch)
        if model_parameters['drop_out'][cnt_layer] is not None:
            swim_style_branch = tf.keras.layers.Dropout(model_parameters['drop_out'][cnt_layer], 
                                                      seed=use_seed and 1337)(swim_style_branch)
        cnt_layer += 1

    # Swim Style Output
    swim_style_output = tf.keras.layers.Dense(
        len(model_parameters['labels']),
        activation="softmax",
        kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
        name="swim_style_output",
    )(swim_style_branch)
    # Stroke Detection Branch (with updates to combat overfitting)
    # First Conv layer with regularization and batch norm
    stroke_branch = tf.keras.layers.Conv2D(
        32, (3, 1), 
        padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337),
        kernel_regularizer=tf.keras.regularizers.l2(0.02),
        kernel_constraint=tf.keras.constraints.MaxNorm(3),
        bias_initializer="zeros"
    )(stroke_branch)
    stroke_branch = tf.keras.layers.BatchNormalization()(stroke_branch)
    stroke_branch = tf.keras.layers.Activation('relu')(stroke_branch)
    stroke_branch = tf.keras.layers.Dropout(0.4)(stroke_branch)
    
    # Attention mechanism with regularization
    attention = tf.keras.layers.Conv2D(
        1, (1, 1), 
        padding='same',
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),
        activity_regularizer=tf.keras.regularizers.l1(0.02),
        bias_initializer="zeros"
    )(stroke_branch)
    
    stroke_branch = tf.keras.layers.Multiply()([stroke_branch, attention])
    
    # Reshape to (batch, time, features)
    temporal_dim = stroke_branch.shape[1]  
    stroke_branch = tf.keras.layers.Reshape((temporal_dim, -1))(stroke_branch)
    
    # Reduced capacity LSTM with dropout
    stroke_branch = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=16, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
                           kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),  # Add LSTM kernel init
                           recurrent_initializer='orthogonal',  # Good for LSTM
                
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        merge_mode='concat'
    )(stroke_branch)

    stroke_branch = tf.keras.layers.Dropout(0.3)(stroke_branch)
    stroke_branch = tf.keras.layers.LayerNormalization()(stroke_branch)
    
    # Final stroke detection output
    stroke_label_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation="sigmoid", 
                            kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),  # Add kernel initializer
                            bias_initializer=output_bias,
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        name="stroke_label_output"
    )(stroke_branch)

    # Create Model
    model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output, stroke_label_output])
    return model


def main():
    input_shape = (180, 6, 1) # Load sensor data, accel, gyro, stroke_labels
    model_parameters = get_default_model_parameters()
    model = cnn_model(input_shape, model_parameters)
    model.summary()

if __name__ == '__main__':
    main()


