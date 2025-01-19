import tensorflow as tf

def get_default_swim_model_parameters():
    """
    Get parameters for swim style branch
    """
    swim_model_parameters = {
        'filters':        [64, 64, 64, 64],
        'kernel_sizes':   [3, 3, 3, 3],
        'strides':        [None, None, None, None],
        'max_pooling':    [3,3,3,3],
        'units':          [128],
        'activation':     ['elu', 'elu', 'elu', 'elu', 'elu'],
        'batch_norm':     [False, False, False, False, False],
        'drop_out':       [0.5, 0.75, 0.25, 0.1, 0.25],
        'max_norm':       [0.1, 0.1, None, 4.0, 4.0],
        'l2_reg':         [None, None, None, None, None],
        'labels':         [0, 1, 2, 3, 4]
    }
    return swim_model_parameters

def get_default_stroke_model_parameters():
    """
    Get parameters for stroke detection branch
    """
    stroke_model_parameters = {
        'filters':        [64, 64],  # First conv, attention conv
        'kernel_sizes':   [3, 1],    # 3x1 for conv, 1x1 for attention
        'strides':        [None, None],
        'max_pooling':    [None, None],
        'units':          [32],      # LSTM units
        'activation':     ['elu', 'sigmoid', 'sigmoid'],  # conv, attention, output
        'batch_norm':     [False, False, False],
        'drop_out':       [0.3, 0.3, 0.4],  # Adjusted dropout rates
        'max_norm':       [0.5, 0.5, 2.0],  # Reduced max norm constraints
        'l2_reg':         [1e-4, 1e-4, 1e-4],  # Add light L2 regularization
        'labels':         ['stroke_labels']     # binary stroke detection
    }
    return stroke_model_parameters

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
                        'stroke_label_output':     False,
                        'swim_style_output':      True
                        }
    return training_parameters

def cnn_model(input_shape, swim_model_parameters, stroke_model_parameters, training_parameters, use_seed=True, output_bias=None):

    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        num_cl = len(swim_model_parameters['filters'])
        num_fcl = len(swim_model_parameters['units'])
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
                tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
                if swim_model_parameters['max_norm'][cnt_layer]
                else None
            )
            kernel_regularizer = (
                tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
                if swim_model_parameters['l2_reg'][cnt_layer]
                else None
            )
            strides = 1 if swim_model_parameters['strides'][i] is None else (swim_model_parameters['strides'][i], 1)

            layer = tf.keras.layers.Conv2D(
                filters=swim_model_parameters['filters'][i],
                kernel_size=(swim_model_parameters['kernel_sizes'][i], 1),
                strides=strides,
                #padding='same',  # Use 'same' padding
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337),
                #kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337), # Add kernel initializer original implementation
                bias_initializer="zeros",
                name=f'swim_style_conv_{i}'
            )(layer)

            if swim_model_parameters['batch_norm'][cnt_layer]:
                layer = tf.keras.layers.BatchNormalization(
                    name=f'swim_style_bn_{i}'
                )(layer)
            layer = tf.keras.layers.Activation(
                swim_model_parameters['activation'][cnt_layer],
                name=f'swim_style_activation_{i}'
            )(layer)
            if swim_model_parameters['max_pooling'][i] is not None:
                layer = tf.keras.layers.MaxPooling2D(
                  #  1,
                    (swim_model_parameters['max_pooling'][i], 1),
                    name=f'swim_style_pool_{i}'
                )(layer)
            if swim_model_parameters['drop_out'][cnt_layer] is not None:
                layer = tf.keras.layers.Dropout(
                    swim_model_parameters['drop_out'][cnt_layer],
                    seed=use_seed and 1337,
                    name=f'swim_style_dropout_{i}'
                )(layer)
            cnt_layer += 1

        # Swim Style Branch
        swim_style_branch = tf.keras.layers.Flatten(
            name='swim_style_flatten'
        )(layer)
        
        for i in range(num_fcl):
            kernel_constraint = (
                tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
                if swim_model_parameters['max_norm'][cnt_layer]
                else None
            )
            kernel_regularizer = (
                tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
                if swim_model_parameters['l2_reg'][cnt_layer]
                else None
            )
            swim_style_branch = tf.keras.layers.Dense(
                units=swim_model_parameters['units'][i],
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
                bias_initializer='zeros',
                name=f'swim_style_dense_{i}'
            )(swim_style_branch)
            
            if swim_model_parameters['batch_norm'][cnt_layer]:
                swim_style_branch = tf.keras.layers.BatchNormalization(
                    name=f'swim_style_dense_bn_{i}'
                )(swim_style_branch)
            swim_style_branch = tf.keras.layers.Activation(
                swim_model_parameters['activation'][cnt_layer],
                name=f'swim_style_dense_activation_{i}'
            )(swim_style_branch)
            if swim_model_parameters['drop_out'][cnt_layer] is not None:
                swim_style_branch = tf.keras.layers.Dropout(
                    swim_model_parameters['drop_out'][cnt_layer],
                    seed=use_seed and 1337,
                    name=f'swim_style_dense_dropout_{i}'
                )(swim_style_branch)
            cnt_layer += 1

        # Swim Style Output
        swim_style_output = tf.keras.layers.Dense(
            len(swim_model_parameters['labels']),
            activation="softmax",
            #kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),

            name="swim_style_output"
        )(swim_style_branch)
        
        # Stroke Detection Branch
        num_stroke_cl = len(stroke_model_parameters['filters'])
        cnt_layer = 0

        # Convolutional layers for stroke detection
        for i in range(num_stroke_cl):
            kernel_constraint = (
                tf.keras.constraints.max_norm(stroke_model_parameters['max_norm'][cnt_layer])
                if stroke_model_parameters['max_norm'][cnt_layer]
                else None
            )
            kernel_regularizer = (
                tf.keras.regularizers.l2(stroke_model_parameters['l2_reg'][cnt_layer])
                if stroke_model_parameters['l2_reg'][cnt_layer]
                else None
            )
            if i == 0: 
                kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337)
            else:
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337)

            stroke_branch = tf.keras.layers.Conv2D(
                filters=stroke_model_parameters['filters'][i],
                kernel_size=(stroke_model_parameters['kernel_sizes'][i], 1),
                padding='same',
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer="zeros",
                name=f'stroke_conv_{i}'
            )(stroke_branch)

            if stroke_model_parameters['batch_norm'][cnt_layer]:
                stroke_branch = tf.keras.layers.BatchNormalization(
                    name=f'stroke_bn_{i}'
                )(stroke_branch)
            stroke_branch = tf.keras.layers.Activation(
                stroke_model_parameters['activation'][cnt_layer],
                name=f'stroke_activation_{i}'
            )(stroke_branch)
            if stroke_model_parameters['max_pooling'][i] is not None:
                stroke_branch = tf.keras.layers.MaxPooling2D(
                    #1,
                    (stroke_model_parameters['max_pooling'][i], 1),
                    name=f'stroke_pool_{i}'
                )(stroke_branch)
            if stroke_model_parameters['drop_out'][cnt_layer] is not None:
                stroke_branch = tf.keras.layers.Dropout(
                    stroke_model_parameters['drop_out'][cnt_layer],
                    seed=use_seed and 1337,
                    name=f'stroke_dropout_{i}'
                )(stroke_branch)
            
            # If this is the first conv layer, create attention mechanism
            if i == 0:
                attention = stroke_branch
            cnt_layer += 1

        # Multiply attention with features
        stroke_branch = tf.keras.layers.Multiply(
            name='stroke_attention'
        )([stroke_branch, attention])
        
        # Reshape to (batch, time, features)
        temporal_dim = stroke_branch.shape[1]
        stroke_branch = tf.keras.layers.Reshape(
            (temporal_dim, -1),
            name='stroke_reshape'
        )(stroke_branch)
        
        # LSTM layer
        stroke_branch = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=stroke_model_parameters['units'][0],
                return_sequences=True,
                recurrent_initializer='orthogonal',  # Good for LSTM
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),  # Add LSTM kernel init
                kernel_constraint=tf.keras.constraints.max_norm(stroke_model_parameters['max_norm'][-1])
                    if stroke_model_parameters['max_norm'][-1] else None,
                kernel_regularizer=tf.keras.regularizers.l2(stroke_model_parameters['l2_reg'][-1])
                    if stroke_model_parameters['l2_reg'][-1] else None,
                bias_initializer='zeros',
                name='stroke_lstm'
            ),
            merge_mode='concat',
            name='stroke_bilstm'
        )(stroke_branch)
        
        if stroke_model_parameters['drop_out'][-1]:
            stroke_branch = tf.keras.layers.Dropout(
                stroke_model_parameters['drop_out'][-1],
                name='stroke_lstm_dropout'
            )(stroke_branch)
        
        stroke_branch = tf.keras.layers.LayerNormalization(
            name='stroke_layer_norm'
        )(stroke_branch)
        
        # Stroke detection output
        stroke_label_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                len(stroke_model_parameters['labels']),
                activation="sigmoid",
                bias_initializer=output_bias,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),
                name='stroke_dense'
            ),
            name="stroke_label_output"
        )(stroke_branch)

        # Create Model
        model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output, stroke_label_output])

        return model
    elif training_parameters['swim_style_output']:
        num_cl = len(swim_model_parameters['filters'])
        num_fcl = len(swim_model_parameters['units'])
        cnt_layer = 0

        inputs = tf.keras.Input(shape=input_shape)
        layer = inputs
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        # Main convolutional layers (for swim style)
        for i in range(num_cl):
            kernel_constraint = (
                tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
                if swim_model_parameters['max_norm'][cnt_layer]
                else None
            )
            kernel_regularizer = (
                tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
                if swim_model_parameters['l2_reg'][cnt_layer]
                else None
            )
            strides = 1 if swim_model_parameters['strides'][i] is None else (swim_model_parameters['strides'][i], 1)

            layer = tf.keras.layers.Conv2D(
                filters=swim_model_parameters['filters'][i],
                kernel_size=(swim_model_parameters['kernel_sizes'][i], 1),
                strides=strides,
                #padding='same',  # Use 'same' padding
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337),
                #kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337), # Add kernel initializer original implementation
                bias_initializer="zeros",
                name=f'swim_style_conv_{i}'
            )(layer)

            if swim_model_parameters['batch_norm'][cnt_layer]:
                layer = tf.keras.layers.BatchNormalization(
                    name=f'swim_style_bn_{i}'
                )(layer)
            layer = tf.keras.layers.Activation(
                swim_model_parameters['activation'][cnt_layer],
                name=f'swim_style_activation_{i}'
            )(layer)
            if swim_model_parameters['max_pooling'][i] is not None:
                layer = tf.keras.layers.MaxPooling2D(
                    #1,
                    (swim_model_parameters['max_pooling'][i], 1),
                    name=f'swim_style_pool_{i}'
                )(layer)
            if swim_model_parameters['drop_out'][cnt_layer] is not None:
                layer = tf.keras.layers.Dropout(
                    swim_model_parameters['drop_out'][cnt_layer],
                    seed=use_seed and 1337,
                    name=f'swim_style_dropout_{i}'
                )(layer)
            cnt_layer += 1

        # Swim Style Branch
        swim_style_branch = tf.keras.layers.Flatten(
            name='swim_style_flatten'
        )(layer)
        
        for i in range(num_fcl):
            kernel_constraint = (
                tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
                if swim_model_parameters['max_norm'][cnt_layer]
                else None
            )
            kernel_regularizer = (
                tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
                if swim_model_parameters['l2_reg'][cnt_layer]
                else None
            )
            swim_style_branch = tf.keras.layers.Dense(
                units=swim_model_parameters['units'][i],
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
                bias_initializer='zeros',
                name=f'swim_style_dense_{i}'
            )(swim_style_branch)
            
            if swim_model_parameters['batch_norm'][cnt_layer]:
                swim_style_branch = tf.keras.layers.BatchNormalization(
                    name=f'swim_style_dense_bn_{i}'
                )(swim_style_branch)
            swim_style_branch = tf.keras.layers.Activation(
                swim_model_parameters['activation'][cnt_layer],
                name=f'swim_style_dense_activation_{i}'
            )(swim_style_branch)
            if swim_model_parameters['drop_out'][cnt_layer] is not None:
                swim_style_branch = tf.keras.layers.Dropout(
                    swim_model_parameters['drop_out'][cnt_layer],
                    seed=use_seed and 1337,
                    name=f'swim_style_dense_dropout_{i}'
                )(swim_style_branch)
            cnt_layer += 1

        # Swim Style Output
        swim_style_output = tf.keras.layers.Dense(
            len(swim_model_parameters['labels']),
            activation="softmax",
            #kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),

            name="swim_style_output"
        )(swim_style_branch)
        # Create Model
        model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output])

        return model
    else:
        # Stroke Detection Branch
        num_stroke_cl = len(stroke_model_parameters['filters'])
        cnt_layer = 0
        inputs = tf.keras.Input(shape=input_shape)
        layer = inputs
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        # First store a copy of the input for the stroke branch
        stroke_branch = inputs
        # Convolutional layers for stroke detection
        for i in range(num_stroke_cl):
            kernel_constraint = (
                tf.keras.constraints.max_norm(stroke_model_parameters['max_norm'][cnt_layer])
                if stroke_model_parameters['max_norm'][cnt_layer]
                else None
            )
            kernel_regularizer = (
                tf.keras.regularizers.l2(stroke_model_parameters['l2_reg'][cnt_layer])
                if stroke_model_parameters['l2_reg'][cnt_layer]
                else None
            )
            if i == 0: 
                kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337)
            else:
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337)

            stroke_branch = tf.keras.layers.Conv2D(
                filters=stroke_model_parameters['filters'][i],
                kernel_size=(stroke_model_parameters['kernel_sizes'][i], 1),
                padding='same',
                kernel_constraint=kernel_constraint,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer="zeros",
                name=f'stroke_conv_{i}'
            )(stroke_branch)

            if stroke_model_parameters['batch_norm'][cnt_layer]:
                stroke_branch = tf.keras.layers.BatchNormalization(
                    name=f'stroke_bn_{i}'
                )(stroke_branch)
            stroke_branch = tf.keras.layers.Activation(
                stroke_model_parameters['activation'][cnt_layer],
                name=f'stroke_activation_{i}'
            )(stroke_branch)
            if stroke_model_parameters['max_pooling'][i] is not None:
                stroke_branch = tf.keras.layers.MaxPooling2D(
                    #1,
                    (stroke_model_parameters['max_pooling'][i], 1),
                    name=f'stroke_pool_{i}'
                )(stroke_branch)
            if stroke_model_parameters['drop_out'][cnt_layer] is not None:
                stroke_branch = tf.keras.layers.Dropout(
                    stroke_model_parameters['drop_out'][cnt_layer],
                    seed=use_seed and 1337,
                    name=f'stroke_dropout_{i}'
                )(stroke_branch)
            
            # If this is the first conv layer, create attention mechanism
            if i == 0:
                attention = stroke_branch
            cnt_layer += 1

        # Multiply attention with features
        stroke_branch = tf.keras.layers.Multiply(
            name='stroke_attention'
        )([stroke_branch, attention])
        
        # Reshape to (batch, time, features)
        temporal_dim = stroke_branch.shape[1]
        stroke_branch = tf.keras.layers.Reshape(
            (temporal_dim, -1),
            name='stroke_reshape'
        )(stroke_branch)
        
        # LSTM layer
        stroke_branch = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=stroke_model_parameters['units'][0],
                return_sequences=True,
                recurrent_initializer='orthogonal',  # Good for LSTM
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),  # Add LSTM kernel init
                kernel_constraint=tf.keras.constraints.max_norm(stroke_model_parameters['max_norm'][-1])
                    if stroke_model_parameters['max_norm'][-1] else None,
                kernel_regularizer=tf.keras.regularizers.l2(stroke_model_parameters['l2_reg'][-1])
                    if stroke_model_parameters['l2_reg'][-1] else None,
                bias_initializer='zeros',
                name='stroke_lstm'
            ),
            merge_mode='concat',
            name='stroke_bilstm'
        )(stroke_branch)
        
        if stroke_model_parameters['drop_out'][-1]:
            stroke_branch = tf.keras.layers.Dropout(
                stroke_model_parameters['drop_out'][-1],
                name='stroke_lstm_dropout'
            )(stroke_branch)
        
        stroke_branch = tf.keras.layers.LayerNormalization(
            name='stroke_layer_norm'
        )(stroke_branch)
        
        # Stroke detection output
        stroke_label_output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                len(stroke_model_parameters['labels']),
                activation="sigmoid",
                bias_initializer=output_bias,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),
                name='stroke_dense'
            ),
            name="stroke_label_output"
        )(stroke_branch)

        # Create Model
        model = tf.keras.Model(inputs=inputs, outputs=[stroke_label_output])

        return model


def main():
    input_shape = (180, 6, 1) # Load sensor data, accel, gyro, stroke_labels
    swim_model_parameters = get_default_swim_model_parameters()
    stroke_model_parameters = get_default_stroke_model_parameters()
    training_parameters = get_default_training_parameters()
    model = cnn_model(input_shape, swim_model_parameters, stroke_model_parameters, training_parameters)
    model.summary()

if __name__ == '__main__':
    main()


