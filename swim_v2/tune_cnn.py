# tune_cnn.py
import tensorflow as tf
from keras_tuner import HyperModel, RandomSearch, BayesianOptimization
from utils_train import compile_model, weighted_binary_crossentropy_smooth_class, EarlyStoppingLogger
import keras_tuner as kt
import pickle
import os
import datetime


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
                        'stroke_label_output':     True,
                        'swim_style_output':      False
                        }
    return training_parameters

def create_swim_model_parameters(hp):
    return {
        'filters': [hp.Int(f'swim_filters_{i}', min_value=32, max_value=128, step=16) for i in range(4)],
        'kernel_sizes': [hp.Choice(f'swim_kernel_size_{i}', values=[1, 3, 5]) for i in range(4)],
        'strides': [hp.Choice(f'swim_stride_{i}', values=[0, 1, 2]) for i in range(4)],
        'max_pooling': [hp.Choice(f'swim_max_pooling_{i}', values=[0, 2, 3]) for i in range(4)],
        'units': [hp.Int(f'swim_dense_units_{i}', min_value=64, max_value=256, step=64) for i in range(1)],
        'activation': [hp.Choice(f'swim_activation_{i}', values=['relu', 'elu', 'leaky_relu']) for i in range(5)],  # Tunable activation for Conv2D layers
        'batch_norm': [hp.Boolean(f'swim_batch_norm_{i}') for i in range(5)],
        'drop_out': [hp.Float(f'swim_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(5)],
        'max_norm': [hp.Choice(f'swim_max_norm_{i}', values=[0.0, 0.1, 0.5, 1.0, 4.0]) for i in range(5)],
        'l2_reg': [hp.Choice(f'swim_l2_reg_{i}', values=[0.0, 1e-4, 1e-3]) for i in range(5)],
        'labels': [0, 1, 2, 3, 4]
    }

def create_stroke_model_parameters(hp):
    return {
        # Filters: Tunable and independent for each layer
        'filters': [hp.Int(f'stroke_filters_{i}', min_value=16, max_value=128, step=16) for i in range(2)],
        # Kernel sizes: Tunable for each Conv2D layer
        'kernel_sizes': [hp.Choice(f'stroke_kernel_size_{i}', values=[1, 3, 5]) for i in range(2)],
        # Strides: Now tunable for each Conv2D layer
        'strides': [hp.Choice(f'stroke_stride_{i}', values=[0, 0, 0]) for i in range(2)],  # Strides of 0 so it set strides = 1 in model, avoid mismatches later in processing
        'max_pooling': [hp.Choice(f'stroke_max_pooling_{i}', values=[0, 0, 0]) for i in range(2)],
        'units': [hp.Int('stroke_lstm_units', min_value=16, max_value=128, step=16)],  # LSTM units
        'lstm_dropout': hp.Float('stroke_lstm_dropout', min_value=0.0, max_value=0.5, step=0.1),  # LSTM dropout
        'lstm_recurrent_dropout': hp.Float('stroke_lstm_recurrent_dropout', min_value=0.0, max_value=0.5, step=0.1),  # LSTM recurrent dropout
        'lstm_l2_reg': hp.Choice('stroke_lstm_l2_reg', values=[0.0, 1e-4, 1e-3]),  # L2 regularization for LSTM
        'lstm_max_norm': hp.Choice('stroke_lstm_max_norm', values=[0.0, 0.1, 0.5, 1.0, 4.0]),  # Max norm for LSTM
        'activation': [hp.Choice(f'stroke_activation_{i}', values=['relu', 'elu', 'leaky_relu']) for i in range(2)],  # Tunable activation for Conv2D layers

        'batch_norm': [hp.Boolean(f'stroke_batch_norm_{i}') for i in range(3)],
        'drop_out': [hp.Float(f'stroke_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(3)],
        'max_norm': [hp.Choice(f'stroke_max_norm_{i}', values=[0.0, 0.1, 0.5, 1.0, 4.0]) for i in range(3)],
        'l2_reg': [hp.Choice(f'stroke_l2_reg_{i}', values=[0.0, 1e-4, 1e-3]) for i in range(3)],
        'final_dropout': hp.Float('stroke_final_dropout', min_value=0.0, max_value=0.5, step=0.1),  # Final dropout before dense
        'stroke_labels': ['stroke_labels']

    }

class InvalidHyperparameterConfiguration(Exception):
    """Custom exception to indicate invalid hyperparameter configuration."""
    pass

def validate_model_parameters(swim_model_parameters=None, stroke_model_parameters=None, input_shape=(2, 6, 16)):
    """
    Validate swim and stroke model parameters to prevent negative dimension errors.
    
    :param swim_model_parameters: Dictionary containing hyperparameters for the swim model.
    :param stroke_model_parameters: Dictionary containing hyperparameters for the stroke model.
    :param input_shape: Tuple representing the shape of the input data (height, width, channels).
    :raises ValueError: If any hyperparameter combination leads to invalid dimensions.
    """
    height, width, channels = input_shape

    # Validate swim model parameters
    if swim_model_parameters is not None:
        for i in range(len(swim_model_parameters['filters'])):
            filter_size = swim_model_parameters['filters'][i]
            kernel_size = swim_model_parameters['kernel_sizes'][i]
            stride = swim_model_parameters['strides'][i]
            if stride == 0:
                stride = 1
            
            max_pooling = swim_model_parameters['max_pooling'][i]

            # Check if kernel size is valid
            if kernel_size > height or kernel_size > width:
                raise ValueError(f"Swim model: Kernel size {kernel_size} exceeds input dimensions {input_shape}.")

            # Check if stride is valid
            if stride > height or stride > width:
                raise ValueError(f"Swim model: Stride {stride} exceeds input dimensions {input_shape}.")

            # Calculate output dimensions after convolution
            output_height = (height - kernel_size) // stride + 1
            output_width = (width - kernel_size) // stride + 1

            # Check if max pooling is valid
            if max_pooling > output_height or max_pooling > output_width:
                raise ValueError(f"Swim model: Max pooling size {max_pooling} exceeds output dimensions ({output_height}, {output_width}).")

            # Update height and width for the next layer
            height = output_height
            width = output_width

    # Reset dimensions for stroke model validation
    height, width = input_shape[:2]

    # Validate stroke model parameters
    if stroke_model_parameters is not None:
        for i in range(len(stroke_model_parameters['filters'])):
            filter_size = stroke_model_parameters['filters'][i]
            kernel_size = stroke_model_parameters['kernel_sizes'][i]
            stride = stroke_model_parameters['strides'][i]
            if stride == 0:
                stride = 1
            max_pooling = stroke_model_parameters['max_pooling'][i]

            # Check if kernel size is valid
            if kernel_size > height or kernel_size > width:
                raise ValueError(f"Stroke model: Kernel size {kernel_size} exceeds input dimensions {input_shape}.")

            # Check if stride is valid
            if stride > height or stride > width:
                raise ValueError(f"Stroke model: Stride {stride} exceeds input dimensions {input_shape}.")

            # Calculate output dimensions after convolution
            output_height = (height - kernel_size) // stride + 1
            output_width = (width - kernel_size) // stride + 1

            # Check if max pooling is valid
            if max_pooling > output_height or max_pooling > output_width:
                raise ValueError(f"Stroke model: Max pooling size {max_pooling} exceeds output dimensions ({output_height}, {output_width}).")

            # Update height and width for the next layer
            height = output_height
            width = output_width

    return True  # If all checks pass

class SwimStrokeHyperModel(HyperModel):
    def __init__(self, input_shape, training_parameters, class_weights, data_parameters):
        self.input_shape = input_shape
        self.training_parameters = training_parameters
        self.class_weights = class_weights
        self.data_parameters = data_parameters

    def build(self, hp):
        swim_model_parameters = (
            create_swim_model_parameters(hp)
            if self.training_parameters['swim_style_output']
            else None
        )
        stroke_model_parameters = (
            create_stroke_model_parameters(hp)
            if self.training_parameters['stroke_label_output']
            else None
        )

        try:
            # Validate hyperparameters before model creation
           # validate_model_parameters(swim_model_parameters, stroke_model_parameters, self.input_shape)

            # Build the CNN model
            model = cnn_model(
                input_shape=self.input_shape,
                swim_model_parameters=swim_model_parameters,
                stroke_model_parameters=stroke_model_parameters,
                training_parameters=self.training_parameters,
            )
            
            if model is None:
                raise InvalidHyperparameterConfiguration("Invalid model configuration. Skipping this trial.")
            
            # Compile the model
            model = compile_model(
                data_parameters=self.data_parameters,
                model=model,
                training_parameters=self.training_parameters,
                class_weights=self.class_weights,
            )

            return model
        
        except (ValueError, InvalidHyperparameterConfiguration) as e:
            print(f"Error in model building: {e}. Skipping this trial.")
            raise Exception(f"TunerError: {str(e)}")  # Re-raise as a generic Exception with a specific message


def cnn_model(input_shape, swim_model_parameters=None, stroke_model_parameters=None, training_parameters=None):

    inputs = tf.keras.Input(shape=input_shape)

    # Build swim style branch if swim_model_parameters are provided
    swim_style_output = None
    if swim_model_parameters is not None and training_parameters['swim_style_output']:
        swim_style_output = swim_style_model(inputs, swim_model_parameters)

    # Build stroke branch if stroke_model_parameters are provided
    stroke_label_output = None
    if stroke_model_parameters is not None and training_parameters['stroke_label_output']:
        stroke_label_output = stroke_model(inputs, stroke_model_parameters)

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


# Define the swim style model (copied from cnn_vanilla_dual.py and modified for tuning)
def swim_style_model(inputs, swim_model_parameters, use_seed=True, output_bias=None):
    """
    Create a CNN model for swim style classification
    :param input_shape: The shape of the input data
    :param swim_model_parameters: A dictionary containing the parameters for the model
    :return: A tf.keras.Model object
    """
    num_cl = len(swim_model_parameters['filters'])
    num_fcl = len(swim_model_parameters['units'])
    cnt_layer = 0

    swim_style_branch = inputs
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Main convolutional layers (for swim style)
    for i in range(num_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
            if swim_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
            if swim_model_parameters['l2_reg'][cnt_layer] != 0
            else None
        )
        strides = swim_model_parameters['strides'][i]
        strides = 1 if strides == 0 else (strides, 1)

        swim_style_branch = tf.keras.layers.Conv2D(
            filters=swim_model_parameters['filters'][i],
            kernel_size=(swim_model_parameters['kernel_sizes'][i], 1),
            strides=strides,
            padding='same',  # Use 'same' padding
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337),
            #kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337), # Add kernel initializer original implementation
            bias_initializer="zeros",
            name=f'swim_style_conv_{i}'
        )(swim_style_branch)

        if swim_model_parameters['batch_norm'][cnt_layer]:
            swim_style_branch = tf.keras.layers.BatchNormalization(
                name=f'swim_style_bn_{i}'
            )(swim_style_branch)
        # Handle different activation functions
        activation_function = swim_model_parameters['activation'][cnt_layer]
        if activation_function == 'leaky_relu':
            swim_style_branch = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'swim_activation_{i}')(swim_style_branch)
        else:
            swim_style_branch = tf.keras.layers.Activation(activation_function, name=f'swim_activation_{i}')(swim_style_branch)
        max_pooling = swim_model_parameters['max_pooling'][i]
        if max_pooling != 0:
            swim_style_branch = tf.keras.layers.MaxPooling2D(
                #1,
                (max_pooling, 1),
                name=f'swim_style_pool_{i}'
            )(swim_style_branch)
        if swim_model_parameters['drop_out'][cnt_layer] is not None:
            swim_style_branch = tf.keras.layers.Dropout(
                swim_model_parameters['drop_out'][cnt_layer],
                seed=use_seed and 1337,
                name=f'swim_style_dropout_{i}'
            )(swim_style_branch)
        cnt_layer += 1

    # Swim Style Branch
    swim_style_branch = tf.keras.layers.Flatten(
        name='swim_style_flatten'
    )(swim_style_branch)
    
    for i in range(num_fcl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][cnt_layer])
            if swim_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][cnt_layer])
            if swim_model_parameters['l2_reg'][cnt_layer] != 0
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

    return swim_style_output

# Define the stroke model (copied from cnn_vanilla_dual.py and modified for tuning)
def stroke_model(inputs, stroke_model_parameters, use_seed=True, output_bias=None):
    """
    Create a CNN model for stroke detection
    :param input_shape: The shape of the input data
    :param stroke_model_parameters: A dictionary containing the parameters for the model
    :return: A tf.keras.Model object
    """
    # Stroke Detection Branch
    num_stroke_cl = len(stroke_model_parameters['filters'])
    cnt_layer = 0
    stroke_branch = inputs
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Convolutional layers for stroke detection
    for i in range(num_stroke_cl):
        kernel_constraint = (
            tf.keras.constraints.max_norm(stroke_model_parameters['max_norm'][cnt_layer])
            if stroke_model_parameters['max_norm'][cnt_layer] != 0
            else None
        )
        kernel_regularizer = (
            tf.keras.regularizers.l2(stroke_model_parameters['l2_reg'][cnt_layer])
            if stroke_model_parameters['l2_reg'][cnt_layer] != 0
            else None
        )
        if i == 0: 
            kernel_initializer=tf.keras.initializers.he_normal(seed=use_seed and 1337)
        else:
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337)

        strides = stroke_model_parameters['strides'][i]
        strides = 1 if strides == 0 else (strides, 1)

        stroke_branch = tf.keras.layers.Conv2D(
            filters=stroke_model_parameters['filters'][i],
            kernel_size=(stroke_model_parameters['kernel_sizes'][i], 1),
            strides=strides,
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
        max_pooling = stroke_model_parameters['max_pooling'][i]
        if max_pooling != 0:
            stroke_branch = tf.keras.layers.MaxPooling2D(
                #1,
                (max_pooling, 1),
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

    # Apply global pooling to the attention tensor
    attention = tf.keras.layers.GlobalAveragePooling2D(name='attention_global_pooling')(attention)

    # Expand dimensions of the attention tensor to match the feature tensor
    attention = tf.keras.layers.Dense(
        units=stroke_branch.shape[-1],  # Match the number of filters in the feature tensor
        activation='sigmoid',  # Use sigmoid to scale attention values between 0 and 1
        name='attention_dense'
    )(attention)
    attention = tf.keras.layers.Reshape((1, 1, stroke_branch.shape[-1]), name='attention_reshape')(attention)

    # Multiply attention with features
    stroke_branch = tf.keras.layers.Multiply(name='stroke_attention')([stroke_branch, attention])

    
    # Reshape to (batch, time, features)
    temporal_dim = stroke_branch.shape[1]
    stroke_branch = tf.keras.layers.Reshape(
        (temporal_dim, -1),
        name='stroke_reshape'
    )(stroke_branch)
    
    # LSTM layer
    stroke_branch = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=stroke_model_parameters['units'][0],  # Tunable LSTM units
            return_sequences=True,
            dropout=stroke_model_parameters['lstm_dropout'],  # Tunable LSTM dropout
            recurrent_dropout=stroke_model_parameters['lstm_recurrent_dropout'],  # Tunable recurrent dropout
            recurrent_initializer='orthogonal',  # Good for LSTM
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
            kernel_constraint=tf.keras.constraints.max_norm(stroke_model_parameters['lstm_max_norm'])  # Tunable max norm
                if stroke_model_parameters['lstm_max_norm'] != 0
                else None,
            kernel_regularizer=tf.keras.regularizers.l2(stroke_model_parameters['lstm_l2_reg'])  # Tunable L2 regularization
                if stroke_model_parameters['lstm_l2_reg'] != 0
                else None,
            bias_initializer='zeros',
            name='stroke_lstm'
        ),
        merge_mode='concat',
        name='stroke_bilstm'
    )(stroke_branch)

    if stroke_model_parameters['final_dropout'] > 0.0:
        stroke_branch = tf.keras.layers.Dropout(
            stroke_model_parameters['final_dropout'],
            name='stroke_lstm_dropout'
        )(stroke_branch)

    stroke_branch = tf.keras.layers.LayerNormalization(
        name='stroke_layer_norm'
    )(stroke_branch)

    # Stroke detection output
    stroke_label_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            len(stroke_model_parameters['stroke_labels']),
            activation="sigmoid",
            bias_initializer=output_bias,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=use_seed and 1337),
            name='stroke_dense'
        ),
        name="stroke_label_output"
    )(stroke_branch)

    return stroke_label_output

class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} Logs:")
        for key, value in logs.items():
            print(f"{key}: {value}")

def get_tensorboard_callback(log_dir):
    """
    Create a TensorBoard callback for logging hyperparameters and metrics.
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Log histograms of weights every epoch
        write_graph=True,  # Log the computation graph
        write_images=True,  # Log model weights as images
        update_freq='epoch',  # Log metrics at the end of each epoch
        profile_batch=0  # Disable profiling to avoid overhead
    )
    return tensorboard_callback


def run_hyperparameter_tuning(input_shape, data_parameters, training_parameters, class_weights, gen, validation_data, experiment_save_path="", callbacks=None):
    # Initialize the tuner
   # tuner = RandomSearch(
   #     hypermodel=SwimStrokeHyperModel(input_shape, training_parameters, class_weights, data_parameters),
   #     objective=kt.Objective('val_f1_score', direction='max'),  # Specify direction
   #     max_trials=20,
   #     executions_per_trial=2,
   #     directory='hyperparameter_tuning',
   #     project_name='swim_stroke_model'
   # )
    # Create the hypermodel
    hypermodel = SwimStrokeHyperModel(input_shape, training_parameters, class_weights, data_parameters)

    # Check if hypermodel is valid
    if hypermodel is None or not isinstance(hypermodel, HyperModel):
        print("Invalid hypermodel configuration. Skipping tuning for this trial.")
        return None  # Skip this trial
     
    if training_parameters['stroke_label_output']:
        objective=kt.Objective('val_weighted_f1_score', direction='max')
    else:
        objective=kt.Objective('val_weighted_categorical_accuracy', direction='max')


    # Define the Bayesian Optimization tuner
    tuner = BayesianOptimization(
        hypermodel=hypermodel,
        objective=objective,
        max_trials=100,  # Number of trials (adjust based on resources)
        num_initial_points=20,  # Number of random points to start the search
        directory='hyperparameter_tuning_stroke100',
        project_name='swim_stroke_model',
        max_consecutive_failed_trials=10 # Increase max_failures

    )

           
    # Set up TensorBoard callback
    log_dir = f"logs/tune_stroke100/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = get_tensorboard_callback(log_dir)

    # Add TensorBoard callback to the list of callbacks
    if callbacks is None:
        callbacks = []
    callbacks.append(tensorboard_callback)
    # Add debugging callback
    #callbacks.append(DebugCallback())
    # Start the search

    tuner.search(
        gen,
        validation_data=validation_data,
        batch_size=training_parameters['batch_size'],
        epochs=training_parameters['max_epochs'],
        steps_per_epoch=training_parameters['steps_per_epoch'],
    #   validation_steps=training_parameters['steps_per_epoch'],  # Define a fixed number of validation steps
        callbacks=callbacks
    )

    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Save the best hyperparameters to a pickle file in the specified directory
    best_hyperparameters_path = os.path.join(experiment_save_path, 'best_hyperparameters.pkl')
    with open(best_hyperparameters_path, 'wb') as f:
        pickle.dump(best_hps.values, f)  # Save the hyperparameters as a dictionary

    print(f"Best hyperparameters saved to: {best_hyperparameters_path}")

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    return best_model

if __name__ == '__main__':
    input_shape = (180, 6, 1)
    data_parameters = {'label_type': 'majority', 'debug': True}

    training_parameters = get_default_training_parameters()
    # Example of class weights (modify it to suit your needs)
    class_weights = [1.0, 15.0] # Example class weights for strokes
    # Generate dummy data for demonstration

    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        def gen():
            while True:
                yield ([tf.random.normal((64, 180, 6, 1)), 
                        [tf.random.uniform((64, 5), minval=0, maxval=4, dtype=tf.int32), tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)]])
        def val_gen():
            while True:
                yield ([tf.random.normal((64, 180, 6, 1)), 
                        [tf.random.uniform((64, 5), minval=0, maxval=4, dtype=tf.int32), tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)]])
    elif training_parameters['swim_style_output']:  # Only swim style output
        def gen():
            while True:
                yield ([tf.random.normal((64, 180, 6, 1)), 
                        tf.random.uniform((64, 5), minval=0, maxval=4, dtype=tf.int32)])
        def val_gen():
            while True:
                yield ([tf.random.normal((64, 180, 6, 1)), 
                        tf.random.uniform((64, 5), minval=0, maxval=4, dtype=tf.int32)])
    else:  # Only stroke label output
        def gen():
            while True:
                yield ([tf.random.normal((64, 180, 6, 1)), 
                        tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)])
        def val_gen():
            while True:
                yield ([tf.random.normal((64, 180, 6, 1)), 
                        tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)])


    best_model = run_hyperparameter_tuning(input_shape, data_parameters, training_parameters, class_weights, gen(), val_gen(), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_f1_score', patience=5, restore_best_weights=True, mode='max')])
    best_model.summary()