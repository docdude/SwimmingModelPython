import tensorflow as tf
from keras_tuner import HyperModel, RandomSearch, BayesianOptimization
from utils_train import compile_model,  CombinedEarlyStopping
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
                        'stroke_label_output':      True,
                        'swim_style_output':        True,
                        'output_bias':              None
                        }
    return training_parameters

def create_swim_model_parameters(hp):
    return {
        'units': [hp.Int(f'swim_dense_units_{i}', min_value=32, max_value=256, step=32) for i in range(1)],
        'activation': [hp.Choice(f'swim_activation_{i}', values=['relu', 'elu', 'leaky_relu']) for i in range(1)],  # Tunable activation for Dense layers
        'batch_norm': [hp.Boolean(f'swim_batch_norm_{i}') for i in range(1)],
        'drop_out': [hp.Float(f'swim_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(1)],
        'max_norm': [hp.Choice(f'swim_max_norm_{i}', values=[0.0, 0.1, 0.5, 1.0, 4.0]) for i in range(1)],
        'l2_reg': [hp.Choice(f'swim_l2_reg_{i}', values=[0.0, 1e-4, 1e-3]) for i in range(1)],
        'labels': [0, 1, 2, 3, 4]
    }

def create_stroke_model_parameters(hp):
    return {
        'drop_out': [hp.Float(f'stroke_dropout{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(1)],
        'stroke_labels': ['stroke_labels']
    }

def create_common_model_parameters(hp):
    num_lstm_layers = 2 #hp.Int('num_lstm_layers', min_value=1, max_value=3)

    return {
        'units': [hp.Int(f'common_lstm_units_{i}', min_value=16, max_value=128, step=16) for i in range(num_lstm_layers)],  # LSTM units
        'lstm_dropout': [hp.Float(f'common_lstm_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(num_lstm_layers)],  # LSTM dropout
        'lstm_recurrent_dropout': [hp.Float(f'common_lstm_recurrent_dropout_{i}', min_value=0.0, max_value=0.5, step=0.1) for i in range(num_lstm_layers)],  # LSTM recurrent dropout
        'lstm_l2_reg': [hp.Choice(f'common_lstm_l2_reg_{i}', values=[0.0, 1e-4, 1e-3]) for i in range(num_lstm_layers)],  # L2 regularization for LSTM
        'lstm_max_norm': [hp.Choice(f'common_lstm_max_norm_{i}', values=[0.0, 0.1, 0.5, 1.0, 4.0]) for i in range(num_lstm_layers)],  # Max norm for LSTM
        'activation': [hp.Choice(f'common_activation_{i}', values=['relu', 'tanh']) for i in range(num_lstm_layers)],  # Tunable activation for LSTM layer

    }


class InvalidHyperparameterConfiguration(Exception):
    """Custom exception to indicate invalid hyperparameter configuration."""
    pass

class SwimStrokeHyperModel(HyperModel):
    def __init__(self, input_shape, training_parameters, class_weights, data_parameters):
        self.input_shape = input_shape
        self.training_parameters = training_parameters
        self.class_weights = class_weights
        self.data_parameters = data_parameters

    def build(self, hp):
        # Create hyperparameters
        common_model_parameters = create_common_model_parameters(hp)

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
        # Add learning rate tuning
        swim_style_lr = (
            hp.Float('swim_style_lr', min_value=1e-5, max_value=1e-2, sampling='log')
            if self.training_parameters['swim_style_output'] 
            else None
        )
        stroke_lr = (
             hp.Float('stroke_lr', min_value=1e-5, max_value=1e-2, sampling='log')
             if self.training_parameters['stroke_label_output']
             else None
        )

        self.training_parameters['swim_style_lr'] = swim_style_lr
        self.training_parameters['stroke_lr']= stroke_lr

        try:
            # Validate hyperparameters before model creation
           # validate_model_parameters(swim_model_parameters, stroke_model_parameters, self.input_shape)

            # Build the CNN model
            model = create_bilstm_model(
                input_shape=self.input_shape,
                common_model_parameters=common_model_parameters,
                swim_model_parameters=swim_model_parameters,
                stroke_model_parameters=stroke_model_parameters,
                training_parameters=self.training_parameters,
            )
            
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

def common_bilstm_model(inputs, common_model_parameters, use_seed=True):
    x = inputs
    num_layers = len(common_model_parameters['units'])  # Get number of LSTM layers from hyperparameters

    for i in range(num_layers):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=common_model_parameters['units'][i],
                return_sequences=True,
                dropout=common_model_parameters['lstm_dropout'][i],
                recurrent_dropout=common_model_parameters['lstm_recurrent_dropout'][i],
                recurrent_initializer='orthogonal',
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
                activation=common_model_parameters['activation'][i],
                kernel_constraint=tf.keras.constraints.max_norm(common_model_parameters['lstm_max_norm'][i])
                    if common_model_parameters['lstm_max_norm'][i] != 0
                    else None,
                kernel_regularizer=tf.keras.regularizers.l2(common_model_parameters['lstm_l2_reg'][i])
                    if common_model_parameters['lstm_l2_reg'][i] != 0
                    else None,
                bias_initializer='zeros',
                name=f"lstm_{i}"
            ),
            name=f"bilstm_{i}",
            merge_mode="concat",
        )(x)
    return x


def swim_style_model(common_model=None, swim_model_parameters=None, use_seed=True, output_bias=None):
    # Swim Style Branch
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    swim_branch = common_model

    swim_branch = tf.keras.layers.GlobalAveragePooling1D(name="swim_style_global_pooling")(swim_branch)

    kernel_constraint = (
        tf.keras.constraints.max_norm(swim_model_parameters['max_norm'][0])
        if swim_model_parameters['max_norm'][0] != 0
        else None
    )
    kernel_regularizer = (
        tf.keras.regularizers.l2(swim_model_parameters['l2_reg'][0])
        if swim_model_parameters['l2_reg'][0] != 0
        else None
    )
    swim_branch = tf.keras.layers.Dense(
        units=swim_model_parameters['units'][0],
        kernel_constraint=kernel_constraint,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=tf.keras.initializers.he_uniform(seed=use_seed and 1337),
        bias_initializer='zeros',
        name=f'swim_style_dense'
    )(swim_branch)
    
    if swim_model_parameters['batch_norm'][0]:
        swim_branch = tf.keras.layers.BatchNormalization(
            name=f'swim_style_dense_bn'
        )(swim_branch)
    swim_branch = tf.keras.layers.Activation(
        swim_model_parameters['activation'][0],
        name=f'swim_style_dense_activation'
    )(swim_branch)
    if swim_model_parameters['drop_out'][0] is not None:
        swim_branch = tf.keras.layers.Dropout(
            swim_model_parameters['drop_out'][0],
            seed=use_seed and 1337,
            name=f'swim_style_dense_dropout'
        )(swim_branch)   


    swim_output = tf.keras.layers.Dense(
        units=len(swim_model_parameters['labels']),
        activation="softmax",
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=use_seed and 1337),
        name="swim_style_output"
    )(swim_branch)

    return swim_output

def stroke_model(common_model=None, stroke_model_parameters=None, use_seed=True, output_bias=None):
    # Stroke Detection Branch
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    stroke_branch = common_model

    stroke_branch = tf.keras.layers.Dropout(
        stroke_model_parameters['drop_out'][0],
        name="stroke_dropout_1"
        )(stroke_branch)
    # Layer normalization
    stroke_branch = tf.keras.layers.LayerNormalization(
        name='stroke_layer_norm'
    )(stroke_branch)

    # TimeDistributed Dense for Per-Timestep Predictions
    stroke_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=len(stroke_model_parameters['stroke_labels']),  # Single output per timestep (stroke/no stroke)
            activation="sigmoid",  # Binary classification
            bias_initializer=output_bias,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=1337),
            name='stroke_dense'
        ),
        name="stroke_label_output"
    )(stroke_branch)

    return stroke_output


def create_bilstm_model(input_shape, common_model_parameters=None, swim_model_parameters=None, stroke_model_parameters=None, use_seed=True, training_parameters=None):

    inputs = tf.keras.Input(shape=input_shape)

    common_model = common_bilstm_model(inputs, common_model_parameters)

    # Build swim style branch if swim_model_parameters are provided
    swim_style_output = None
    if swim_model_parameters is not None and training_parameters['swim_style_output']:
        swim_style_output = swim_style_model(common_model, swim_model_parameters, use_seed=True, output_bias=None)

    # Build stroke branch if stroke_model_parameters are provided
    stroke_label_output = None
    if stroke_model_parameters is not None and training_parameters['stroke_label_output']:
        stroke_label_output = stroke_model(common_model, stroke_model_parameters, use_seed=True, output_bias=training_parameters['output_bias'])

    # Combine outputs based on the branches enabled
    if swim_style_output is not None and stroke_label_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output, stroke_label_output])
    elif swim_style_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=swim_style_output)
    elif stroke_label_output is not None:
        model = tf.keras.Model(inputs=inputs, outputs=stroke_label_output)
    else:
        raise ValueError("No outputs selected for the model.")
    model.summary()
    return model

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



def run_hyperparameter_tuning(input_shape, data_parameters, training_parameters, class_weights, gen, validation_data, run_name="debug", callbacks=None):
    # Create the hypermodel
    hypermodel = SwimStrokeHyperModel(input_shape, training_parameters, class_weights, data_parameters)

    # Check if hypermodel is valid
    if hypermodel is None or not isinstance(hypermodel, HyperModel):
        print("Invalid hypermodel configuration. Skipping tuning for this trial.")
        return None  # Skip this trial
    
    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        objective = kt.Objective('val_combined_metric', direction='max')
    elif training_parameters['swim_style_output']:
        objective=kt.Objective('val_weighted_categorical_accuracy', direction='max')
    else:
        objective=kt.Objective('val_weighted_f1_score', direction='max')



    # Define the Bayesian Optimization tuner
    tuner = BayesianOptimization(
        hypermodel=hypermodel,
        objective=objective,
        max_trials=100,  # Number of trials (adjust based on resources)
        num_initial_points=20,  # Number of random points to start the search
        directory=f'hyperparameter_{run_name}',
        project_name='swim_stroke_model',
        max_consecutive_failed_trials=10 # Increase max_failures

    )

           
    # Set up TensorBoard callback
    log_dir = f"logs/{run_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = get_tensorboard_callback(log_dir)

    # Add TensorBoard callback to the list of callbacks
    if callbacks is None:
        callbacks = []
    callbacks.append(tensorboard_callback)
    # Add debugging callback
    callbacks.append(DebugCallback())
    # Start the search

    tuner.search(
        gen,
        validation_data=validation_data,
        batch_size=training_parameters['batch_size'],
        epochs=training_parameters['max_epochs'],
        steps_per_epoch=training_parameters['steps_per_epoch'],
   #  validation_steps=training_parameters['steps_per_epoch'],  # Define a fixed number of validation steps
        callbacks=callbacks
    )

    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    return best_model, best_hps

if __name__ == '__main__':
    input_shape = (180, 6)
    data_parameters = {'label_type': 'majority', 'debug': True}

    training_parameters = get_default_training_parameters()
    # Example of class weights (modify it to suit your needs)
    class_weights = [1.0, 15.0] # Example class weights for strokes
    # Generate dummy data for demonstration

    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        def gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6)),  # Features
                    {  # Labels
                        'swim_style_output': tf.random.uniform((64, 5), minval=0, maxval=5, dtype=tf.int32),
                        'stroke_label_output': tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32),
                    }
                )

        def val_gen():
            while True:
                yield (
                    tf.random.normal((64, 180, 6)),  # Features
                    { 
                        'swim_style_output': tf.random.uniform((64, 5), minval=0, maxval=5, dtype=tf.int32),
                        'stroke_label_output': tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32),
                    }
                )
        # Set up the combined early stopping callback
        callbacks = [CombinedEarlyStopping(
            monitor1='val_stroke_label_output_weighted_f1_score',
            monitor2='val_swim_style_ouput_weighted_categorical_accuracy',
            mode1='max',
            mode2='max',
            patience=5,
            restore_best_weights=True
        )]
    elif training_parameters['swim_style_output']:  # Only swim style output
        def gen():
            while True:
                yield ([tf.random.normal((64, 180, 6)), 
                        tf.random.uniform((64, 5), minval=0, maxval=4, dtype=tf.int32)])
        def val_gen():
            while True:
                yield ([tf.random.normal((64, 180, 6)), 
                        tf.random.uniform((64, 5), minval=0, maxval=4, dtype=tf.int32)])

        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_categorical_accuracy', patience=10, restore_best_weights=True, mode='max')]
    else:  # Only stroke label output
        def gen():
            while True:
                yield ([tf.random.normal((64, 180, 6)), 
                        tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)])
        def val_gen():
            while True:
                yield ([tf.random.normal((64, 180, 6)), 
                        tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.int32)])
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_weighted_f1_score', patience=10, restore_best_weights=True, mode='max')]

    best_model = run_hyperparameter_tuning(input_shape, data_parameters, training_parameters, class_weights, gen(), val_gen(), callbacks=callbacks)
    best_model.summary()