import tensorflow as tf
import os
import tensorflow_addons as tfa
def weighted_binary_crossentropy(weight_zero=1.0, weight_one=15.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Reshape y_true to match y_pred if needed
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        
        # Calculate weights maintaining the shape
        weights = y_true * weight_one + (1 - y_true) * weight_zero
        
        # Calculate binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Ensure shapes match for multiplication
        weights = tf.squeeze(weights, axis=-1)
        
        # Calculate weighted loss
        weighted_loss = weights * bce
        return tf.reduce_mean(weighted_loss)
    return loss

 
def weighted_binary_crossentropy_smooth_class(class_weights):  
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        
        # Use calculated class weights
        weight_zero, weight_one = class_weights
        weights = y_true * weight_one + (1 - y_true) * weight_zero
        weights = tf.squeeze(weights, axis=-1)

        # BCE with label smoothing
        bce = tf.keras.losses.binary_crossentropy(
            y_true, 
            y_pred, 
            label_smoothing=0.1  # Smooth labels directly
        )
        
        return tf.reduce_mean(weights * bce)
    return loss


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
        if logs is None:
            logs = {}
        
        # Get learning rate
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'optimizers_and_layers'):
            # Multi-optimizer case
            for i, (opt, _) in enumerate(optimizer.optimizers_and_layers):
                if isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr = opt.learning_rate(opt.iterations)
                else:
                    lr = opt.learning_rate
                lr_value = float(lr)
                logs[f'lr_optimizer_{i}'] = lr_value
                
                # Write to TensorBoard
                tf.summary.scalar(f'learning_rate_optimizer_{i}', data=lr_value, step=epoch)
        else:
            # Single optimizer case
            if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = optimizer.learning_rate(optimizer.iterations)
            else:
                lr = optimizer.learning_rate
            lr_value = float(lr)
            logs['lr'] = lr_value
            
            # Write to TensorBoard
            tf.summary.scalar('learning_rate', data=lr_value, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Ensure learning rate is in logs for CSV logger
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'optimizers_and_layers'):
            for i, (opt, _) in enumerate(optimizer.optimizers_and_layers):
                if isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr = opt.learning_rate(opt.iterations)
                else:
                    lr = opt.learning_rate
                logs[f'lr_optimizer_{i}'] = float(lr)
        else:
            if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = optimizer.learning_rate(optimizer.iterations)
            else:
                lr = optimizer.learning_rate
            logs['lr'] = float(lr)

class LearningRateLogger_new(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self._current_epoch = 0
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        optimizer = self.model.optimizer

        # Handle multi-optimizer case
        if hasattr(optimizer, 'optimizers_and_layers'):
            for i, (opt, _) in enumerate(optimizer.optimizers_and_layers):
                if isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr = opt.learning_rate(opt.iterations)
                else:
                    lr = opt.learning_rate
                lr_value = float(lr)

                # Write custom scalar for TensorBoard
                with self.writer.as_default():
                    tf.summary.scalar(f'learning_rate_optimizer_{i}', data=lr_value, step=epoch)
        else:
            # Handle single optimizer case
            if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = optimizer.learning_rate(optimizer.iterations)
            else:
                lr = optimizer.learning_rate
            lr_value = float(lr)

            # Write custom scalar for TensorBoard
            with self.writer.as_default():
                tf.summary.scalar('learning_rate', data=lr_value, step=epoch)

class EarlyStoppingLogger(tf.keras.callbacks.EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        val_weighted_f1_score = logs.get('val_weighted_f1_score', None)
        if val_weighted_f1_score is not None:
            print(f" Epoch {epoch + 1}: val_weighted_f1_score = {val_weighted_f1_score:.4f}, patience = {self.wait}")
        else:
            print(f" Epoch {epoch + 1}: val_weighted_f1_score not available.")

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

class F1ScoreMultiClass(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', average='macro', threshold=0.5, num_classes=None, **kwargs):
        """
        Multi-class F1 score metric with support for various averaging methods.

        Args:
            name (str): Name of the metric.
            average (str): Averaging method - "macro", "weighted", or "micro".
            threshold (float): Threshold for binary classification (for each class in multi-class case).
            num_classes (int): Number of classes (required for "macro" or "weighted").
        """
        super().__init__(name=name, **kwargs)
        self.average = average
        self.threshold = threshold
        self.num_classes = num_classes

        # Initialize precision and recall per class
        self.precision = [tf.keras.metrics.Precision(thresholds=threshold) for _ in range(num_classes)]
        self.recall = [tf.keras.metrics.Recall(thresholds=threshold) for _ in range(num_classes)]

        # Accumulators for "micro" averaging
        if average == "micro":
            self.micro_tp = self.add_weight(name="micro_tp", initializer="zeros")
            self.micro_fp = self.add_weight(name="micro_fp", initializer="zeros")
            self.micro_fn = self.add_weight(name="micro_fn", initializer="zeros")

        # Weight storage for "weighted" averaging
        if average == "weighted":
            self.class_weights = self.add_weight(name="class_weights", shape=(num_classes,), initializer="zeros")
            self.class_totals = self.add_weight(name="class_totals", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state with true and predicted labels.
        """
        # Ensure y_true and y_pred are the correct shape and type
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # If predictions are probabilities, apply thresholding
        if y_pred.shape[-1] == self.num_classes:  # Probabilistic predictions
            y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.one_hot(y_true, depth=self.num_classes)
        y_pred = tf.one_hot(y_pred, depth=self.num_classes)

        if self.average == "micro":
            # Micro-averaged counts
            tp = tf.reduce_sum(y_true * y_pred)
            fp = tf.reduce_sum((1 - y_true) * y_pred)
            fn = tf.reduce_sum(y_true * (1 - y_pred))
            self.micro_tp.assign_add(tp)
            self.micro_fp.assign_add(fp)
            self.micro_fn.assign_add(fn)
        else:
            # Update per-class precision and recall
            for i in range(self.num_classes):
                self.precision[i].update_state(y_true[..., i], y_pred[..., i], sample_weight)
                self.recall[i].update_state(y_true[..., i], y_pred[..., i], sample_weight)

                if self.average == "weighted":
                    # Track weights for weighted averaging
                    class_total = tf.reduce_sum(y_true[..., i])
                    self.class_weights[i].assign_add(class_total)
                    self.class_totals[i].assign_add(1.0)

    def result(self):
        """
        Compute the F1 score based on the averaging method.
        """
        if self.average == "micro":
            # Micro-averaged F1
            precision = self.micro_tp / (self.micro_tp + self.micro_fp + tf.keras.backend.epsilon())
            recall = self.micro_tp / (self.micro_tp + self.micro_fn + tf.keras.backend.epsilon())
            return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        else:
            # Compute F1 for each class
            f1_scores = []
            for i in range(self.num_classes):
                p = self.precision[i].result()
                r = self.recall[i].result()
                f1 = 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
                f1_scores.append(f1)

            f1_scores = tf.stack(f1_scores)

            if self.average == "macro":
                return tf.reduce_mean(f1_scores)
            elif self.average == "weighted":
                weights = self.class_weights / (tf.reduce_sum(self.class_weights) + tf.keras.backend.epsilon())
                return tf.reduce_sum(weights * f1_scores)

    def reset_state(self):
        """
        Reset all metrics and accumulators.
        """
        for i in range(self.num_classes):
            self.precision[i].reset_state()
            self.recall[i].reset_state()
        if self.average == "micro":
            self.micro_tp.assign(0.0)
            self.micro_fp.assign(0.0)
            self.micro_fn.assign(0.0)
        if self.average == "weighted":
            self.class_weights.assign(tf.zeros_like(self.class_weights))
            self.class_totals.assign(tf.zeros_like(self.class_totals))

    def get_config(self):
        """
        Return the configuration of the metric.
        """
        config = super().get_config()
        config.update({
            "average": self.average,
            "threshold": self.threshold,
            "num_classes": self.num_classes
        })
        return config

def combined_metric(logs, alpha=0.5):
    """
    Combine two metrics with a weighted harmonic mean.
    
    :param logs: Dictionary containing logged metrics (e.g., logs from callbacks).
    :param alpha: Weight for the first metric (0 ≤ alpha ≤ 1). The second metric weight will be 1 - alpha.
    :return: Combined metric value.
    """
    metric1 = logs.get('val_stroke_label_output_weighted_f1_score', 0.0)  # Stroke branch
    metric2 = logs.get('val_swim_style_output_weighted_categorical_accuracy', 0.0)  # Swim style branch
    
    # Avoid division by zero
    if metric1 == 0 or metric2 == 0:
        return 0.0
    
    # Calculate the weighted harmonic mean
    harmonic_mean = 2 / ((alpha / metric1) + ((1 - alpha) / metric2))
    return harmonic_mean

class CombinedMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        combined = combined_metric(logs, alpha=self.alpha)
        logs['val_combined_metric'] = combined
        print(f"Epoch {epoch + 1}: val_combined_metric = {combined:.4f}")
           
class CombinedEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor1, monitor2, mode1='max', mode2='max', patience=5, restore_best_weights=True):
        super(CombinedEarlyStopping, self).__init__()
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.mode1 = mode1
        self.mode2 = mode2
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best1 = -float('inf') if mode1 == 'max' else float('inf')
        self.best2 = -float('inf') if mode2 == 'max' else float('inf')
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current1 = logs.get(self.monitor1)
        current2 = logs.get(self.monitor2)

        if current1 is None or current2 is None:
            print(f"Warning: Monitor metrics {self.monitor1} or {self.monitor2} not found in logs.")
            return

        improved1 = (current1 > self.best1 if self.mode1 == 'max' else current1 < self.best1)
        improved2 = (current2 > self.best2 if self.mode2 == 'max' else current2 < self.best2)

        if improved1 or improved2:
            self.best1 = max(current1, self.best1) if self.mode1 == 'max' else min(current1, self.best1)
            self.best2 = max(current2, self.best2) if self.mode2 == 'max' else min(current2, self.best2)
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}.")
   
def get_callbacks(model_save_path, user, log_dir):
    callbacks = [
        # Early Stopping Logger
        EarlyStoppingLogger(
            monitor='val_weighted_f1_score',  # Use weighted F1 score if appropriate
            patience=30,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1,
            mode='max'
        ),
   
        # Reduce Learning Rate on Plateau - More conservative
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_weighted_f1_score',  # Align with early stopping
            factor=0.4,  # Slightly less aggressive reduction
            patience=10,  # Reduced patience
            min_delta=0.001,
            verbose=1,
            mode='max',  # Changed to max
            min_lr=5e-5  # Slightly higher minimum learning rate
        ),

        # Model Checkpoint - More specific
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, f'model_{user}_best'),
            monitor='val_weighted_f1_score',  # Specific to stroke output
            save_best_only=True,
            mode='max',  # Changed to max
            verbose=1,
            save_weights_only=False  # Save full model
        ),
        
        # TensorBoard with more detailed configuration
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
            profile_batch=0  # Disable profiling
        ),
        
        # Updated Learning Rate Logger
        LearningRateLogger_new(log_dir=log_dir),
        
        # CSV Logger with append mode
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_save_path, f'training_history_{user}.csv'),
            separator=',',
            append=True  # Preserve history
        )
    ]
    
    return callbacks

def get_layers(model, layer_name_filter=None):
    """
    Recursively retrieve layers from a model, including wrapped layers in constructs like TimeDistributed or Bidirectional.
    
    :param model: The model or layer to retrieve layers from.
    :param layer_name_filter: A substring to filter layers by name.
    :return: A list of layers matching the filter.
    """
    layers = []
    if hasattr(model, 'layers'):  # Check if the model or layer has sub-layers
        for layer in model.layers:
            layers.extend(get_layers(layer, layer_name_filter))
    else:
        layers = [model]  # Base case: This is a single layer

    # Apply name filter if provided
    if layer_name_filter:
        layers = [layer for layer in layers if layer_name_filter in layer.name]
    return layers


class MultiOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizers_and_layers):
        super(MultiOptimizer, self).__init__(name="MultiOptimizer")
        self.optimizers_and_layers = optimizers_and_layers
        
        # Set a default learning rate for the main optimizer
        self._learning_rate = self.optimizers_and_layers[0][0].learning_rate
        
    def _create_slots(self, var_list):
        for optimizer, _ in self.optimizers_and_layers:
            optimizer._create_slots(var_list)
            
    def apply_gradients(self, grads_and_vars, **kwargs):
        optimizer_grads = {opt: [] for opt, _ in self.optimizers_and_layers}
        
        for grad, var in grads_and_vars:
            for optimizer, layers in self.optimizers_and_layers:
                if any(var.name.startswith(layer.name) for layer in layers):
                    optimizer_grads[optimizer].append((grad, var))
                    break
                    
        for optimizer, grads in optimizer_grads.items():
            if grads:
                optimizer.apply_gradients(grads, **kwargs)

    @property
    def learning_rate(self):
        # Return the swim style learning rate for logging
        return self._learning_rate

    def get_config(self):
        return {"name": self.name}


def compile_model(data_parameters, model, training_parameters, class_weights=None):

    if data_parameters['label_type'] == 'sparse':
        type_categorical_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy()#'sparse_categorical_crossentropy'
    else:
        type_categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy() #'categorical_crossentropy'

    if training_parameters['swim_style_output'] and training_parameters['stroke_label_output']:
        # Get layers for each branch
        swim_style_layers = [layer for layer in model.layers 
                            if 'swim_style' in layer.name]
        stroke_layers     = [layer for layer in model.layers 
                            if 'stroke' in layer.name]
        common_layers     = [layer for layer in model.layers
                            if 'bilstm' in layer.name]
         # Debug print
        if data_parameters['debug']:
            print("\nSwim Style Layers:")
            for layer in swim_style_layers:
                print(f"  {layer.name, layer.trainable}")
        
            print("\nStroke Layers:")
            for layer in stroke_layers:
                print(f"  {layer.name, layer.trainable}")

            print("\nCommon Layers:")
            for layer in common_layers:
                print(f"  {layer.name, layer.trainable}")

        # Create optimizers with explicit learning rates
        swim_style_optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_parameters['swim_style_lr'],  
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )

        stroke_optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_parameters['stroke_lr'],  
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        """
        stroke_optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=training_parameters['stroke_lr']['initial_lr'],
                decay_steps=training_parameters['stroke_lr']['decay_steps'],
                decay_rate=training_parameters['stroke_lr']['decay_rate']
            ),
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        """
        # Combine optimizers
 #       optimizer = MultiOptimizer([
  #          (swim_style_optimizer, swim_style_layers),
   #         (stroke_optimizer, stroke_layers)
    #    ])
        optimizers_and_layers = [(swim_style_optimizer, common_layers + swim_style_layers), (stroke_optimizer, common_layers + stroke_layers)]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        loss={
            'swim_style_output': type_categorical_crossentropy,
            'stroke_label_output': tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
                                #weighted_binary_crossentropy_smooth_class(class_weights)
        }
        metrics={
            'swim_style_output': [
                tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ],
            'stroke_label_output': [
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5),
                F1Score(name='f1_score', threshold=0.5)
            ]
        }
        weighted_metrics={
            'swim_style_output': [
                tf.keras.metrics.CategoricalAccuracy(name='weighted_categorical_accuracy'),
                tf.keras.metrics.Precision(name='weighted_precision'),
                tf.keras.metrics.Recall(name='weighted_recall')
            ],
            'stroke_label_output': [
                tf.keras.metrics.BinaryAccuracy(name='weighted_accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='weighted_precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='weighted_recall', thresholds=0.5),
                F1Score(name='weighted_f1_score', threshold=0.5)
            ]
        }

    elif training_parameters['swim_style_output']:
        # Create optimizers with explicit learning rates
        swim_style_optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_parameters['swim_style_lr'],  
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        optimizer = swim_style_optimizer
        loss = {
            'swim_style_output': type_categorical_crossentropy
        }
        metrics = {
            'swim_style_output': [
                tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                tf.keras.metrics.CategoricalCrossentropy(name='cross_entropy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        }
        weighted_metrics = {
            'swim_style_output': [
                tf.keras.metrics.CategoricalAccuracy(name='weighted_categorical_accuracy'),
                tf.keras.metrics.Precision(name='weighted_precision'),
                tf.keras.metrics.Recall(name='weighted_recall')
            ]
        }
    else:
        stroke_optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_parameters['stroke_lr'],  
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        """
        stroke_optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=training_parameters['stroke_lr']['initial_lr'],
                decay_steps=training_parameters['stroke_lr']['decay_steps'],
                decay_rate=training_parameters['stroke_lr']['decay_rate']
            ),
            beta_1=training_parameters['beta_1'],
            beta_2=training_parameters['beta_2']
        )
        """
        optimizer = stroke_optimizer
        loss = {
            'stroke_label_output': tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
#weighted_binary_crossentropy_smooth_class(class_weights)
        }
        metrics = {
            'stroke_label_output': [
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5),
                F1Score(name='f1_score', threshold=0.5)
            ]
        }
        weighted_metrics = {
            'stroke_label_output': [
                tf.keras.metrics.BinaryAccuracy(name='weighted_accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='weighted_precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='weighted_recall', thresholds=0.5),
                F1Score(name='weighted_f1_score', threshold=0.5)
            ]
        }



    # Then use it in model compilation
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics

    )

    return model
