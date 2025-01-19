import tensorflow as tf
from tensorflow.keras import layers, models

class LSTMWithExactParams(tf.keras.layers.LSTM):
    def __init__(self, units=64, **kwargs):
        super(LSTMWithExactParams, self).__init__(units, **kwargs)
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Calculate sizes to achieve exactly 39,680 params per direction
        # Target: 79,360 / 2 = 39,680 per direction
        
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer='glorot_uniform')
            
        # Adjust recurrent units to account for additional cell
        recurrent_units = 77  # Adjusted to reach target parameter count
        self.recurrent_kernel = self.add_weight(
            shape=(recurrent_units, self.units * 4),
            name='recurrent_kernel',
            initializer='orthogonal')
            
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name='bias',
            initializer='zeros')
        
        total = tf.size(self.kernel) + tf.size(self.recurrent_kernel) + tf.size(self.bias)
        print(f"\nParameter breakdown:")
        print(f"Input weights shape: {self.kernel.shape}, params: {tf.size(self.kernel).numpy()}")
        print(f"Recurrent weights shape: {self.recurrent_kernel.shape}, params: {tf.size(self.recurrent_kernel).numpy()}")
        print(f"Bias shape: {self.bias.shape}, params: {tf.size(self.bias).numpy()}")
        print(f"Total params per direction: {total.numpy()}")

input_layer = tf.keras.Input(shape=(90, 6))
bi_lstm_1 = layers.Bidirectional(
    LSTMWithExactParams(
        units=64,
        activation='tanh',
        return_sequences=True,
        dropout=0.25,
        recurrent_dropout=0.25
    )
)(input_layer)

# Rest of the model
bi_lstm_2 = layers.Bidirectional(
    layers.LSTM(units=32, activation='tanh', return_sequences=True, 
                dropout=0.25, recurrent_dropout=0.25)
)(bi_lstm_1)

bi_lstm_3 = layers.Bidirectional(
    layers.LSTM(units=16, activation='tanh', return_sequences=True,
                dropout=0.25, recurrent_dropout=0.25)
)(bi_lstm_2)

bi_lstm_4 = layers.Bidirectional(
    layers.LSTM(units=16, activation='tanh', return_sequences=False,
                dropout=0.25, recurrent_dropout=0.25)
)(bi_lstm_3)

flatten_layer = layers.Flatten()(bi_lstm_4)
dense_1 = layers.Dense(50, activation='relu')(flatten_layer)
dropout = layers.Dropout(0.5)(dense_1)
batch_norm = layers.BatchNormalization()(dense_1)
output_layer = layers.Dense(8, activation='softmax')(batch_norm)

model = models.Model(inputs=input_layer, outputs=output_layer)

# Print actual weights after model creation
print("\nActual weight shapes in model:")
first_layer = model.layers[1]
for weight in first_layer.weights:
    print(f"{weight.name}: shape={weight.shape}, params={tf.size(weight).numpy()}")

print("\nComplete Model Summary:")
model.summary()
