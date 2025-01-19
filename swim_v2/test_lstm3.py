import tensorflow as tf

def lstm_model(input_shape):
    # Define the input shape (batch_size, 180, 5)
    input_layer = tf.keras.Input(shape=input_shape)

    # Bi-LSTM Layer 1
    bi_lstm_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=64,
            activation='tanh',
            return_sequences=True,
            dropout=0.25,
            recurrent_dropout=0.25
        )
    )(input_layer)

    # Bi-LSTM Layer 2
    bi_lstm_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=32,
            activation='tanh',
            return_sequences=True,
            dropout=0.25,
            recurrent_dropout=0.25
        )
    )(bi_lstm_1)

    # Bi-LSTM Layer 3
    bi_lstm_3 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=16,
            activation='tanh',
            return_sequences=True,
            dropout=0.25,
            recurrent_dropout=0.25
        )
    )(bi_lstm_2)

    # Shared features for both outputs
    bi_lstm_4 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=16,
            activation='tanh',
            return_sequences=False,
            dropout=0.25,
            recurrent_dropout=0.25
        )
    )(bi_lstm_3)

    # Flatten layer for Swim Style Classification
    flatten_layer = tf.keras.layers.Flatten()(bi_lstm_4)
    shared_features = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=16,
            activation='tanh',
            return_sequences=True,
            dropout=0.25,
            recurrent_dropout=0.25
        )
    )(bi_lstm_3)

    # Shared Dense Layer for Swim Style Classification
    shared_dense = tf.keras.layers.Dense(50, activation='relu')(flatten_layer)
    shared_dense = tf.keras.layers.Dropout(0.5)(shared_dense)
    shared_dense = tf.keras.layers.BatchNormalization()(shared_dense)
    shared_dense = tf.keras.layers.Dropout(0.5)(shared_dense)

    # Output 1: Swim Style Classification
    swim_style_output = tf.keras.layers.Dense(5, activation='softmax', name='swim_style_output')(shared_dense)

    # Output 2: Stroke Detection (Sequence-Based)
    stroke_detection_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='sigmoid'), name='stroke_label_output'
    )(shared_features)

    # Define the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=[swim_style_output, stroke_detection_output])

    return model

def main():
    input_shape = (180, 6)  # Sensor data: accel, gyro, stroke_labels
    model = lstm_model(input_shape)
    model.summary()

if __name__ == '__main__':
    main()
