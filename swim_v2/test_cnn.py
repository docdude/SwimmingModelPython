import tensorflow as tf
def cnn_model(input_shape=(180, 7, 1), num_classes=5):
    """
    CNN model to predict swim style, lap counts, and stroke labels.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Shared convolutional layers
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 1), activation='elu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 1), activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 1), activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 1), activation='elu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Output 1: Swim Style Prediction (Flatten + Dense)
    flat_features = tf.keras.layers.Flatten()(x)
    swim_style_output = tf.keras.layers.Dense(num_classes, activation='softmax', name="swim_style_output")(flat_features)

    # Output 2: Stroke Label Prediction (TimeDistributed)
    reshaped_features = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)
    stroke_label_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='sigmoid'), name="stroke_label_output"
    )(reshaped_features)

    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=[swim_style_output, stroke_label_output])

    return model

def main():

    model = cnn_model()
    model.summary()

if __name__ == '__main__':
    main()
