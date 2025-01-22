import tensorflow as tf

def minimal_gen():
    while True:
 
        yield (
            tf.random.normal((64, 180, 6, 1)),
            {'stroke_label_output': tf.random.uniform((64, 180, 1), minval=0, maxval=2, dtype=tf.float32)}
        )

inputs = tf.keras.Input(shape=(180, 6, 1))
outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='stroke_label_output')(inputs)
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
dataset = tf.data.Dataset.from_generator(
    minimal_gen,
    output_signature=(
        tf.TensorSpec(shape=(64, 180, 6, 1), dtype=tf.float32),
        {'stroke_label_output': tf.TensorSpec(shape=(64, 180, 1), dtype=tf.float32)}
    )
)
for features, labels in dataset.take(1):
    print("Features dtype:", features.dtype)
    print("Features shape:", features.shape)
    print("Labels keys:", labels.keys())
    print("Labels['stroke_label_output'] dtype:", labels['stroke_label_output'].dtype)
    print("Labels['stroke_label_output'] shape:", labels['stroke_label_output'].shape)
print("Model output names:", model.output_names)
model.fit(dataset, steps_per_epoch=1, epochs=1, verbose=1)
