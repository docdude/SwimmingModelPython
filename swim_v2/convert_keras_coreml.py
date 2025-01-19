import coremltools as ct
import tensorflow as tf

#from keras.models import load_model
#from tensorflow.python.keras.models import load_model
#import keras

import os
print(tf.version.VERSION)
#print(keras.__version__)
#print(h5py.__version__)

model_path = 'tutorial_save_path_epoch60/27/model_best.keras'
model = tf.keras.models.load_model(model_path,compile=False)

model.summary()

# Define class labels if known
class_labels = ['turn', 'freestyle', 'breaststroke', 'backstroke', 'butterfly']
model = ct.convert(model,convert_to='mlprogram',source="tensorflow",
                   inputs=[ct.TensorType(name="input_21", shape=(1,180,9,1))],
                        classifier_config=ct.ClassifierConfig(class_labels)
                   #output_names="SwimType",
    #  class_labels=["turn", "freestyle", "breaststroke", "backstroke", "butterfly"],  # Define class labels
   #  output_shapes="1,5" # Output the raw probabilities (one per class)
     )
swim_model_best = 'swim_model_best.mlpackage'
model.save(swim_model_best)
see_model = ct.models.MLModel(swim_model_best)
print(see_model)
print("Done")
