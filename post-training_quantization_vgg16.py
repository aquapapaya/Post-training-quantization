"""
Post Training Quantization using TensorFlow for VGG-16
======================================================
# The script will generate a full integer quantized 
  VGG-16 .tflite model
# By Kuen-Wey Lin
# pip3 show tensorflow => Version: 2.5.0
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
image_shape = (224, 224)
data_type = 'float32' #expected type FLOAT32 for input of already-trained float TensorFlow model

# a generator function to load images for calibration
def representative_dataset():
    num_calibration_steps = 50 # number of images for calibration
    imgs = []
    batch_size = 1
    for sn in range(num_calibration_steps):
        image_data = tf.keras.preprocessing.image.load_img('./test/' + str(sn) + '.jpg', target_size=image_shape)
        image_data = tf.keras.preprocessing.image.img_to_array(image_data)
        image_data = tf.keras.applications.vgg16.preprocess_input(image_data)
        #image_data = image_data.astype(np.float32)
        image_data = np.reshape(image_data, (224, 224, 3))
        imgs.append(image_data)
  
    imgs = np.array(imgs)
    images = tf.data.Dataset.from_tensor_slices(imgs).batch(1)
    for i in images.take(batch_size):
      yield [i]

# a Keras image classification model, loaded with weights pre-trained on ImageNet
# You can find the downloaded Keras files in $HOME/.keras
model = tf.keras.applications.vgg16.VGG16(weights="imagenet", input_shape=(224, 224, 3))

# convert a tf.Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# "DEFAULT" quantizes model weights
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# tf.lite.RepresentativeDataset requires a generator function, so use Python's "yield"
converter.representative_dataset = representative_dataset

# The following three lines are used for full integer quantization (input/output/activation tensors are int8)
# To generate a model with float32 input, remark the following three lines
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

tflite_quant_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

