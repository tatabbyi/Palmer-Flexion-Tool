import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

interpreter = tf.lite.Interpreter(model_path="palm_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()