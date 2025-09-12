import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from detect import detect_creases
from utils import show-image

model = tf.keras.models.load_model("palm_model.h5")
class_names = ["normal", "simian", "sydney"]
user_image = "images/text_user.jpg"

