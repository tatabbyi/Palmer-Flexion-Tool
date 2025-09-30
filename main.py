import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


model = tf.keras.models.load_model("palm_model.h5")
class_names = ["simian", "sydney"]
user_image = "images/simian/simian-01.jpg"


img = image.load_img(user_image, target_size=(128,128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

pred = model.predict(x)
predicted_class = class_names[np.argmax(pred)]
print(f"Predicted Class: {predicted_class}")

