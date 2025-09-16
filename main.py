import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from detect import detect_creases
from utils import show-image

model = tf.keras.models.load_model("palm_model.h5")
class_names = ["normal", "simian", "sydney"]
user_image = "images/text_user.jpg"
result_img = detect_creases(user_image)
show_image(result_img, "Detected Creases")

img = image.load_img(user_image, target_size=(128,128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

pred = model.predict(X)
predicted_class = class_names[np.argmax(pred)]
print(f"Predicted Class: {predicted_class}")

