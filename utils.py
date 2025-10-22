import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2

def load_data(data_dir, img_size=(128,128), batch_size=32, seed=1337):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )
    
    class_names = train_ds.class_names
    print("Loaded classes:", class_names)

    train_ds = train_ds.map(lambda x, y:(x / 255.0, y))

    return train_ds, None, class_names

import cv2
import numpy as np

def preprocess_image(frame):

    img = cv2.resize(frame, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img 

def show_image_cv2(window_title, frame):
    cv2.imshow(window_title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()