import tensorflow as tf

def load_data(data_dir, img_size=(128,128), batch_size=32):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size
        batch_size=batch_size
    )
    
    class_names = dataset.class_names
    print("Loaded classes:", class_names)

    dataset = dataset.map(lamda x, y: (x/255.0, y))

    return dataset, class_names

import cv2
import numpy as np