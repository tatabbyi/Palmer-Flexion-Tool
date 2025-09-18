import cv2
import tensorflow as tf
import numpy as np
from utils import preprocess_image_bytes

Feedback = {
    "simian": "To be added later with research",
    "sydney": "To be added later with research"
}

def detect_local():
    model = tf.keras.models.load_model("palm_model.h5")
    classes = ["simian", "sydney"]

    cap = cv2.VideoCapture(0)
    print("Starting Camera,")