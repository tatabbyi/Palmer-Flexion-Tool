import cv2
import tensorflow as tf
import numpy as np
from utils import preprocess_image

Feedback = {
    "simian": "To be added later with research",
    "sydney": "To be added later with research"
}

def detect_local():
    model = tf.keras.models.load_model("palm_model.h5")
    classes = ["simian", "sydney"]

    cap = cv2.VideoCapture(0)
    print("Starting Camera,")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam.")
            break

        cv2.imshow("palm Scanner (Press s)", frame)
# listens for key q or esc exits. shows camera and closes window
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        cap.release()
        cv2.destroyAllWindows()
        
        
