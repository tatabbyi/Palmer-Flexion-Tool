import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 3), num_classes=2):
    model = models.Sequential([
        #for creases and edge detection 32 filters
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        #views image with less detail
        layers.MaxPooling2D((2,2)),
        #convolutional layer 64 filters
        layers.Conv2D(64, (3,3), activation='relu'),
        #shrinks image focuses on important creases
        layers.MaxPooling2D((2,2)),
        #128 filters
        layers.Conv2D(128, (3,3), activation='relu'),
        #2d grid of numbers flattens to 1D list
        layers.Flatten(),
        #128 neurons connects to all features found connecting them
        layers.Dense(128, activation='relu'),
        #turns output into probabilties
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
