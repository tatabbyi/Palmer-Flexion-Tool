from model_cnn import build_model
from utils import load_data

train_ds, test_ds, class_names =  load_data("images/")

#building the model
model = build_model(num_classes=len(class_names))
#training the modal
model.fit(train_ds, epochs=10, validation_data=test_ds)
#saving the model
model.save("palm_model.h5")
print("Model trained and saved as palm_model.h5")
