from model_cnn import build_model, load_data

dataset = load_data("image/")

train_size = int(0.8 * len(dataset))
train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)
#building the model
model = build_model(num_classes=len(dataset.class_names))
#training the modal
model.fit(train_ds, epochs=10, validation_data=test_ds)
#saving the model
model.save("palm_model.h5")
print("Model trained and saved as palm_model.h5")
