import tensorflow as tf
from tensorflow import keras 
import numpy as np
import kagglehub

# Downloads images directly from Kaggle
# Download latest version
path = kagglehub.dataset_download("andrewmvd/medical-mnist")

print("Path to dataset files:", path)


# Create dataset
BATCHSIZE = 64
IMG_HEIGHT = 64
IMG_WIDTH = 64

train_ds = keras.utils.image_dataset_from_directory(
    path,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCHSIZE
)

validation_ds = keras.utils.image_dataset_from_directory(
    path,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCHSIZE
)

class_names = train_ds.class_names

# FIXME: Somehow, I need the datasets to be in a numpy array unless there is another way.
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break



# CNN Model
# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D())

# model.add(keras.layers.Flatten())

# model.add(keras.layers.Dense())
# model.add(keras.layers.Dense(units = 6, activation="softmax"))


# model.compile(
#     optimizer = "adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# history = model.fit()

# score = model.evaluate()

# MAKE PREDICTIONS