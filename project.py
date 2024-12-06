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

train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCHSIZE
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCHSIZE
)

# CNN Model
model = keras.Sequential()

