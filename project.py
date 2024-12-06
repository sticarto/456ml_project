import tensorflow as tf
from tensorflow import keras 
import numpy as np
import kagglehub

# Downloads images directly from Kaggle
# Download latest version
path = kagglehub.dataset_download("alessiocorrado99/animals10")

print("Path to dataset files:", path)

