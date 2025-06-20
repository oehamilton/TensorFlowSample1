import platform
import tensorflow as tf
from tensorflow import keras 

import numpy as np
print("Platform ", platform.machine()) 
print("TensorFlow Version: ", tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist