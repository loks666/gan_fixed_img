import tensorflow as tf

# 设定TensorFlow只使用CPU
# tf.config.set_visible_devices([], 'GPU')

# print("TensorFlow version:", tf.__version__)
# print("Available devices:", tf.config.list_physical_devices())


import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras loaded from TensorFlow")

import numpy as np
import matplotlib
print("NumPy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)

