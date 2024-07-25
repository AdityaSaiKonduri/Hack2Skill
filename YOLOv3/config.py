import tensorflow as tf

print(tf.__version__)
print(len(tf.config.list_physical_devices("GPU")))