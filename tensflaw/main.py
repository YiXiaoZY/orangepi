import tensorflow as tf

print("TensorFlow 版本:", tf.__version__)
print(" ",tf.__version__)
print("GPU 可用:", len(tf.config.list_physical_devices("GPU")) > 0)

# 最简单的张量运算
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])
print("a + b =", a + b)