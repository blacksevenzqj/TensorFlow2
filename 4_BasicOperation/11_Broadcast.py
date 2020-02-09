import tensorflow as tf

a = tf.random.normal([4, 32, 32, 3])
b = tf.ones([3])
c = tf.fill([32, 32, 1], 2.)
d = tf.random.uniform([4, 1, 1, 1])

(a+b).shape   # (4, 32, 32, 3)
(a+c).shape   # (4, 32, 32, 3)
(a+d).shape   # (4, 32, 32, 3)