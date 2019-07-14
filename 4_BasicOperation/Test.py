import numpy as np
import tensorflow as tf

y = tf.convert_to_tensor([3,2,8])
print(y)
y_onehot = tf.one_hot(y, depth=10)
print(y_onehot)

out = tf.convert_to_tensor([[0.3, 0.1, 0.2, 1, 0.2, 0.4, 0.5, 0.6, 0.3, 0.2],
                            [0.6, 0.8, 0.9, 0.1, 1, 0.3, 0.5, 0.6, 0.7, 0.3],
                            [0.6, 0.8, 0.9, 0.1, 1, 0.3, 0.5, 0.6, 0.7, 0.3]])
print(out)


a = tf.square(y_onehot - out)
print(a)
print(tf.reduce_mean(a)) # 求2次平均
print(tf.reduce_sum(a) / 30) # 所以是除以元素总个数