import numpy as np
import  tensorflow as tf
from  tensorflow.keras import layers, optimizers, datasets


print(np.zeros([2,3]))
print(np.ones([2,3]))
a = tf.convert_to_tensor([[1,2,3],[4,5,6]])
print(a)

b = np.array([[1,2,3],[4,5,6]])
print(b.shape)

c = tf.zeros([2,3,3])
print(c)
print(tf.zeros_like(c))
print(tf.zeros(c.shape))

print("====================================================================")

print(tf.fill([2,2],9))

print("====================================================================")

z = tf.random.normal([2,2], mean=0, stddev=1)
print(z)
z = tf.random.truncated_normal([2,2], mean=0, stddev=1)
print(z)

z = tf.random.uniform([2,2], minval=0, maxval=1)
print(z)

idx = tf.range(10)
idx = tf.random.shuffle(idx)
a = tf.random.normal([10, 784], mean=0, stddev=1)
b = tf.random.uniform([10], maxval=10, dtype=tf.int32)
a = tf.gather(a, idx)
b = tf.gather(b, idx)
print(a)
print(b)

print("====================================================================")

a = tf.constant(1)
print(a)
b = tf.constant([1])
print(b)
c = tf.constant([[1,2,3],[4,5,6]])
print(c)

print("====================================================================")

out = tf.random.uniform([2, 5], seed=1)
print(out)
y = tf.range(2)
y = tf.one_hot(y, depth=5)
print(y)
loss = tf.keras.losses.mse(y, out) # mse是均方差；rmse才是均方根。
print(loss)
loss = tf.reduce_mean(loss) # 求2次平均（最小二乘）
print(loss)
print("--------------------------------------------------------------------")
# 1、直接一步求
loss = tf.reduce_mean(tf.square(y - out)) # 直接求2次平均（最小二乘）
print(loss)
# 2、分开2步求：
loss = tf.reduce_mean(tf.square(y - out), axis=1)
loss = tf.reduce_mean(loss, axis=0)
print(loss)
# 3、分开2步求：
loss = tf.reduce_sum(tf.square(y - out))
loss = loss / 10 # 10为矩阵的元素个数
print(loss)

print("====================================================================")

'''
这里有坑：
1、layers.Dense(10)是指输出的列个数为10，由于输入是4:8的，又由于使用的是x * w：(4:8) * (?:10)，那么w的行个数就为8。
2、bias就是偏移量b，维度为10。
'''
net = layers.Dense(10)
net.build((4, 8)) # 输入shape
print(net.kernel)
print(net.bias)