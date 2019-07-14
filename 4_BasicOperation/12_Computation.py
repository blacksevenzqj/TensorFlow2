import tensorflow as tf

b = tf.fill([2,2],2.)
a = tf.ones([2,2])
# print(a+b, b//2, b%a)

print("====================================================================")

c = tf.math.log(a)
# print(c)
d = tf.exp(a)
# print(d)

tf.math.log(8.) / tf.math.log(2.) # log以2为底
tf.math.log(100.) / tf.math.log(10.) # log以10为底

print("====================================================================")

print(tf.pow(b, 3))
print(b**3)
print(tf.sqrt(b))

print("====================================================================")

print(a@b)
print(tf.matmul(a,b))

# 第一个索引4 表示 batch批次：也就是4个矩阵点乘
a = tf.ones([4,2,3])
b = tf.fill([4,3,5], 2.)
print(a@b)
print(tf.matmul(a,b))

print("====================================================================")

a = tf.ones([4,2,3])
b = tf.fill([3,5], 2.)
bb = tf.broadcast_to(b, [4,3,5])
print(a@b)
print(tf.matmul(a,b))

print("====================================================================")

x = tf.ones([4,2])
w = tf.ones([2,1])
b = tf.constant(0.1)
print(b.dtype, tf.is_tensor(b))
print(x@w+b)

