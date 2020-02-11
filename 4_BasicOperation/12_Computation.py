import tensorflow as tf

b = tf.fill([2,2],2.)
a = tf.ones([2,2])
print(a+b, b//2, b%a)

# In[]:
# 对数
a = tf.ones([2,2])
c = tf.math.log(a)
print(c)
d = tf.exp(a)
print(d)

# tf.math.log只有以e为底的公式
tf.math.log(8.) / tf.math.log(2.) # 除法实现 log以2为底8
tf.math.log(100.) / tf.math.log(10.) # 除法实现 log以10为底100

# In[]:
b = tf.fill([2,2],2.)
a = tf.ones([2,2])

# 平方、开方
print(tf.pow(b, 3))
print(b**3)
print(tf.sqrt(b))

# In[]:
# @ 和 matmul
print(a@b)
print(tf.matmul(a,b))

# In[]:
# 第一个索引4 表示 batch批次： 也就是4个矩阵点乘
a = tf.ones([4,2,3])
b = tf.fill([4,3,5], 2.)
print(a@b)
print(tf.matmul(a,b))

# In[]:
a = tf.ones([4,2,3])
b = tf.fill([3,5], 2.)
bb = tf.broadcast_to(b, [4,3,5]) # 广播后 可直接矩阵乘法
print(a@b)
print(tf.matmul(a,b))

# In[]:
x = tf.ones([4,2])
w = tf.ones([2,1])
b = tf.constant(0.1)
print(b.dtype, tf.is_tensor(b))
print(x@w+b)

out = tf.nn.relu(x@w+b)
print(out)

