import numpy as np
import  tensorflow as tf
from    tensorflow import keras
from  tensorflow.keras import layers, optimizers, datasets


a = tf.constant(1) # 标量
print(a.device) # 运行设备：CPU/GUP
b = tf.range(4)
print(b)
print(b.device)
print(b.numpy())
print(b.ndim) # 维度
print(tf.rank(b)) # 维度（Tensor变量表示）
print(tf.ones([3,4,2])) # 3维数组
print(tf.ones([3,4,2]).ndim, tf.rank(tf.ones([3,4,2]))) # 维度

# In[]:
a = tf.constant([1., 2.]) # 标量
b = tf.constant([True, False])
c = tf.constant('hello world')
d = np.arange(4)
print(isinstance(a, tf.Tensor)) # 不推荐使用
print(tf.is_tensor(b)) # 数据类型判断
print(tf.is_tensor(d))
print(a.dtype, b.dtype, c.dtype, d.dtype)
print(a.dtype == tf.float32, c.dtype == tf.string)

# In[]:
a = np.arange(5)
print(a.dtype)

aa = tf.convert_to_tensor(a, dtype=tf.int32)
print(aa.dtype, tf.is_tensor(aa))

# In[]:
# 数据类型转换
b = tf.cast(aa, dtype=tf.float32)
aaa = tf.cast(aa, dtype=tf.double)
print(aaa)
print(tf.cast(aaa, dtype=tf.int32))

b = tf.constant([0,1])
bb = tf.cast(b, dtype=tf.bool)
print(bb)
print(tf.cast(bb, dtype=tf.int32))

# In[]:
# 梯度求导变量： tf.Variable
a = tf.range(5) # Tensor 转 Variable
b = tf.Variable(a, name="input_data")
print(b.dtype, b.name, b.trainable) # 自动记录梯度信息

print(isinstance(b, tf.Tensor)) # 错误判断，不推荐使用isinstance
print(isinstance(b, tf.Variable))
print(tf.is_tensor(b))

c = np.arange(5) # Numpy 转 Variable
d = tf.Variable(c, name="input_data")
print(d.dtype, d.name, d.trainable) # 自动记录梯度信息

print(isinstance(d, tf.Tensor)) # 错误判断，不推荐使用isinstance
print(isinstance(d, tf.Variable))
print(tf.is_tensor(d))

print(b.numpy())
print(d.numpy())

a = tf.ones([])
print(a, a.numpy())
print(int(a), float(a)) # 直接转换，a必须为标量

# In[]:
print(tf.convert_to_tensor(np.ones([2,3,3])))
print(tf.convert_to_tensor([1, 2.])) # 数据类型自动升级为float32
print(tf.convert_to_tensor([[1], [2.]]))

print("-"*30)

print(tf.zeros([]))
print(tf.zeros([1]))
print(tf.zeros([1,2]))
print(tf.zeros([2,3,3]))

# In[]:
a = tf.zeros([2,3,3])
print(tf.zeros_like(a)) # 相同
print(tf.zeros(a.shape))

print("-"*30)

print(tf.ones(1))
print(tf.ones([]))
print(tf.ones([2]))
print(tf.ones([2,3]))
print(tf.ones_like(a))

# In[]:
print(tf.fill([2,2], 0))
print(tf.fill([2,2], 9.0))

# In[]:
print(tf.random.normal([2,2], mean=1, stddev=1)) # 正太分布
print(tf.random.truncated_normal([2,2], mean=0, stddev=1)) # 截断正太分布

print(tf.random.uniform([3,4], minval=0, maxval=1, dtype=tf.float32)) # 均匀分布

# In[]:
idx = tf.range(10)
idx = tf.random.shuffle(idx)
a = tf.random.normal([10, 784], mean=0, stddev=1)
b = tf.random.uniform([10], maxval=10, dtype=tf.int32)

# tf.gather：用一个一维的索引数组，将张量中对应索引的向量提取出来
a = tf.gather(a, idx)
b = tf.gather(b, idx)
print(a)
print(b)

# In[]:
# tf.convert_to_tensor 和 tf.constant 功能是重合的
a = tf.constant(1)
print(a)
b = tf.constant([1])
print(b)
c = tf.constant([[1,2,3],[4,5,6]])
print(c)

print("-"*30)

print(tf.convert_to_tensor(1))
print(tf.convert_to_tensor([1])) # 数据类型自动升级为float32
print(tf.convert_to_tensor([[1,2,3],[4,5,6]]))

# In[]:
out = tf.random.uniform([2, 3], seed=1)
print(out)
y = tf.range(2)
y = tf.one_hot(y, depth=3)
print(y)
loss = tf.keras.losses.mse(y, out) # mse是均方差； rmse才是均方根。
print(loss)
loss = tf.reduce_mean(loss) # 求平均
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
loss = tf.reduce_sum(tf.square(y - out), axis=1) / y.shape[1]
loss = tf.reduce_sum(loss) / y.shape[0] 
print(loss)

# In[]:
'''
#这里有坑：
#1、layers.Dense(10)是指输出的列个数为10，由于输入x是4:8的，又由于使用的是x * w = x(4:8) * w(?:10)，那么w的行个数就为8。
#2、bias就是偏移量b，维度为10。
'''
net = layers.Dense(10)
net.build((4, 8)) # 输入shape
print(net.kernel)
print(net.bias)

# In[]:
x = tf.random.normal([4, 784])

net = layers.Dense(10) # 全连接层
net.build((4, 784)) # 输入

print(net(x).shape) # 输出
print(net.kernel.shape) # 隐藏层
print(net.bias.shape) # 偏移项

# In[]:
#(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=10000)
#x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80) # 每个评论80个单词
#print(x_train.shape)
#
#emb = embedding(x_train)
#print(emb.shape)
#
#out = rnn(emb[:4])
#print(out.shape)






