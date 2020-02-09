import tensorflow as tf

a = tf.random.normal([4, 28, 28, 3])
a.shape   # TensorShape([4, 28, 28, 3])
a.ndim   # 4 (int)

# In[]:
a = tf.random.normal([4, 28, 28, 3])
a.shape   # TensorShape([4, 28, 28, 3])
a.ndim   # 4

tf.reshape(a, [4, 28*28, 3]).shape   #  TensorShape([4, 784, 3])
tf.reshape(a, [4, -1, 3]).shape   # TensorShape([4, 784, 3])	

tf.reshape(a, [4, 784*3]).shape   # TensorShape([4, 2352])
tf.reshape(a, [4, -1]).shape   # TensorShape([4, 2352])

# Reshape is flexible
tf.reshape(tf.reshape(a, [4, -1]),[4, 28, 28, 3]).shape   # TensorShape([4, 28, 28, 3])
tf.reshape(tf.reshape(a, [4, -1]),[4, -1, 3]).shape   # TensorShape([4, 784, 3])	

# In[]:
# tf.transpose 转置
a = tf.random.normal([4, 3, 2, 1])
a.shape   # TensorShape([4, 3, 2, 1])
tf.transpose(a).shape   # TensorShape([1, 2, 3, 4])

a = tf.random.normal([4, 28, 28, 3]) 
a.shape   # TensorShape([4, 28, 28, 3])
tf.transpose(a, perm=[0,3,1,2]).shape # TensorShape([4, 3, 28, 28]) perm原始轴索引的新顺序
# tensorflow tensor to pytorch tensor

# In[]:
# tf.expand_dim 增加维度
a = tf.ones([4,35,10])
a.shape   # TensorShape([4, 35, 10])
tf.expand_dims(a, axis=0).shape  # front   TensorShape([1, 4, 35, 10])
tf.expand_dims(a, axis=-1).shape  # behind   TensorShape([4, 35, 10, 1])
tf.expand_dims(a, axis=-2).shape  # behind   TensorShape([4, 35, 1, 10])

# In[]:
# tf.squeeze
a = tf.ones([1, 4, 1, 35, 1, 10])
print(a.shape)
tf.squeeze(a).shape   # # TensorShape([4, 35, 10])


print(tf.squeeze(tf.zeros([1,2,1,1,3])).shape) # 全部删除

a = tf.zeros([1,2,1,3])
print(tf.squeeze(a, axis=0).shape) # 指定0轴删除
print(tf.squeeze(a, axis=2).shape) # 指定2轴删除
print(tf.squeeze(a, axis=-2).shape) # 指定2轴删除
print(tf.squeeze(a, axis=-4).shape) # 指定0轴删除











