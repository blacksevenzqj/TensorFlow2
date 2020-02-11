import tensorflow as tf

a = tf.ones([2,2])

# L2范数
b = tf.norm(a) # 默认L2范数
print(b)
b = tf.sqrt(tf.reduce_sum(tf.square(a)))
print(b)

a = tf.ones([4,28,28,3])
b = tf.norm(a)
print(b)
b = tf.sqrt(tf.reduce_sum(tf.square(a)))
print(b)

# In[]:
a = tf.ones([2,2])

b = tf.norm(a) # 默认L2范数
print(b)
b = tf.norm(a, ord=2, axis=0) # 行向：从上到下
print(b)
b = tf.norm(a, ord=2, axis=1) # 列向：从左到右
print(b)

c = tf.norm(a, ord=1) # L1范数
print(c)
c = tf.norm(a, ord=1, axis=0) # 行向：从上到下
print(c)
c = tf.norm(a, ord=1, axis=1) # 列向：从左到右
print(c)

# In[]:
a = tf.random.normal([4,10])
print(tf.reduce_min(a), tf.reduce_max(a), tf.reduce_mean(a))
print(tf.reduce_min(a, axis=1), tf.reduce_max(a, axis=1), tf.reduce_mean(a, axis=1))

# In[]:
# 最大值索引
a = tf.convert_to_tensor([[1,3,5,2,9], [8,6,3,9,1]])
b = tf.argmax(a, axis=1) # 列向：从左到右，每一行的最大值索引：[4 3]代表每一行的行索引
print(b, b.shape)
b = tf.argmax(a, axis=0) # 行向：从上到下，每一列的最大值索引：[1 1 0 1 0]代表每列比较，第几行是最大的
print(b, b.shape)

# In[]:
a = tf.constant([1,2,3,2,4])
b = tf.range(5)
print(tf.equal(a,b)) # 一样
a = tf.convert_to_tensor([1,2,3,2,4]) # 因为 tf.constant 和 tf.convert_to_tensor 是相同的功能
b = tf.range(5)
print(tf.equal(a,b)) # 一样

res = tf.equal(a, b) # tf.Tensor([False False False False  True])
# tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，转换之后就变为0/1了
print(tf.reduce_sum(tf.cast(res, dtype=tf.int32)))

# In[]:
# 每一行为一个样本，每一行的行索引号即为 所选类别
a = tf.convert_to_tensor([[0.1,0.2,0.7], [0.9,0.05,0.05]]) 
pred = tf.cast(tf.argmax(a, axis=1), dtype=tf.int32)
print(pred) # tf.Tensor([2 0]) 即为索引
y = tf.convert_to_tensor([[2,0]])
z = tf.equal(pred, y)
print(z) # tf.Tensor([[ True  True]])

correct = tf.reduce_sum(tf.cast(z, dtype=tf.int32))
print(correct) # tf.Tensor(2)
print(correct / a.shape[0]) # tf.Tensor(1.0)

# In[]:
a = tf.range(5)
print(tf.unique(a)) #不重复的数据

a = tf.constant([4,2,4,3,3])
# 返回两个部分：
# 1、去重后的值的列表：[4, 2, 3]
# 2、去重后的值的列表：[4, 2, 3] 对应到 原列表[4,2,4,3,3]的值， 在原列表相同值上 使用 新列表的索引0,1,2填充， 形成原列表的新索引。     
print(tf.unique(a))
