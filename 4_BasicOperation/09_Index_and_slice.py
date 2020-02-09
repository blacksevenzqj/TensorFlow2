import tensorflow as tf

a = tf.ones([1,5,5,3])
print(a)
print(a[0])
print(a[0][0])
print(a[0][0][0])
print(a[0][0][0][0])

# In[]:
a = tf.convert_to_tensor([2,5,6,8,9,10])
# print(a)
# print(a[1:3])
# print(a[3::-1])

# In[]:
b = tf.random.normal([4, 28, 28, 3])
print(b[1].shape)   # (28, 28, 3)
print(b[1, 2].shape)   # (28, 3)
print(b[1, 2, 3].shape)   # (3,)
print(b[1, 2, 3, 2].shape)   # ()

# In[]:
c = tf.range(10)
print(c[-1:])   # tf.Tensor([9], shape=(1,), dtype=int32)
print(c[-2:])   # tf.Tensor([8 9], shape=(2,), dtype=int32)
print(c[:2])   # tf.Tensor([0 1], shape=(2,), dtype=int32)
print(c[:-1])   # tf.Tensor([0 1 2 3 4 5 6 7 8], shape=(9,), dtype=int32)

# In[]:
d = tf.random.normal([4, 28, 28, 3])
print(d[0].shape)   # (28, 28, 3)
print(d[0,:,:,:].shape)    # (28, 28, 3)
print(d[0,1,:,:].shape)   # (28, 3)
print(d[:,:,:,0].shape)   # (4, 28, 28)
print(d[:,:,:,2].shape)   # (4, 28, 28)
print(d[:,0,:,0].shape)   # (4, 28)

print("-"*30)

print(d.shape)   # (4, 28, 28, 3)
print(d[0:2,:,:,:].shape)   # (2, 28, 28, 3)
print(d[:,14:,14:,:].shape)   # (4, 14, 14, 3)
print(d[:,0:28:2,0:28:2,:].shape)   # (4, 14, 14, 3)
print(d[:,::2,::2,:].shape)   # (4, 14, 14, 3)

print("-"*30)

e = tf.range(10)
print(e) # tf.Tprint(ensor([0 1 2 3 4 5 6 7 8 9], shapprint(e=(10,), dtypprint(e=int32)
print(e[::-1]) # tf.Tprint(ensor([9 8 7 6 5 4 3 2 1 0], shapprint(e=(10,), dtypprint(e=int32)
print(e[::-2]) # tf.Tprint(ensor([9 7 5 3 1], shapprint(e=(5,), dtypprint(e=int32)
print(e[2::-2]) # 从正向的下标2开始倒序检索 tf.Tprint(ensor([2 0], shapprint(e=(2,), dtypprint(e=int32)

# In[]:
f = tf.random.normal([2, 4, 28, 28, 3])
print(f[0].shape)   # (4, 28, 28, 3)
print(f[0,:,:,:,:].shape)   # (4, 28, 28, 3)
print(f[0,...].shape)   # (4, 28, 28, 3)

print(f[:,:,:,:,0].shape)   # (2, 4, 28, 28)
print(f[...,0].shape)   # (2, 4, 28, 28)

print(f[0,...,0].shape)   # (4, 28, 28)
print(f[0,2,...,0].shape)   # (28, 28)

# In[]:
'''
索引采样
tf.gather函数：默认axis=0；indices关键字可以省略，直接给出列表
'''
a = tf.random.normal([4,6,5], mean=0, stddev=1, seed=1)
# print(a)

# 1、单轴操作
print(tf.gather(a, axis=0, indices=[2,3])) # axis=0 采集的就是第一个维度
print(tf.gather(a, axis=1, indices=[3,5])) # axis=1 采集的就是第二个维度
print(tf.gather(a, axis=2, indices=[2,4])) # axis=2 采集的就是第三个维度

idx = tf.range(4)
print(idx)
a1 = tf.gather(a, idx[:4]) # 默认axis=0
a2 = tf.gather(a, axis=0, indices=idx[:4])
print(tf.equal(a1, a2)) # a1 等于 a2

# In[]:
# 2.1、多轴操作1：一个索引代表一个维度
a = tf.random.normal([4,6,5], mean=0, stddev=1, seed=1)

b1 = tf.gather_nd(a, [0]) # [0轴]
#print(b1)
b2 = tf.gather_nd(a, [0,1]) # [0轴1轴]
#print(b2)
# b3 等同于 b4
b3 = tf.gather_nd(a, [0,1,2]) # [0轴1轴2轴]：标量
#print(b3)
b4 = tf.gather_nd(a, [[0,1,2]]) # [[0轴1轴2轴]]：一维向量（元素是标量）
#print(b4)

print("-"*30)

# 2.2、多轴操作2：
c1 = tf.gather_nd(a, [[0,0],[1,1]]) # [[0轴1轴] [0轴1轴]]
print(c1)
c2 = tf.gather_nd(a, [[0,0],[1,1],[2,2]]) # [[0轴1轴] [0轴1轴] [0轴1轴]]
#print(c2)
# c3 等同于 c4
c3 = tf.gather_nd(a, [[0,0,0],[1,1,1],[2,2,2]]) # [[0轴1轴2轴] [0轴1轴2轴] [0轴1轴2轴]]：一维向量
#print(c3)
c4 = tf.gather_nd(a, [[[0,0,0],[1,1,1],[2,2,2]]]) # [[[0轴1轴2轴] [0轴1轴2轴] [0轴1轴2轴]]]：二维向量
#print(c4)

# In[]:
# 
w = tf.random.normal([4,6,6,3], mean=0, stddev=1, seed=1)
print(w.shape) # (4, 6, 6, 3)

# mask一维： 默认操作0轴
d1 = tf.boolean_mask(w, mask=[True, True, False, False]) # 默认操作0轴： [0轴真 0轴真 0轴假 0轴假] 相当于取0轴的0和1索引
print(d1.shape) # (2, 6, 6, 3)

# 选中 3轴 作为操作轴，其中 3轴 有3列： 0、1列为真，2列为假，所以3轴的维度是2
# 明确axis=3指定轴， mask一维操作相应轴
d2 = tf.boolean_mask(w, mask=[True, True, False], axis=3)
print(d2.shape) # (4, 6, 6, 2)

# mask二维： 操作1轴，1轴有3列，所以向量长度为3
e = tf.random.normal([2,3,4], mean=0, stddev=1, seed=1)
print(e.shape)
e1 = tf.boolean_mask(e, mask=[[True, False, False],[False, True, True]]) # [[1轴真 1轴假 1轴假], [1轴假 1轴真 1轴真]]
print(e1.shape)









