import tensorflow as tf

a = tf.random.normal([4, 32, 32, 3])
b = tf.ones([3])
c = tf.fill([32, 32, 1], 2.)
d = tf.random.uniform([4, 1, 1, 1])

print((a+b).shape)   # (4, 32, 32, 3)
print((a+c).shape)   # (4, 32, 32, 3)
print((a+d).shape)   # (4, 32, 32, 3)

# 注意： [4, 32, 14, 14] 与 [2, 32, 14, 14] 是无法传播的，除非将2换为1。

# In[]:
# 显示指定 广播维度：
x = tf.ones([4, 32, 32, 3])
y = tf.ones([1, 32, 1 ])

(x+y).shape   # (4, 32, 32, 3) 自动广播
tf.broadcast_to(y, x.shape).shape   # (4, 32, 32, 3) 显示广播
tf.ones_like(x)   # (4, 32, 32, 3)

# In[]:
# Broadcast VS Tile 
# broadcasting 内存无关， 而tile内存相关（tf.tile的multiple参数是扩展倍数）
a = tf.ones([3,4])
a1 = tf.broadcast_to(a, [2,3,4])
a1.shape   # (2, 3, 4)

a2 = tf.expand_dims(a, axis=0) # 在最前面新增0轴： 1维
a2 = tf.tile(a2, [2,1,1])  # 每一维的倍数： 1维*2倍, 3维*1倍, 4维*1倍
a2.shape   # (2, 3, 4)
