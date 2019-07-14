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

print("====================================================================")

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

print("====================================================================")

a = tf.random.normal([4,10])
print(tf.reduce_min(a), tf.reduce_max(a), tf.reduce_mean(a))
print(tf.reduce_min(a, axis=1), tf.reduce_max(a, axis=1), tf.reduce_mean(a, axis=1))

print("====================================================================")

a = tf.convert_to_tensor([[1,3,5,2,9], [8,6,3,9,1]])
b = tf.argmax(a, axis=1) # 列向：从左到右，每一行的最大值索引：[4 3]代表每一行的行索引
print(b, b.shape)
b = tf.argmax(a, axis=0) # 行向：从上到下，每一列的最大值索引：[1 1 0 1 0]代表每列比较，第几行是最大的
print(b, b.shape)

print("====================================================================")

a = tf.constant([1,2,3,2,4])
b = tf.range(5)
print(tf.equal(a,b))
a = tf.convert_to_tensor([1,2,3,2,4])
b = tf.range(5)
print(tf.equal(a,b)) # 一样

res = tf.equal(a, b)
print(tf.reduce_sum(tf.cast(res, dtype=tf.int32))) # tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换

print("--------------------------------------------------------------------")

a = tf.convert_to_tensor([[0.1,0.2,0.7], [0.9,0.05,0.05]]) # 每一行为一个样本，每一行的行索引号即为 所选类别
pred = tf.cast(tf.argmax(a, axis=1), dtype=tf.int32)
print(pred)
y = tf.convert_to_tensor([[2,0]])
z = tf.equal(pred, y)
print(z)

correct = tf.reduce_sum(tf.cast(z, dtype=tf.int32))
print(correct)
print(correct / a.shape[0])



print("====================================================================")

a = tf.range(5)
print(tf.unique(a))

a = tf.constant([4,2,4,3,3])
# 返回两个部分：
# 1、去重后的值的列表：[4, 2, 3]
# 2、去重后的值的列表：[4, 2, 3] 对应到 原列表[4,2,4,3,3]的值，按照新列表的索引 以原列表的长度 取索引。
print(tf.unique(a))
