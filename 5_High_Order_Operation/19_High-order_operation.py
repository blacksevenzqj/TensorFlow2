import tensorflow as tf

# 一、where
a = tf.random.normal([3,3], seed=1)

mask = a > 0
print(mask)

# 方式1：
b = tf.boolean_mask(a, mask)
print(b)

# 方式2：
indices = tf.where(mask) # 返回索引
c = tf.gather_nd(a, indices)
print(c)

print("--------------------------------------------------------------------")

print(mask)
A = tf.convert_to_tensor([[1,2,3],[4,5,6],[7,8,9]])
B = tf.convert_to_tensor([[11,22,33],[44,55,66],[77,88,99]])
print(tf.where(mask, A, B)) # 按照mask中的True/False 选择A/B中元素。True选择A，False选择B


print("====================================================================")


# 二、scatter_nd
indices = tf.constant([[4],[3],[1],[7]])
updates = tf.constant([9,10,11,12])
shape = tf.constant([8])
a = tf.scatter_nd(indices, updates, shape)
print(a)

print("--------------------------------------------------------------------")

indices = tf.constant([[0], [2]])
updates = tf.constant([[[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]], [[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]]])
print(updates.shape)
shape = tf.constant([4,4,4])
res = tf.scatter_nd(indices, updates, shape)
print(res)


print("====================================================================")


# 三、meshgrid






