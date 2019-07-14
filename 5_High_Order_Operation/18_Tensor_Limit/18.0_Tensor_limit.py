import tensorflow as tf

# 一、clip_by_value
a = tf.convert_to_tensor([0,1,2,3,4,5,6,7,8,9])
b = tf.maximum(a, 2) # 小于2的 变为2
c = tf.minimum(a, 8) # 大于8的 变为8
print(b, c)
cc = tf.minimum(tf.maximum(a, 2), 8) # 等同于tf.clip_by_value函数
print(cc)

d = tf.clip_by_value(a, 2,8)
print(d)

print("====================================================================")

# 二、relu函数
a = a - 5
print(a)
print(tf.nn.relu(a))
print(tf.maximum(a, 0)) # 和relu函数效果相同

print("====================================================================")

# 三、clip_by_norm
a = tf.random.normal([2,2], mean=10, seed=1)
print(a)
print(tf.norm(a)) # L2范数
# a = tf.convert_to_tensor([[9.188682, 11.484599],
#                          [10.06533, 7.557296]])
# print(tf.sqrt(tf.reduce_sum(tf.square(a))))

# clip_by_norm
'''
1、平面外任意一点x - 平面上任意一点x' = a向量
2、w / ||w||  =  w法向量 的 单位长度。
3、w / ||w|| · a向量 =  a向量 在 单位法向量 上的投影的矢量长度。即：平面外任意一点x 到平面的 垂直 距离。
注意，皮尔森相似度转换为 余弦相似度 公式为：△ai = ai - μ, △bi = bi - v
△a·△b  /  ||△a||·||△b||
4、而 clip_by_norm 就是 w / ||w|| · 标量x（注意，不是点乘向量）：
向量单位化后（值在0-1范围内），点乘 标量x 进行 等比例放缩（值得范围限制在 标量x之内）
'''
aa = tf.clip_by_norm(a, 15)
print(aa)
print((a / tf.norm(a)) * 15) # w / ||w|| · 标量x

print(tf.norm(aa)) # a -> aa 这个过程向量保持了方向不变，只改变了向量模。 L2范数为 标量x 的限定范围之内 = 15
print(tf.sqrt(tf.reduce_sum(tf.square(aa))))

print("====================================================================")

# 四、Gradient clipping 梯度渐变

