import tensorflow as tf

a = tf.ones([1,5,5,3])
# print(a)
# print(a[0])
# print(a[0][0])
# print(a[0][0][0])
# print(a[0][0][0][0])

print("====================================================================")

a = tf.convert_to_tensor([2,5,6,8,9,10])
# print(a)
# print(a[1:3])
# print(a[3::-1])

print("====================================================================")

'''
tf.gather函数：默认axis=0；indices关键字可以省略，直接给出列表
'''
a = tf.random.normal([4,6,5], mean=0, stddev=1, seed=1)
# print(a)

# 一、单轴操作
# print(tf.gather(a, axis=0, indices=[2,3]))
# print(tf.gather(a, axis=1, indices=[3,5]))
# print(tf.gather(a, axis=2, indices=[2,4]))

# a1 = tf.gather(a, axis=0, indices=[2,3])
# print(a1)
# a2 = tf.gather(a1, axis=1, indices=[1,3])
# print(a2)

idx = tf.range(4)
print(idx)
a1 = tf.gather(a, idx[:4]) # 默认
a2 = tf.gather(a, axis=0, indices=idx[:4])
# print(tf.equal(a1, a2)) # a1 等于 a2
print("--------------------------------------------------------------------")

# 二、多轴操作1：一个索引代表一个维度
b1 = tf.gather_nd(a, [0]) # [0轴]
# print(b1)
b2 = tf.gather_nd(a, [0,1]) # [0轴1轴]
# print(b2)
# b3 等同于 b4
b3 = tf.gather_nd(a, [0,1,2]) # [0轴1轴2轴]：标量
# print(b3)
b4 = tf.gather_nd(a, [[0,1,2]]) # [[0轴1轴2轴]]：一维向量
# print(b4)

# 多轴操作2：
c1 = tf.gather_nd(a, [[0,0],[1,1]]) # [[0轴1轴] [0轴1轴]]
# print(c1)
c2 = tf.gather_nd(a, [[0,0],[1,1],[2,2]]) # [[0轴1轴] [0轴1轴] [0轴1轴]]
# print(c2)
# c3 等同于 c4
c3 = tf.gather_nd(a, [[0,0,0],[1,1,1],[2,2,2]]) # [[0轴1轴2轴] [0轴1轴2轴] [0轴1轴2轴]]：一维向量
# print(c3)
c4 = tf.gather_nd(a, [[[0,0,0],[1,1,1],[2,2,2]]]) # [[[0轴1轴2轴] [0轴1轴2轴] [0轴1轴2轴]]]：二维向量
# print(c4)


print("====================================================================")


w = tf.random.normal([4,6,6,3], mean=0, stddev=1, seed=1)
# print(w)
d1 = tf.boolean_mask(w, mask=[True, True, False, False]) # [0轴真 0轴真 0轴假 0轴假]
# print(d1)
# 选中 3轴 作为操作轴，其中 3轴 有3列：0、1列为真，2列为假
d2 = tf.boolean_mask(w, mask=[True, True, False],axis=3)
# print(d2)


e = tf.random.normal([2,3,4], mean=0, stddev=1, seed=1)
# print(e)
e1 = tf.boolean_mask(e, mask=[[True, False, False],[False, True, True]]) # [[1轴真 1轴假 1轴假], [1轴假 1轴真 1轴真]]
# print(e1)









