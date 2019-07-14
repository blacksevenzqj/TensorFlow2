import tensorflow as tf

a = tf.ones([4,35,8])
b = tf.ones([2,35,8])
c = tf.concat([a,b], axis=0) # 1维和2维必须相等
print(c.shape)

a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
c = tf.concat([a,b], axis=1) # 35这个轴1维：0维2维必须相等
print(c.shape)
d = tf.concat([a,b], axis=2) # 8这个轴2维：0维1维必须相等
print(d.shape)

print("====================================================================")

'''
School1:[classes, students, scores]
School2:[classes, students, scores]
[schools, classes, students, scores]
'''
# stack 所有维度都必须相等
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
c = tf.stack([a,b], axis=0)
print(c.shape)
d = tf.stack([a,b], axis=3)
print(d.shape)

print("====================================================================")

# unstack的 “分解维度” 只能是1维
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
c = tf.stack([a,b], axis=0) # (2, 4, 35, 8)
print(c.shape)

aa, bb = tf.unstack(c, axis=0) # (2, 4, 35, 8)分解0轴：维度为2，所以有分解为2个tensor (4, 35, 8)
print(aa.shape, bb.shape)

res = tf.unstack(c, axis=1) # (2, 4, 35, 8)分解1轴：维度为4，所以有分解为4个tensor (2, 35, 8)
print(res[0].shape, res[1].shape, res[3].shape, len(res))

res = tf.unstack(c, axis=2) # (2, 4, 35, 8)分解2轴：维度为35，所以有分解为35个tensor (2, 4, 8)
print(res[0].shape, res[1].shape, res[34].shape,len(res))

res = tf.unstack(c, axis=3) # (2, 4, 35, 8)分解3轴：维度为8，所以有分解为8个tensor (2, 4, 35)
print(res[0].shape, res[1].shape, res[7].shape, len(res))

print("====================================================================")

a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
c = tf.stack([a,b], axis=0) # (2, 4, 35, 8)
print(c.shape)

res = tf.unstack(c, axis=3) # (2, 4, 35, 8)分解3轴：维度为8，所以有分解为8个tensor (2, 4, 35)
print(res[0].shape, res[1].shape, res[7].shape, len(res))
print("--------------------------------------------------------------------")
# split分解更灵活，可以指定“分解维度”
res = tf.split(c, axis=3, num_or_size_splits=2) # 分解3轴：由于num_or_size_splits=2 即3轴的8平均分为2份：(2, 4, 35, 4)
for i in range(0, len(res)):
    print(res[i].shape)
print("--------------------------------------------------------------------")
# 分解3轴：由于num_or_size_splits=[2,2,4] 即3轴的8分为3份：(2, 4, 35, 2) (2, 4, 35, 2) (2, 4, 35, 4)
res = tf.split(c, axis=3, num_or_size_splits=[2,2,4])
for i in range(0, len(res)):
    print(res[i].shape)

