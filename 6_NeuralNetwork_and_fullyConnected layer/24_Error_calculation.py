import tensorflow as tf

# 最后一层没有经过激活函数的数据称为Logit
y = tf.constant([1, 2, 3, 0, 2])
y = tf.one_hot(y, depth=4)
y = tf.cast(y, dtype=tf.float32)

out = tf.random.normal([5, 4])

loss1 = tf.reduce_mean(tf.square(y-out)) # 直接求2次平均（最小二乘）
loss2 = tf.square(tf.norm(y-out))/(5*4) # L2范数开平方 / 5*4（5*4是矩阵元素个数）
# MSE先在axis=1上求均值，后再使用reduce_mean在axis=0求均值。
loss3 = tf.reduce_mean(tf.losses.MSE(y, out)) # VS MeanSquaredError is a class；

print(loss1)
print(loss2)
print(loss3)


print("====================================================================")


# 交叉熵
a = tf.fill([4], 0.25) # 数据越平均 不确定性越大 熵越大
j = a * tf.math.log(a) / tf.math.log(2.) # 普通乘法，不是矩阵乘法
print(j)
print(-tf.reduce_sum(j)) # -∑p*logp

a = tf.constant([0.1,0.1,0.1,0.7]) # 数据越不平均 确定性越大 熵越小
print(tf.math.log(a) / tf.math.log(2.))
j = a * tf.math.log(a) / tf.math.log(2.)
print(j)
print(-tf.reduce_sum(j)) # -∑p*logp

a = tf.constant([0.01,0.01,0.01,0.97]) # 数据越不平均 确定性越大 熵越小
j = a * tf.math.log(a) / tf.math.log(2.)
print(-tf.reduce_sum(j)) # -∑p*logp

print("--------------------------------------------------------------------")

# 使用tf的方法计算交叉熵：
# 交叉熵需要求和：1、所有样本 的 所有类别；2、最后再求平均
# 交叉熵的输入：必须是概率形式，求和等于1
j = tf.losses.categorical_crossentropy([0,1,0,0], [0.25,0.25,0.25,0.25])
print(j)
j = tf.losses.categorical_crossentropy([0,1,0,0], [0.01,0.97,0.01,0.01])
print(j)
j = tf.losses.BinaryCrossentropy()([1],[0.1])
print(j)
j = tf.losses.binary_crossentropy([1],[0.1])
print(j)