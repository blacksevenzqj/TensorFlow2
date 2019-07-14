import tensorflow as tf

x = tf.random.normal([2, 4]) # 2个样本，4个特征
w = tf.random.normal([4, 3]) # 4个特征，3个分类
b = tf.zeros([3]) # 3个分类的偏移
y = tf.constant([2, 0]) # 2个样本标签

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.nn.softmax(x@w+b, axis=1)
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))

grads = tape.gradient(loss, [w,b])
print(grads[0]) # w 的梯度
print(grads[1]) # b 的梯度


print("====================================================================")


x = tf.random.normal([2, 4], seed=1, mean=0, stddev=1) # 2个样本，4个特征
w = tf.random.normal([4, 3], seed=1, mean=0, stddev=1) # 4个特征，3个分类
b = tf.zeros([3]) # 3个分类的偏移
y = tf.constant([2, 0]) # 2个样本标签

with tf.GradientTape() as tape:
    tape.watch([w, b])
    logits = x@w + b
    print("logits is", logits)
    res = tf.nn.softmax(logits, axis=1)
    print("res is", res)
    print("res sum is", tf.reduce_sum(res, axis=1))
    # 交叉熵需要求和：1、所有样本 的 所有类别；2、最后再求平均
    # 交叉熵的输入：必须是概率形式，求和等于1
    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y, depth=3), res))

    # logits直接在categorical_crossentropy函数内部自动计算softmax之后再求交叉熵，必须设置from_logits=True，结果同上。
    # loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y, depth=3), logits, from_logits=True))
    print("loss is ", loss)

grades = tape.gradient(loss, [w,b])
print(grades[0])
print(grades[1])

