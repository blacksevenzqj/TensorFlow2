import tensorflow as tf

# 单输出感知机（Sigmoid）：它的PPT中梯度求导写的很不清楚，没有清楚的写出链式求导过程
x = tf.random.normal([1,3], seed=1, mean=0, stddev=1)
w = tf.random.normal([3,1], seed=1) * tf.sqrt(1/3)
b = tf.zeros([1])
y = tf.constant([1])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    logits = tf.sigmoid(x@w + b)
    loss = tf.reduce_mean(tf.losses.MSE(y, logits))

grads = tape.gradient(loss, [w, b])
print(grads[0])
print(grads[1])


print("====================================================================")


# 多输出感知机（Softmax）：
x = tf.random.normal([2,4], seed=1, mean=0, stddev=1)
w = tf.random.normal([4,3], seed=1) * tf.sqrt(1/4)
b = tf.zeros([3])
y = tf.constant([2,0])

with tf.GradientTape() as tape:
    tape.watch([w,b])
    prob = tf.nn.softmax(x@w + b)
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))

grads = tape.gradient(loss, [w,b])
print(grads[0])
print(grads[1])
