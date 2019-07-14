import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets
import  os

'''
    base_loging	屏蔽信息	             输出信息
“0”	INFO	  无	          INFO + WARNING + ERROR + FATAL
“1”	WARNING	 INFO	          WARNING + ERROR + FATAL
“2”	ERROR	 INFO + WARNING	       ERROR + FATAL
“3”	FATAL	 INFO + WARNING + ERROR	   FATAL
'''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape)
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)
print('tensor:', x.shape, y.shape)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# 输入x：[b, 784] => [b, 256] => [b, 128] => [b, 10]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

# 注意这个损失函数 是随意测试用的，softmax不应该是最小二乘法的。
for epoch in range(10): # iterate db for 10
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28*28])

        with tf.GradientTape() as type:
            # h1 = x@w1+b1   [128:784] @ [784:256] + [256]
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256]) # h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3

            # compute loss
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)

            grads = type.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))
