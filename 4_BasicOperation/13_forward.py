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
print('datasets:', x.shape, y.shape, type(x), x.dtype, type(y), y.dtype)
# In[]:
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)
print('tensor:', x.shape, y.shape, type(x), x.dtype, type(y), y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# In[]:
# 创建数据集： 
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)
sample1 = next(train_iter)
print('batch1:', sample1[0].shape, sample1[1].shape)

# In[]:
# 输入x：[b, 784] => [b, 256] => [b, 128] => [b, 10]
# w1:[784, 256] => w2:[256, 128] => w3:[128, 10]
# x@w1=[b, 256] => x@w2=[b, 128] => x@w3=[b, 10]
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

        # 自动计算梯度
        with tf.GradientTape() as type:
            # h1 = x@w1+b1   [128:784] @ [784:256] + [256]
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256]) # h1 = x@w1 + b1 其会自动广播，这里只是为了演示
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3

            # compute loss
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)

            grads = type.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 数学公式： w1 = w1 - lr * w1_grad 
        # 注意： tf.Variable 直接进行 +-*/ 运算后会转换为 tf.tensor类型，再计算梯度就报错。
        # 所以为了不改变tf.Variable的类型，需要使用 原地更新： assign_sub原地减函数（变量地址不变）
        # w1 = w1 - lr * grads[0] 其中=号右边的w1是原始变量，而经过计算后得到=号左边w1是新变量。
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))
