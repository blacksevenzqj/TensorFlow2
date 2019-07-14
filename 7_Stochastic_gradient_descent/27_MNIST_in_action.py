import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

'''
    base_loging	屏蔽信息	             输出信息
“0”	INFO	  无	          INFO + WARNING + ERROR + FATAL
“1”	WARNING	 INFO	          WARNING + ERROR + FATAL
“2”	ERROR	 INFO + WARNING	       ERROR + FATAL
“3”	FATAL	 INFO + WARNING + ERROR	   FATAL
'''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x,y

# (x, y), (x_test, y_test) = datasets.fashion_mnist.load_data() # 下载不了
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print(x.shape, y.shape)

batchsz = 128

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz) # 把数据进行10000个为单位的乱序，生成批次为128的batch

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz) # 测试数据不用打乱顺序

db_iter = iter(db)
sample = next(db_iter)
print('batch:', sample[0].shape, sample[1].shape)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu), # [b, 784] => [b, 256]：w1 = [784, 256]
    layers.Dense(128, activation=tf.nn.relu), # [b, 256] => [b, 128]：w2 = [256, 128]
    layers.Dense(64, activation=tf.nn.relu), # [b, 128] => [b, 64]：w3 = [128, 64]
    layers.Dense(32, activation=tf.nn.relu), # [b, 64] => [b, 32]：w4 = [64, 32]
    layers.Dense(10) # [b, 32] => [b, 10]：w5 = [32, 10]. parameter：w5(32*10) + b5(10) = 330
])
model.build(input_shape=[None, 28*28])
model.summary()
# w = w - lr*grad
optimizer = optimizers.Adam(lr=1e-3)

'''
def main():
    for epoch in range(2):  # 30次epoch
        for step, (x, y) in enumerate(db): # 两层循环，step当然会重新计数。对比使用了repeat函数的区别
            print(step, x.shape, y.shape)
'''
def main():
    for epoch in range(3): # 30次epoch
        for step, (x,y) in enumerate(db):
            x = tf.reshape(x, [-1, 28*28])
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                # from_logits=True：函数内部自动求Softmax，再求交叉熵。
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss', float(loss_ce), float(loss_mse))

        # 每次epoch进行一次test测试
        total_correct = 0
        total_num = 0
        for x,y in db_test:
            x = tf.reshape(x, [-1, 28*28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)


if __name__ == '__main__':
    main()
