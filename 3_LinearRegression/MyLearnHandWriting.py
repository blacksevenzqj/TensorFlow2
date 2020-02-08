import  os
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets

'''
    base_loging	屏蔽信息	             输出信息
“0”	INFO	  无	          INFO + WARNING + ERROR + FATAL
“1”	WARNING	 INFO	          WARNING + ERROR + FATAL
“2”	ERROR	 INFO + WARNING	       ERROR + FATAL
“3”	FATAL	 INFO + WARNING + ERROR	   FATAL
'''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, type(x), type(y))
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255 # tensor数据格式为了GPU运算
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print('datasets:', x.shape, y.shape, type(x), type(y))
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)
print(type(train_dataset))

'''
定义时 不进行build，不创建w和b
network.build(input_shape=(None, 28 * 28))
network.summary()
'''
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)



def train_epoch(epoch):
    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28*28))
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x) # 直接在运行时 模型中根据输入x的维度创建w和b
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())



def train(epoch_num=30):
    for epoch in range(epoch_num):
        train_epoch(epoch)



if __name__ == '__main__':
    train(1)