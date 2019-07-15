import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os


'''
注意：
1、Padding：p=(f−1)/2 不向下取整，结果是多少就是多少。
2、strides：strides>1 向下取整>。
'''

# Conv层：padding=VALID、SAME
# 一、strides=1
# 1.1、默认padding=VALID, strides=1：(n+2p−f)/s + 1 × (n+2p−f)/s + 1
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], activation=tf.nn.relu),
    # layers.Conv2D(64, kernel_size=[3, 3], padding="VALID", strides=1, activation=tf.nn.relu),
]
conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 32, 32, 3])
x = tf.random.normal([4, 32, 32, 3])
out = conv_net(x)
print(out.shape) # (4, 30, 30, 64)：(32+2*0-3)/1 + 1 = 30

print("--------------------------------------------------------------------")

# 1.2、padding=SAME：(n+2p−f)/s + 1 × (n+2p−f)/s + 1; p=(f−1)/2（SAME自动设置了，不向下取整）
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding="SAME", strides=1, activation=tf.nn.relu),
]
conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 32, 32, 3])
x = tf.random.normal([4, 32, 32, 3])
out = conv_net(x)
print(out.shape) # (4, 32, 32, 64)：先算p=(3-1)/2=1; 得 (32+2*1-3)/1 + 1 = 32


print("====================================================================")


# 二、strides>1 向下取整
# 2.1、padding=VALID, strides=2：(n+2p−f)/s + 1 × (n+2p−f)/s + 1;
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding="VALID", strides=2, activation=tf.nn.relu),
]
conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 32, 32, 3])
x = tf.random.normal([4, 32, 32, 3])
out = conv_net(x)
print(out.shape) # (4, 15, 15, 64)：(32+2*0-3)/2 + 1 = 15（29/2向下取整=14）

print("--------------------------------------------------------------------")

# 2.2、padding=SAME, strides=2：(n+2p−f)/s + 1 × (n+2p−f)/s + 1;  p=(f−1)/2（SAME自动设置了，不向下取整）
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding="SAME", strides=2, activation=tf.nn.relu),
]
conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 32, 32, 3])
x = tf.random.normal([4, 32, 32, 3])
out = conv_net(x)
print(out.shape) # (4, 16, 16, 64)：先算p=(3-1)/2=1; 得 (32+2*1-3)/2 + 1 = 16（31/2向下取整=15）



print("====================================================================")



# 池化层 padding=VALID、SAME
# 三、strides=1
# 3.1、padding=VALID, strides=1：(n+2p−f)/s + 1 × (n+2p−f)/s + 1
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu), # (4, 32, 32, 64)
    layers.MaxPool2D(pool_size=[2, 2], strides=1, padding='VALID')
]
conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 32, 32, 3])
x = tf.random.normal([4, 32, 32, 3])
out = conv_net(x)
print(out.shape) # (4, 31, 31, 64)：(32+2*0-2)/1 + 1 = 31

print("--------------------------------------------------------------------")

# 3.2、padding=SAME, strides=1：(n+2p−f)/s + 1 × (n+2p−f)/s + 1;  p=(f−1)/2（SAME自动设置了，不向下取整）
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu), # (4, 32, 32, 64)
    layers.MaxPool2D(pool_size=[2, 2], strides=1, padding='SAME')
]
conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 32, 32, 3])
x = tf.random.normal([4, 32, 32, 3])
out = conv_net(x)
print(out.shape) # (4, 32, 32, 64)：先算p=(2-1)/2=0.5（不向下取整）; 得 (32+2*0.5-2)/1 + 1 = 32


print("====================================================================")


# 四、strides>1 向下取整
# 4.1、padding=VALID, strides=2：(n+2p−f)/s + 1 × (n+2p−f)/s + 1
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu), # (4, 32, 32, 64)
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='VALID')
]
conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 32, 32, 3])
x = tf.random.normal([4, 32, 32, 3])
out = conv_net(x)
print(out.shape) # (4, 16, 16, 64)：(32+2*0-2)/2 + 1 = 16

print("--------------------------------------------------------------------")

# 4.2、padding=SAME, strides=2：(n+2p−f)/s + 1 × (n+2p−f)/s + 1;  p=(f−1)/2（SAME自动设置了，不向下取整）
conv_layers = [
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu), # (4, 32, 32, 64)
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME')
]
conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 32, 32, 3])
x = tf.random.normal([4, 32, 32, 3])
out = conv_net(x)
print(out.shape) # (4, 32, 32, 64)：先算p=(2-1)/2=0.5（不向下取整）; 得 (32+2*0.5-2)/2 + 1 = 16（31/2向下取整=15）