import  os
import tensorflow as tf
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

(x, y), (x_val, y_val) = datasets.mnist.load_data() # 得到的是Numpy的数据格式，不是tensor的
# (x, y), (x_val, y_val) = datasets.cifar10.load_data() # 得到的是Numpy的数据格式，不是tensor的
print('trainDataSets:', x.shape, y.shape, type(x), type(y))
print('trainMin:', x.min(), 'trainMax:', x.max())
print('testDataSets', x_val.shape, y_val.shape)
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255 # 可以转换为tensor格式
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)

# Numpy格式转换为Dataset格式，可迭代、可批量、可多线程、可多数据加载、可shuffle打散
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128) # 可以接收tensor格式、也可以接收Numpy格式
print(type(train_db))
train_db = train_db.shuffle(10000) # 数据打散
train_iter = iter(train_db)
sample = next(train_iter)
print(sample[0].shape, sample[1].shape)


print("====================================================================")


# 数据预处理：.map
(x, y), (x_val, y_val) = datasets.mnist.load_data() # 得到的是Numpy的数据格式，不是tensor的
print(type(x_val), type(y_val))
train_db = tf.data.Dataset.from_tensor_slices((x_val,y_val)) # 可以接收tensor格式、也可以接收Numpy格式

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

db2 = train_db.map(preprocess)
res = next(iter(db2))
print(res[0].shape, res[1].shape)

print("--------------------------------------------------------------------")

db3 = db2.batch(32)
res = next(iter(db3))
print(res[0].shape, res[1].shape)

print("--------------------------------------------------------------------")

# StopIteration
'''
db_iter = iter(db3)
while True:
    next(db_iter)
如果这样写会报错，
'''

print("====================================================================")

# For example
def prepare_mnist_features_and_labels(x,y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def minist_dataset():
    (x, y), (x_val, y_val) = datasets.mnist.load_data() # 返回的是Numpy数据格式
    # (x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
    print(type(x), type(y))
    y = tf.one_hot(y, depth=10)
    y_val = tf.one_hot(y_val, depth=10)

    ds = tf.data.Dataset.from_tensor_slices((x, y)) # 可以接收tensor格式、也可以接收Numpy格式
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.shuffle(60000).batch(100)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(prepare_mnist_features_and_labels)
    ds_val = ds_val.shuffle(60000).batch(100)
    return ds, ds_val


if __name__ == '__main__':
    ds, ds_val = minist_dataset()
    res = next(iter(ds))
    print(res[0].shape, res[1].shape)

    res = next(iter(ds_val))
    print(res[0].shape, res[1].shape)








