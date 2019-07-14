import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


'''
工业化的 通用 保存方式，Python开发保存后，生产上使用C++部署
'''
tf.saved_model.save(m, '/tmp/saved_model/')
imported = tf.saved_model.load(path)
f = imported.signatures["serving_default"]
print(f(x=tf.ones([1, 28, 28, 3])))