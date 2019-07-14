import tensorflow as tf
from 	tensorflow import keras


x = tf.random.normal([2, 3])

model = keras.Sequential([
		keras.layers.Dense(2, activation='relu'), # w1维度：[3,2]，param：3*2+2（w的元素个数+b的元素个数）
		keras.layers.Dense(2, activation='relu'), # w2维度：[2,2],param：2*2+2（w的元素个数+b的元素个数）
		keras.layers.Dense(2) # w3维度：[2,2],param：2*2+2（w的元素个数+b的元素个数）
	])
model.build(input_shape=[None, 3]) # [None, 3]输入x的维度：None代表多个样本，3表示x的特征维度
model.summary()

for p in model.trainable_variables:
	print(p.name, p.shape)

