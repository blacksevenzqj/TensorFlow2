import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('trainDatasets:', x.shape, y.shape, x.min(), x.max()) # (60000, 28, 28) (60000,) 0 255
print('testDatasets:', x_val.shape, y_val.shape, x_val.min(), x_val.max()) # (10000, 28, 28) (10000,) 0 255

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz) # map(preprocess) 将y转了one-hot
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz) # map(preprocess) 将y转了one-hot

sample = next(iter(db))
print(sample[0].shape, sample[1].shape) # (128, 784) (128, 10) 只是一个batch

sampleTest = next(iter(ds_val))
print(sampleTest[0].shape, sampleTest[1].shape)


network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
network.summary()

# from_logits=True：函数内部自动使用了Softmax函数
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

# 1.1、训练中进行test：
'''
epochs=5：代替了之前的 repeat(5)函数。
validation_freq=2：每2次 epoch（db）训练，进行一次test测试。格式如：
Epoch 2/5
469/469 [==============================] - 3s 7ms/step - loss: 0.1356 - accuracy: 0.9583 - val_loss: 0.1161 - val_accuracy: 0.9671
训练db共60000样本 / 128batch ≈ 468.75，在Epoch第2次时进行一次训练中的test。
'''
network.fit(db, epochs=5, validation_data=ds_val, validation_freq=2)

print("--------------------------------------------------------------------")

# 1.2、训练外进行test：
'''
训练完成之后，进行一次test。格式如：
79/79 [==============================] - 0s 6ms/step - loss: 0.1252 - accuracy: 0.9698
测试样本共10000样本 / 128batch ≈ 78.125
'''
network.evaluate(ds_val) # 在训练之外的测试


print("====================================================================")


# 2、最后的预测：本应该用全新的测试集，但是还是用之前的ds_val
sample = next(iter(ds_val)) # 就是一个batch的ds_val
x = sample[0]
y = sample[1]  # 数据加载时的 map(preprocess) 将y转了one-hot
print(x.shape, y.shape) # (128, 784) (128, 10)
pred = network.predict(x)  # [b, 10]
# convert back to number
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)

print(pred)
print(y)