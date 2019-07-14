import tensorflow as tf

a = tf.random.shuffle(tf.range(5))

b = tf.sort(a, direction='DESCENDING')
print(b)

idx = tf.argsort(a, direction='DESCENDING')
c = tf.gather(a, idx[0:2])
print(c)

print("====================================================================")

# 默认是按照最后一个维度进行排序（列维度）
a = tf.random.uniform([3,3], maxval=10, dtype=tf.int32)
print(tf.sort(a))
print(tf.sort(a, direction='DESCENDING'))
print(tf.argsort(a))

print("====================================================================")

a = tf.convert_to_tensor([[6, 7, 2],
                          [8, 5, 5],
                          [1, 2, 1]])
# 默认是按照最后一个维度进行排序（列维度）
res = tf.math.top_k(a, 2) # 只是选出前n个最大值，并排序
print(res)
print(res.indices)
print(res.values)

print("====================================================================")

# Top-k accuracy
prob = tf.constant([[0.1,0.2,0.7], [0.2,0.7,0.1]])
target = tf.constant([2,0])

k_b = tf.math.top_k(prob, 3).indices
print(k_b)
k_b = tf.transpose(k_b, [1,0]) # [1, 0]表示 将原1轴和0轴互换位置
print(k_b)

target = tf.broadcast_to(target, k_b.shape) # [3,2]
print(target)

print("--------------------------------------------------------------------")


# Top-k Accuracy 函数：
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, [1,0])
    target_ = tf.broadcast_to(target, pred.shape)
    # print(pred, target_)
    correct = tf.equal(pred, target_)
    # print(correct)

    res = []
    for k in topk:
        print("********************************************************")
        print(correct[:k])
        print(tf.reshape(correct[:k], [-1]))
        print(tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32))
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / batch_size)
        res.append(acc)

    return res


# 简单测试：
# prob = tf.constant([[0.1,0.2,0.7], [0.2,0.7,0.1]])
# target = tf.constant([2,0])
# print(accuracy(prob, target, topk=(3,2,1)))

# 1、最大值 Accuracy
output = tf.random.normal([10, 6], seed=2)
output = tf.math.softmax(output, axis=1) # 将数值转换为概率
target = tf.random.uniform([10], maxval=6, dtype=tf.int32, seed=2)
print("prob", output)
pred = tf.cast(tf.argmax(output, axis=1), tf.int32)
print("pred", pred)
print("label", target)
z = tf.equal(pred, target)
print(z)
correct = tf.reduce_sum(tf.cast(z, dtype=tf.int32))
print(correct)
print(float(correct / output.shape[0]))

# 2、Top-k Accuracy
acc = accuracy(output, target, topk=(1,))
print('top-1-6 acc', acc)
