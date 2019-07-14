import tensorflow as tf

# 1、sigmoid
x = tf.linspace(-10., 10., 10) # startNum浮点数, endNum浮点数, Number of elements
print(x)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.sigmoid(x)

print(y)
grads = tape.gradient(y, [x])
print(grads)
'''
x:
[-10.         -7.7777777  -5.5555553  -3.333333   -1.1111107   1.1111116
3.333334    5.5555563   7.7777786  10.       ]
y:
[4.5418739e-05 4.1875243e-04 3.8510859e-03 3.4445167e-02 2.4766389e-01 7.5233626e-01 
9.6555483e-01 9.9614894e-01 9.9958128e-01 9.9995458e-01]
grads:
[4.5416677e-05, 4.1857705e-04, 3.8362551e-03, 3.3258699e-02, 1.8632649e-01, 1.8632641e-01, 
3.3258699e-02, 3.8362255e-03, 4.1854731e-04, 4.5416677e-05]

当 x = -3.333333 时，y = 3.4445167e-02已经很接近于0了。梯度为3.3258699e-02也接近于0.
当 x = 3.333334 时，y = 9.6555483e-01已经很接近于1了。梯度为3.3258699e-02也很接近于0.
'''


print("====================================================================")


# 2、tanh：tanh = 2sigmoid(2x) - 1
x = tf.linspace(-5., 5., 10) # startNum浮点数, endNum浮点数, Number of elements
y = tf.tanh(x) # 值在 -1 到 1 之间
print(y)


print("====================================================================")


# 3、relu：当 x < 0 时梯度为0；当 x > 0 时，梯度为1。不会放大也不会缩小，搜索最优解时导数计算简单。
x = tf.linspace(-1., 1., 10)
y = tf.nn.relu(x)
print(y)

y = tf.nn.leaky_relu(x)
print(y)



