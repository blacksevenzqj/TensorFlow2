import tensorflow as tf

x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)

with tf.GradientTape(persistent=True) as tape: # 设置persistent GradientTape 实现多次求导
    tape.watch([w1, b1, w2, b2])
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2

dy2_dy1 = tape.gradient(y2, [y1])[0] # ∂y2 / ∂y1
dy1_dw1 = tape.gradient(y1, [w1])[0] # ∂y1 / w1
# ∂y2 / w1  =  ∂y2 / ∂y1  ·  ∂y1 / w1
dy2_dw1 = tape.gradient(y2, [w1])[0] # ∂y2 / w1
print(dy2_dy1, dy1_dw1, dy2_dy1*dy1_dw1)
print(dy2_dw1)
