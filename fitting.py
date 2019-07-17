import tensorflow as tf
import numpy as np

# 拟合 y = x * 0.1 + 0.3

# 生成训练数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 创建 tensorflow 结构
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# 拟合函数
y = weights * x_data + biases

# 代价函数
loss = tf.reduce_mean(tf.square(y - y_data))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始器
init = tf.initialize_all_variables()

# 训练
sess = tf.Session()
sess.run(init)

for step in range(201):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(weights), sess.run(biases))

sess.close()
