import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function = None):
  weights = tf.Variable(tf.random_normal([in_size, out_size]))
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
  weights_plus_biases = tf.matmul(inputs, weights) + biases
  if activation_function is None:
    outputs = weights_plus_biases
  else:
    outputs = activation_function(weights_plus_biases)
  return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
layer = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
predication = add_layer(layer, 10, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predication), reduction_indices = [1]))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init)

  for step in range(1000):
    sess.run(train, feed_dict = { xs: x_data, ys: y_data })
    if step % 50 == 0:
      # print(sess.run(loss, feed_dict = { xs: x_data, ys: y_data }))
      try:
        ax.lines.remove(lines[0])
      except Exception:
        pass
      predication_value = sess.run(predication, feed_dict = { xs: x_data, ys: y_data })
      lines = ax.plot(x_data, predication_value, 'r-', lw = 5)
      plt.pause(0.1)

plt.pause(0)
