# usage:
# $ python3 graph.py => generate graph file
# $ tensorboard --logdir=. => run tensorboard

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, layer_n, activation_function = None):
  layer_name = 'layer_%i' % layer_n

  with tf.name_scope(layer_name):
    weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'weights')
    tf.summary.histogram('weights', weights)

    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'biases')
    tf.summary.histogram('biases', biases)

    weights_plus_biases = tf.matmul(inputs, weights) + biases

    if activation_function is None:
      outputs = weights_plus_biases
    else:
      outputs = activation_function(weights_plus_biases)
    tf.summary.histogram('outputs', outputs)

    return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
  xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
  ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')

layer = add_layer(xs, 1, 10, layer_n = 1, activation_function = tf.nn.relu)
predication = add_layer(layer, 10, 1, layer_n = 2, activation_function = None)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predication), reduction_indices = [1]))
  tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
  train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

merged = tf.summary.merge_all()

with tf.Session() as sess:
  writer = tf.summary.FileWriter('.', sess.graph)
  sess.run(tf.initialize_all_variables())

  for step in range(1000):
    sess.run(train, feed_dict = { xs: x_data, ys: y_data })
    if step % 50 == 0:
      rs = sess.run(merged, feed_dict = { xs: x_data, ys: y_data })
      writer.add_summary(rs, step)
