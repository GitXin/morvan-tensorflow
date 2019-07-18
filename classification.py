import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs, in_size, out_size, activation_function = None):
  weights = tf.Variable(tf.random_normal([in_size, out_size]))
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
  weights_plus_biases = tf.matmul(inputs, weights) + biases
  if activation_function is None:
    outputs = weights_plus_biases
  else:
    outputs = activation_function(weights_plus_biases)
  return outputs

def compute_accuracy(x_data, y_data):
  predication_ys = sess.run(predication, feed_dict = { xs: x_data })
  correct_predicition = tf.equal(tf.argmax(predication_ys, 1), tf.argmax(y_data, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
  return sess.run(accuracy, feed_dict = { xs: x_data, ys: y_data })

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

predication = add_layer(xs, 784, 10, activation_function = tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predication), reduction_indices = [1]))

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict = { xs: batch_xs, ys: batch_ys })

    if step % 50 ==0:
      print(compute_accuracy(mnist.test.images, mnist.test.labels))