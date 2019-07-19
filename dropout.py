import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, activation_function = None):
  weights = tf.Variable(tf.random_normal([in_size, out_size]))
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
  weights_plus_biases = tf.matmul(inputs, weights) + biases
  weights_plus_biases = tf.nn.dropout(weights_plus_biases, keep_prob)
  if activation_function is None:
    outputs = weights_plus_biases
  else:
    outputs = activation_function(weights_plus_biases)
  tf.summary.histogram('outputs', outputs)
  return outputs

digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

layer = add_layer(xs, 64, 100, activation_function = tf.nn.tanh)
predication = add_layer(layer, 100, 10, activation_function = tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predication), reduction_indices = [1]))
tf.summary.scalar('loss', cross_entropy)
merged = tf.summary.merge_all()

train = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  train_writer = tf.summary.FileWriter('train', sess.graph)
  test_writer = tf.summary.FileWriter('test', sess.graph)

  for step in range(500):
    sess.run(train, feed_dict = { xs: x_train, ys: y_train, keep_prob: 0.8 })

    if step % 50 == 0:
      train_result = sess.run(merged, feed_dict = { xs: x_train, ys: y_train, keep_prob: 1 })
      test_result = sess.run(merged, feed_dict = { xs:x_test, ys: y_test, keep_prob: 1 })
      train_writer.add_summary(train_result, step)
      test_writer.add_summary(test_result, step)

