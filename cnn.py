import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def biase_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

'''
padding:
SAME: 通过在边界补零的方式，来适应过滤器的大小，从而使卷积/池化前后的大小不发生改变
VALID: 当边界不足以被过滤器扫描时，将该部分进行丢弃，所以卷积/池化后的大小会小于等于原图
'''

# convolution(卷积)
def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

# pooling(池化)
def max_pooling(x):
  return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def compute_accuracy(x_data, y_data):
  # tips: don't forget to feed placeholder
  predication_ys = sess.run(predication, feed_dict = { xs: x_data, keep_prob: 1 })
  correct_predicition = tf.equal(tf.argmax(predication_ys, 1), tf.argmax(y_data, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
  return sess.run(accuracy)

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# reshape => (28, 28, 1)
# 1 为图片厚度，即 rgb 通道，此处 1 代表着黑白，因颜色对图片识别没有用处
x_image = tf.reshape(xs,[-1, 28, 28, 1])

# convolution layer 1 => (28, 28, 32)
weight_conv1 = weight_variable([5, 5, 1, 32])
biase_conv1 = biase_variable([32])
conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + biase_conv1)

# pooling layer 1 => (14, 14, 32)
pool1 = max_pooling(conv1)

# convolution layer 2 => (14, 14, 64)
weight_conv2 = weight_variable([5, 5, 32, 64])
biase_conv2 = biase_variable([64])
conv2 = tf.nn.relu(conv2d(pool1, weight_conv2) + biase_conv2)

# pooling layer 2 => (7, 7, 64)
pool2 = max_pooling(conv2)

# flatten => 7 * 7 * 64
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# fully connected layer 1 => 1024
weight_fc1 = weight_variable([7 * 7 * 64, 1024])
biase_fc1 = biase_variable([1024])
fc1 = tf.nn.relu(tf.matmul(pool2_flat, weight_fc1) + biase_fc1)
fc1_dropout = tf.nn.dropout(fc1, keep_prob)

# fully connected layer 2 => 10
weight_fc2 = weight_variable([1024, 10])
biase_fc2 = biase_variable([10])
predication = tf.nn.softmax(tf.matmul(fc1_dropout, weight_fc2) + biase_fc2)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predication), reduction_indices = [1]))

# tips: adjust learning rate
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict = { xs: batch_xs, ys: batch_ys, keep_prob: 0.7 })

    if step % 50 == 0:
      print(compute_accuracy(mnist.test.images, mnist.test.labels))
