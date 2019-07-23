import tensorflow as tf
import numpy as np

# save
save_weights = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype = tf.float32, name = 'weights')
save_biases = tf.Variable([[1, 2, 3]], dtype = tf.float32, name = 'biases')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)
  save_path = saver.save(sess, 'saver/data.ckpt')
  print("Save to path: ", save_path)

# restore
load_weights = tf.Variable(np.arange(6).reshape((2, 3)), dtype = tf.float32, name = 'weights')
load_biases = tf.Variable(np.arange(3).reshape((1, 3)), dtype = tf.float32, name = 'biases')

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, 'saver/data.ckpt')
  print('weights:', sess.run(load_weights))
  print('biases:', sess.run(load_biases))
