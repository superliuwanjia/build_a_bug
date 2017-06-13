import tensorflow as tf


a = tf.Variable(tf.ones([2, 20, 1]))
b = tf.Variable(tf.ones([20, 1, 20]))
# c = tf.matmul(b,a)
c = tf.einsum("aij,ijk->ajk", a, b)
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	s = sess.run([c,b,a], feed_dict = {})

for i in s:
	print i.shape
	break
