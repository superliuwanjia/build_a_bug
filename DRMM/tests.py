import tensorflow as tf
import numpy as np
import theano as th
import time
from theano.tensor import *
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, sigmoid
import theano.tensor as T
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############ to create initial prior ##########
# amps_val = np.random.rand(50)
# amps_val /= np.sum(amps_val)
# b = tf.Variable(tf.convert_to_tensor(amps_val))
# c = 2*b

# a  =tf.Variable(tf.random_uniform([50]))
# d = tf.nn.softmax(a)
# e = 2*d

# f = tf.Variable(0.5*tf.ones([2], dtype = tf.float32))

####################to test maxpool ############

# a =tf.Variable(tf.ones([12,5,5,3]))
# d = tf.nn.avg_pool(a, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID")
# e = tf.tile(a,[1,1,1,10])
# f = d.dtype
# print f==tf.float32
# b = theano.shared(np.ones([12,3,5,5], dtype=theano.config.floatX), name = "dd", borrow = True)
# inc = T.iscalar('inc')
# output = pool.pool_2d(input=b, ds=(2,2),ignore_border=True, mode='average_exc_pad')
# accumulator = theano.function([inc], output, updates=[(b, b+inc)])

###################### to test repeat################

# b = theano.shared(np.ones([12,3,5,5], dtype=theano.config.floatX), name = "dd", borrow = True)
# inc = T.iscalar('inc')
# output = T.repeat(b,10, axis=1)
# accumulator = theano.function([inc], output, updates=[(b, b+inc)])
# print accumulator(1).shape


# with tf.Session() as sess:
# 	sess.run(tf.initialize_all_variables())
	# t1 = time.time()	
	# print sess.run(c, feed_dict = {})
	# print time.time()-t1
	# t1 = time.time()
	# print sess.run(e, feed_dict = {})
	# print time.time() - t1
	# print sess.run(f, feed_dict = {})
	# print sess.run(d,feed_dict={}).shape
	# print sess.run(e,feed_dict={}).shape
# print accumulator(1).shape
# print accumulator(1).shape

#####tets convolutions
# def get_important_latents_BU_th(self, input, betas):
# 	# if self.border_mode != 'FULL':
# 	latents_before_BN = tf.nn.conv2d(tf.transpose(input, [0,2,3,1]), betas, strides=[1, 1, 1, 1], padding="VALID")
# 	latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])

# def get_important_latents_BU_tf(self, input, betas):

# 	latents_before_BN = conv2d(
# 		input=input,
# 		filters=betas,
# 		filter_shape=(self.K, self.Cin, self.h, self.w),
# 		input_shape=(self.Ni, self.Cin, self.H, self.W),
# 		filter_flip=False,
# 		border_mode="valid"
# 	)

# def compare(a, b):
# 	print "ff", a.shape
# 	a = np.array(a).reshape((-1))
# 	print "hh"
# 	b = np.array(b).reshape((-1))
# 	print "hi"
# 	for i in range(a.shape[0]):
# 		print "hi",a[i],b[i]
# 		if (a[i]-b[i] > 1e-3):
# 			print a[i],b[i]
# 			print i
# 			return False
# 	return True

# t = np.random.rand(2,3,2,4)
# tth = t.repeat(2,axis =2)
# img = np.random.rand(1,3,9,8)
# a = np.random.rand(8,3,3,3)
# b = theano.shared(value = a, borrow = True)
# c = theano.shared(value = img, borrow = True)
# conv = conv2d(c,b, border_mode = "valid", filter_flip = False)
# fn = theano.function([], conv)
# th_ans = fn()
# sd = tf.convert_to_tensor(t,tf.float32)
# sd_shape = sd.get_shape().as_list()
# print sd_shape
# ttf = tf.reshape(tf.tile(tf.reshape(sd,[sd_shape[0],-1,1,sd_shape[3]]),[1,1,2,1]),[sd_shape[0],sd_shape[1],-1,sd_shape[3]])
# # ttf = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(sd,[0,2,3,1]), [2*int(sd_shape[2]), int(sd_shape[3])]),[0,3,1,2])
# d = tf.Variable(tf.convert_to_tensor(img, dtype = tf.float32))
# e = tf.transpose(tf.Variable(tf.convert_to_tensor(a, dtype = tf.float32)),[2,3,1,0])
# print e.get_shape()
# tf_a = tf.nn.conv2d(d,e, strides = [1,1,1,1], padding = "VALID", data_format = "NCHW")
# with tf.Session() as sess:
# 	sess.run(tf.initialize_all_variables())
# 	tf_ans = sess.run(ttf, feed_dict = {})

# print np.array_equal(tth,tf_ans)
# print tth.shape, tf_ans.shape
# print compare(tth, tf_ans)

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('save/testing/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('save/testing'))