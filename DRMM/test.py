import theano
import theano.tensor as T
import tensorflow as tf
import numpy as np
from layer import *
from utils import *
from model_v1 import *
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


save_dir = "save/testing"

def reshape_data(data, to_shape = (1, 28, 28)):
	data_sh = data.shape
	return data.reshape((data_sh[0], to_shape[0], to_shape[1], to_shape[2]))

# def CreateModel():
	
batch_size =  100
Cin = 1
W = 28
H = 28
seed = 23

model = Model(batch_size, Cin, W, H, seed, is_sup = True)
model.add(noise_weight = 0.0, noise_std = 0.01, K = 1, W = 28, H = 28, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
model.add(noise_weight = 0.0, noise_std = 0.01, K = 1, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
model.add(noise_weight = 0.0, noise_std = 0.01, K = 1, W = 64, H = 64, M = 1, w = 3, h = 3, Ni = 100, Cin = 1, border_mode = "VALID")
# print tf.trainable_variables()
# print model.layer.output_lab.get_shape()
model.Compile()
model.Optimize()

epochs = 100
with  tf.Session() as sess:
	# try:
	# 	saver = tf.train.import_meta_graph('save/testing/model.ckpt.meta')
	# 	saver.restore(sess,tf.train.latest_checkpoint('save/testing'))
	# except Exception as e:
		
	sess.run(tf.global_variables_initializer())
	# print model.grads
	for grad, var in model.grads:
		try:
			tf.summary.histogram(var.name + '/gradient', grad)
		except Exception, e:
			continue
		
	for var in tf.trainable_variables():
		tf.summary.histogram(var.name, var)
	#create a saver
	# saver = tf.train.Saver()
	#Summary collection
	summary_op = tf.summary.merge_all()
	writer = tf.summary.FileWriter("logs")
	writer.add_graph(sess.graph)
	batch_no = 0
	for epoch in range(epochs):
		mnist = input_data.read_data_sets('MNIST_data')		
		print "epoch number is", epoch
		while True:

			try:
				train = mnist.train.next_batch(100)
				# print train[1]
				# break
				feed_dict = {
							model.x_lab : reshape_data(train[0]),
							model.y_lab : train[1],
							model.lr : 0.00001,
							model.momentum_bn : 0.99,
							model.is_train : 1
							# self.model.x_unl : self.reshape_data(train[0]),
							# self.model.x_clean : self.reshape_data(train[0])
				}
				# print "here"
				_, train_loss, err, summ, probs = sess.run([model.train_ops, model.cost, model.classification_error, summary_op, model.softmax_layer_nonlin.gammas_lab], feed_dict = feed_dict)			
				# print "gogo"
				writer.add_summary(summ, batch_no)
				# err = sess.run(self.model.cost, feed_dict={self.model.x_lab: self.reshape_data(self.mnist.test.images), self.model.y_lab : self.mnist.test.labels})
															# self.model.x_unl : None, self.model.x_clean : None											
# self.																	})
				# if ((batch_no*batch_size)%10000 == 0):
				# 	save_path = saver.save(save_dir+"/model.ckpt")
				# 	print "latest model saved on the path ", save_path
				if ((batch_no*batch_size)%100000 == 0):

					print train_loss, err,batch_no*100
					print "train error after training %s batches is %s".format(batch_no, train_loss)
				batch_no += 1

			except Exception as e:
				break
				# raise e	

	
print model.layer.output_lab.get_shape().as_list()
