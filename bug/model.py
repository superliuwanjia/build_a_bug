import tensorflow as tf

import sys
import os 

sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmm_tensorflow')
sys.path.insert(0, '/home/rgoel/drmm/EDRNN')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



from retina import *
from layer  import *
from utils import *
from bug_utils import *
class Bug():
	"""docstring for Bug"""
	def __init__(self, inp_size, out_size, conv_layers, lrcn_layers, num_classes = 6): # num_classes set to no of classes in kth dataset
		self.x = tf.placeholder(tf.float32, inp_size)
		self.Y = tf.placeholder(tf.float32, out_size)
		self.lr = tf.placeholder(tf.float32, [])
		self.inp_size = inp_size
		self.out_size = out_size
		self.conv_layers = conv_layers
		self.lrcn_layers = lrcn_layers
		self.num_classes  = num_classes
		self.get_retina_output(tf.reshape(self.x,[inp_size[0], inp_size[1], -1]))
		self.get_conv_output()

		self.get_lrcn_output()
		# self.get_lrrmn_output()
		self.get_softmax()
		self.get_loss()

	def get_retina_output(self, inp):
		with tf.name_scope("retina"):
			inp_size = inp.get_shape()
			retina = Retina(inp, inp_size)		
			print retina.get_output()
			self.retina_output = tf.reshape(retina.get_output(), 
											[self.inp_size[0]*self.inp_size[1],self.inp_size[2],self.inp_size[3], self.inp_size[4]*2])

	def get_conv_output(self, ):
		with tf.name_scope("convnet"):
			# convnet = Convnet(2, [[1,1,1,1], [1,1,1,1]], [], ["VALID", "VALID"])
			convnet = Convnet(self.conv_layers, [[1,1,1,1]], [[1,1, self.inp_size[4]*2, 1]], ["VALID"])
			convnet.build(self.retina_output)
			self.convnet_out = convnet.get_output()
			self.convnet_out_sh = self.convnet_out.get_shape().as_list()[1:]
			# self.convnet_out = tf.reshape(conv_out,
			# 								[self.inp_size[0], self.inp_size[1], -1])
			# 								# [self.inp_size[0], self.inp_size[1], self.inp_size[2], self.inp_size[3], self.inp_size[4]*2])
			

	
	def get_lrcn_output(self):
		inp = self.convnet_out
		for layer_no in range(self.lrcn_layers):
			with tf.variable_scope("lrcn"+str(layer_no)):
				convnet = Convnet(1, [[1,1,1,1]], [[1,1, self.convnet_out_sh[-1], 1]], ["VALID"])
				convnet.build(inp)
				conv_out = convnet.get_output()
				conv_out_shape = conv_out.get_shape().as_list()
				conv_out = tf.reshape(conv_out, [self.inp_size[0], self.inp_size[1], -1])

				print conv_out.get_shape(),"hi"
				hidden_units = conv_out_shape[1]*conv_out_shape[2]*conv_out_shape[3]
				lrcn_cell = LRCNcell(hidden_units, 1, 2, str(layer_no))

				inp, state = tf.nn.dynamic_rnn(lrcn_cell, conv_out, dtype = tf.float32)
				inp = tf.reshape(inp, conv_out_shape)
				# h = tf.split(state,2)[0]
				# print h.get_shape(), "lo"

		# self.output = tf.reshape(inp,[self.inp_size[0], self.inp_size[1], -1])
		self.output = tf.split(state,2)[0]


	def build_lrrmn(self, ):
		pass

	def get_lrrmn_output(self, ):
		with tf.name_scope("lrrmn"):
			lrrmn_cell = LRRMNcell([2], [self.inp_size[0], self.inp_size[4]*2*self.convnet_out_sh[-1]]+self.convnet_out_sh[:-1])
			outputs, state = tf.nn.dynamic_rnn(lrrmn_cell, self.convnet_out, dtype = tf.float32)
			self.output = state

	def get_softmax(self):
		lrcn_out_sh = self.output.get_shape().as_list()
		with tf.variable_scope("softmax_layer"):

			W = tf.get_variable("W",[lrcn_out_sh[1], self.num_classes], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			b = tf.get_variable("b",[self.num_classes], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			self.softmax = tf.nn.softmax(tf.add(tf.matmul(self.output, W), b))
			print self.softmax.get_shape()
			self.argmax_output = tf.argmax(self.softmax, 1)
			print self.argmax_output.get_shape()

	def get_loss(self):
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.Y, logits = self.softmax)
		# self.classification_acc = tf.metrics.accuracy(self.Y, sel)
		self.train_ops = optimizer.minimize(self.loss)



			
bug = Bug([2,5,4,5,3], [2,10], 1, 1, 10)