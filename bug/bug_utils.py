import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import sys

sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmm_tensorflow')
from layer import *

class Convnet(object):
	"""docstring for Convnet"""
	def __init__(self, num_layers, strides, filtersizes, padding):
		self.num_layers = num_layers
		self.strides = strides
		self.filtersizes = filtersizes
		self.padding = padding

	def build(self, input):
		out = input

		for layer_num in range(self.num_layers):
			with tf.variable_scope("conv_layer_"+str(layer_num)):
				filter_var = tf.get_variable("filter", self.filtersizes[layer_num], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
				out = tf.nn.relu(tf.nn.conv2d(out, filter_var, self.strides[layer_num], self.padding[layer_num]))				
		self.out = out

	def get_output(self):
		return self.out

				
		
class LRRMNcell(RNNCell):
	"""docstring for EMA"""
	def __init__(self, hidden_size,inp_size = None, num_drmm_lay = 1):
		self._hidden_size = hidden_size
		self.inp_size = inp_size
		self.num_drmm_lay = num_drmm_lay

	@property
	def state_size(self):
		return self._hidden_size

	@property
	def output_size(self):
		return self._hidden_size

	def _build_drmm_network(self, name, noise_weight, noise_std, K, M, W, H, w, h, Cin, Ni,
			lambdas_t_val_init=None, gamma_val_init=None, beta_val_init=None,prun_mat_init=None, prun_synap_mat_init=None,mean_bn_init=None, var_bn_init=None,pool_t_mode='max_t', 
			border_mode='VALID', nonlin='abs', mean_pool_size=[2, 2],max_condition_number=1.e3,weight_init="xavier",is_noisy=False, is_bn_BU=True,
			epsilon=1e-10,momentum_pi_t=0.99,momentum_pi_a=0.99,momentum_pi_ta=0.99,momentum_pi_synap=0.99, is_prun=True, is_prun_synap=True,
			is_dn=True,sigma_dn_init=None,b_dn_init=None,alpha_dn_init=None,update_mean_var_with_sup=True, survive_thres = 0.0):
		with tf.variable_scope(name):
			layer = None
			for layer_num in range(self.num_drmm_lay):
				# Building the forward model
				with tf.variable_scope("drmm_layer_no_" + str(self.N_layers)):
					if layer != None:
						shapes = layer.output_lab.get_shape().as_list()
						# print shapes,"fffffff\n\n\n\n\n", self.N_layers
						layer = Layer(
							data_4D_lab=layer.output_lab,
							data_4D_unl=layer.output,
							data_4D_unl_clean=layer.output_clean,
							# data_4D_unl=None,
							# data_4D_unl_clean=None,
							noise_weight=noise_weight,
							noise_std=noise_std,
							is_train=self.is_train, momentum_bn=self.momentum_bn,
							K=K, M=M, W=shapes[2], H=shapes[3],
							w=w, h=h, Cin=shapes[1], Ni=self.batch_size,
							lambdas_t_val_init=lambdas_t_val_init,
							gamma_val_init=gamma_val_init, beta_val_init=beta_val_init,
							prun_mat_init=prun_mat_init, prun_synap_mat_init=prun_synap_mat_init,
							mean_bn_init=mean_bn_init, var_bn_init=var_bn_init,
							sigma_dn_init=sigma_dn_init, b_dn_init=b_dn_init, alpha_dn_init=alpha_dn_init,
							update_mean_var_with_sup=self.update_mean_var_with_sup,
							pool_t_mode=pool_t_mode,
							border_mode=border_mode,
							max_condition_number=1.e3, weight_init=weight_init,
							is_noisy=is_noisy, is_bn_BU=self.is_bn_BU,
							nonlin=self.nonlin,
							is_prun=self.is_prun,
							is_prun_synap=self.is_prun_synap,
							is_dn=self.is_dn,
							name = "drmm_layer_no_" + str(layer_num+1)
						)
					else:
						#create x variables for the first time

						layer = Layer(
							data_4D_lab=x_lab,
							data_4D_unl=x_unl,
							data_4D_unl_clean=x_clean,
							# data_4D_unl=None,
							# data_4D_unl_clean=None,

							noise_weight=noise_weight,
							noise_std=noise_std,
							is_train=self.is_train, momentum_bn=self.momentum_bn,
							K=K, M=M, W=W, H=H,
							w=w, h=h, Cin=Cin, Ni=self.batch_size,
							lambdas_t_val_init=lambdas_t_val_init,
							gamma_val_init=gamma_val_init, beta_val_init=beta_val_init,
							prun_mat_init=prun_mat_init, prun_synap_mat_init=prun_synap_mat_init,
							mean_bn_init=mean_bn_init, var_bn_init=var_bn_init,
							sigma_dn_init=sigma_dn_init, b_dn_init=b_dn_init, alpha_dn_init=alpha_dn_init,
							update_mean_var_with_sup=self.update_mean_var_with_sup,
							pool_t_mode=pool_t_mode,
							border_mode=border_mode,
							max_condition_number=1.e3, weight_init=weight_init,
							is_noisy=is_noisy, is_bn_BU=self.is_bn_BU,
							nonlin=self.nonlin,
							is_prun=self.is_prun,
							is_prun_synap=self.is_prun_synap,
							is_dn=self.is_dn,
							name = "drmm_layer_no_0"
						)

			

	def __call__(self, input, state, scope = None):
		inp = tf.reshape(input, self.inp_size)
		r_channel,g_channel,b_channel = tf.split(inp, 3, axis = 1)

		r_channel_


		print r_channel.get_shape()

		return state, state
	
class LRCNcell(RNNCell):
	"""docstring for LRCNcelll"""
	def __init__(self, hidden_units, num_channels, batch_size, scope = None):
		self._hidden_units = hidden_units
		self.num_channels = num_channels
		self.batch_size = batch_size
		self.scope = scope
		with tf.variable_scope("lstm_"+scope):
			# Input state variables
			self.i_gate_vars = tf.get_variable("i_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			self.f_gate_vars = tf.get_variable("f_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			self.o_gate_vars = tf.get_variable("o_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			self.g_gate_vars = tf.get_variable("w_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

			# Hidden state variables
			self.i_hidden_gate_vars = tf.get_variable("i_hidden_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			self.f_hidden_gate_vars = tf.get_variable("f_hidden_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			self.o_hidden_gate_vars = tf.get_variable("o_hidden_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			self.g_hidden_gate_vars = tf.get_variable("w_hidden_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())


	@property
	def state_size(self):
		return self._hidden_units

	@property
	def output_size(self):
		return self._hidden_units
	
	


	def __call__(self, input, state, scope = None):
		
		with tf.variable_scope(self.scope):
			print state.get_shape(), input.get_shape()
			h, c = tf.split(state,2)
			print h.get_shape()
			inp = tf.reshape(input, [self.batch_size, -1, self.num_channels])
			h = tf.reshape(h, inp.get_shape())
			print h.get_shape(), inp.get_shape(), self.i_hidden_gate_vars.get_shape()
			i = tf.sigmoid(tf.einsum("aij,ijk->ajk", inp, self.i_gate_vars, ) + tf.einsum("aij,ijk->ajk", h, self.i_hidden_gate_vars))
			f = tf.sigmoid(tf.einsum("aij,ijk->ajk", inp, self.f_gate_vars, ) + tf.einsum("aij,ijk->ajk", h, self.f_hidden_gate_vars))
			o = tf.sigmoid(tf.einsum("aij,ijk->ajk", inp, self.o_gate_vars, ) + tf.einsum("aij,ijk->ajk", h, self.o_hidden_gate_vars))
			g = tf.nn.relu(tf.einsum("aij,ijk->ajk", inp, self.i_gate_vars, ) + tf.einsum("aij,ijk->ajk", h, self.i_hidden_gate_vars))
			print g.get_shape(), self._hidden_units
			new_c = c*f + i*g
			new_h = tf.reshape(o*tf.nn.relu(new_c), [self.batch_size, self._hidden_units])
			new_c = tf.reshape(new_c, [self.batch_size, self._hidden_units])
			print new_h.get_shape()
			print "ho"
			return new_h, tf.concat([new_h, new_c], axis = 0)
		
		