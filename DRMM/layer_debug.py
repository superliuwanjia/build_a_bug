import tensorflow as tf
import numpy as np
from utils import BatchNormalization
class Layer():

	"""
	Arguments : 
		internal variables:
		`data_4D_lab`:  labeled images
		`data_4D_unl`:  unlabeled images
		`data_4D_unl_clean`: unlabeled images used for the clean pass without noise. This is used when we add noise to the
								the model to make it robust. We are not doing this now.
		`noise_weight`:control how much noise we want to add to the model. Since we don't add noise to the model, we always 
					   noise_weight to 0
		`noise_std`: the standard deviation of the noise
		`K`:           Number of the rendering matrices lambdas_t
		`M`:           Latent dimensionality (set to 1 when the DRMM is applied at the patch level. In general, can be set 
					   to any value, and we save this for the future work)
		`W`:           Width of the input
		`H`:           Height of the input
		`w`:           Width of the rendering matrices
		`h`:           Height of the rendering matrices
		`Cin`:         Number of channels of the input = Number of channels of the rendering matrices
		`Ni`:          Number of input images
		`momentum_bn`: control how to update the batch mean and var in batch normalization (BatchNorm) which will be used in testing
		`is_train`:     {0,1} where 0: testing mode and 1: training mode
		`lambdas_t_val_init`: Initial values for rendering matrices
		`gamma_val_init`: Initial values for the correction term gamma in BatchNorm
		`beta_val_init`: Initial values for the correction term beta in BatchNorm
		`mean_bn_init`: Initial values for the batch mean in BatchNorm
		`var_bn_init`: Initial values for the batch variance in BatchNorm
		`prun_mat_init`: Initial values for the neuron pruning masking matrices
		`prun_synap_mat_init`: Initial values for the synaptic pruning masking matrices
		`sigma_dn_init`: Initial values for the sigma term in DivNorm
		`b_dn_init`: Initial values for the bias term in DivNorm
		`alpha_dn_init`: Initial values for the alpha term in DivNorm
		`pool_t_mode`: Pooling mode={'max_t','mean_t`,None}
		`border_mode`: {'valid`, 'full', `half`} mode for the convolutions
		`nonlin`: {`relu`, `abs`, None}
		`mean_pool_size`: size of the mean pooling layer before the softmax regression
		`max_condition_number`: a condition number to make computations more stable. Set to 1.e3
		`xavier_init`: {True, False} use Xavier initialization or not
		`is_noisy`: {True, False} add noise to the model or not. Since we don't add noise to the model, we always set this 
					to False
		`is_bn_BU`: {True, False} do batch normalization in the bottom-up E step or not
		`epsilon`: used to avoid divide by 0
		`momentum_pi_t`: used to update pi_t during training (Exponential Moving Average (EMA) update)
		`momentum_pi_a`: used to update pi_a during training (EMA update)
		`momentum_pi_ta`: used to update pi_ta during training (EMA update)
		`momentum_pi_synap`: used to update pi_synap during training (EMA update)
		`is_prun`: {True, False} apply neuron pruning or not
		`is_prun_synap`: {True, False} apply synaptic pruning or not
		`is_dn`: {True, False} apply divisive normalization (DivNorm) or not
		`update_mean_var_with_sup`: {True, False} if True, only use data_4D_lab to update the mean and var in BatchNorm
	"""
	def __init__(self, data_4D_lab,
				 data_4D_unl,
				 data_4D_unl_clean,
				 noise_weight,
				 noise_std,
				 K, M, W, H, w, h, Cin, Ni,
				 momentum_bn, is_train,
				 lambdas_t_val_init=None, 
				 gamma_val_init=None, beta_val_init=None,
				 prun_mat_init=None, prun_synap_mat_init=None,
				 mean_bn_init=None, var_bn_init=None,
				 pool_t_mode='max_t', border_mode='VALID', nonlin='relu', 
				 mean_pool_size=(2, 2),
				 max_condition_number=1.e3,
				 weight_init="xavier",
				 is_noisy=False,
				 # is_noisy=True,
				 # is_bn_BU=False,
				 is_bn_BU=True,
				 epsilon=1e-10,
				 momentum_pi_t=0.99,
				 momentum_pi_a=0.99,
				 momentum_pi_ta=0.99,
				 momentum_pi_synap=0.99,
				 # is_prun=False,
				 is_prun=True,
				 # is_prun_synap=False,
				 is_prun_synap=True,
				 # is_dn=False,
				 is_dn=True,
				 sigma_dn_init=None,
				 b_dn_init=None,
				 alpha_dn_init=None,
				 # update_mean_var_with_sup=False,
				 update_mean_var_with_sup=True, 
				 name = "conv_layer",
				 beta_init = None):
		# self.beta = tf.Variable(tf.convert_to_tensor(beta_init, dtype = tf.float32))
		self.name = name
		self.K = K  # number of lambdas_t/filter
		self.M = M  # latent dimensionality. Set to 1 for our model
		self.data_4D_lab = tf.convert_to_tensor(data_4D_lab, dtype = tf.float32)  # labeled data
		self.data_4D_unl = tf.convert_to_tensor(data_4D_unl, dtype = tf.float32) # unlabeled data
		self.data_4D_unl_clean = tf.convert_to_tensor(data_4D_unl_clean, dtype = tf.float32)  # unlabeled data used for the clean path
		self.noise_weight = noise_weight  # control how much noise we want to add to the model. Set to 0 for our model
		self.noise_std = noise_std  # the standard deviation of the noise
		self.Ni = Ni  # no. of labeled examples = no. of unlabeled examples
		self.w = w  # width of filter
		self.h = h  # height of filter
		self.Cin = Cin  # number of channels in the image
		self.D = self.h * self.w * self.Cin  # patch size
		self.W = W  # width of image
		self.H = H  # height of image
		if border_mode == 'VALID':  # convolution mode. Output size is smaller than input size
			self.Np = (self.H - self.h + 1) * (self.W - self.w + 1)  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H - self.h + 1, self.W - self.w + 1)
		elif border_mode == 'HALF':  # convolution mode. Output size is the same as input size
			self.Np = self.H * self.W  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H, self.W)
		elif border_mode == 'SAME':  # # convolution mode. Output size is greater than input size
			self.Np = (self.H + self.h - 1) * (self.W + self.w - 1)  # no. of patches per image
			self.latents_shape = (self.Ni, self.K, self.H + self.h - 1, self.W + self.w - 1)
		else:
			raise
		print "latent_shape : ", self.latents_shape, data_4D_lab.get_shape()
		self.N = self.Ni * self.Np  # total no. of patches and total no. of hidden units
		self.mean_pool_size = mean_pool_size  # size of the mean pooling layer before the softmax regression

		self.lambdas_t_val_init = lambdas_t_val_init  # Initial values for rendering matrices
		self.gamma_val_init = gamma_val_init  # Initial values for the correction term gamma in BatchNorm
		self.beta_val_init = beta_val_init  # Initial values for the correction term beta in BatchNorm
		self.prun_synap_mat_init = prun_synap_mat_init  # Initial values for the synaptic pruning masking matrices
		self.prun_mat_init = prun_mat_init  # Initial values for the neuron pruning masking matrices
		self.sigma_dn_init = sigma_dn_init  # Initial values for the sigma term in DivNorm
		self.mean_bn_init = mean_bn_init  # Initial values for the batch mean in BatchNorm
		self.var_bn_init = var_bn_init  # Initial values for the batch variance in BatchNorm
		self.b_dn_init = b_dn_init  # Initial values for the bias term in DivNorm
		self.alpha_dn_init = alpha_dn_init  # Initial values for the alpha term in DivNorm

		self.pool_t_mode = pool_t_mode  # Pooling mode={'max_t','mean_t`,None}
		self.nonlin = nonlin  # {`relu`, `abs`, None}
		
		self.border_mode = border_mode  # {'valid`, 'full', `half`} mode for the convolutions
		self.max_condition_number = max_condition_number  # a condition number to make computations more stable. Set to 1.e3
		self.weight_init = weight_init  # type of weight initialisation
		self.is_noisy = is_noisy  # {True, False} add noise to the model or not.
		self.is_bn_BU = is_bn_BU  # {True, False} do batch normalization in the bottom-up E step or not
		self.momentum_bn = momentum_bn  # control how to update the batch mean and var in batch normalization (BatchNorm)
		self.momentum_pi_t = momentum_pi_t  # used to update pi_t during training (Exponential Moving Average (EMA) update)
		self.momentum_pi_a = momentum_pi_a  # used to update pi_a during training (EMA update)
		self.momentum_pi_ta = momentum_pi_ta  # used to update pi_ta during training (EMA update)
		self.momentum_pi_synap = momentum_pi_synap  # used to update pi_synap during training (EMA update)
		self.is_train = is_train  # {0,1} where 0: testing mode and 1: training mode
		self.epsilon = epsilon  # used to avoid divide by 0
		self.is_prun = is_prun  # {True, False} apply neuron pruning or not
		self.is_prun_synap = is_prun_synap  # {True, False} apply synaptic pruning or not
		self.is_dn = is_dn  # {True, False} apply divisive normalization (DivNorm) or not
		self.update_mean_var_with_sup = update_mean_var_with_sup  # {True, False} if True, only use data_4D_lab to update
																	# the mean and var in BatchNorm
		self._initialize()
	
	def _initialize(self):
		#
		# initialize the model parameters.
		# all parameters involved in the training are collected in self.params for the gradient descent step

		# add noise to the input if is_noisy
		if self.is_noisy:
			if self.data_4D_unl is not None:
				self.data_4D_unl = self.data_4D_unl + \
								   self.noise_weight * tf.random_normal([self.Ni, self.Cin, self.H, self.W],
																		mean=0.0, stddev=self.noise_std)
			if self.data_4D_lab is not None:
				self.data_4D_lab = self.data_4D_lab + \
								   self.noise_weight * tf.random_normal([self.Ni, self.Cin, self.H, self.W],
																		mean=0.0, stddev=self.noise_std)


		# initialize t and a priors
		print self.latents_shape[1:]
		self.pi_t = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_t")
		self.pi_a = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_a")
		self.pi_a_old = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_a_old")
		self.pi_ta = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_ta")
		

		# initialize the pruning masking matrices
		if self.prun_mat_init is None:
			self.prun_mat = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "prun_mat")
		else:
			self.prun_mat = tf.Variable(tf.convert_to_tensor(np.asarray(self.prun_mat_init), dtype = tf.float32, name = "prun_mat"))
			
		if self.prun_synap_mat_init is None:
			self.prun_synap_mat = tf.Variable(tf.ones([self.h, self.w, self.Cin, self.K]), dtype=tf.float32,name='prun_synap_mat')
		else:
			self.prun_synap_mat = tf.Variable(tf.convert_to_tensor(np.asarray(self.prun_synap_mat_init)), dtype=tf.float32, name='prun_mat')

		# initialize synapse prior (its probability to be ON or OFF)
		if self.is_prun_synap:
			self.pi_synap = tf.Variable(tf.ones([self.h, self.w, self.Cin, self.K], dtype = tf.float32), name = 'pi_synap')
			self.pi_synap_old = tf.Variable(tf.ones([self.h, self.w, self.Cin, self.K], dtype = tf.float32), name = 'pi_synap_old')
			
		# pi_t_final and pi_a_final are used after training for sampling
		self.pi_t_final = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_t_final")
		self.pi_a_final = tf.Variable(tf.ones(self.latents_shape[1:], dtype = tf.float32), name = "pi_a_final")
		
		# initialize the lambdas_t
		# if initial values for lambdas_t are not provided, randomly initialize lambdas_t
		initialised = 0
		if self.lambdas_t_val_init is None:
			if self.weight_init == "xavier":
				initialised = 1
				self.lambdas_t = tf.get_variable(shape = [self.K, self.D, self.M],name = "lambdas_t",  initializer = tf.contrib.layers.xavier_initializer())
			else:
				lambdas_t_value = np.random.randn(self.K, self.D, self.M) / \
								np.sqrt(self.max_condition_number)
		else:
			lambdas_t_value = self.lambdas_t_val_init
		if initialised == 0:
			self.lambdas_t = tf.Variable(tf.convert_to_tensor(lambdas_t_value, dtype=tf.float32), name='lambdas_t')

		# Initialize BatchNorm
		if self.is_bn_BU:
			self.bn_BU = BatchNormalization(insize=self.K, mode=1, momentum=self.momentum_bn, is_train=self.is_train,
											epsilon=self.epsilon,
											gamma_val_init=self.gamma_val_init, beta_val_init=self.beta_val_init,
											mean_init=self.mean_bn_init, var_init=self.var_bn_init)

			self.params = [self.lambdas_t, self.bn_BU.gamma, self.bn_BU.beta]
		else:
			self.params = [self.lambdas_t, ]

		# Initialize the output
		self.output_lab = None
		self.output_clean = None
		self.output_1 = None
		self.output_2 = None

		# Initialize the sigma parameter in DivNorm
		if self.sigma_dn_init is None:
			self.sigma_dn = tf.Variable(0.5*tf.ones([1], dtype=tf.float32), name='sigma_dn')
		else:
			self.sigma_dn = tf.Variable(tf.convert_to_tensor(self.sigma_dn_init, dtype=tf.float32), name='sigma_dn')

		# Initialize the bias in DivNorm
		if self.b_dn_init is None:
			self.b_dn = tf.Variable(0. * tf.ones([1], dtype=tf.float32), name='b_dn')
		else:
			self.b_dn = tf.Variable(tf.convert_to_tensor(self.b_dn_init, dtype=tf.float32), name='b_dn')

		# Initialize the alpha parameter in DivNorm
		if self.alpha_dn_init is None:
			self.alpha_dn = tf.Variable(tf.ones([1], dtype=tf.float32), name='alpha_dn')
		else:
			self.alpha_dn = tf.Variable(tf.convert_to_tensor(self.alpha_dn_init, dtype=tf.float32), name='alpha_dn')

		if self.is_dn:
			self.params.append(self.sigma_dn)
			self.params.append(self.alpha_dn)
			self.params.append(self.b_dn)

		self.betas = tf.transpose(self.lambdas_t, [0, 2, 1])

		betas = tf.reshape(tf.squeeze(self.betas,[1]), [self.K, self.Cin, self.h, self.w])
		latents_before_BN = tf.nn.conv2d(tf.transpose(self.data_4D_lab, [0,2,3,1]), tf.transpose(betas,[2,3,1,0]), strides=[1, 1, 1, 1], padding=self.border_mode)
		self.latents_before_BN_lab = tf.transpose(latents_before_BN, [0, 3, 1, 2])
		
		# self.betas = tf.transpose(self.lambdas_t, [0, 2, 1])
		# self.betas = tf.reshape(self.betas, shape=(self.h, self.w, self.Cin, self.K))
		# print self.betas.get_shape(),"here", self.prun_synap_mat

		# self.betas = self.betas * self.prun_synap_mat
		# self.latents_before_BN_lab = tf.nn.conv2d(self.data_4D_lab,self.betas, strides = [1,1,1,1], padding = "VALID", data_format = "NCHW")

	def get_important_latents_BU(self, input, betas):
		""""This function is used in the _E_step_Bottom_Up to compute latent representations
		
		Return:
			latents_before_BN: activations after convolutions
			latents: activations after BatchNorm/DivNorm
			max_over_a_mask: masking tensor results from ReLU
			max_over_t_mask: masking tensor results from max-pooling
			latents_masked: activations after BatchNorm/DivNorm masked by a and t
			masked_mat: max_over_t_mask * max_over_a_mask
			output: output of the layer, a.k.a. the downsampled activations
			mask_input: the input masked by the ReLU
			scale_s: the scale latent variable in DivNorm
			latents_demeaned: activations after convolutions whose means are removed
			latents_demeaned_squared: (activations after convolutions whose means are removed)^2

		"""
		# compute the activations after convolutions
		print "i am called"
		print "here",self.border_mode,input.get_shape(), betas.get_shape()
		self.betas111 = betas
		self.iiii = input
		if self.border_mode != 'FULL':
			latents_before_BN = tf.nn.conv2d(input,betas, strides = [1,1,1,1], padding = "VALID", data_format = "NCHW")
			# latents_before_BN = tf.nn.conv2d(, betas, strides=[1, 1, 1, 1], padding=self.border_mode)
			# latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])
		else:
			latents_before_BN = tf.nn.conv2d_transpose(tf.transpose(input, [0,2,3,1]), tf.transpose(betas,[2,3,0,1])[::-1,::-1,:,:], strides=[1, 1, 1, 1], padding="VALID",
													   output_shape=[self.latents_shape[0], self.latents_shape[2],
																	 self.latents_shape[3], self.latents_shape[1]])
			latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])
		self.latents_before_BN = latents_before_BN
		print latents_before_BN.get_shape()
		# do batch normalization or divisive normalization
		if self.is_bn_BU: # do batch normalization
			latents_after_BN = self.bn_BU.get_result(input=latents_before_BN, input_shape=self.latents_shape)
			scale_s = tf.ones(latents_before_BN.get_shape().as_list(), dtype = tf.float32)
			latents_demeaned = latents_before_BN
			latents_demeaned_squared = latents_demeaned ** 2
		elif self.is_dn: # do divisive normalization
			filter_for_norm_local = tf.Variable(tf.ones((1, self.K, self.h, self.w), dtype=tf.float32), name='filter_norm_local')

			sum_local = tf.nn.conv2d(
				data_format = "NCHW",
				strides = [1,1,1,1],
				input=latents_before_BN,
				filter=filter_for_norm_local,
				padding='half'
			)

			mean_local = sum_local/(self.K*self.h*self.w)

			latents_demeaned = latents_before_BN - tf.tile(mean_local, [1,self.K,1,1])

			latents_demeaned_squared = latents_demeaned**2

			norm_local = tf.nn.conv2d(
				data_format = "NCHW",
				strides = [1,1,1,1],
				input=latents_demeaned_squared,
				filter=filter_for_norm_local,
				padding='half'
			)
			norm_local_shp_ln = len(norm_local_shp.get_shape().as_list())
			multiples = [1]*norm_local_shp_ln
			multiples[1] *= self.K
			scale_s = (tf.reshape(tf.tile((self.alpha_dn + 1e-10), [self.K]),[1,-1,1,1])
					   + tf.tile(norm_local / (self.K * self.h * self.w), multiples)/((tf.reshape(tf.tile((self.sigma_dn + 1e-5), [self.K]),[1, -1, 1, 1])) ** 2)) / 2.


			latents_after_BN = (latents_demeaned / tf.sqrt(scale_s)) + tf.reshape(tf.tile(self.b_dn, self.K),[1,-1,1,1])

		else:
			latents_after_BN = latents_before_BN
			scale_s = tf.ones(latents_before_BN.get_shape())
			latents_demeaned = latents_before_BN
			latents_demeaned_squared = latents_demeaned ** 2

		print latents_after_BN.get_shape(), self.prun_mat.get_shape()
		latents = latents_after_BN * self.prun_mat # masking the activations by the neuron pruning mask.
													# self.prun_mat is all 1's if no neuron pruning

		mask_input = tf.cast(tf.greater(input, 0.), tf.float32) # find positive elements in the input

		# find activations survive after max over a
		if self.nonlin == 'relu':
			max_over_a_mask = tf.cast(tf.greater(latents, 0.), tf.float32)
		else:
			max_over_a_mask = tf.cast(tf.ones(latents.get_shape()), tf.float32)
		print "max_over_mask",max_over_a_mask.get_shape()
		# find activations survive after max over t
		if self.pool_t_mode == 'max_t' and self.nonlin == 'relu':
			# print('Do max over t')
			max_over_t_mask = tf.greater_equal(latents,
											   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents, [0,2,3,1]), [1, 2, 2, 1], strides=[1,2,2,1], padding='VALID'),
															[self.latents_shape[2], self.latents_shape[3]]),
															[0,3,1,2]))
			max_over_t_mask = tf.cast(max_over_t_mask, dtype=tf.float32)
		elif self.pool_t_mode == 'max_t' and self.nonlin == 'abs': # still in the beta state
			# print('Do max over t')
			latents_abs = tf.abs(latents)
			max_over_t_mask = tf.greater_equal(latents_abs,
											   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents_abs, [0, 2, 3, 1]), [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'),
												   [self.latents_shape[2], self.latents_shape[3]]),
															[0, 3, 1, 2]))
			max_over_t_mask = tf.cast(max_over_t_mask, dtype=tf.float32)
		else:
			# print('No max over t')
			# compute latents masked by a
			max_over_t_mask = tf.cast(tf.ones_like(latents), dtype=tf.float32)

		# mask the activations by t and a
		if self.nonlin == 'relu':
			latents_masked = tf.nn.relu(latents) * max_over_t_mask  # * max_over_a_mask
		elif self.nonlin == 'abs':
			latents_masked = tf.abs(latents) * max_over_t_mask
		else:
			latents_masked = latents * max_over_t_mask

		# find activations survive after max over a and t
		masked_mat = max_over_t_mask * max_over_a_mask  # * max_over_a_mask


		# downsample the activations
		if self.pool_t_mode == 'max_t':
			output = tf.nn.max_pool(latents_masked, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID")
			output = output * 4.0
		elif self.pool_t_mode == 'mean_t':
			output = tf.nn.avg_pool(latents_masked, ksize = [1,self.mean_pool_size[0],self.mean_pool_size[1],1], strides = [1,self.mean_pool_size[0],self.mean_pool_size[1],1], padding = "VALID")
			
		else:
			output = latents_masked

		return latents_before_BN, latents, max_over_a_mask, max_over_t_mask, latents_masked, masked_mat, output, mask_input, scale_s, latents_demeaned, latents_demeaned_squared


	def EBottomUp(self):
		"""
		E-step bottom-up infers the latents in the images
		"""

		# reshape lambdas_t into the filter

		self.betas = tf.transpose(self.lambdas_t, [0, 2, 1])
		betas = tf.reshape(self.betas[:, 0, :], shape=(self.h, self.w, self.Cin, self.K))
		print betas.get_shape(),"here", self.prun_synap_mat

		betas = betas * self.prun_synap_mat

		# run bottom up for labeled examples.
		if self.is_bn_BU:
			if self.data_4D_lab is not None and self.data_4D_unl_clean is None: # supervised training mode
				self.bn_BU.set_runmode(0)
			elif self.data_4D_lab is not None and self.data_4D_unl_clean is not None: # semisupervised training mode
				if self.update_mean_var_with_sup:
					self.bn_BU.set_runmode(0)
				else:
					self.bn_BU.set_runmode(1)

		[self.latents_before_BN_lab, self.latents_lab, self.max_over_a_mask_lab,
		 self.max_over_t_mask_lab, self.latents_masked_lab, self.masked_mat_lab, self.output_lab, self.mask_input_lab, self.scale_s_lab,
		 self.latents_demeaned_lab, self.latents_demeaned_squared_lab] \
			= self.get_important_latents_BU(input=self.data_4D_lab, betas=betas)


# __author__ = 'minhtannguyen'

# #######################################################################################################################
# # get rid of no rendering channel
# # do both soft and hard on c
# #######################################################################################################################

# import numpy as np

# import tensorflow as tf

# from utils import BatchNormalization


# # from lasagne.layers import InputLayer, FeatureWTALayer

# class Layer(object):
# 	"""
# 	EM Algorithm for the Convolutional Mixture of Factor Analyzers.

# 	All of the equation numbers in this code follows those in "The EM Algorithm for Mixtures of Factor Analyzers" by
# 	Zoubin Ghahramani and Geoffrey E. Hinton

# 	calling arguments:

# 	[TBA]

# 	internal variables:

# 	`K`:           Number of components
# 	`M`:           Latent dimensionality
# 	`D`:           Data dimensionality
# 	`N`:           Number of data points
# 	`data`:        (N,D) array of observations
# 	`latents`:     (K,M,N) array of latent variables
# 	`latent_covs`: (K,M,M,N) array of latent covariances
# 	`lambdas`:     (K,M,D) array of loadings
# 	`psis`:        (K,D) array of diagonal variance values
# 	`rs`:          (K,N) array of responsibilities
# 	`amps`:        (K) array of component amplitudes
# 	'data_4D: data in 4-D (N,D,H,W)

# 	"""

# 	def __init__(self, data_4D, 
# 				 data_4D_unl,
# 				 data_4D_unl_clean,
# 				 noise_weight,
# 				 noise_std,
# 				 K, M, W, H, w, h, Cin, Ni,
# 				 momentum_bn, is_train,
# 				 data_4D_clean=None,
# 				 lambdas_t_val_init=None, amps_val_init=None,
# 				 gamma_bn_init=None, beta_bn_init=None, mean_bn_init=None, var_bn_init=None,
# 				 PPCA=False, lock_psis=True,
# 				 em_mode='hard', layer_loc='intermediate',
# 				 pool_t_mode='max_t', border_mode='VALID', pool_a_mode='relu', nonlin='relu',
# 				 mean_pool_size=[1, 6, 6, 1],
# 				 rs_clip=0.0,
# 				 max_condition_number=1.e3,
# 				 init_ppca=False,
# 				 init_Bengio=False,
# 				 is_noisy=False,
# 				 is_bn_BU=False,
# 				 is_bn_TD=False,
# 				 epsilon=1e-10,
# 				 momentum_pi_t=0.99,
# 				 momentum_pi_a=0.99,
# 				 is_tied_bn=False,
# 				 is_Dg=False):

# 		## required
# 		self.K = K  # number of clusters
# 		self.M = M  # latent dimensionality
# 		self.data_4D = data_4D
# 		self.data_4D_clean = data_4D_clean
# 		# self.labels = labels
# 		self.Ni = Ni  # no. of images
# 		self.w = w  # width of filters
# 		self.h = h  # height of filters
# 		self.Cin = Cin  # number of channels in the image
# 		self.D = self.h * self.w * self.Cin  # patch size
# 		self.W = W  # width of image
# 		self.H = H  # height of image
# 		if border_mode == 'VALID':
# 			self.Np = (self.H - self.h + 1) * (self.W - self.w + 1)  # no. of patches per image
# 			self.latents_shape = (self.Ni, self.K, self.H - self.h + 1, self.W - self.w + 1)
# 		elif border_mode == 'SAME':
# 			self.Np = self.H * self.W  # no. of patches per image
# 			self.latents_shape = (self.Ni, self.K, self.H, self.W)
# 		elif border_mode == 'FULL':
# 			self.Np = (self.H + self.h - 1) * (self.W + self.w - 1)  # no. of patches per image
# 			self.latents_shape = (self.Ni, self.K, self.H + self.h - 1, self.W + self.w - 1)
# 		else:
# 			print('Please specify self.Np and self.latents_shape in CRM_no_factor_latest.py')

# 		self.N = self.Ni * self.Np  # total no. of patches and total no. of hidden units
# 		self.mean_pool_size = mean_pool_size

# 		# self.means_val_init = means_val_init
# 		self.lambdas_t_val_init = lambdas_t_val_init
# 		self.amps_val_init = amps_val_init
# 		self.gamma_bn_init = gamma_bn_init
# 		self.beta_bn_init = beta_bn_init
# 		self.mean_bn_init = mean_bn_init
# 		self.var_bn_init = var_bn_init

# 		# options
# 		self.em_mode = em_mode
# 		self.layer_loc = layer_loc
# 		self.pool_t_mode = pool_t_mode
# 		self.pool_a_mode = pool_a_mode
# 		self.nonlin = nonlin
# 		self.border_mode = border_mode
# 		self.PPCA = PPCA
# 		self.lock_psis = lock_psis
# 		self.rs_clip = rs_clip
# 		self.max_condition_number = max_condition_number
# 		self.init_Bengio = init_Bengio
# 		self.is_noisy = is_noisy
# 		self.is_bn_BU = is_bn_BU
# 		self.is_bn_TD = is_bn_TD
# 		self.momentum_bn = momentum_bn
# 		self.momentum_pi_t = momentum_pi_t
# 		self.momentum_pi_a = momentum_pi_a
# 		self.is_train = is_train
# 		self.epsilon = epsilon
# 		self.is_tied_bn = is_tied_bn
# 		self.is_Dg = is_Dg
# 		assert rs_clip >= 0.0

# 		self._initialize(init_ppca)

# 	def _initialize(self, init_ppca):
# 		#
# 		# initialize pi's, means, lambdas, psis, lambda_covs, covs, and inv_covs
# 		#

# 		# initialize the pi's (a.k.a the priors)
# 		# if initial values for pi's are not provided, randomly initialize pi's

# 		if self.amps_val_init == None:
# 			amps_val = np.random.rand(self.K)
# 			amps_val /= np.sum(amps_val)

# 		else:
# 			amps_val = self.amps_val_init

# 		self.amps = tf.Variable(amps_val, name='amps', dtype=tf.float32)

# 		self.pi_t = tf.Variable(tf.zeros(self.latents_shape[1:]),name='pi_t', dtype=tf.float32)

# 		self.pi_a = tf.Variable(tf.zeros(self.latents_shape[1:]), name='pi_a', dtype=tf.float32)

# 		self.pi_t_final = tf.Variable(tf.zeros(self.latents_shape[1:]), name='pi_t_final', dtype=tf.float32)

# 		self.pi_a_final = tf.Variable(tf.zeros(self.latents_shape[1:]), name='pi_a_final', dtype=tf.float32)

# 		# initialize the lambdas
# 		# if initial values for lambdas are not provided, randomly initialize lambdas
# 		if self.lambdas_t_val_init == None:
# 			if self.init_Bengio:
# 				print('Do init_Bengio')
# 				fan_in = self.D
# 				if self.pool_t_mode == None:
# 					fan_out = self.K * self.h * self.w
# 				else:
# 					fan_out = self.K * self.h * self.w / 4

# 				lambdas_bound = np.sqrt(6. / (fan_in + fan_out))
# 				lambdas_value = np.random.uniform(low=-lambdas_bound, high=lambdas_bound, size=(self.K, self.D, self.M))
# 			else:
# 				lambdas_value = np.random.randn(self.K, self.D, self.M) / \
# 								np.sqrt(self.max_condition_number)
# 		else:
# 			lambdas_value = self.lambdas_t_val_init

# 		self.lambdas_t = tf.Variable(np.asarray(lambdas_value), name='lambdas', dtype=tf.float32)

# 		if self.is_bn_BU:
# 			print('do batch_normalization in Bottom Up')
# 			self.bn_BU = BatchNormalization(insize=self.K, mode=1, momentum=self.momentum_bn, is_train=self.is_train,
# 											epsilon=self.epsilon,
# 											gamma_init=self.gamma_bn_init,
# 											beta_init=self.beta_bn_init,
# 											mean_init=self.mean_bn_init,
# 											var_init=self.var_bn_init)
# 			self.params = [self.lambdas_t, self.bn_BU.gamma, self.bn_BU.beta]
# 		else:
# 			self.params = [self.lambdas_t, ]

# 		if self.is_Dg:
# 			print('use Dg')
# 			self.Dg = tf.Variable(tf.ones((self.K)), name='Dg', dtype=tf.float32)
# 			self.dg = tf.Variable(tf.zeros((self.K)), name='dg', dtype=tf.float32)
# 			self.params.append(self.Dg)
# 			self.params.append(self.dg)

		# self.betas = tf.transpose(self.lambdas_t, [0, 2, 1])

		# betas = tf.reshape(tf.squeeze(self.betas,[1]), [self.K, self.Cin, self.h, self.w])

# 		# self.betas = tf.transpose(self.lambdas_t, [0, 2, 1])
# 		# self.betas = tf.reshape(self.betas, shape=(self.h, self.w, self.Cin, self.K))
# 		# print self.betas.get_shape(),"here", self.prun_synap_mat

# 		# self.betas = self.betas * self.prun_synap_mat
		# latents_before_BN = tf.nn.conv2d(tf.transpose(self.data_4D, [0,2,3,1]), tf.transpose(betas,[2,3,1,0]), strides=[1, 1, 1, 1], padding=self.border_mode)
		# self.latents_before_BN_lab = tf.transpose(latents_before_BN, [0, 3, 1, 2])
		# # self.latents_before_BN_lab = tf.nn.conv2d(self.data_4D,self.betas, strides = [1,1,1,1], padding = "VALID", data_format = "NCHW")

# 	def get_important_latents_BU(self, input, betas):
# 		# compute E[z|x] using eq. 13

# 		if self.border_mode != 'FULL':
# 			latents_before_BN = tf.nn.conv2d(tf.transpose(input, [0,2,3,1]), tf.transpose(betas,[2,3,1,0]), strides=[1, 1, 1, 1], padding=self.border_mode)
# 			latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])
# 		else:
# 			latents_before_BN = tf.nn.conv2d_transpose(tf.transpose(input, [0,2,3,1]), tf.transpose(betas,[2,3,0,1])[::-1,::-1,:,:], strides=[1, 1, 1, 1], padding="VALID",
# 													   output_shape=[self.latents_shape[0], self.latents_shape[2],
# 																	 self.latents_shape[3], self.latents_shape[1]])
# 			latents_before_BN = tf.transpose(latents_before_BN, [0, 3, 1, 2])

# 		# do Batch normalization
# 		if self.is_bn_BU:
# 			latents = self.bn_BU.get_result(input=latents_before_BN, input_shape=self.latents_shape)
# 		elif self.is_Dg: # still in the beta state
# 			latents = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.Dg, 0),2),3) * \
# 					  (latents_before_BN - tf.expand_dims(tf.expand_dims(tf.expand_dims(self.Dg, 0),2),3))
# 		else:
# 			latents = latents_before_BN

# 		# max over a
# 		if self.pool_a_mode == 'relu':
# 			print('Do max over a')
# 			max_over_a_mask = tf.cast(tf.greater(latents, 0.), dtype=tf.float32)
# 		else:
# 			print('No max over a')
# 			max_over_a_mask = tf.cast(tf.ones_like(latents), dtype=tf.float32)

# 		# max over t
# 		if self.pool_t_mode == 'max_t' and self.nonlin == 'relu':
# 			print('Do max over t')
# 			max_over_t_mask = tf.greater_equal(latents,
# 											   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents, [0,2,3,1]), [1, 2, 2, 1], strides=[1,2,2,1], padding='VALID'),
# 															[self.latents_shape[2], self.latents_shape[3]]),
# 															[0,3,1,2]))
# 			max_over_t_mask = tf.cast(max_over_t_mask, dtype=tf.float32)
# 		elif self.pool_t_mode == 'max_t' and self.nonlin == 'abs': # still in the beta state
# 			print('Do max over t')
# 			latents_abs = tf.abs(latents)
# 			max_over_t_mask = tf.greater_equal(latents_abs,
# 											   tf.transpose(tf.image.resize_nearest_neighbor(tf.nn.max_pool(tf.transpose(latents_abs, [0, 2, 3, 1]), [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'),
# 												   [self.latents_shape[2], self.latents_shape[3]]),
# 															[0, 3, 1, 2]))
# 			max_over_t_mask = tf.cast(max_over_t_mask, dtype=tf.float32)
# 		else:
# 			print('No max over t')
# 			# compute latents masked by a
# 			max_over_t_mask = tf.cast(tf.ones_like(latents), dtype=tf.float32)

# 		# compute latents masked by a and t
# 		if self.nonlin == 'relu':
# 			print('Nonlinearity is ReLU')
# 			latents_masked = latents * max_over_t_mask * max_over_a_mask  # * max_over_a_mask
# 		elif self.nonlin == 'abs':
# 			print('Nonlinearity is abs')
# 			latents_masked = tf.abs(latents) * max_over_t_mask
# 		else:
# 			print('Please specify your nonlin in CRM_no_factor_latest')
# 			raise

# 		masked_mat = max_over_t_mask * max_over_a_mask  # * max_over_a_mask

# 		# latents_rs = self.latents[:,0,:] * self.rs
# 		if self.layer_loc == 'intermediate':
# 			output_before_pool = latents_masked
# 		else:
# 			print('Please specify your output in CRM_no_factor_latest.py')

# 		if self.pool_t_mode == 'max_t':
# 			output = tf.nn.avg_pool(tf.transpose(output_before_pool, [0,2,3,1]), [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# 			output = output * 4.0
# 			output = tf.transpose(output, [0,3,1,2])
# 		elif self.pool_t_mode == 'mean_t':
# 			output = tf.nn.avg_pool(tf.transpose(output_before_pool, [0,2,3,1]), self.mean_pool_size, strides=self.mean_pool_size, padding='VALID')
# 			output = tf.transpose(output, [0, 3, 1, 2])
# 		else:
# 			output = output_before_pool

# 		return latents_before_BN, latents, max_over_a_mask, max_over_t_mask, latents_masked, masked_mat, output

# 	def _E_step_Bottom_Up(self):
# 		"""
# 		Expectation step through all clusters.
# 		Compute responsibilities, likelihoods, lambda dagger, latents, latent_covs, pi's
# 		"""

# 		# Bottom-Up

# 		# compute the lambda dagger
# 		self.betas = tf.transpose(self.lambdas, [0, 2, 1])

# 		betas = tf.reshape(tf.squeeze(self.betas,[1]), [self.K, self.Cin, self.h, self.w])

# 		# Batch Normalization
# 		if self.is_noisy:
# 			print('Compute the clean path since the clean path is not the same as the noisy path')
# 			if self.is_bn_BU:
# 				self.bn_BU.set_runmode(1)

# 			[self.latents_before_BN, self.latents, self.max_over_a_mask, self.max_over_t_mask, self.latents_masked, self.masked_mat,
# 			 self.output] \
# 				= self.get_important_latents_BU(input=self.data_4D, betas=betas)

# 			if self.is_bn_BU:
# 				self.bn_BU.set_runmode(0) # define constant name for run mode

# 			[self.latents_before_BN_clean, self.latents_clean, self.max_over_a_mask_clean, self.max_over_t_mask_clean, self.latents_masked_clean,
# 			 self.masked_mat_clean, self.output_clean] \
# 				= self.get_important_latents_BU(input=self.data_4D_clean, betas=betas)

# 		else:
# 			print('The clean path is the same as the noisy path')
# 			if self.is_bn_BU:
# 				self.bn_BU.set_runmode(0)

# 			[self.latents_before_BN, self.latents, self.max_over_a_mask, self.max_over_t_mask, self.latents_masked, self.masked_mat,
# 			 self.output] \
# 				= self.get_important_latents_BU(input=self.data_4D, betas=betas)
# 			self.output_clean = self.output

# 		self.pi_t_minibatch = tf.reduce_mean(self.max_over_t_mask, 0)
# 		self.pi_a_minibatch = tf.reduce_mean(self.max_over_a_mask, 0)

# 		self.pi_t_new = self.momentum_pi_t*self.pi_t + (1 - self.momentum_pi_t)*self.pi_t_minibatch
# 		self.pi_a_new = self.momentum_pi_a*self.pi_a + (1 - self.momentum_pi_a)*self.pi_a_minibatch

# 		self.rs = self.masked_mat
# 		self.logLs = 0.5 * tf.reduce_sum(self.masked_mat, 1)

# 		self.amps_new = tf.reduce_sum(self.masked_mat, [0,2,3])/float(self.N/4)
# 		################################################################################################################

# 		# IF DO M STEP, REMEMBER TO RESHAPE SELF.LATENTS TO (K, M, N)