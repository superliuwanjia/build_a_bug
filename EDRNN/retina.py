import tensorflow as tf
from rnn_cell import EMAcell


class Retina():
	"""docstring for Retina"""
	def __init__(self, inp_size, beta = [0.5, 0.7, 0.9], alphas = [1.5, 2.0, 2.5], mu_off = None, mu_on = None):
		self.inp_size = inp_size
		# self.beta = tf.Variable(tf.random_normal[])
		self.beta = beta
		self.alphas = alphas
		self.mu_off = mu_off
		self.mu_on = mu_on
		self.hidden_units = inp_size[2]
		# self.beta_sl = beta_sh
		# self.beta_sl = beta_med
		# self.beta_lg = beta_lg
		self.input = tf.placeholder(tf.float32, self.inp_size)

		self.get_ema()
		self.get_rel_changes()
		self.threshold()
		

	def get_ema(self):

		ema_cell_sh = EMAcell(self.hidden_units, self.alphas[0])
		state_sh = self.input[:,0,:]
		outputs_sh, state_sh = tf.nn.dynamic_rnn(ema_cell_sh, self.input, initial_state = state_sh)
		self.i_hat_sh = outputs_sh	

		ema_cell_md = EMAcell(self.hidden_units, self.alphas[1])
		state_md = self.input[:,0,:]
		outputs_md, state_md = tf.nn.dynamic_rnn(ema_cell_md, self.input, initial_state = state_md)
		self.i_hat_md = outputs_md
		

		ema_cell_lg = EMAcell(self.hidden_units, self.alphas[2])
		state_lg = self.input[:,0,:]
		outputs_lg, state_lg = tf.nn.dynamic_rnn(ema_cell_lg, self.input, initial_state = state_lg)
		self.i_hat_lg = outputs_lg
		

	def get_rel_changes(self):
		self.r_x = self.beta[0]*tf.log(tf.divide(self.input, self.i_hat_sh)) + self.beta[1]*tf.log(tf.divide(self.input, self.i_hat_md))\
					+ self.beta[2]*tf.log(tf.divide(self.input, self.i_hat_lg))

		

	def threshold(self):
		e_on = tf.nn.relu(self.r_x - (1 + self.mu_on))
		e_off = tf.nn.relu(self.r_x - (1 - self.mu_off))
		self.out = tf.concat([e_on, e_off], axis = 2)


	def get_output(self):
		return self.out




		