from tensorflow.contrib.rnn import RNNCell



class EMAcell(RNNCell):
	"""docstring for EMA"""
	# def __init__(self, inp_size, tao = 1.5):
	def __init__(self, inp_size, alpha = 1.5):
		self._input_size = inp_size
		self.alpha = alpha
		# self.alpha = 1.0/tao

	@property
	def state_size(self):
		return self._input_size

	@property
	def output_size(self):
		return self._input_size

	def __call__(self, input, state, scope = None):
		i_hat = (1-self.alpha)*state + self.alpha*input

		return i_hat, i_hat
	
