import unittest
# import tenosrflow as tf
# import theano

import numpy as np
# from .. layer import *
from .. import utils
class FunctTest(unittest.TestCase):
	"""docstring for LayerTest"""
	def test_BN(self):
		pass
	def test_BuildCost(self):
		pass
	def test_DN(self):
		pass
	def test_HiddenLayerSoftmax(self):
		pass
	def test_SoftmaxNonlinearity(self):
		pass
	
if __name__ == "__main__":
	# unittest.main()
	print np.random.rand(100,8,8,3).shape