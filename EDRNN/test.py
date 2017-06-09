import tensorflow as tf
import numpy as np
import os
import argparse

from rnn_cell import EMAcell
from retina import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


retina = Retina([2,5,4], mu_off = 0.01, mu_on = 0.01)
#testing the cell
# x = tf.placeholder(tf.float32, [2,5,4])
# cell = EMAcell(4)
# outputs, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)

x_i = np.random.rand(2,5,4)
# x_i = np.ones((2,5,4))
print x_i
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	feed_dict = {
				retina.input : x_i
	}
	ema = sess.run([retina.i_hat_sh, retina.i_hat_md, retina.i_hat_lg, retina.out], feed_dict = feed_dict)

for i in ema:
	print i
