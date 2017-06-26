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
from lenet import *

class Bug():
    """docstring for Bug"""
    def __init__(self, inp_size, out_size, conv_layers, lrcn_layers, num_classes = 6, lstm_hidden_units = 84, is_edr = True): # num_classes set to no of classes in kth dataset
        self.x = tf.placeholder(tf.float32, inp_size)
        self.Y = tf.placeholder(tf.float32, out_size)
        self.lr = tf.placeholder(tf.float32, [])
        self.inp_size = inp_size
        
        if is_edr:
            print "using event driven representation"
            self.get_retina_output(tf.reshape(self.x,[inp_size[0], inp_size[1], -1]))
        else:
            print "using frame representation"
            self.retina_output = tf.reshape(self.x, 
                                [self.inp_size[0]*self.inp_size[1],self.inp_size[2],self.inp_size[3], self.inp_size[4]])
        
        self.get_conv_output()

    def get_retina_output(self, inp):
        with tf.name_scope("retina"):
            retina = Retina(inp, self.inp_size)     
            self.retina_output = tf.reshape(retina.get_output(), 
                                            [self.inp_size[0]*self.inp_size[1],self.inp_size[2],self.inp_size[3], self.inp_size[4]*2])


    def get_conv_output(self, conv_type = "lenet"):
        with tf.name_scope("convnet"):
            with tf.variable_scope("convnet"):
                if conv_type == "lenet":
                    print "lenet conv type chosen ..."
                    lenet = Lenet(self.retina_output)
                    self.convnet_out = lenet.output
                else:
                    print "nin conv type chosen ..."
                    nin = NiN(self.retina_output)
                    self.convnet_out = nin.output
                self.convnet_out_sh = self.convnet_out.get_shape().as_list()[1:]
                

    
    