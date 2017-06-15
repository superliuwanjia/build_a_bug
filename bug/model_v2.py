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
    def __init__(self, inp_size, out_size, conv_layers, lrcn_layers, num_classes = 6, lstm_hidden_units = 84): # num_classes set to no of classes in kth dataset
        self.x = tf.placeholder(tf.float32, inp_size)
        self.Y = tf.placeholder(tf.float32, out_size)
        self.lr = tf.placeholder(tf.float32, [])
        self.inp_size = inp_size
        self.out_size = out_size
        self.conv_layers = conv_layers
        self.lrcn_layers = lrcn_layers
        self.num_classes  = num_classes
        self.hidden_units = 84
        self.get_retina_output(tf.reshape(self.x,[inp_size[0], inp_size[1], -1]))
        self.get_conv_output()

        self.get_lrcn_output()
        self.get_softmax()
        self.get_loss()

    def get_retina_output(self, inp):
        with tf.name_scope("retina"):
            retina = Retina(inp, self.inp_size)     
            print retina.get_output()
            self.retina_output = tf.reshape(retina.get_output(), 
                                            [self.inp_size[0]*self.inp_size[1],self.inp_size[2],self.inp_size[3], self.inp_size[4]*2])


    def get_conv_output(self, ):
        with tf.name_scope("convnet"):
            with tf.variable_scope("convnet"):
                lenet = Lenet(self.retina_output)
                self.convnet_out = lenet.output
                self.convnet_out_sh = self.convnet_out.get_shape().as_list()[1:]
                

    
    def get_lrcn_output(self):
        for layer_no in range(self.lrcn_layers):
            with tf.variable_scope("lrcn"+str(layer_no)):
                conv_out_shape = self.convnet_out.get_shape().as_list()
                channels = conv_out_shape[-1]
                inp = tf.reshape(self.convnet_out, [self.inp_size[0], self.inp_size[1],-1, channels])
                inp_shape = inp.get_shape().as_list()
                inp_split = tf.split(inp, channels, axis = 3)
                outputs = []
                for channel in range(channels):
                    with tf.variable_scope("channel_lstm"+str(channel)):
                        lstm_inp = tf.reshape(inp_split[channel], inp_shape[:-1])
                        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units)
                        out, state = tf.nn.dynamic_rnn(lstm_cell, lstm_inp, dtype = tf.float32)                     

                        outputs.append(tf.reshape(out[:,-1,:],[self.inp_size[0], self.hidden_units, 1]))
                outputs = tf.concat(outputs, axis = 2)
                self.output = tf.reshape(outputs,[self.inp_size[0], -1])



    

    def get_softmax(self):
        lrcn_out_sh = self.output.get_shape().as_list()
        with tf.variable_scope("softmax_layer"):
            W = tf.get_variable("W",[lrcn_out_sh[1], self.num_classes], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

            b = tf.get_variable("b",[self.num_classes], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.softmax = tf.nn.softmax(tf.add(tf.matmul(self.output, W), b))
            self.argmax_output = tf.argmax(self.softmax, 1)


    def get_loss(self):
        with tf.variable_scope("loss_optimization"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Y, logits = self.softmax))
            tf.summary.scalar("loss", self.loss)
            self.classification_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.Y, axis = 1), tf.argmax(self.softmax, axis = 1)), tf.float32))/self.inp_size[0]
            grads = tf.gradients(self.loss, tf.trainable_variables())
            self.grads = list(zip(grads, tf.trainable_variables()))
            # Op to update all variables according to their gradient
            self.train_ops = optimizer.apply_gradients(grads_and_vars=self.grads)
            # self.train_ops = optimizer.minimize(self.loss)
