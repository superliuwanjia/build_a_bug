import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import sys
import multiprocessing
# sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmm_tensorflow')
# from layer import *

class Convnet():
    """docstring for Convnet"""
    def __init__(self, num_layers, strides, filtersizes, padding, name = "conv_filters", pool_strides = None,  pool_pad = None):
        self.num_layers = num_layers
        self.strides = strides
        self.filtersizes = filtersizes
        self.padding = padding
        self.name = name
        self.pool_strides = pool_strides
        self.pool_pad = pool_pad

    def build(self, input):
        out = input

        for layer_num in range(self.num_layers):
            with tf.variable_scope("conv_layer_"+str(layer_num)):
                filter_var = tf.get_variable("filterss"+str(layer_num), self.filtersizes[layer_num], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
                out = tf.nn.relu(tf.nn.conv2d(out, filter_var, self.strides[layer_num], self.padding[layer_num]))               
                if self.pool_strides:
                    out = tf.nn.max_pool(out, [1,self.pool_strides[layer_num][0],self.pool_strides[layer_num][0],1], [1,self.pool_strides[layer_num][1],self.pool_strides[layer_num][1],1], self.pool_pad[layer_num])
                if self.filtersizes[layer_num][-1] < 4:
                    tf.summary.image(self.name, filter_var)
                # out = 
        self.out = out

    def get_output(self):
        return self.out

# def convolution(input, strides, filtersizes, padding, name = "conv_filters"):
#   out = input 
#   with tf.variable_scope("conv_layer_"+name):

#       filter_var = tf.get_variable("filterss"+name, filtersizes[0], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
#       out = tf.nn.relu(tf.nn.conv2d(out, filter_var, strides[0], padding[0]))             
#       tf.summary.image(name, filter_var)
#   return out

        
    
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
            
            self.i_gate_vars_bias = tf.get_variable("i_bias", [num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.f_gate_vars_bias = tf.get_variable("f_bias", [num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.o_gate_vars_bias = tf.get_variable("o_bias", [num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.g_gate_vars_bias = tf.get_variable("w_bias", [num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            # Hidden state variables
            self.i_hidden_gate_vars = tf.get_variable("i_hidden_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.f_hidden_gate_vars = tf.get_variable("f_hidden_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.o_hidden_gate_vars = tf.get_variable("o_hidden_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.g_hidden_gate_vars = tf.get_variable("w_hidden_weights", [hidden_units/num_channels, num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

            # self.i_hidden_gate_vars_bias = tf.get_variable("i_hidden_bias", [num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            # self.f_hidden_gate_vars_bias = tf.get_variable("f_hidden_bias", [num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            # self.o_hidden_gate_vars_bias = tf.get_variable("o_hidden_bias", [num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            # self.g_hidden_gate_vars_bias = tf.get_variable("w_hidden_bias", [num_channels, hidden_units/num_channels], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            

    @property
    def state_size(self):
        return self._hidden_units

    @property
    def output_size(self):
        return self._hidden_units
    
    


    def __call__(self, input, state, scope = None):
        
        with tf.variable_scope(self.scope):
            h, c = tf.split(state,2)
            # print h.get_shape()
            inp = tf.reshape(input, [self.batch_size, -1, self.num_channels])
            h = tf.reshape(h, inp.get_shape())
            inp_sh = inp.get_shape().as_list()

            c = tf.reshape(c, [inp_sh[0], inp_sh[2], inp_sh[1]])
            # print h.get_shape(), inp.get_shape(), self.i_hidden_gate_vars.get_shape()
            i = tf.sigmoid(tf.einsum("aij,ijk->ajk", inp, self.i_gate_vars, ) + tf.einsum("aij,ijk->ajk", h, self.i_hidden_gate_vars) + self.i_gate_vars_bias )
            f = tf.sigmoid(tf.einsum("aij,ijk->ajk", inp, self.f_gate_vars, ) + tf.einsum("aij,ijk->ajk", h, self.f_hidden_gate_vars) + self.f_gate_vars_bias )
            o = tf.sigmoid(tf.einsum("aij,ijk->ajk", inp, self.o_gate_vars, ) + tf.einsum("aij,ijk->ajk", h, self.o_hidden_gate_vars) + self.o_gate_vars_bias )
            g = tf.nn.relu(tf.einsum("aij,ijk->ajk", inp, self.i_gate_vars, ) + tf.einsum("aij,ijk->ajk", h, self.i_hidden_gate_vars) + self.g_gate_vars_bias )
            # print g.get_shape(), self._hidden_units
            new_c = c*f + i*g
            print  new_c.get_shape(),"c"
            print (o*tf.nn.relu(new_c)).get_shape(), "h"

            new_h = tf.reshape(o*tf.nn.relu(new_c), [self.batch_size, self._hidden_units])
            new_c = tf.reshape(new_c, [self.batch_size, self._hidden_units])
            # print new_h.get_shape()
            # print "ho"
            return new_h, tf.concat([new_h, new_c], axis = 0)
        
class Worker():
    """docstring for Worker"""
    def __init__(self,):
        pass
        