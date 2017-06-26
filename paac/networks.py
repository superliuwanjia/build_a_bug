import tensorflow as tf
import logging
import numpy as np

import sys
import os 

# from model_v2 import *

sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmm_tensorflow')
sys.path.insert(0, '/home/rgoel/drmm/EDRNN')
sys.path.insert(0, '/home/rgoel/drmm/bug')
# sys.path.insert(0, '/home/rgoel/drmm/bug')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from retina import *
from dataLoader import *

def flatten(_input):
    shape = _input.get_shape().as_list()
    dim = shape[1]*shape[2]*shape[3]
    return tf.reshape(_input, [-1,dim], name='_flattened')


def conv2d(name, _input, filters, size, channels, stride, padding = 'VALID', init = "torch"):
    w = conv_weight_variable([size,size,channels,filters],
                             name + '_weights', init = init)
    b = conv_bias_variable([filters], size, size, channels,
                           name + '_biases', init = init)
    conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1],
                        padding=padding, name=name + '_convs')
    out = tf.nn.relu(tf.add(conv, b),
                     name='' + name + '_activations')
    return w, b, out


def conv_weight_variable(shape, name, init = "torch"):
    if init == "glorot_uniform":
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        w = shape[0]
        h = shape[1]
        input_channels = shape[2]
        d = 1.0 / np.sqrt(input_channels * w * h)

    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def conv_bias_variable(shape, w, h, input_channels, name, init= "torch"):
    if init == "glorot_uniform":
        initial = tf.zeros(shape)
    else:
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def fc(name, _input, output_dim, activation = "relu", init = "torch"):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim],
                           name + '_weights', init = init)
    b = fc_bias_variable([output_dim], input_dim,
                         '' + name + '_biases', init = init)
    out = tf.add(tf.matmul(_input, w), b, name= name + '_out')

    if activation == "relu":
        out = tf.nn.relu(out, name='' + name + '_relu')

    return w, b, out


def fc_weight_variable(shape, name, init="torch"):
    if init == "glorot_uniform":
        fan_in = shape[0]
        fan_out = shape[1]
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def fc_bias_variable(shape, input_channels, name, init= "torch"):
    if init=="glorot_uniform":
        initial = tf.zeros(shape, dtype='float32')
    else:
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name + '_weights')
    b = fc_bias_variable([output_dim], input_dim, name + '_biases')
    out = tf.nn.softmax(tf.add(tf.matmul(_input, w), b), name= name + '_policy')
    return w, b, out


def log_softmax( name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name + '_weights')
    b = fc_bias_variable([output_dim], input_dim, name + '_biases')
    out = tf.nn.log_softmax(tf.add(tf.matmul(_input, w), b), name= name + '_policy')
    return w, b, out


class Network(object):

    def __init__(self, conf):

        self.name = conf['name']
        self.num_actions = conf['num_actions']
        self.clip_norm = conf['clip_norm']
        self.clip_norm_type = conf['clip_norm_type']
        self.device = conf['device']

        with tf.device(self.device):
            with tf.name_scope(self.name):
                self.loss_scaling = 5.0
                self.input_ph = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='input')
                self.selected_action_ph = tf.placeholder("float32", [None, self.num_actions], name="selected_action")
                self.input = tf.scalar_mul(1.0/255.0, tf.cast(self.input_ph, tf.float32))

                # This class should never be used, must be subclassed

                # The output layer
                self.output = None

    def init(self, checkpoint_folder, saver, session):
        last_saving_step = 0

        with tf.device('/cpu:0'):
            # Initialize network parameters
            path = tf.train.latest_checkpoint(checkpoint_folder)
            if path is None:
                logging.info('Initializing all variables')
                session.run(tf.global_variables_initializer())
            else:
                logging.info('Restoring network variables from previous run')
                saver.restore(session, path)
                last_saving_step = int(path[path.rindex('-')+1:])
        return last_saving_step


# class NIPSNetwork(Network):

#     def __init__(self, conf):
#         super(NIPSNetwork, self).__init__(conf)
#         print("NipsNetwork selected, input shape",self.input.get_shape())
#         self.output_retina = self.input
#         with tf.device(self.device):
#             with tf.name_scope(self.name):
#                 _, _, conv1 = conv2d('conv1', self.input, 16, 8, 4, 4)

#                 _, _, conv2 = conv2d('conv2', conv1, 32, 4, 16, 2)

#                 _, _, fc3 = fc('fc3', flatten(conv2), 256, activation="relu")

#                 self.output = fc3


class NatureNetwork(Network):

    def __init__(self, conf):
        super(NatureNetwork, self).__init__(conf)
        print("NatureNetwork selected, input shape",self.input.get_shape())
        with tf.device(self.device):
            with tf.name_scope(self.name):
                _, _, conv1 = conv2d('conv1', self.input, 32, 8, 4, 4)

                _, _, conv2 = conv2d('conv2', conv1, 64, 4, 32, 2)

                _, _, conv3 = conv2d('conv3', conv2, 64, 3, 64, 1)

                _, _, fc4 = fc('fc4', flatten(conv3), 512, activation="relu")

                self.output = fc4

# def conv2d(name, _input, filters, size, channels, stride, padding = 'VALID', init = "torch"):
# class EDRNN(Network):
# class NIPSNetwork(Network):
#     """docstring for EDRNN"""
#     def __init__(self, conf):
#         # super(EDRNN, self).__init__(conf)
#         super(NIPSNetwork, self).__init__(conf)
#         print("ho")
#         self.output_viz_folder = "/home/rgoel/atatri/paac/visualisation"
#         with tf.device(self.device):
#             with tf.name_scope(self.name):
#                 inp_shape = [1] + self.input.get_shape().as_list()
#                 print(inp_shape)
#                 inp = tf.reshape(self.input, [1, -1,  inp_shape[3]* inp_shape[2]*inp_shape[4]])
#                 retina = Retina(inp, inp_shape)
#                 inp = tf.reshape(retina.get_output(),[-1, inp_shape[2], inp_shape[3], inp_shape[4]*2])
#                 self.output_retina = inp
#                 # print(inp.get_shape())
#                 # print(hi)
#                 _, _, conv1 = conv2d('conv1', inp, 32, 8, 8, 4)
#                 print(conv1.get_shape(),"conv1")
#                 _, _, conv2 = conv2d('conv2', conv1, 64, 4, 32, 2)
#                 print(conv2.get_shape(),"conv2")
#                 _, _, conv3 = conv2d('conv3', conv2, 64, 3, 64, 1)
#                 print(conv3.get_shape(),"conv3")
#                 _, _, fc4 = fc('fc4', flatten(conv3), 512, activation="relu")
#                 # print(hi)
#                 self.output = fc4


class NIPSNetwork(Network):
    """docstring for EDRNN"""
    def __init__(self, conf):
        # super(EDRNN, self).__init__(conf)
        super(NIPSNetwork, self).__init__(conf)
        print("ho")
        self.output_viz_folder = "/home/rgoel/atatri/paac/visualisation"
        with tf.device(self.device):
            with tf.name_scope(self.name):
                inp_shape = self.input.get_shape().as_list()
                print(inp_shape)
                inp = tf.reshape(tf.transpose(self.input,[0,3,1,2]), [-1, inp_shape[3],  inp_shape[1]* inp_shape[2]])
                retina = Retina(inp, inp_shape)
                # inp = tf.reshape(retina.get_output(),[-1, inp_shape[2], inp_shape[3], inp_shape[4]*2])
                inp = retina.get_output()
                # try:
                #     self.output_retina = tf.reshape(tf.transpose(inp,[0,3,1,2]), [inp_shape[3], inp_shape[1], inp_shape[2]])
                # except Exception:
                    
                self.output_retina = inp
                # print(inp.get_shape())
                # print(hi)
                _, _, conv1 = conv2d('conv1', inp, 32, 8, 8, 4)
                print(conv1.get_shape(),"conv1")
                _, _, conv2 = conv2d('conv2', conv1, 64, 4, 32, 2)
                print(conv2.get_shape(),"conv2")
                _, _, conv3 = conv2d('conv3', conv2, 64, 3, 64, 1)
                print(conv3.get_shape(),"conv3")
                _, _, fc4 = fc('fc4', flatten(conv3), 512, activation="relu")
                # print(hi)
                self.output = fc4
        