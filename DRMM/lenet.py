import tensorflow as tf
import numpy as np

class Lenet():
    """docstring for Lenet"""
    # def __init__(self, inp, num_layers = 2, conv_filters = [[5, 5, 3, 6], [5, 5, 6, 16]], conv_strides = [[1,1,1,1], [1,1,1,1]], conv_pads = ["VALID", "VALID"],
    def __init__(self, inp, num_layers = 2, conv_filters = [[5, 5, 2, 6], [5, 5, 6, 16]], conv_strides = [[1,1,1,1], [1,1,1,1]], conv_pads = ["VALID", "VALID"],
                pool_ksizes = [[1, 2, 2, 1], [1, 2, 2, 1]], pool_strides = [[1,2,2,1], [1,2,2,1]], pool_pads = ["VALID", "VALID"]):
        max_pool = inp
        with tf.variable_scope("lenet"):
            for layer_num in range(num_layers):
                filter_var = tf.get_variable("filterss"+str(layer_num), conv_filters[layer_num], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable("bias"+str(layer_num), conv_filters[layer_num][-1], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
                out = tf.nn.relu(tf.nn.conv2d(max_pool, filter_var, conv_strides[layer_num], conv_pads[layer_num]) + bias)
                max_pool = tf.nn.max_pool(out, ksize = pool_ksizes[layer_num], strides = pool_strides[layer_num], padding = pool_pads[layer_num])
                if conv_filters[layer_num][-1] < 4:
                    tf.summary.image(self.name, filter_var)
        self.output = max_pool

class NiN():
  """docstring for NiN"""
  def __init__(self, inp, num_layers = 3, conv_filters = [[[5,5,6,192],[1,1,192,160],[1,1,160,96]], [[5,5,96,192],[1,1,192,192],[1,1,192,192]], [[3,3,192,192],[1,1,192,192]]],
                conv_strides = [[[1,1,1,1],[1,1,1,1],[1,1,1,1]], [[1,1,1,1],[1,1,1,1],[1,1,1,1]], [[1,1,1,1],[1,1,1,1]]], 
                pad_mode = "VALID",
                conv_pads = [[[2,2],None,None],[[2,2], None, None],[[1,1],None]],
                pool_ksizes = [[1, 3, 3, 1], [1, 3, 3, 1], [1, 2, 2, 1]], 
                pool_strides = [[1,2,2,1], [1,2,2,1], [1, 2, 2, 1]], 
                pool_pads = ["VALID", "VALID", "VALID"],
                fcn_= None):
    out = tf.convert_to_tensor(inp, tf.float32)
    with tf.variable_scope("NiN"):
        for layer_num in range(num_layers):
            conv1_out = self._block(out, conv_filters[layer_num][0], conv_strides[layer_num][0], conv_pads[layer_num][0], pad_mode,str(layer_num)+"_0")
            # print conv1_out.get_shape(), "conv1"
            conv2_out = self._block(conv1_out, conv_filters[layer_num][1], conv_strides[layer_num][1], conv_pads[layer_num][1], pad_mode, str(layer_num)+"_1")
            # print conv2_out.get_shape(), "conv2"
            if layer_num != num_layers-1:
                conv3_out = self._block(conv2_out, conv_filters[layer_num][2], conv_strides[layer_num][2], conv_pads[layer_num][2], pad_mode, str(layer_num)+"_2")
                # print conv3_out.get_shape(), "conv3"
                out = tf.nn.max_pool(conv3_out, ksize = pool_ksizes[layer_num], strides = pool_strides[layer_num], padding = pool_pads[layer_num])
                # print out.get_shape(),"maxpool"
            else:
                out = conv2_out
                # print out.get_shape(),"final"
        self.output = out

  def _block(self, inp, filter_size, strides, padding, pad_mode, scope):
      with tf.variable_scope("conv_"+scope):
        filter_var = tf.get_variable("filterss"+scope, filter_size, tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias"+scope, filter_size[-1], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        if padding:
            inp = tf.pad(inp, [[0,0],[padding[0], padding[0]],[padding[1], padding[1]], [0,0]])
        out = tf.nn.relu(tf.nn.conv2d(inp, filter_var, strides, pad_mode) + bias)
        if filter_size[-1] < 4:
            tf.summary.image(self.name, filter_var)
        return out
# if __name__ == "__main__":
#     s = np.random.rand(10,32,32,6)
#     # print "this shouldnt be called"
#     a = NiN(s)