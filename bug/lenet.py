import tensorflow as tf


class Lenet():
    """docstring for Lenet"""
    def __init__(self,
                 inp,
                 num_layers=2,
                 conv_filters=[[5, 5, 6, 6], [5, 5, 6, 16]],
                 conv_strides=[[1,1,1,1], [1,1,1,1]],
                 conv_pads=["SAME", "SAME"],
                 pool_ksizes=[[1, 2, 2, 1], [1, 2, 2, 1]],
                 pool_strides=[[1,2,2,1], [1,2,2,1]],
                 pool_pads=["VALID", "VALID"],
                 trainable=True,
                 first=True):
        max_pool = inp
        self.name = "lenet"
        with tf.variable_scope("lenet", reuse=not first):
            for layer_num in range(num_layers):
                filter_var = tf.get_variable("filters" + str(layer_num),
                                             conv_filters[layer_num],
                                             tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             trainable=trainable
                                             )
                bias = tf.get_variable("bias" + str(layer_num),
                                       conv_filters[layer_num][-1],
                                       tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=trainable)
                out = tf.nn.relu(tf.nn.conv2d(max_pool, filter_var, conv_strides[layer_num], conv_pads[layer_num]) + bias)
                max_pool = tf.nn.max_pool(out, ksize = pool_ksizes[layer_num], strides = pool_strides[layer_num], padding = pool_pads[layer_num])
                if conv_filters[layer_num][-1] < 4:
                    tf.summary.image(self.name, filter_var)
        self.output = max_pool

# class NiN():
# 	"""docstring for NiN"""
# 	def __init__(self, inp, num_layers = 2, conv_filters = [[11, 11, 6, 96], [1, 1, 96, 96]], conv_strides = [[1,4,4,1], [1,1,1,1]], conv_pads = ["VALID", "VALID"],
# 				pool_ksizes = [[1, 2, 2, 1], [1, 2, 2, 1]], pool_strides = [[1,2,2,1], [1,2,2,1]], pool_pads = ["VALID", "VALID"],
# 				fcn_= None):
# 		pass


