import state
import math
import numpy as np
import os
import tensorflow as tf
import sys
import pdb

from networks import fc, flatten
from retina import Retina
from lenet import Lenet

gamma = .99

class GradientClippingOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, use_locking=False, name="GradientClipper"):
        super(GradientClippingOptimizer, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = self.optimizer.compute_gradients(*args, **kwargs)
        clipped_grads_and_vars = []
        for (grad, var) in grads_and_vars:
            if grad is not None:
                clipped_grads_and_vars.append((tf.clip_by_value(grad, -1, 1), var))
            else:
                clipped_grads_and_vars.append((grad, var))
        return clipped_grads_and_vars

    def apply_gradients(self, *args, **kwargs):
        return self.optimizer.apply_gradients(*args, **kwargs)

class DeepQNetwork:
    def __init__(self, numActions, baseDir, args):
        self.args = args
        self.numActions = numActions
        self.baseDir = baseDir
        self.saveModelFrequency = args.save_model_freq
        self.targetModelUpdateFrequency = args.target_model_update_freq
        self.normalizeWeights = args.normalize_weights
        self.preprocess = args.preprocess

        self.staleSess = None

        tf.set_random_seed(123456)

        self.sess = tf.Session()
        self.input = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
        self.x_normalized = tf.to_float(self.input) / 255.0
        print self.preprocess
        # pdb.set_trace()
        self.i_hat_s = tf.zeros_like(self.x_normalized)[:, :, :, 0]
        self.i_hat_s = tf.reshape(self.i_hat_s, [-1, 84 * 84])
        self.i_hat_m = tf.zeros_like(self.x_normalized)[:, :, :, 0]
        self.i_hat_m = tf.reshape(self.i_hat_m, [-1, 84 * 84])
        self.i_hat_l = tf.zeros_like(self.x_normalized)[:, :, :, 0]
        self.i_hat_l = tf.reshape(self.i_hat_l, [-1, 84 * 84])
        self.first_time = True

        assert (len(tf.global_variables()) == 0),"Expected zero variables"
        self.x, self.y = self.buildNetwork('policy', True, numActions)
        self.first_time = False
        if not self.preprocess:
            assert (len(tf.trainable_variables()) == 10),"Expected 10 trainable_variables"
            assert (len(tf.global_variables()) == 10),"Expected 10 total variables"

        self.x_target, self.y_target = self.buildNetwork('target', False, numActions)

        if not self.preprocess:
            assert (len(tf.trainable_variables()) == 10),"Expected 10 trainable_variables"
            assert (len(tf.global_variables()) == 20),"Expected 20 total variables"

        # build the variable copy ops
        self.update_target = []
        trainable_variables = tf.trainable_variables()
        all_variables = tf.global_variables()

        if args.testing:
            for t in all_variables:
                print(t, t.get_shape().as_list())
            print('length of all variables (global): {}'.format(len(all_variables)))
            print
            for t in trainable_variables:
                print(t, t.get_shape().as_list())
            print('length of trainable: {}'.format(len(trainable_variables)))
            print

        start = 4 if self.preprocess == 'stack' or self.preprocess == 'ema' else 0
        for i in range(start, len(trainable_variables)):
            if args.testing: print('global: {}, assigned from trainabled {}'.format(all_variables[len(trainable_variables) + i - start], trainable_variables[i]))
            self.update_target.append(all_variables[len(trainable_variables) + i - start].assign(trainable_variables[i]))

        self.a = tf.placeholder(tf.float32, shape=[None, numActions])
        print('a %s' % (self.a.get_shape()))
        self.y_ = tf.placeholder(tf.float32, [None])
        print('y_ %s' % (self.y_.get_shape()))

        self.y_a = tf.reduce_sum(tf.multiply(self.y, self.a), reduction_indices=1)
        print('y_a %s' % (self.y_a.get_shape()))

        difference = tf.abs(self.y_a - self.y_)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = difference - quadratic_part
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.loss = tf.reduce_sum(errors)
        tf.summary.scalar('Loss', self.loss)
        #self.loss = tf.reduce_mean(tf.square(self.y_a - self.y_))

        # (??) learning rate
        # Note tried gradient clipping with rmsprop with this particular loss function and it seemed to suck
        # Perhaps I didn't run it long enough
        #optimizer = GradientClippingOptimizer(tf.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01))
        optimizer = tf.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01)
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=25)

        self.merged = tf.summary.merge_all()
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_target) # is this necessary?


        self.summary_writer = tf.summary.FileWriter(self.baseDir + '/tensorboard', self.sess.graph)

        if args.model is not None:
            print('Loading from model file %s' % (args.model))
            self.saver.restore(self.sess, args.model)

    def applyPreprocess(self, input_tensor, inp_shape):
        inputs = []
        channels = []
        channel = 1
        # pdb.set_trace()
        if self.preprocess == 'ema' or self.preprocess == 'stack':
            frames = tf.transpose(input_tensor, [0, 3, 1, 2])
            inp = tf.reshape(
                    frames,
                    [-1, inp_shape[3], inp_shape[1] * inp_shape[2]]
            )

            if self.args.testing: print('inp to retina: {}\n from frames {}'.format(inp.get_shape().as_list(), frames.get_shape().as_list()))
            if self.first_time:
                retina = Retina(inp,
                                inp_shape,
                                is_lrcn=True,
                                i_hat_lg=self.i_hat_l,
                                i_hat_md=self.i_hat_m,
                                i_hat_sh=self.i_hat_s
                                )
            else:
                retina = Retina(inp, inp_shape, is_lrcn=True)

            self.i_hat_s = retina.i_hat_sh[:, -1, :]
            self.i_hat_m = retina.i_hat_md[:, -1, :]
            self.i_hat_l = retina.i_hat_lg[:, -1, :]
            self.sd = retina.i_s

            # None, 4, 84, 84 ,1
            frames = tf.expand_dims(frames, -1)
            # move frames into batch size
            frame_shape = tf.shape(frames)

            frames = tf.reshape(frames, [frame_shape[0] * frame_shape[1], frame_shape[2], frame_shape[3], frame_shape[4]])

            event = retina.get_output()
            # event = tf.stack([event for _ in range(frame_shape[1])], axis=1)
            if self.args.testing: print('Retina Output Shape {}'.format(event.get_shape().as_list()))

            event_channel = 2
            inputs += [event]
            channels += [event_channel]

            if self.preprocess == 'stack':
                inputs += [frames]
                channels += [channel]

            # None, 4, 84, 84, 3
            retina_out = tf.concat(inputs, axis=-1, name='retina_output')
            retina_channels = np.sum(channels)
            lenet = Lenet(retina_out,
                          conv_filters=[[5, 5, retina_channels, 6], [2, 2, 6, 16]],
                          trainable=self.trainable,
                          first=self.first_time)
            inp = lenet.output
            # new_inp_sh = tf.shape(inp)
            new_inp_sh = inp.get_shape().as_list()
            if self.args.testing: print("Lenet Output Shape: {}".format(inp.get_shape().as_list()))
            # inp = tf.reshape(inp, [new_inp_sh[0] / 4, new_inp_sh[1], new_inp_sh[2], new_inp_sh[3], 4])
            # inp = tf.reshape(inp, [new_inp_sh[0] / 4, 84, 84, 4])
            inp = tf.reshape(inp, [-1, new_inp_sh[1] * 4, new_inp_sh[2] * 4, 1])
            splits = tf.split(inp, 4)
            preproc_out = tf.concat(splits, -1)
            if self.args.testing: print(preproc_out.get_shape().as_list())
            return preproc_out
            # lstm_inp = tf.reshape(inp,[-1, inp_shape[-1], new_inp_sh[1]* new_inp_sh[2]*new_inp_sh[3]])
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(new_inp_sh[1]*new_inp_sh[2]*new_inp_sh[3])
            # out, state = tf.nn.dynamic_rnn(lstm_cell, lstm_inp, dtype = tf.float32)
            # outputs = tf.reshape(out[:,-1,:],[-1, new_inp_sh[1], new_inp_sh[2], new_inp_sh[3]])
            # print("LSTM Out shape: {}".format(outputs.get_shape().as_list()))
            # _, _, fc4 = fc('fc4', flatten(outputs), 512, activation="relu")
            # print("Fully connected Shape: {}".format(fc4.get_shape().as_list()))
            # return fc4
        elif self.process is None:
            # return tf.reshape(input_tensor, [-1, inp_shape[1], inp_shape[2],1])
            return input_tensor
        else:
            raise NotImplementedError('Preprocessing input not implemented')

    def buildNetwork(self, name, trainable, numActions):
        self.trainable = trainable
        print("Building network for %s trainable=%s" % (name, trainable))
        x = self.input
        x_norm = self.x_normalized
        inp_shape = x_norm.get_shape().as_list()
        # None, 84, 84, 4
        print("Input shape to network {}".format(inp_shape))
        # Apply EDR
        dqn_in = self.applyPreprocess(x_norm, inp_shape)
        if self.args.testing: print("After preprocessing input shape {}".format(dqn_in.get_shape().as_list()))

        # Second layer convolves 32 8x8 filters with stride 4 with relu
        with tf.variable_scope("cnn1_" + name):
            W_conv1, b_conv1 = self.makeLayerVariables([8, 8, 4, 32], trainable, "conv1")

            h_conv1 = tf.nn.relu(tf.nn.conv2d(dqn_in, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)

        # Third layer convolves 64 4x4 filters with stride 2 with relu
        with tf.variable_scope("cnn2_" + name):
            W_conv2, b_conv2 = self.makeLayerVariables([4, 4, 32, 64], trainable, "conv2")

            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)

        # Fourth layer convolves 64 3x3 filters with stride 1 with relu
        with tf.variable_scope("cnn3_" + name):
            W_conv3, b_conv3 = self.makeLayerVariables([3, 3, 64, 64], trainable, "conv3")

            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
            print(h_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64], name="h_conv3_flat")
        print(h_conv3_flat)

        # Fifth layer is fully connected with 512 relu units
        with tf.variable_scope("fc1_" + name):
            W_fc1, b_fc1 = self.makeLayerVariables([7 * 7 * 64, 512], trainable, "fc1")

            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)

        # Sixth (Output) layer is fully connected linear layer
        with tf.variable_scope("fc2_" + name):
            W_fc2, b_fc2 = self.makeLayerVariables([512, numActions], trainable, "fc2")

            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            print(y)

        return x, y

    def makeLayerVariables(self, shape, trainable, name_suffix):
        if self.normalizeWeights:
            # This is my best guess at what DeepMind does via torch's Linear.lua and SpatialConvolution.lua (see reset methods).
            # np.prod(shape[0:-1]) is attempting to get the total inputs to each node
            stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
            weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
            biases  = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
        else:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), trainable=trainable, name='W_' + name_suffix)
            biases  = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name='W_' + name_suffix)
        return weights, biases

    def inference(self, screens):
        y = self.sess.run([self.y], {self.x: screens})
        q_values = np.squeeze(y)
        return np.argmax(q_values)

    def train(self, batch, stepNumber):

        x2 = [b.state2.getScreens() for b in batch]
        y2 = self.y_target.eval(feed_dict={self.x_target: x2}, session=self.sess)

        x = [b.state1.getScreens() for b in batch]
        a = np.zeros((len(batch), self.numActions))
        y_ = np.zeros(len(batch))

        for i in range(0, len(batch)):
            a[i, batch[i].action] = 1
            if batch[i].terminal:
                y_[i] = batch[i].reward
            else:
                y_[i] = batch[i].reward + gamma * np.max(y2[i])

        _, sum_str = self.sess.run([self.train_step, self.merged], feed_dict={
            self.x: x,
            self.a: a,
            self.y_: y_
        })
        # self.train_step.run(feed_dict={
        #     self.x: x,
        #     self.a: a,
        #     self.y_: y_
        # }, session=self.sess)
        # sum_str = self.sess.run(self.merged)
        self.summary_writer.add_summary(sum_str, stepNumber)
        self.summary_writer.flush()

        if stepNumber % self.targetModelUpdateFrequency == 0:
			self.sess.run(self.update_target)

        if stepNumber % self.targetModelUpdateFrequency == 0 or stepNumber % self.saveModelFrequency == 0:
            dir = self.baseDir + '/models'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            savedPath = self.saver.save(self.sess, dir + '/model', global_step=stepNumber)
