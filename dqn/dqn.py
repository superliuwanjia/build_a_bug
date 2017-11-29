import state
import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import sys
import pdb
import time

from networks import fc, flatten
from retina import Retina
from lenet import Lenet

import scipy.misc

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
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        self.input = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")

        # pdb.set_trace()
        self.i_hat_s = tf.zeros_like(self.input, dtype=tf.float32)[:, :, :, 0]
        self.i_hat_s = tf.reshape(self.i_hat_s, [-1, 84 * 84])
        self.i_hat_m = tf.zeros_like(self.input, dtype=tf.float32)[:, :, :, 0]
        self.i_hat_m = tf.reshape(self.i_hat_m, [-1, 84 * 84])
        self.i_hat_l = tf.zeros_like(self.input, dtype=tf.float32)[:, :, :, 0]
        self.i_hat_l = tf.reshape(self.i_hat_l, [-1, 84 * 84])
        self.first_time = True

        assert (len(tf.global_variables()) == 0),"Expected zero variables"

        if self.preprocess == 'ema':
            self.feat_size = 3 * 2
        elif self.preprocess == 'stack':
            self.feat_size = 3 * 3
        elif self.preprocess is None:
            self.feat_size = 4
        else:
            raise NotImplementedError('Preprocess not implemented')


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

        # start = 4 if self.args.conv_preprocess else 0
        start = 0
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

    def tf_apply_preprocess(self, input_frames):
        if self.preprocess == 'ema' or self.preprocess == 'stack':
            # Using retina or stacking event with raw frames
            frames_shape = input_frames.get_shape().as_list()
            t_frames = tf.transpose(input_frames, (3, 0, 1, 2))

            event = self.tf_retina(t_frames)
            if self.preprocess == 'stack':
                t_frames = tf.expand_dims(t_frames, len(frames_shape))
                event = tf.concat((event, t_frames), len(frames_shape))
            event_shape = event.get_shape().as_list()
            # print(np.amax(event), np.amin(event), np.mean(event), event.shape)
            event = tf.transpose(event, (1, 2, 3, 4, 0))
            event = event[:, :, :, :, 1:]
            # print
            # for i in range(4):
            #     print('two', np.amax(event[0,:,:,1,i]), np.amin(event[0,:,:,1,i]), np.mean(event[0,:,:,1,i]), event[0,:,:,1,i].shape)
            event_out = tf.reshape(event, (event_shape[1], event_shape[2], event_shape[3], event_shape[4] * (event_shape[0] - 1)))
            
        elif self.preprocess is None:
            event_out = tf.to_float(input_frames) / 255.0
        else:
            raise NotImplementedError('Preprocess type not implemented')

        return event_out

    def np_retina(self, video, alpha=0.5, mu_on=0.00, mu_off=-0.1):
        """
        video is a numpy array, where the first dimention is time T
        """
        # decay constant must be smaller than 1
        assert alpha <= 1 and alpha >= 0, 'decay constant must be positive and less than 1'

        ema = np.zeros(video.shape, dtype=float)

        # normalize video pixel value to 0 ~ 1
        video = (video.astype(np.float32)) + 1e-3

        # EMA filtering
        ema[0, :, :, :] = video[0, :, :, :]

        for t in range(1, video.shape[0]):
            ema[t, :, :, :] = (1 - alpha) * ema[t - 1, :, :, :] + \
                              alpha * video[t, :, :, :]

        # compute relative change
        change = np.tanh(np.log(np.divide(video, ema + 1e-5)))

        # thresholding
        on = np.expand_dims(np.maximum(0, change - mu_on),
                            len(video.shape))
        off = np.expand_dims(np.maximum(0, - (change - mu_off)),
                             len(video.shape))

        return np.concatenate((on, off), axis=4)

    def tf_retina(self, video, alpha=0.5, mu_on=0.00, mu_off=-0.1):
        """
        video is a numpy array, where the first dimention is time T
        """
        # decay constant must be smaller than 1
        # assert alpha <= 1 and alpha >= 0, 'decay constant must be positive and less than 1'

        ema = np.zeros(video.shape, dtype=float)

        # normalize video pixel value to 0 ~ 1
        video = tf.to_float(video) + 1e-3

        # EMA filtering
        ema[0, :, :, :] = video[0, :, :, :]

        for t in range(1, video.shape[0]):
            ema[t, :, :, :] = (1 - alpha) * ema[t - 1, :, :, :] + \
                              alpha * video[t, :, :, :]

        # compute relative change
        change = tf.tanh(tf.log(tf.divide(video, ema + 1e-5)))

        # thresholding
        on = tf.expand_dims(tf.nn.relu(change - mu_on), 4)
        off = tf.expand_dims(tf.nn.relu(-(change - mu_off)), 4)

        return tf.concat((on, off), axis=4)

    def apply_preprocess(self, input_frames):
        if self.preprocess == 'ema' or self.preprocess == 'stack':
            # Using retina or stacking event with raw frames
            frames_shape = input_frames.shape
            t_frames = np.transpose(input_frames, (3, 0, 1, 2))

            event = self.retina(t_frames)
            if self.preprocess == 'stack':
                t_frames = np.expand_dims(t_frames, len(frames_shape))
                event = np.concatenate((event, t_frames), len(frames_shape))
            event_shape = event.shape
            # print(np.amax(event), np.amin(event), np.mean(event), event.shape)
            event = np.transpose(event, (1, 2, 3, 4, 0))
            event = event[:, :, :, :, 1:]
            # print
            # for i in range(4):
            #     print('two', np.amax(event[0,:,:,1,i]), np.amin(event[0,:,:,1,i]), np.mean(event[0,:,:,1,i]), event[0,:,:,1,i].shape)
            event_out = np.reshape(event, (event_shape[1], event_shape[2], event_shape[3], event_shape[4] * (event_shape[0] - 1)))
            # print('three', np.amax(event_out[0,:,:,0]), np.amin(event_out[0,:,:,0]), np.mean(event_out[0,:,:,0]), event_out[0,:,:,0].shape)
        elif self.preprocess is None:
            event_out = input_frames / 255.0
        else:
            raise NotImplementedError('Preprocess type not implemented')

        return event_out

    def buildNetwork(self, name, trainable, numActions):
        self.trainable = trainable
        print("Building network for %s trainable=%s" % (name, trainable))
        # First layer takes a screen, and shrinks by 2x
        x = tf.placeholder(tf.float32, shape=[None, 84, 84, self.feat_size], name="preproc_screens")
        # None, 84, 84, 4
        # print("Input shape to preproc {}".format(inp_shape))
        # dqn_in = self.applyPreprocess(x, inp_shape)
        # dqn_in_shape = dqn_in.get_shape().as_list()
        # self.feat_size = dqn_in_shape[-1]
        # if self.args.testing: print("After preprocessing input shape {}".format(dqn_in_shape))

        # Second layer convolves 32 8x8 filters with stride 4 with relu
        with tf.variable_scope("cnn1_" + name):
            W_conv1, b_conv1 = self.makeLayerVariables([8, 8, self.feat_size, 32], trainable, "conv1")

            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
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
        proc_screens = self.apply_preprocess(screens)
        y = self.sess.run([self.y], {self.x: proc_screens})
        q_values = np.squeeze(y)
        return np.argmax(q_values)

    def train(self, batch, stepNumber):

        x2 = [b.state2.getScreens() for b in batch]
        x = [b.state1.getScreens() for b in batch]

        '''
        start = time.time()
        x2 = np.asarray(x2)
        x = np.asarray(x)
        end = time.time()
        # print('CASTING ARRAY TIME: {}'.format(end - start))

        start = time.time()
        proc_x2 = self.apply_preprocess(x2)
        end = time.time()
        # print('APPLYING PREPROCESS TIME: {}'.format(end - start))
        proc_x = self.apply_preprocess(x)
        # print('three', np.amax(proc_x), np.amin(proc_x), np.mean(proc_x), proc_x.shape)
        '''

        if self.args.save_imgs and stepNumber % 100 == 0:
            print('Saving debug images to debugImages')
            dir_save_main = self.baseDir + '/debugImages'
            if not os.path.isdir(dir_save_main):
                os.makedirs(dir_save_main)

            dir_save = dir_save_main + '/{}/'.format(stepNumber)
            if not os.path.isdir(dir_save):
                os.makedirs(dir_save)
            scipy.misc.imsave(dir_save + 'normFrame{}.png'.format(stepNumber), x[0, :, :, 0])
            scipy.misc.imsave(dir_save + 'retinaFramesOn{}.png'.format(stepNumber), proc_x[0, :, :, 0])
            scipy.misc.imsave(dir_save + 'retinaFramesOff{}.png'.format(stepNumber), proc_x[0, :, :, 1])
            r_off = proc_x[0, :, :, 1]
            r_on = proc_x[0, :, :, 0]
            print("TEST TEST", np.amin(r_off[np.nonzero(r_off)]))
            assert np.amin(r_off) >= 0 and np.amin(r_on) >= 0, 'retina output < 0'
            print("RETINA ON -  MEAN: {}, MAX: {}, MIN: {}".format(np.mean(r_on), np.amax(r_on), np.amin(r_on)))
            print("RETINA OFF-  MEAN: {}, MAX: {}, MIN: {}".format(np.mean(r_off), np.amax(r_off), np.amin(r_off)))

        y2 = self.y_target.eval(feed_dict={self.x_target: proc_x2}, session=self.sess)
        a = np.zeros((len(batch), self.numActions))
        y_ = np.zeros(len(batch))

        for i in range(0, len(batch)):
            a[i, batch[i].action] = 1
            if batch[i].terminal:
                y_[i] = batch[i].reward
            else:
                y_[i] = batch[i].reward + gamma * np.max(y2[i])

        _, sum_str = self.sess.run([self.train_step, self.merged], feed_dict={
                                                                            self.x: proc_x,
                                                                            self.a: a,
                                                                            self.y_: y_
                                                                        })
        self.summary_writer.add_summary(sum_str, stepNumber)
        self.summary_writer.flush()

        if stepNumber % self.targetModelUpdateFrequency == 0:
			self.sess.run(self.update_target)

        if stepNumber % self.saveModelFrequency == 0:
            dir = self.baseDir + '/models'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            savedPath = self.saver.save(self.sess, dir + '/model', global_step=stepNumber)
