import tensorflow as tf
from rnn_cell import EMAcell


class Retina():
    """docstring for Retina"""
    def __init__(self, inp, inp_size, beta = [0.5, 0.7, 0.9], alphas = [1.5, 2.0, 2.5], mu_off = 0.01, mu_on = 0.01, is_lrcn = False, i_hat_lg = None,
                i_hat_md = None, i_hat_sh = None):
        self.inp_size = inp_size
        # self.beta = tf.Variable(tf.random_normal[])
        # self.beta = beta
        # self.alphas = alphas
        # self.mu_off = mu_off
        # self.mu_on = mu_on
        try:
            self.hidden_units = inp_size[2]*inp_size[3]*inp_size[4]    
        except Exception:
            self.hidden_units = inp_size[2]*inp_size[1]
        
        # self.beta_sl = beta_sh
        # self.beta_sl = beta_med
        # self.beta_lg = beta_lg
        self.input = tf.convert_to_tensor(inp, tf.float32)
        self.i_s = i_hat_sh
        self.i_m = i_hat_md
        self.i_l = i_hat_lg
        self.create_vars()
        self.get_ema()
        self.get_rel_changes()
        if is_lrcn:
            self.threshold(is_lrcn)
        else:
            self.threshold()
        


    def create_vars(self,):
        # self.alpha_s = tf.Variable(0.3)
        # self.alpha_m = tf.Variable(0.5)
        # self.alpha_l = tf.Variable(0.7)
        self.alpha_s = 0.3
        self.alpha_m = 0.5
        self.alpha_l = 0.7
        self.alphas = [self.alpha_s, self.alpha_m, self.alpha_l]
        # self.beta_s = tf.Variable(0.33)
        self.beta_s = 0.33
        # self.beta_m = tf.Variable(0.33)
        self.beta_m = 0.33
        # self.beta_l = tf.Variable(0.33)
        self.beta_l = 0.33
        self.beta = [self.beta_s, self.beta_m, self.beta_l]
        #self.mu_off = tf.Variable(1.0)
        #self.mu_on= tf.Variable(-1.0)
        self.mu_off = 1.96
        self.mu_on = 1.96
        # self.mu_on= 1.0
        # print(self.mu_on)
        # print(li)
        # print(self.mu_on, self.mu_off,"gogogogogo")
        # print(df)
        # with tf.device("gpu:0"):
            # tf.summary.scalar("alphas", self.alpha_s)
            # tf.summary.scalar("alpham", self.alpha_m)
            # tf.summary.scalar("alphal", self.alpha_l)
            # tf.summary.scalar("betal", self.beta_l)
            # tf.summary.scalar("betam", self.beta_m)
            # tf.summary.scalar("betas", self.beta_s)
            # tf.summary.scalar("mu_off", self.mu_off)
            # tf.summary.scalar("mu_on", self.mu_on)
        print("here")

    def get_ema(self):

        ema_cell_sh = EMAcell(self.hidden_units, self.alphas[0])
        if self.i_s is not None:
            state_sh = self.i_s
        else:
            state_sh = self.input[:,0,:]
        outputs_sh, state_sh = tf.nn.dynamic_rnn(ema_cell_sh, self.input, initial_state = state_sh)
        self.i_hat_sh = outputs_sh + 1e-7

        ema_cell_md = EMAcell(self.hidden_units, self.alphas[1])
        if self.i_m is not None:
            state_md = self.i_m
        else:
            state_md = self.input[:,0,:]
        
        outputs_md, state_md = tf.nn.dynamic_rnn(ema_cell_md, self.input, initial_state = state_md)
        self.i_hat_md = outputs_md  + 1e-7
        

        ema_cell_lg = EMAcell(self.hidden_units, self.alphas[2])
        if self.i_l is not None:
            state_lg = self.i_l
        else:
            state_lg = self.input[:,0,:]
        outputs_lg, state_lg = tf.nn.dynamic_rnn(ema_cell_lg, self.input, initial_state = state_lg)
        self.i_hat_lg = outputs_lg + 1e-7
        

    def get_rel_changes(self):
        # self.r_x = self.beta[0]*tf.log(tf.divide(self.input, self.i_hat_sh)) + self.beta[1]*tf.log(tf.divide(self.input, self.i_hat_md))\
        #             + self.beta[2]*tf.log(tf.divide(self.input, self.i_hat_lg))

        # Code for normalization

        log_sh = tf.log(tf.divide(self.input, self.i_hat_sh))
        mean_sh, var_sh = tf.nn.moments(log_sh, [2], keep_dims=True)
        normalized_sh = tf.divide(tf.subtract(log_sh, mean_sh), tf.sqrt(var_sh+1e-7))

        log_md = tf.log(tf.divide(self.input, self.i_hat_md))
        mean_md, var_md = tf.nn.moments(log_md, [2], keep_dims=True)
        normalized_md = tf.divide(tf.subtract(log_md, mean_md), tf.sqrt(var_md+1e-7))

        log_lg = tf.log(tf.divide(self.input, self.i_hat_lg))
        mean_lg, var_lg = tf.nn.moments(log_lg, [2], keep_dims=True)
        normalized_lg = tf.divide(tf.subtract(log_lg, mean_lg), tf.sqrt(var_lg+1e-7))

        self.r_x = self.beta[0] * normalized_sh + self.beta[1] * normalized_md + self.beta[2] * normalized_lg
         
    def threshold(self, is_lrcn = False):
        if is_lrcn:
            # if we use lrcn then we cannot use the 4 frames provided as channels. we need to keep it at time index
            e_on = tf.reshape(tf.nn.relu(self.r_x - (1 + self.mu_on)), [-1, self.inp_size[3], self.inp_size[1], self.inp_size[2], 1])
            e_off = tf.reshape(tf.nn.relu(-(self.r_x - (1 - self.mu_off))),  [-1, self.inp_size[3], self.inp_size[1], self.inp_size[2], 1])    
            self.out = tf.reshape(tf.concat([e_on, e_off], axis = 4),[-1, self.inp_size[1], self.inp_size[2], 2])
        else:
            # if we are not using lrcn we can use the frames as channels.
            e_on = tf.reshape(tf.nn.relu(self.r_x - self.mu_on),[-1, self.inp_size[1], self.inp_size[2], self.inp_size[3],self.inp_size[4]])
            e_off = tf.reshape(tf.nn.relu(-(self.r_x + self.mu_off)), [-1, self.inp_size[1], self.inp_size[2], self.inp_size[3], self.inp_size[4]])
            print(e_on.get_shape(), e_off.get_shape())        
            self.out = tf.concat([e_on, e_off], axis = 4)    
        # self.out = tf.stack([e_on, e_off], axis = 2)    
        # print(self.out.get_shape(),"here111")
        # except Exception:
        # print(s)
            # e_on = tf.reshape(tf.nn.relu(self.r_x - (1 + self.mu_on)),self.inp_size)
            # e_off = tf.reshape(tf.nn.relu(-(self.r_x - (1 - self.mu_off))), self.inp_size)
        #     self.out = tf.concat([e_on, e_off], axis = 4)

        

    def get_output(self):
        return self.out




