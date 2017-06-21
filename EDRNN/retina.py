import tensorflow as tf
from rnn_cell import EMAcell


class Retina():
    """docstring for Retina"""
    def __init__(self, inp, inp_size, beta = [0.5, 0.7, 0.9], alphas = [1.5, 2.0, 2.5], mu_off = 0.01, mu_on = 0.01):
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
        self.create_vars()
        self.get_ema()
        self.get_rel_changes()
        self.threshold()
        


    def create_vars(self,):
        self.alpha_s = tf.Variable(2.0)
        self.alpha_m = tf.Variable(2.0)
        self.alpha_l = tf.Variable(2.0)
        self.alphas = [self.alpha_s, self.alpha_m, self.alpha_l]
        self.beta_s = tf.Variable(0.33)
        self.beta_m = tf.Variable(0.33)
        self.beta_l = tf.Variable(0.33)
        self.beta = [self.beta_s, self.beta_m, self.beta_l]
        self.mu_off = tf.Variable(0.05)
        self.mu_on= tf.Variable(0.05)
        print("here")

    def get_ema(self):

        ema_cell_sh = EMAcell(self.hidden_units, self.alphas[0])
        state_sh = self.input[:,0,:]
        outputs_sh, state_sh = tf.nn.dynamic_rnn(ema_cell_sh, self.input, initial_state = state_sh)
        self.i_hat_sh = outputs_sh  

        ema_cell_md = EMAcell(self.hidden_units, self.alphas[1])
        state_md = self.input[:,0,:]
        outputs_md, state_md = tf.nn.dynamic_rnn(ema_cell_md, self.input, initial_state = state_md)
        self.i_hat_md = outputs_md
        

        ema_cell_lg = EMAcell(self.hidden_units, self.alphas[2])
        state_lg = self.input[:,0,:]
        outputs_lg, state_lg = tf.nn.dynamic_rnn(ema_cell_lg, self.input, initial_state = state_lg)
        self.i_hat_lg = outputs_lg
        

    def get_rel_changes(self):
        self.r_x = self.beta[0]*tf.log(tf.divide(self.input, self.i_hat_sh)) + self.beta[1]*tf.log(tf.divide(self.input, self.i_hat_md))\
                    + self.beta[2]*tf.log(tf.divide(self.input, self.i_hat_lg))

        

    def threshold(self):
        # try:
        e_on = tf.reshape(tf.nn.relu(self.r_x - (1 + self.mu_on)),[1, -1, self.inp_size[2], self.inp_size[3], self.inp_size[4]])
        e_off = tf.reshape(tf.nn.relu(-(self.r_x - (1 - self.mu_off))), [1, -1, self.inp_size[2], self.inp_size[3], self.inp_size[4]])
        self.out = tf.concat([e_on, e_off], axis = 4)    
        # self.out = tf.stack([e_on, e_off], axis = 2)    
        print(self.out.get_shape())
        # except Exception:

        #     e_on = tf.reshape(tf.nn.relu(self.r_x - (1 + self.mu_on)),self.inp_size)
        #     e_off = tf.reshape(tf.nn.relu(-(self.r_x - (1 - self.mu_off))), self.inp_size)
        #     self.out = tf.concat([e_on, e_off], axis = 4)

        

    def get_output(self):
        return self.out




