from numpy import dtype
import tensorflow as tf



class LSTMForgetBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes):
        super().__init__(name='LSTMForgetBlock')
        self.W_f = tf.keras.layers.Dense(num_nodes, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, name='U_f')
        self.W_i = tf.keras.layers.Dense(num_nodes, name='W_i', use_bias=False)
        self.U_i = tf.keras.layers.Dense(num_nodes, name='U_i')
        self.W_o = tf.keras.layers.Dense(num_nodes, name='W_o', use_bias=False)
        self.U_o = tf.keras.layers.Dense(num_nodes, name='U_o')
        self.W_c = tf.keras.layers.Dense(num_nodes, name='W_c', use_bias=False)
        self.U_c = tf.keras.layers.Dense(num_nodes, name='U_c')

    def call(self, x, h, c):
        f = tf.keras.activations.sigmoid(self.W_f(x) + self.U_f(h))
        i = tf.keras.activations.sigmoid(self.W_i(x) + self.U_i(h))
        o = tf.keras.activations.sigmoid(self.W_o(x) + self.U_o(h))
        c_ = tf.keras.activations.tanh(self.W_c(x) + self.U_c(h))
        c = f*c + i*c_
        return o*tf.keras.activations.tanh(c), c


class LSTMPeepholeBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes):
        super().__init__(name='LSTMPeepholeBlock')
        self.W_f = tf.keras.layers.Dense(num_nodes, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, name='U_f')
        self.W_i = tf.keras.layers.Dense(num_nodes, name='W_i', use_bias=False)
        self.U_i = tf.keras.layers.Dense(num_nodes, name='U_i')
        self.W_o = tf.keras.layers.Dense(num_nodes, name='W_o', use_bias=False)
        self.U_o = tf.keras.layers.Dense(num_nodes, name='U_o')
        self.W_c = tf.keras.layers.Dense(num_nodes, name='W_c')

    def call(self, x, h, c):
        f = tf.keras.activations.sigmoid(self.W_f(x) + self.U_f(h))
        i = tf.keras.activations.sigmoid(self.W_i(x) + self.U_i(h))
        o = tf.keras.activations.sigmoid(self.W_o(x) + self.U_o(h))
        c = f*c + i * tf.keras.activations.tanh(self.W_c(x))
        return o*c, c

class DGMBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes):
        super().__init__(name='DGMBlock')
        self.W_z = tf.keras.layers.Dense(num_nodes, name='W_f', use_bias=False)
        self.U_z = tf.keras.layers.Dense(num_nodes, name='U_f')
        self.W_g = tf.keras.layers.Dense(num_nodes, name='W_i', use_bias=False)
        self.U_g = tf.keras.layers.Dense(num_nodes, name='U_i')
        self.W_r = tf.keras.layers.Dense(num_nodes, name='W_o', use_bias=False)
        self.U_r = tf.keras.layers.Dense(num_nodes, name='U_o')
        self.W_h = tf.keras.layers.Dense(num_nodes, name='W_c', use_bias=False)
        self.U_h = tf.keras.layers.Dense(num_nodes, name='U_c')

    def call(self, x, S):
        Z = tf.keras.activations.sigmoid(self.W_z(S) + self.U_z(x))
        G = tf.keras.activations.sigmoid(self.W_g(S) + self.U_g(x))
        R = tf.keras.activations.sigmoid(self.W_r(S) + self.U_r(x))
        H = tf.keras.activations.tanh(self.W_h(S*R) + self.U_h(x))
        return (1.0 - G)*H + Z*S

class FPForget(tf.keras.models.Model):
    """
    Description: 
        LSTM Forget architecture 
    Args:
        num_nodes: number of nodes in each LSTM layer
        num_layers: number of LSTM layers
    """
    def __init__(self, num_nodes, num_layers, name = 'FPForget', save_path=None):
        super().__init__(name=name)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.lstm_layers = [LSTMForgetBlock(num_nodes) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
        
        

    def call(self, *args):
        x = tf.concat(args, axis=1)
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes))
        for i in range(self.num_layers):
            h, c = self.lstm_layers[i](x, h, c)
            #h = self.batch_norm(h)
            #c = self.batch_norm(c)
        y = self.final_dense(h)
        return y


class FPPeephole(tf.keras.models.Model):
    """
    Description: 
        LSTM Peephole architecture 
    Args:
        num_nodes: number of nodes in each LSTM layer
        num_layers: number of LSTM layers
    """
    def __init__(self, num_nodes, num_layers, name = 'FPPeephole', save_path=None):
        super().__init__(name=name)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.lstm_layers = [LSTMPeepholeBlock(num_nodes) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
        
        

    def call(self, *args):
        x = tf.concat(args, axis=1)
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes))
        for i in range(self.num_layers):
            h, c = self.lstm_layers[i](x, h, c)
            #h = self.batch_norm(h)
            #c = self.batch_norm(c)
        y = self.final_dense(h)
        return y

class FPDGM(tf.keras.models.Model):
    """
    Description: 
        DGM architecture 
    Args:
        num_nodes: number of nodes in each DGM layer
        num_layers: number of DGM layers
    """
    def __init__(self, num_nodes, num_layers, name = 'FPDGM', save_path=None):
        super().__init__(name=name)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.initial_dense = tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh)
        self.lstm_layers = [DGMBlock(num_nodes) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
         

    def call(self, *args):
        x = tf.concat(args, axis=1)
        S = self.initial_dense(x)
        for i in range(self.num_layers):
            S = self.lstm_layers[i](x, S)
        y = self.final_dense(S)
        return y


class FPVanilla(tf.keras.models.Model):
    """
    Description: 
        Vanilla architecture 
    Args:
        num_nodes: number of nodes in each Vanilla layer
        num_layers: number of Vanilla layers
    """
    def __init__(self, num_nodes, num_layers, name = 'FPVanilla', save_path=None):
        super().__init__(name=name)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.initial_dense = tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh)
        self.middle_layers = [tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
         

    def call(self, *args):
        x = self.initial_dense(tf.concat(args, axis=1))
        for i in range(self.num_layers):
            x = self.middle_layers[i](x)
        x = self.final_dense(x)
        return x