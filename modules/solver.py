import numpy as np
import tensorflow as tf
import time
import nnplot as nplt
import os

class Domain:
    """
    Description:
        A class for implementing box domains
    Attrs:
        low: an array of floats specifying min in each dimension
        high: an array of floats specifying max in each dimension
    """
    def __init__(self, low, high):
        self.low = low 
        self.high = high
        self.dim = len(low)

    def sample(self, num_sample):
        return tf.random.uniform(shape=(num_sample, self.dim), minval=self.low, maxval=self.high)

class SolverPlotFunc:
    def __init__(self, solver):
        self.dtype = solver.dtype
        self.name = solver.name
        self.domain = solver.domain
        self.solver = solver 

    def __call__(self, t, *args):
        self.solver.load_weights(t.numpy()[0][0])
        X = tf.concat(args, axis=1)
        return self.solver.u0(X)


class DeepBSDE(tf.keras.Model):
    def __init__(self, t0=0.0,\
                 t1=1.0,\
                 domain=[np.array([-1., -1.], dtype='float32'), np.array([1., 1.], dtype='float32')],\
                 time_steps=20,\
                 sigma=np.sqrt(2),\
                 num_hidden_layers=2,\
                 num_neurons=200,\
                 mu = None,
                 fun_f=None,
                 fun_g=None,
                 name='DeepBSDESolver',
                 dir=None, 
                 **kwargs):
        """Set up basic architecture of deep BSDE NN model."""
        
        super().__init__(dtype='float32', name=name, **kwargs)
        
        self.t0 = t0
        self.t1 = t1
        self.N = time_steps
        self.domain = Domain(low=domain[0], high=domain[1])
        self.dim = self.domain.dim
        self.mu = mu
        self.sigma = sigma
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.fun_f = fun_f
        self.fun_g = fun_g
        self.dt = (t1 - t0)/(self.N)
        self.sqrt_dt = np.sqrt(self.dt)
        
        self.t_space = np.linspace(self.t0, self.t1, self.N + 1)[:-1]

        if dir is not None:
            self.folder = '{}/{}'.format(dir, self.name)
        else:
            self.folder = self.name

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        
        # Initialize value and gradient of u(t_0,X_{t_0}) by zeros
        #self.u0 = tf.Variable(np.zeros((1),dtype=DTYPE))
        #self.gradu0 = tf.Variable(np.zeros((1,self.dim),dtype=DTYPE))
        
        # Alternatively, initialize both randomly
        #self.u0 = self.subnet(out_dim=1)#tf.Variable(np.random.uniform(.3, .5, size=(1)).astype(DTYPE))
        #self.gradu0 = tf.Variable(np.random.uniform(-1e-1, 1e-1, size=(1, dim)).astype(DTYPE))
        
        # Create template of dense layer without bias and activation
        self._dense = lambda dim: tf.keras.layers.Dense(
            units=dim,
            activation=None,
            use_bias=False)
        
        # Create template of batch normalization layer
        self._bn = lambda : tf.keras.layers.BatchNormalization(
            momentum=.99,
            epsilon=1e-6,
            beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
            gamma_initializer=tf.random_uniform_initializer(0.1, 0.5))
        
        self.u0 = self.u0_net()
        # Initialize a list of networks approximating the gradient of u(t, x) at t_i
        self.gradui = []
        
        # Loop over number of time steps
        for _ in range(self.N-1):
            self.gradui.append(self.subnet(out_dim=self.dim))
      
        self.plt_fn = SolverPlotFunc(self)

    def u0_net(self):
        """
        # Batch normalization on dim-dimensional input
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.dim))
        #model.add(self._bn())
        
        # Hidden layers of type (Dense -> Batch Normalization -> ReLU)
        for _ in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(
            units=self.num_neurons,
            activation=tf.keras.activations.tanh))
            #model.add(self._bn())
            
        # Dense layer followed by batch normalization for output
        model.add(tf.keras.layers.Dense(
            units=1,
            activation=tf.keras.activations.exponential,
            use_bias=False))
        #model.add(self._bn())
        """
        return FPForget(num_nodes=self.num_neurons, num_layers=self.num_hidden_layers)

    @tf.function
    def grad_u0(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            u0 = self.u0(x)
        return tape.gradient(u0, x)

            
    def subnet(self, out_dim):
        # Batch normalization on dim-dimensional input
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.dim))
        model.add(self._bn())
        
        # Hidden layers of type (Dense -> Batch Normalization -> ReLU)
        for _ in range(self.num_hidden_layers):
            model.add(self._dense(self.num_neurons))
            model.add(self._bn())
            model.add(tf.keras.layers.ReLU())
            
        # Dense layer followed by batch normalization for output
        model.add(self._dense(out_dim))
        model.add(self._bn())
        return model

    def draw_X_and_dW(self, num_sample, x=None):
        """ Method to draw num_sample paths of X. """
        
        # Draw increments of Brownian motion
        dW = np.random.normal(loc=0.0, scale=self.sqrt_dt, size=(num_sample, self.dim, self.N)).astype(self.dtype)
        
        # Initialize the array X
        X = np.zeros((num_sample, self.dim, self.N+1), dtype=self.dtype)
        # Initialize and set array of paths
        if x is None:
            X[:, :, 0] = np.zeros((num_sample, self.dim, self.N+1), dtype=self.dtype)
        else:
            X[:, :, 0] = np.ones((num_sample, self.dim)) * x
        
        
        for i in range(self.N):
            # This corresponds to the Euler-Maruyama Scheme
            X[:, :, i+1] = X[:, :, i] + self.mu(X[:, :, i]) * self.dt + self.sigma * dW[:, :, i]
            
        # Return simulated paths as well as increments of Brownian motion
        return X, dW

    
    
    @tf.function
    def call(self, inp, training=False):
        """
        Method to perform one forward sweep through the network
        given inputs: inp - (X, dW)
                      training - boolean variable indicating training
        """
        X, dW = inp
        #num_sample = X.shape[0]
        
        
        #e_num_sample = tf.ones(shape=[num_sample, 1], dtype=self.dtype)
        
        # Value approximation at t0
        y = self.u0(X[:, :, 0])
        
        # Gradient approximation at t0
        z = self.grad_u0(X[:, :, 0])
        
        for i in range(self.N-1):
            
            t = self.t_space[i]
            
            # Optimal control is attained by gradient
            eta1 = - self.fun_f(t, X[:, :, i], y, z) * self.dt
            eta2 = tf.reduce_sum(z * dW[:, :, i], axis=1, keepdims=True)

            y = y + eta1 + eta2

            # New gradient approximation
            # The division by self.dim acts as a stabilizer
            z = self.gradui[i](X[:, :, i + 1], training) / self.dim

        # Final step
        eta1 = - self.fun_f(self.t_space[self.N-1], X[:, :, self.N-1], y, z) * self.dt
        eta2 = tf.reduce_sum(z * dW[:, :, self.N-1], axis=1, keepdims=True)

        y = y + eta1 + eta2

        return y
    
    def loss_fn(self, inputs, training=False):
        X, _ = inputs
        # Forward pass to compute value estimates
        y_pred = self.call(inputs, training)
        #print(y_pred)
        # Exact values at final time
        y = self.fun_g(X[:, :, -1])
        y_diff = y-y_pred
        loss = tf.reduce_mean(tf.square(y_diff))
        
        return loss
    
    @tf.function
    def train_step(self, inp):
        loss, grad = self.grad(inp, training=True)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss
    
    @tf.function
    def grad(self, inputs, training=False):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.trainable_variables)
        return loss, grad
    
    #def fun_f(self, t, x, y, z):
    #    raise NotImplementedError
        
    #def fun_g(self, t, x, y, z):
    #    raise NotImplementedError

    def train(self, epochs, batch_size=64, num_domain_sample=100, learning_rate=1e-2, gap=10):
        start_time = time.time()
        # Set optimizer
        self.optimizer=tf.keras.optimizers.Adam(learning_rate, epsilon=1e-8)
        history = []
        domain_sample = self.domain.sample(num_domain_sample)
        print('  Iter        Loss|   Time')
        for i in range(epochs):
            # Each epoch we draw a batch of 64 random paths
            j = np.random.randint(num_domain_sample)
            X, dW = self.draw_X_and_dW(batch_size, domain_sample[j:j+1].numpy())

            # Compute the loss as well as the gradient
            loss = self.train_step((X, dW))
            curr_time = time.time() - start_time
            hentry = (i, loss.numpy(), curr_time)
            history.append(hentry)
            if i%gap == 0:
                print('{:5d} {:12.4f}   | {:6.1f}'.format(*hentry))
        self.save_weights()
        return history

    def save_weights(self):
        super().save_weights(self.folder + '\model_{}'.format(self.t1))

    def load_weights(self, t):
        super().load_weights(self.folder + '\model_{}'.format(t)).expect_partial()

    def plot(self, low=None, high=None, t=None):
        if t is None:
            t=self.t1
        if low is None:
            low = self.domain.low
        if high is None:
            high = self.domain.high
        plotter = nplt.NNPlotter(funcs=[self.plt_fn], low=low, high=high, num_pts_per_dim=30)
        plotter.plot(file_path='{}/sol.png'.format(self.folder), t=t)




class LSTMForgetBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float64):
        super().__init__(name='LSTMForgetBlock', dtype=dtype)
        self.W_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_i', use_bias=False)
        self.U_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_i')
        self.W_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_o', use_bias=False)
        self.U_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_o')
        self.W_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_c', use_bias=False)
        self.U_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_c')

    def call(self, x, h, c):
        f = tf.keras.activations.sigmoid(self.W_f(x) + self.U_f(h))
        i = tf.keras.activations.sigmoid(self.W_i(x) + self.U_i(h))
        o = tf.keras.activations.sigmoid(self.W_o(x) + self.U_o(h))
        c_ = tf.keras.activations.tanh(self.W_c(x) + self.U_c(h))
        c = f*c + i*c_
        return o*tf.keras.activations.tanh(c), c


class LSTMPeepholeBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float32):
        super().__init__(name='LSTMPeepholeBlock', dtype=dtype)
        self.W_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_i', use_bias=False)
        self.U_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_i')
        self.W_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_o', use_bias=False)
        self.U_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_o')
        self.W_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_c')

    def call(self, x, h, c):
        f = tf.keras.activations.sigmoid(self.W_f(x) + self.U_f(h))
        i = tf.keras.activations.sigmoid(self.W_i(x) + self.U_i(h))
        o = tf.keras.activations.sigmoid(self.W_o(x) + self.U_o(h))
        c = f*c + i * tf.keras.activations.tanh(self.W_c(x))
        return o*c, c

class DGMBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float32):
        super().__init__(name='DGMBlock', dtype=dtype)
        self.W_z = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_z = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_g = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_i', use_bias=False)
        self.U_g = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_i')
        self.W_r = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_o', use_bias=False)
        self.U_r = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_o')
        self.W_h = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_c', use_bias=False)
        self.U_h = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_c')

    def call(self, x, S):
        Z = tf.keras.activations.sigmoid(self.W_z(S) + self.U_z(x))
        G = tf.keras.activations.sigmoid(self.W_g(S) + self.U_g(x))
        R = tf.keras.activations.sigmoid(self.W_r(S) + self.U_r(x))
        H = tf.keras.activations.tanh(self.W_h(S*R) + self.U_h(x))
        return (1.0 - G)*H + Z*S

class FPForget(tf.keras.models.Model):
    """
    Description: 
        LSTM Forget architecture for the FPSolver, inherits from FPSolver
    Args:
        num_nodes: number of nodes in each LSTM layer
        num_layers: number of LSTM layers
        ------ parent args ------
        diff_ops: a tensorflow layer object representing the space differential operator L
        ens_file: path to ensemble evolution file
        sinkhorn_epsilon: regularization constant for Sinkhorn algorithm
        sinkhorn_iters: number of iterations for Sinkhorn algorithm
        #dtype: tf.float32 or tf.float64
        name: name of the FPSolver network
    """
    def __init__(self, num_nodes, num_layers, dtype=tf.float32, name = 'FPForget'):
        super().__init__(dtype=dtype, name=name)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.lstm_layers = [LSTMForgetBlock(num_nodes, dtype=dtype) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.exponential, dtype=dtype)
        #self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
        
        

    def call(self, x):
        #x = tf.concat(args, axis=1)
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_layers):
            h, c = self.lstm_layers[i](x, h, c)
            #h = self.batch_norm(h)
            #c = self.batch_norm(c)
        y = self.final_dense(h)
        return y



class DeepBSDEPt(tf.keras.Model):
    def __init__(self, t0=0.0,\
                 t1=1.0,\
                 dim=100,\
                 time_steps=20,\
                 sigma=np.sqrt(2),\
                 num_hidden_layers=2,\
                 num_neurons=200,\
                 mu = None,
                 fun_f=None,
                 fun_g=None,
                 name='DeepBSDESolver',
                 dir=None, 
                 **kwargs):
        """Set up basic architecture of deep BSDE NN model."""
        
        super().__init__(dtype='float32', name=name, **kwargs)
        
        self.t0 = t0
        self.t1 = t1
        self.N = time_steps
        self.dim = dim
        self.mu = mu
        self.sigma = sigma
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.fun_f = fun_f
        self.fun_g = fun_g
        self.dt = (t1 - t0)/(self.N)
        self.sqrt_dt = np.sqrt(self.dt)
        
        self.t_space = np.linspace(self.t0, self.t1, self.N + 1)[:-1]

        if dir is not None:
            self.folder = '{}/{}'.format(dir, self.name)
        else:
            self.folder = self.name

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        
        # Alternatively, initialize both randomly
        # Initialize the value u(0, x) randomly
        u0 = np.random.uniform(.1, .3, size=(1)).astype(self.dtype)
        self.u0 = tf.Variable(u0)
        
        # Initialize the gradient nabla u(0, x) randomly
        gradu0 = np.random.uniform(-1e-1, 1e-1, size=(1, self.dim)).astype(self.dtype)
        self.gradu0 = tf.Variable(gradu0)
        
        # Create template of dense layer without bias and activation
        self._dense = lambda dim: tf.keras.layers.Dense(
            units=dim,
            activation=None,
            use_bias=False)
        
        # Create template of batch normalization layer
        self._bn = lambda : tf.keras.layers.BatchNormalization(
            momentum=.99,
            epsilon=1e-6,
            beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
            gamma_initializer=tf.random_uniform_initializer(0.1, 0.5))
    
        # Initialize a list of networks approximating the gradient of u(t, x) at t_i
        self.gradui = []
        
        # Loop over number of time steps
        for _ in range(self.N-1):
            self.gradui.append(self.subnet(out_dim=self.dim))

            
    def subnet(self, out_dim):
        # Batch normalization on dim-dimensional input
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.dim))
        model.add(self._bn())
        
        # Hidden layers of type (Dense -> Batch Normalization -> ReLU)
        for _ in range(self.num_hidden_layers):
            model.add(self._dense(self.num_neurons+10))
            model.add(self._bn())
            model.add(tf.keras.layers.ReLU())
            
        # Dense layer followed by batch normalization for output
        model.add(self._dense(out_dim))
        model.add(self._bn())
        return model

    def draw_X_and_dW(self, num_sample, x):
        """ Method to draw num_sample paths of X. """
        
        # Draw increments of Brownian motion
        dW = np.random.normal(loc=0.0, scale=self.sqrt_dt, size=(num_sample, self.dim, self.N)).astype(self.dtype)
        
        # Initialize the array X
        X = np.zeros((num_sample, self.dim, self.N+1), dtype=self.dtype)
        # Initialize and set array of paths
        if x is None:
            X[:, :, 0] = np.zeros((num_sample, self.dim, self.N+1), dtype=self.dtype)
        else:
            X[:, :, 0] = np.ones((num_sample, self.dim)) * x
        
        
        for i in range(self.N):
            # This corresponds to the Euler-Maruyama Scheme
            X[:, :, i+1] = X[:, :, i] + self.mu(X[:, :, i]) * self.dt + self.sigma * dW[:, :, i]
            
        # Return simulated paths as well as increments of Brownian motion
        return X, dW

    
    
    @tf.function
    def simulate_Y(self, inp, training=False):
        """
        Method to perform one forward sweep through the network
        given inputs: inp - (X, dW)
                      training - boolean variable indicating training
        """
        X, dW = inp
        num_sample = X.shape[0]
        
        
        e_num_sample = tf.ones(shape=[num_sample, 1], dtype=self.dtype)

        # Value approximation at t0
        y = e_num_sample * self.u0

        # Gradient approximation at t0
        z = e_num_sample * self.gradu0
        
        for i in range(self.N-1):
            
            t = self.t_space[i]
            
            # Optimal control is attained by gradient
            eta1 = - self.fun_f(t, X[:, :, i], y, z) * self.dt
            eta2 = tf.reduce_sum(z * dW[:, :, i], axis=1, keepdims=True)

            y = y + eta1 + eta2

            # New gradient approximation
            # The division by self.dim acts as a stabilizer
            z = self.gradui[i](X[:, :, i + 1], training) / self.dim

        # Final step
        eta1 = - self.fun_f(self.t_space[self.N-1], X[:, :, self.N-1], y, z) * self.dt
        eta2 = tf.reduce_sum(z * dW[:, :, self.N-1], axis=1, keepdims=True)

        y = y + eta1 + eta2

        return y
    
    def loss_fn(self, inputs, training=False):
        X, _ = inputs
        # Forward pass to compute value estimates
        y_pred = self.simulate_Y(inputs, training)
        #print(y_pred)
        # Exact values at final time
        y = self.fun_g(X[:, :, -1])
        y_diff = y-y_pred
        loss = tf.reduce_mean(tf.square(y_diff))
        
        return loss
    
    @tf.function
    def train_step(self, inp):
        loss, grad = self.grad(inp, training=True)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss
    
    @tf.function
    def grad(self, inputs, training=False):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.trainable_variables)
        return loss, grad
    
    #def fun_f(self, t, x, y, z):
    #    raise NotImplementedError
        
    #def fun_g(self, t, x, y, z):
    #    raise NotImplementedError

    def train(self, x, epochs, batch_size=64, learning_rate=1e-2, gap=10):
        start_time = time.time()
        # Set optimizer
        self.optimizer=tf.keras.optimizers.Adam(learning_rate, epsilon=1e-8)
        history = []
        print('  Iter        Loss|   Time')
        for i in range(epochs):
            # Each epoch we draw a batch of 64 random paths
            X, dW = self.draw_X_and_dW(batch_size, x)

            # Compute the loss as well as the gradient
            loss = self.train_step((X, dW))
            curr_time = time.time() - start_time
            hentry = (i, loss.numpy(), curr_time)
            history.append(hentry)
            if i%gap == 0:
                print('{:5d} {:12.4f}   | {:6.1f}'.format(*hentry))
        self.save_weights()
        return history

    def save_weights(self):
        super().save_weights(self.folder + '\model_{}'.format(self.t1))

    def load_weights(self, t):
        super().load_weights(self.folder + '\model_{}'.format(t)).expect_partial()