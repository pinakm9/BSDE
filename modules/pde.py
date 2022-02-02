from cv2 import repeat
import numpy as np
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import utility as ut
from matplotlib.animation import FuncAnimation
import pandas as pd

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




class SemilinearPDE:

    def __init__(self, f=None, g=None, mu=0., sigma=np.sqrt(2.), t0=0., t1=1.,
                 low=[-1., -1.], high=[1., 1.], name='semilinear_pde', folder='.'):
        self.mu = mu 
        self.sigma = sigma 
        self.f = f 
        self.g = g 
        self.t0 = t0
        self.t1 =  t1
        self.dim = len(low)
        self.domain = Domain(low=low, high=high)
        self.name = name
        self.folder = folder + '/' + name
        self.dtype = 'float32'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    @ut.timer
    def evolve(self, X0=500, n_repeats=1000, time_steps=20, invert_mu=False, save=True, animate=False, prune=None):
        self.dt = (self.t1 - self.t0) / time_steps
        self.sqrt_dt = np.sqrt(self.dt)
        self.N = time_steps
        if isinstance(X0, int):
            X0 = self.domain.sample(X0)
        n_particles = X0.shape[0]
        self.n_particles = n_particles 
        self.n_repeats = n_repeats
        dW = np.random.normal(loc=0.0, scale=self.sqrt_dt, size=(n_particles * n_repeats, self.N, self.dim)).astype(self.dtype)
        # Initialize the array X
        X = np.zeros((n_particles * n_repeats, self.N+1, self.dim), dtype=self.dtype)
        X[:, 0, :] = np.repeat(X0, [n_repeats]*n_particles, axis=0)
        # fix sign of mu 
        if invert_mu:
            s = -1.
        else:
            s = 1.
        # evolve in time
        for i in range(self.N):
            # This corresponds to the Euler-Maruyama Scheme
            X[:, i+1, :] = X[:, i, :] + s * self.mu(X[:, i, :]) * self.dt + self.sigma * dW[:, i, :]
        
        Y = tf.reduce_mean(self.g(X.reshape(n_particles, n_repeats, self.N+1, self.dim)), axis=1)
        if save:
            t = np.arange(self.t0, self.t1 + self.dt, self.dt)
            t = list(t) * n_particles
            t = np.array(t).reshape((-1, 1))
            spacetime = np.concatenate([t, np.repeat(X0, [self.N+1]*n_particles, axis=0)], axis=1)
            values = Y.numpy().reshape((-1, 1))
            if prune is not None:
                idx, _ = np.where(values>prune)
                spacetime = np.delete(spacetime, idx, axis=0)
                values = np.delete(values, idx, axis=0)

            pd.DataFrame(spacetime)\
                .to_csv(self.folder + '/spacetime_sample_{}_rep_{}.csv'.format(n_particles, n_repeats), index=None, header=None, sep=',')
            pd.DataFrame(values)\
                .to_csv(self.folder + '/values_sample_{}_rep_{}.csv'.format(n_particles, n_repeats), index=None, header=None, sep=',')
            print('generated values for {} spacetime points'.format(spacetime.shape[0]))

        if animate:
            self.animate(X)
        # Return simulated paths as well as increments of Brownian motion
        return X, Y, dW, values

 
    
    def animate(self, X):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        idx = [i*self.n_repeats for i in range(self.n_particles)]
        def animator(j):
            ax.clear()
            ax.scatter(np.take(X[:, j, 0], idx), np.take(X[:, j, 1], idx))
            ax.set_title('time = {:.3f}'.format(j*self.dt))

        animation = FuncAnimation(fig=fig, func=animator, frames = self.N + 1, interval=50, repeat=False)
        animation.save(self.folder + '/evolution_n_particles_{}.gif'.format(self.n_particles), writer='pillow')
        #plt.show()

        

    def kde(self, X = None, t=0):
        if X is None:
            X = np.load(self.folder + '/evolution.npy')
        var1 = X[:, 0, t]
        var2 = X[:, 1, t]
        sns.kdeplot(x=var1, y=var2, fill=True, thresh=0.01)
        plt.show()



