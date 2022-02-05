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
    def __init__(self, low, high, dtype='float32'):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.dim = len(low)
        self.dtype = dtype

    def sample(self, num_sample):
        return tf.random.uniform(shape=(num_sample, self.dim), minval=self.low, maxval=self.high)

    def grid_sample_2d(self, resolution, indices=[0, 1]):
        i, j = indices[0], indices[1]
        x = np.linspace(self.low[i], self.high[i], num=resolution, endpoint=True)
        y = np.linspace(self.low[j], self.high[j], num=resolution, endpoint=True)
        x = np.repeat(x, resolution, axis=0).reshape((-1, 1))
        y = np.array(list(y) * resolution).reshape((-1, 1))
        return np.hstack([x, y]).astype(self.dtype)


class SemilinearPDE:

    def __init__(self, f=None, g=None, mu=0., sigma=np.sqrt(2.), t0=0., t1=1.,
                 low=[-1., -1.], high=[1., 1.], name='semilinear_pde', folder='.', dtype='float32'):
        self.mu = mu 
        self.sigma = sigma 
        self.f = f 
        self.g = g 
        self.t0 = t0
        self.t1 =  t1
        self.dim = len(low)
        self.domain = Domain(low=low, high=high, dtype=dtype)
        self.name = name
        self.folder = folder + '/' + name
        self.dtype = 'float32'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    @ut.timer
    def evolve(self, X0=500, n_repeats=1000, time_steps=20, invert_mu=False, save=True, animate=False, prune=None, interpolate=False):
        self.dt = (self.t1 - self.t0) / time_steps
        self.sqrt_dt = np.sqrt(self.dt)
        self.N = time_steps
        if isinstance(X0, int):
            X0 = self.domain.sample(X0).numpy()
        n_particles = X0.shape[0]
        self.n_particles = n_particles 
        self.n_repeats = n_repeats
        dW = np.random.normal(loc=0.0, scale=self.sqrt_dt, size=(self.N, n_particles * n_repeats, self.dim)).astype(self.dtype)
        # Initialize the array X
        X = np.zeros((self.N+1, n_particles * n_repeats, self.dim), dtype=self.dtype)
        X[0, :,  :] = np.repeat(X0, [n_repeats]*n_particles, axis=0)
        # fix sign of mu 
        if invert_mu:
            s = -1.
        else:
            s = 1.
        # evolve in time
        for i in range(self.N):
            # This corresponds to the Euler-Maruyama Scheme
            X[i+1, :, :] = X[i, :, :] + s * self.mu(X[i, :, :]) * self.dt + self.sigma * dW[i, :, :]
        
        values = tf.reduce_mean(self.g(X.reshape(self.N+1, n_particles, n_repeats, self.dim)), axis=2)
        values = tf.squeeze(values).numpy().reshape((-1, 1))

        
        t = np.repeat(np.arange(self.t0, self.t1 + self.dt, self.dt), n_particles, axis=0).reshape((-1, 1))
        X0 = np.array(list(X0) * (self.N+1))
        spacetime = np.hstack([t, X0])

        if prune is not None:
            idx = np.argwhere(values>prune)
            spacetime = np.delete(spacetime, idx, axis=0)
            values = np.delete(values, idx, axis=0)
        
        if interpolate:
            values = self.interpolate_outliers(values)

        if save:
            pd.DataFrame(spacetime)\
                .to_csv(self.folder + '/spacetime.csv', index=None, header=None, sep=',')
            pd.DataFrame(values)\
                .to_csv(self.folder + '/values.csv', index=None, header=None, sep=',')
            print('generated values for {} spacetime points'.format(spacetime.shape[0]))
            

        if animate:
            self.animate(X)
        
        return spacetime, values

 
    
    def animate(self, X):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        idx = [i*self.n_repeats for i in range(self.n_particles)]
        def animator(j):
            ax.clear()
            ax.scatter(np.take(X[j, :, 0], idx), np.take(X[j, :, 1], idx))
            ax.set_title('time = {:.3f}'.format(j*self.dt))

        animation = FuncAnimation(fig=fig, func=animator, frames = self.N + 1, interval=50, repeat=False)
        animation.save(self.folder + '/evolution.mp4', writer='ffmpeg')
        #plt.show()

        
    def read_data(self):
        spacetime = np.genfromtxt(self.folder + '/spacetime.csv', delimiter=',', dtype=self.dtype)
        values = np.genfromtxt(self.folder + '/values.csv', delimiter=',', dtype=self.dtype)
        return spacetime, values

    def get_time_slice(self, slice):
        spacetime, values = self.read_data()
        a = self.n_particles * slice
        b = a + self.n_particles
        return spacetime[a:b, :], values[a:b]

    def plot_slice(self, slice, indices=[0, 1]):
        spacetime, values = self.get_time_slice(slice)
        #values = self.interpolate_outliers(values)
        X = spacetime[:, indices[0] + 1]
        Y = spacetime[:, indices[1] + 1]

        grid = len(np.unique(X)), len(np.unique(Y))
        print(grid)

        X = X.reshape(grid)
        Y = Y.reshape(grid)
        Z = values.reshape(grid)
    
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title('time = {:.4f}s'.format(slice * self.dt))
        plt.savefig(self.folder + '/slice_{}.png'.format(slice))

    def interpolate_outliers(self, values):
        y = values.reshape((-1,))
        idx = np.argwhere(np.abs(y - np.mean(y)) > 2. * np.std(y))
        print(idx)
        for i in idx:
            if i > 0:
                y[i] = (y[i-1] + y[i+1]) / 2.
            else:
                y[i] = y[1]
        return y.reshape((-1, 1))
            


