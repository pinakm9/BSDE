import os, sys
from pathlib import Path 
script_dir = Path(os.path.abspath(''))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import pde
import tensorflow as tf 
import numpy as np 
from scipy.special import erf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
import utility as ut

class Lorenz63(pde.SemilinearPDE):
    def __init__(self, p0, sigma=np.sqrt(0.1), t1=1., low=[-1., -1.], high=[1., 1.], name='CircleBackward2D', folder='.', dtype='floaat32'):
        _sigma = 10.
        beta = 8./3.
        rho = 28.

        def mu(X):
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            a = (_sigma * (x - y)).reshape(-1, 1)
            b = (y - x * (rho - z)).reshape(-1, 1)
            c = (beta * z -x * y).reshape(-1, 1)
            
            return np.hstack([a, b, c]) 
        
        

        g = lambda x: p0(x) 
        f = lambda x: 0.

        super().__init__(f=f, g=g, mu=mu, sigma=sigma, t0=0., t1=t1, low=low, high=high, name=name, folder=folder, dtype=dtype)


    def collect_data(self, X0='grid_20', n_repeats=1000, time_steps=20, invert_mu=False, save=True, animate=False, prune=None, interpolate=False):
        spacetime, values = super().evolve(X0=X0, n_repeats=n_repeats, time_steps=time_steps, invert_mu=invert_mu,\
                                         save=False, animate=animate, prune=prune, interpolate=interpolate)
        

    @ut.timer
    def evol(self, X0=500, n_repeats=1000, final_time=1.0, time_steps=20, invert_mu=False, save=True, prune=None, interpolate=False):
        self.dt = (final_time - self.t0) / time_steps
        self.sqrt_dt = np.sqrt(self.dt)
        self.N = time_steps
        if isinstance(X0, int):
            X0 = self.domain.sample(X0).numpy()
        elif isinstance(X0, str):
            resolution = int(X0.split('_')[-1])
            X0 = self.domain.grid_sample_2d(resolution=resolution, indices=[0, 1])
        n_particles = X0.shape[0]
        self.n_particles = n_particles 
        self.n_repeats = n_repeats
        # Initialize the array X
        X = np.repeat(X0, [n_repeats]*n_particles, axis=0)
        t = np.ones((X0.shape[0], 1)) * final_time
        spacetime = np.hstack([t, X0])

        # fix sign of mu 
        if invert_mu:
            s = -1.
        else:
            s = 1.
        # evolve in time
        for i in range(self.N):
            # This corresponds to the Euler-Maruyama Scheme
            dW = np.random.normal(loc=0.0, scale=self.sqrt_dt, size=(n_particles * n_repeats, self.dim)).astype(self.dtype)
            X = X + s * self.mu(X) * self.dt + self.sigma * dW
    
        values = tf.reduce_mean(self.g(tf.reshape(X, (n_particles, n_repeats, self.dim))), axis=1)
        values = tf.squeeze(values).numpy().reshape((-1, 1))
        print(values)
        
        if prune is not None:
            idx = np.argwhere(values>prune)
            spacetime = np.delete(spacetime, idx, axis=0)
            values = np.delete(values, idx, axis=0)
        
        if interpolate:
            values = self.interpolate_outliers(values)

        if save:
            pd.DataFrame(spacetime)\
                .to_csv(self.folder + '/spacetime{}.csv'.format(self.time_tag(final_time)), index=None, header=None, sep=',')
            pd.DataFrame(values)\
                .to_csv(self.folder + '/values{}.csv'.format(self.time_tag(final_time)), index=None, header=None, sep=',')
            print('generated values for {} spacetime points'.format(spacetime.shape[0]))
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2])
            ax.set_title('time = {:.2f}'.format(final_time))
            plt.savefig(self.folder + '/final_time{}.png'.format(self.time_tag(final_time)))
        return spacetime, values