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

class CircleBackward2D(pde.SemilinearPDE):
    def __init__(self, p0, sigma=np.sqrt(0.1), t1=1., low=[-1., -1.], high=[1., 1.], name='CircleBackward2D', folder='.', dtype='floaat32'):
        self.D = sigma**2 / 2.
        self.Z = np.sqrt(np.pi**3 * self.D) * (1. + erf(1./np.sqrt(self.D)))
        #log_Z = np.log(self.Z) 
        #log_2_pi = np.log(2.*np.pi)

        V = lambda x: tf.square((tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) - 1.))
        mu = lambda x: -4. * x * (tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) - 1.)
        self.p_inf = lambda x: tf.exp(-V(x)/self.D) / self.Z
        g = lambda x: p0(x) / self.p_inf(x)
        f = lambda x: 0.

        super().__init__(f=f, g=g, mu=mu, sigma=sigma, t0=0., t1=t1, low=low, high=high, name=name, folder=folder, dtype=dtype)


    def collect_data(self, X0='grid_20', n_repeats=1000, time_steps=20, invert_mu=False, save=True, animate=False, prune=None, interpolate=False):
        if isinstance(X0, str):
            resolution = int(X0.split('_')[-1])
            X0 = self.domain.grid_sample_2d(resolution=resolution, indices=[0, 1])

        spacetime, values = super().evolve(X0=X0, n_repeats=n_repeats, time_steps=time_steps, invert_mu=invert_mu,\
                                         save=False, animate=animate, prune=None, interpolate=False)
        values = (self.p_inf(spacetime) * values).numpy()
        if prune is not None:
            idx = np.argwhere(values>prune)
            spacetime = np.delete(spacetime, idx, axis=0)
            values = np.delete(values, idx, axis=0)
        if interpolate:
            values = super().interpolate_outliers(values)
        if save:
            pd.DataFrame(spacetime)\
                .to_csv(self.folder + '/spacetime.csv', index=None, header=None, sep=',')
            pd.DataFrame(values)\
                .to_csv(self.folder + '/values.csv', index=None, header=None, sep=',')
            print('generated values for {} spacetime points'.format(spacetime.shape[0]))
        return spacetime, values


    def plot_slice(self, slice, indices=[0, 1], steady=True):
        spacetime, values = self.get_time_slice(slice)
        #values = self.interpolate_outliers(values)
        X = spacetime[:, indices[0] + 1]
        Y = spacetime[:, indices[1] + 1]
        if steady:
         Z_inf = self.p_inf(np.hstack([X.reshape((-1, 1)), Y.reshape((-1, 1))])).numpy()
        grid = len(np.unique(X)), len(np.unique(Y))
        print(grid)

        X = X.reshape(grid)
        Y = Y.reshape(grid)
        Z = values.reshape(grid)
        if steady:
            Z_inf = Z_inf.reshape(grid)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        if steady:
            ax.plot_wireframe(X, Y, Z_inf, color='deeppink')
        ax.set_title('time = {:.4f}s'.format(slice * self.dt))
        plt.savefig(self.folder + '/slice_{}.png'.format(slice))