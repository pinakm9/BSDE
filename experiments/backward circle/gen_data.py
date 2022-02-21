import os, sys
from pathlib import Path 
script_dir = Path(os.path.abspath(''))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/equations')

import tensorflow as tf 
import numpy as np 
from circle_bw import CircleBackward2D
from scipy.special import gamma

sigma = np.sqrt(2.)
two_pi = 2. * np.pi 
s = 0.5
def p0(x):
    return tf.exp(-0.5*tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)/(2.*s**2)) / (two_pi*s)

Z = np.pi*gamma(4./3.)
def p1(x):
    return tf.exp(-tf.pow(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True), 3)) / Z


high = 2. * np.ones(2)
low = -high
tag = '{:.2f}'.format(sigma**2).replace('.', '_')
name = 'CircleBackward2D_sigma2_{}'.format(tag)
eqn = CircleBackward2D(p0=p0, sigma=sigma, t1=.5, low=low, high=high, folder='.', name=name, dtype='float64')
t = 50.0
s, values = eqn.evol(X0='grid_30', n_repeats=100, final_time=t, time_steps=10000, prune=None, interpolate=True, invert_mu=False)
eqn.plot_slice(t, steady=True)
