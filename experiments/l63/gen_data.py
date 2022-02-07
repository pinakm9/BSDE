import os, sys
from pathlib import Path 
script_dir = Path(os.path.abspath(''))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/equations')

import tensorflow as tf 
import numpy as np 
from l63 import Lorenz63
from scipy.special import gamma

sigma = np.sqrt(.0)
two_pi = 2. * np.pi 
s = 0.5
def p0(x):
    return tf.exp(-0.5*tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)/(2.*s**2)) / (two_pi*s)

Z = 2.*gamma(7./6.)
def p1(x):
    return tf.exp(-tf.pow(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True), 3)) / Z


high = 0.2 * np.ones(3)
low = -high
tag = '{:.2f}'.format(sigma**2).replace('.', '_')
name = 'Lorenz63_sigma2_{}'.format(tag)
eqn = Lorenz63(p0=p1, sigma=sigma, t1=.5, low=low, high=high, folder='.', name=name, dtype='float64')
s, values = eqn.evol(X0=250, n_repeats=1, final_time=0.1, time_steps=10000, prune=None, interpolate=True, invert_mu=False)

#print(tf.reduce_mean(tf.reduce_sum(x[:, -1, :]**2, axis=-1, keepdims=True)))
#print(np.where(values > 10.))