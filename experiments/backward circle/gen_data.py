import os, sys
from pathlib import Path 
script_dir = Path(os.path.abspath(''))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/equations')

import tensorflow as tf 
import numpy as np 
from circle_bw import CircleBackward2D

sigma = np.sqrt(20.)
two_pi = 2. * np.pi 
s = 0.5
def p0(x):
    return tf.exp(-0.5*tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)/(2.*s**2)) / (two_pi*s)
high = 3.0 * np.ones(2)
low = -high
tag = '{:.2f}'.format(sigma**2).replace('.', '_')
name = 'CircleBackward2D_sigma2_{}'.format(tag)
eqn = CircleBackward2D(p0=p0, sigma=sigma, t1=10.0, low=low, high=high, folder='.', name=name, dtype='float64')
s, values = eqn.collect_data(X0='grid_10', n_repeats=10, time_steps=100000, prune=None, interpolate=False)

#print(tf.reduce_mean(tf.reduce_sum(x[:, -1, :]**2, axis=-1, keepdims=True)))
#print(np.where(values > 10.))
eqn.plot_slice(0)
eqn.plot_slice(1)
eqn.plot_slice(10)
eqn.plot_slice(20)
eqn.plot_slice(50)
eqn.plot_slice(100)
eqn.plot_slice(200)
eqn.plot_slice(500)
eqn.plot_slice(5000)
eqn.plot_slice(8000)
eqn.plot_slice(100000)