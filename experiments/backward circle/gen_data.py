import os, sys
from pathlib import Path 
script_dir = Path(os.path.abspath(''))
module_dir = str(script_dir.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/equations')

import tensorflow as tf 
import numpy as np 
from circle_bw import CircleBackward2D

sigma = np.sqrt(20.0)
two_pi = 2. * np.pi
def p0(x):
    return tf.exp(-0.5*tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)) / two_pi
high = 2. * np.ones(2)
low = -high
tag = '{:.2f}'.format(sigma**2).replace('.', '_')
name = 'CircleBackward2D_sigma2_{}'.format(tag)
eqn = CircleBackward2D(p0=p0, sigma=sigma, low=low, high=high, folder='.', name=name)
x, y, _ = eqn.evolve(X0=500, n_repeats=1000, time_steps=100, animate=True)
print(x)
print(y)
print(tf.reduce_mean(tf.reduce_sum(x[:, -1, :]**2, axis=-1, keepdims=True)))