import os, sys
from pathlib import Path 
script_dir = Path(os.path.abspath(''))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import pde
import tensorflow as tf 
import numpy as np 
from scipy.special import erf


class CircleBackward2D(pde.SemilinearPDE):
    def __init__(self, p0, sigma=np.sqrt(0.1), t1=1., low=[-1., -1.], high=[1., 1.], name='CircleBackward2D', folder='.'):
        self.D = sigma**2 / 2.
        self.Z = np.sqrt(np.pi**3 * self.D) * (1. + erf(1./np.sqrt(self.D)))
        #log_Z = np.log(self.Z) 
        #log_2_pi = np.log(2.*np.pi)

        V = lambda x: tf.square((tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) - 1.))
        mu = lambda x: -4. * x * (tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) - 1.)
        self.p_inf = lambda x: tf.exp(-V(x)/self.D) / self.Z
        g = lambda x: p0(x) / self.p_inf(x)
        f = lambda x: 0.

        super().__init__(f=f, g=g, mu=mu, sigma=sigma, t0=0., t1=t1, low=low, high=high, name=name, folder=folder)