import numpy as np
import tensorflow as tf 
import os, sys 
from pathlib import Path 

script_dir = Path(os.path.abspath(''))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import pde


def mu(X):
    x, y = X[:, 0], X[:, 1]
    z = 4. * (x**2 + y**2 - 1.)
    mu_1 = (x*z).reshape((-1, 1))
    mu_2 = (y*z).reshape((-1, 1))
    return np.hstack([mu_1, mu_2])

mean = np.zeros(2)
cov = 0.1 * np.identity(2)
initial_distribution = np.random.multivariate_normal(mean, cov, size=500)

eqn = pde.SemilinearPDE(dim=2, mu = mu, t1=1000.0)
mcs = pde.MonteCarloSolver(eqn, initial_distribution) 
mcs.evolve(time_steps=50000, invert_mu=True)
mcs.kde(t=50000)