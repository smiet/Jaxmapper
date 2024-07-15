from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as onp

from tests import run_test
from maps import original_standard_map, standard_map, Nmap, no_modulo
from maps import standard_map_theta_modulo as no_p_modulo
from maps import standard_map_modulo as modulo
from methods import grid_starting_points, linear_starting_points, step_NM

from plotting import plot_newtons_fractal, plot_fixed_points
from maps import standard_map_theta_modulo as no_p_mod

##################################################################################################################
############################################ MAP DIFFERENCE AGAINST N ############################################
##################################################################################################################
from convergence import map_difference_against_N
starts = onp.random.rand(1000, 2)

k=0.5
N=20

iterations = map_difference_against_N(points=starts,
                            map=standard_map, map_modulo=modulo, 
                            step=step_NM, step_modulo=no_p_mod, 
                            max_N=N,
                            k=k)
N_array = np.arange(N+1)
for i in range(iterations.shape[0]):
    plt.scatter(N_array, iterations[i, :], color='black', s=1)
##################################################################################################################
##################################################################################################################

plt.xscale('log')
plt.yscale('log')
plt.show()