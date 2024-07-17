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
from methods import grid_starting_points, linear_starting_points, step_NM, step_LGTNMx

from plotting import plot_newtons_fractal, plot_fixed_points
from maps import standard_map_theta_modulo as no_p_mod

map2 = Nmap(standard_map, 2)

##################################################################################################################
################################################# DMIN AGAINST N #################################################
##################################################################################################################
from convergence import dmin_against_N
from groundtruth import sm2_fixed_points
from plotting import expand_fixed_points
expanded_fixed_points = expand_fixed_points(sm2_fixed_points, 0, 1, 0, 1)[0]
from groundtruth import starts50 as starts

k=0.5
N=100

iterations = dmin_against_N(points=starts,
                            map=map2, map_modulo=modulo, 
                            step=step_LGTNMx, step_modulo=modulo, 
                            max_N=N, fixed_points=expanded_fixed_points,
                            k=k)
N_array = np.arange(N+1)
for i in range(iterations.shape[0]):
    plt.plot(N_array, iterations[i, :], 
             color= onp.random.rand(3), lw = 0.1,  
             marker = 'o', ms=1)
##################################################################################################################
##################################################################################################################

plt.xlabel('N')
plt.ylabel('dmin')

plt.ylim(1E-16, 1E1)

plt.xscale('log')
plt.yscale('log')
plt.show()
