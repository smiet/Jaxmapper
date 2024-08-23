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
############################################ MAP DIFFERENCE AGAINST N ############################################
##################################################################################################################
from convergence import map_difference_against_N
from groundtruth import starts50 as starts

# starts = np.array([[0.48747675, 0.04207196]])

k=0.5
N=300

iterations = map_difference_against_N(points=starts,
                            map=map2, map_modulo=modulo, 
                            step=step_LGTNMx, step_modulo=modulo, 
                            max_N=N,
                            k=k)
N_array = np.arange(N+1)
for i in range(iterations.shape[0]):
    plt.plot(N_array, iterations[i, :], 
             color= onp.random.rand(3), lw = 0.1,  
             marker = 'o', ms=1)
##################################################################################################################
##################################################################################################################

plt.xlabel('N')
plt.ylabel('||f(x)-x||')

plt.ylim(1E-16, 1E1)

plt.xscale('log')
plt.yscale('log')
plt.show()

# print(iterations[0,:])

# from methods import fixed_point_trajectory

# print(starts[0,:])

# test = fixed_point_trajectory(starts, map2, modulo, step_LGTNMx, modulo, N, k=k)
# print(test[0,:,:].swapaxes(0,1))

