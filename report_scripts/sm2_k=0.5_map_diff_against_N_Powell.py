from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 15})

import numpy as onp

from tests import run_test
from maps import original_standard_map, standard_map, Nmap, no_modulo
from maps import standard_map_theta_modulo as no_p_modulo
from maps import standard_map_modulo as modulo
from methods import grid_starting_points, linear_starting_points, step_NM, step_AGTNMx, step_LGTNMx

from plotting import plot_newtons_fractal, plot_fixed_points
from maps import standard_map_theta_modulo as no_p_mod

map2 = Nmap(standard_map, 2)
map5 = Nmap(standard_map, 5)

##################################################################################################################
############################################ MAP DIFFERENCE AGAINST N ############################################
##################################################################################################################
from convergence import map_difference_against_N_scipy
from methods import traj_powell, fixed_point_trajectory_scipy

starts = np.load('starts1000.npy')
starts = starts[300:350,:]

k=0.5
N=500

steps = fixed_point_trajectory_scipy(xy=starts,
                                     map=map2, modulo=modulo,
                                     traj=traj_powell,
                                     k=k)

iterations = map_difference_against_N_scipy(iterations=steps,
                            map=map2, map_modulo=modulo, 
                            k=k)
N_array = np.arange(steps.shape[2])
for i in range(iterations.shape[0]):
    plt.plot(N_array, iterations[i, :], 
             color= onp.random.rand(3), lw = 0.1,  
             marker = 'o', ms=1)
##################################################################################################################
##################################################################################################################

plt.xlabel(r'$N$')
plt.ylabel(r'$|\mathbf{f}(\mathbf{x})-\mathbf{x}|$')

plt.xlim(left=0)
plt.ylim(1E-16, 1E1)

plt.xscale('symlog')
plt.yscale('log')
# plt.show()
plt.savefig('sm2_k=0.5_map_diff_against_N_Powell.png', bbox_inches='tight', dpi=300)


