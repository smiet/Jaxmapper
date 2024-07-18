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


from convergence import map_difference_against_N, N_against_epsilon
from groundtruth import starts50 as starts
starts = np.load('starts1000.npy')
k=0.5
sm2 = Nmap(standard_map, 2)
N=100

iterations = map_difference_against_N(points=starts,
                                      map=sm2, map_modulo=modulo,
                                      step=step_LGTNMx, step_modulo=modulo,
                                      max_N=N,
                                      k=k)


e_list = np.logspace(-14, 1, 16)

test = N_against_epsilon(iterations=iterations, epsilon_list=e_list)
plt.boxplot(test, positions = e_list, widths = e_list)

plt.xticks(ticks=e_list, labels=e_list)
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$\epsilon$')
plt.ylabel('N', rotation=0)

plt.show()