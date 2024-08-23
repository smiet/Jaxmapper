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
from maps import standard_map, Nmap, original_standard_map
from methods import grid_starting_points, linear_starting_points
from methods import step_NM, calculate_poincare_section

from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod

k=1.0

map5 = Nmap(standard_map, 5)


##################################################################################################################
############################################## PLOT NEWTONS FRACTAL ##############################################
##################################################################################################################
from methods import find_unique_fixed_points, step_NM, step_AGTNMx, apply_finder_to_grid
from plotting import expand_fixed_points, assign_colours_to_grid
# load fixed points from groundtruth.py.
from groundtruth import sm5_fixed_points_1
unique_fixed_points_array = sm5_fixed_points_1
# use expand_fixed_points to generate expanded list of fixed points as well as corresponding colour array.
expanded_fixed_points, colour_array = expand_fixed_points(fixed_points=unique_fixed_points_array,
                                                          x_min=0, x_max=1,
                                                          y_min=0, y_max=1)
# initialise grid of starting points for newton's fractal.
starts = grid_starting_points(xy_start=(0,0), xy_end=(1,1), 
                              x_points=1000, y_points=1000)
# use newton's fractal function to get coordinate array with indices of fixed points as elements.
fixed_point_index_grid = apply_finder_to_grid(map=map5, map_modulo=modulo, 
                                              step=step_NM, step_modulo=no_p_mod,
                                              startpoints=starts, x_points=1000, y_points=1000, 
                                              fixedpoints=expanded_fixed_points, 
                                              Niter=20, 
                                              k=k)
# replace each index with rgb value.
colour_grid = assign_colours_to_grid(fixed_point_index_grid, colour_array)
# plot.
plt.imshow(colour_grid, origin = 'lower', extent=(0, 1, 0, 1))
##################################################################################################################
##################################################################################################################

###################################################################################################################
################################################ PLOT FIXED POINTS ################################################
###################################################################################################################
# load fixed points from groundtruth.py.
from groundtruth import sm5_fixed_points_1
unique_fixed_points_array = sm5_fixed_points_1
expanded_fixed_points, colour_array = expand_fixed_points(fixed_points=unique_fixed_points_array,
                                                          x_min=0, x_max=1,
                                                          y_min=0, y_max=1)
colour_list = (colour_array/255).tolist()
plt.scatter(expanded_fixed_points[:, 0], expanded_fixed_points[:, 1], facecolors=colour_list, marker='o', 
            edgecolor='black', s=50, linewidth = 2)
###################################################################################################################
###################################################################################################################

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.xlim([0,1])
plt.ylim([0,1])
# plt.show()
plt.savefig('sm5_k=1.0_nf_fp_NM.pdf', bbox_inches='tight', dpi=300)