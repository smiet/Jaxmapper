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

from maps import basecase, no_modulo

##################################################################################################################
############################################## PLOT NEWTONS FRACTAL ##############################################
##################################################################################################################
from methods import find_unique_fixed_points, step_NM, step_AGTNMx, apply_finder_to_grid
grid = grid_starting_points((-1.2, -1.2), (1.2,1.2), x_points=100, y_points=100)
test = find_unique_fixed_points(basecase, no_modulo, 100)
unique_fixed_points_array = test(grid, step_NM, no_modulo)

colour_array = np.array([[30,144,255], [50,205,50], [220,20,60]])



# initialise grid of starting points for newton's fractal.
starts = grid_starting_points(xy_start=(-1.2,-1.2), xy_end=(1.2,1.2), 
                              x_points=1000, y_points=1000)
# use newton's fractal function to get coordinate array with indices of fixed points as elements.
fixed_point_index_grid = apply_finder_to_grid(map=basecase, map_modulo=no_modulo, 
                                              step=step_NM, step_modulo=no_modulo,
                                              startpoints=starts, x_points=1000, y_points=1000, 
                                              fixedpoints=unique_fixed_points_array, 
                                              Niter=20)
# replace each index with rgb value.
from plotting import assign_colours_to_grid
colour_grid = assign_colours_to_grid(fixed_point_index_grid, colour_array)
# plot.
plt.imshow(colour_grid, origin = 'lower', extent=(-1.2, 1.2, -1.2, 1.2))
##################################################################################################################
##################################################################################################################

###################################################################################################################
################################################ PLOT FIXED POINTS ################################################
###################################################################################################################
colour_list = (colour_array/255).tolist()
plt.scatter(unique_fixed_points_array[:, 0], unique_fixed_points_array[:, 1], facecolors=colour_list, marker='o', 
            edgecolor='black', s=50, linewidth = 2)
###################################################################################################################
###################################################################################################################

plt.xlabel(r'Re($z$)')
plt.ylabel(r'Im($z$)')

plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
# plt.show()
plt.savefig('bc_nf_fp_NM.pdf', bbox_inches='tight', dpi=300)