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
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap, no_modulo
from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod
from maps import original_standard_map as sm
from methods import grid_starting_points, linear_starting_points, step_NM, step_LGTNMx,step_LGTNMo

k=0.5

map2 = Nmap(sm, 2)

###########################
## PLOT POINCARE SECTION ##
###########################
from methods import calculate_poincare_section
line = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=100)
poincare = calculate_poincare_section(starts=line, niter=1000, map=sm, modulo=modulo, k=k)
for i in range(poincare.shape[0]):
    plt.scatter(poincare[i,0,:], poincare[i,1,:], 
                color='black', s=0.001, marker ='.')
###########################
###########################

##################################################################################################################
############################################## PLOT NEWTONS FRACTAL ##############################################
##################################################################################################################
from methods import find_unique_fixed_points, step_NM, step_LGTNMx, apply_finder_to_grid
from plotting import expand_fixed_points, assign_colours_to_grid
# initialise grid to find fixed points.
grid = grid_starting_points(xy_start=(0,0), xy_end=(1,1), x_points=100, y_points=100)
# initialise function to find fixed points.
map_fixed_points = find_unique_fixed_points(map=map2, map_modulo=modulo, Niter=50)
# use map_fixed_points and find fixed points.
unique_fixed_points_array = map_fixed_points(grid, step=step_NM, step_modulo=no_p_mod, k=k)
# use expand_fixed_points to generate expanded list of fixed points as well as corresponding colour array.
expanded_fixed_points, colour_array = expand_fixed_points(fixed_points=unique_fixed_points_array,
                                                          x_min=0, x_max=1,
                                                          y_min=0, y_max=1)
# initialise grid of starting points for newton's fractal.
starts = grid_starting_points(xy_start=(0,0), xy_end=(1,1), 
                              x_points=1000, y_points=1000)
# use newton's fractal function to get coordinate array with indices of fixed points as elements.
fixed_point_index_grid = apply_finder_to_grid(map=map2, map_modulo=modulo, 
                                              step=step_LGTNMx, step_modulo=modulo,
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
from methods import find_unique_fixed_points
grid = grid_starting_points(xy_start=(0,0), xy_end=(1,1), x_points=100, y_points=100)
map_fixed_points = find_unique_fixed_points(map=map2, map_modulo=modulo, Niter=50)
# use lambda to roll in the kwargs
unique_fixed_points = map_fixed_points(grid, step=step_NM, step_modulo=no_p_mod, k=k)
expanded_fixed_points, colour_array = expand_fixed_points(fixed_points=unique_fixed_points,
                                                          x_min=0, x_max=1,
                                                          y_min=0, y_max=1)
colour_list = (colour_array/255).tolist()
plt.scatter(expanded_fixed_points[:, 0], expanded_fixed_points[:, 1], facecolors=colour_list, marker='o', 
            edgecolor='black', s=50, linewidth = 2)
###################################################################################################################
###################################################################################################################

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title(f'OOTNMx to find fixed points of 2-Map. k={k}.')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()
# plt.savefig('sm2_ps_nf_fp_OOTNMx.png', bbox_inches='tight', dpi=300)