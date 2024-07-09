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
from maps import standard_map, Nmap
from methods import grid_starting_points, linear_starting_points
from methods import step_NM, step_TNM

from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod



k=1

map2 = Nmap(standard_map, 2)

##################################################################################################################
############################################## PLOT NEWTONS FRACTAL ##############################################
##################################################################################################################
from methods import find_unique_fixed_points, step_NM, step_TNM, apply_finder_to_grid
from plotting import expand_fixed_points, assign_colours_to_grid
# initialise grid to find fixed points.
grid = grid_starting_points(xy_start=(0,0), xy_end=(1,1), x_points=100, y_points=100)
# initialise function to find fixed points.
map_fixed_points = find_unique_fixed_points(map=map2, map_modulo=modulo, Niter=70)
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
                                              step=step_NM, step_modulo=no_p_mod,
                                              startpoints=starts, x_points=1000, y_points=1000, 
                                              fixedpoints=expanded_fixed_points, 
                                              Niter=10, 
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
grid = grid_starting_points(xy_start=(0,0), xy_end=(1,1), x_points=100, y_points=100)
map_fixed_points = find_unique_fixed_points(map=map2, map_modulo=modulo, Niter=70)
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


#########################################################################################################
##################################### PLOT FIXED POINT TRAJECTORIES #####################################
#########################################################################################################
from methods import fixed_point_trajectory
starts = linear_starting_points((0.1, 0.1), (0.1, 0.9), npoints=10)
steps = fixed_point_trajectory(xy=starts, 
                               map=map2, map_modulo=modulo,
                               step=step_TNM, step_modulo=modulo,
                               niter=30,
                               k=k)

cmap = colormaps['PiYG']
colors = cmap(np.linspace(0, 1, steps.shape[0]))

for j in range(steps.shape[0]): # for each fixed point
    for i in range(steps.shape[2]): # for each line segment
        plt.plot(steps[j, 0, i:i+2], steps[j, 1, i:i+2],
                color=colors[j],
                ms=5, marker ='o', markerfacecolor=colors[j], markeredgecolor=colors[j])
    plt.scatter(steps[j,0,0], steps[j,1,0], s=100, marker='.', color=colors[j])
    plt.scatter(steps[j,0,-1], steps[j,1,-1], s=100, marker='x', color='cyan')
#########################################################################################################
#########################################################################################################



plt.xlim([0,1])
plt.ylim([0,1])
plt.tight_layout()
plt.show()
#plt.savefig('sm2_nf_fp_fptraj_TNM.pdf', bbox_inches='tight', dpi=300)