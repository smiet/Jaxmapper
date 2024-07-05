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
from methods import grid_starting_points, linear_starting_points, step_NM, apply_step_no_modulo

from plotting import plot_newtons_fractal, plot_point_trajectories_to_fixed_points

k=1


##########################
## PLOT NEWTONS FRACTAL ##
##########################
from methods import find_unique_fixed_points, step_NM, step_TNM, apply_finder_to_grid
from plotting import expand_fixed_points, assign_colours_to_grid
# initialise grid to find fixed points.
grid = grid_starting_points(xy_start=(0,0), xy_end=(1,1), x_points=100, y_points=100)
# initialise function to find fixed points.
map_fixed_points = find_unique_fixed_points(map=standard_map, modulo=modulo)
# use map_fixed_points and find fixed points.
unique_fixed_points_array = map_fixed_points(grid, step=step_NM, k=k)
# use expand_fixed_points to generate expanded list of fixed points as well as corresponding colour array.
expanded_fixed_points, colour_array = expand_fixed_points(fixed_points=unique_fixed_points_array,
                                                          x_min=0, x_max=1,
                                                          y_min=0, y_max=1)
# initialise grid of starting points for newton's fractal.
starts = grid_starting_points(xy_start=(0,0), xy_end=(1,1), 
                              x_points=1000, y_points=1000)
# use newton's fractal function to get coordinate array with indices of fixed points as elements.
fixed_point_index_grid = apply_finder_to_grid(map=standard_map, modulo=modulo, 
                                              step=step_NM, 
                                              startpoints=starts, x_points=1000, y_points=1000, 
                                              fixedpoints=expanded_fixed_points, 
                                              Niter=15, 
                                              k=k)
# replace each index with rgb value.
colour_grid = assign_colours_to_grid(fixed_point_index_grid, colour_array)
# plot.
plt.imshow(colour_grid, origin = 'lower', extent=(0, 1, 0, 1))
##########################
##########################


###################################
## PLOT FIXED POINT TRAJECTORIES ##
###################################
niter = 1
initial_points_1 = linear_starting_points((0.6,0.2), (0.9, 0.2), npoints=10)
initial_points_2 = linear_starting_points((0.6,0.1), (0.9, 0.1), npoints=10)
initial_points = np.append(initial_points_1, initial_points_2, axis=0)

step_for_map = step_NM(standard_map, modulo)
apply_step_for_map = apply_step_no_modulo(step_for_map)
# use lambda to "roll-in" the mapping kwargs
rolled_apply_step = lambda xy: apply_step_for_map(xy, k=k)
# jit rolled_delta
applydelta = jit(vmap(rolled_apply_step, in_axes=0))
# initialize results array
iterations = [initial_points, ]
# calculate mapping of previous mappings niter times
for _ in range(niter):
    old_point = iterations[-1]
    new_point = applydelta(old_point)
    iterations.append(new_point)

steps = np.stack(iterations, axis=-1)

cmap = colormaps['PiYG']
colors = cmap(np.linspace(0, 1, steps.shape[0]))

for j in range(steps.shape[0]): # for each fixed point
    for i in range(steps.shape[2]): # for each line segment
        plt.plot(steps[j, 0, i:i+2], steps[j, 1, i:i+2],
                color='blue',
                ms=10, marker ='.', markerfacecolor=colors[j], markeredgecolor='blue')
###################################
###################################


plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title(f'NM to find fixed points of 2-cycle. k={k}.')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
plt.tight_layout
plt.show()

from methods import mapping_vector_modulo, mapping_vector_full

# xy = np.array([0.8, 0.08])

# rolled_map = lambda xy: standard_map(xy, k=k)
# M = jacfwd(rolled_map)(xy)
# A = M - np.eye(2)
# diff1 = mapping_vector_modulo(rolled_map, modulo)
# b1 = -diff1(xy)
# delta1 = np.linalg.solve(A,b1)

# diff2 = mapping_vector_full(rolled_map)
# b2 = -diff2(xy)
# delta2 = np.linalg.solve(A,b2)
# print(b1, b2)
# print(delta1, delta2)
