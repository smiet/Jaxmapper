from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as onp
from scipy import optimize

import math

from tests import run_test
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap, basecase, no_modulo
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section
from methods import step_NM, step_TNM, apply_step, fixed_point_finder, fixed_point_trajectory, find_unique_fixed_points
from methods import theta, test_isotrope, isotrope, theta_comparison
from methods import newton_fractal

from maps import standard_map_modulo as modulo

k=0.5

grid = onp.random.rand(9,2)
grid = grid_starting_points((0,0), (1,1), 3, 3)

rolled_map = lambda xy: standard_map(xy, k)

apply_standard_map = vmap(rolled_map)
test1 = apply_standard_map(grid)
test2 = apply_standard_map(test1)

map2 = Nmap(standard_map, 2)
rolled_map_2 = lambda xy: map2(xy, k=k)
apply_map2 = vmap(rolled_map_2)
test3 = apply_map2(grid)
#print(test3-test2)


Niter = 10

###############################
## from apply_finder_to_grid ##
###############################
grid_points = grid.reshape(3,3, 2)
# initialise fixed point finder to use Niter argument.
map_fixed_point_finder = fixed_point_finder(standard_map, modulo, step_NM, Niter)
# use lambda to roll in the kwargs
rolled_fixed_point_finder = lambda xy: map_fixed_point_finder(xy, k=k)
# vmap the fixed_point_finder function. 
vmap_fixed_point_finder = vmap(vmap(rolled_fixed_point_finder))
# apply vmap_fixed_point_finder to the grid of starting points.
end_points = vmap_fixed_point_finder(grid_points)


#################################
## from fixed_point_trajectory ##
#################################
step_for_map = step_NM(standard_map, modulo)
apply_step_for_map = apply_step(step_for_map, modulo)
# use lambda to "roll-in" the mapping kwargs
rolled_apply_step = lambda xy: apply_step_for_map(xy, k=k)
# jit rolled_delta
applydelta = jit(vmap(rolled_apply_step, in_axes=0))
# initialize results array
iterations = [grid, ]
# calculate mapping of previous mappings niter times
for _ in range(Niter):
    old_point = iterations[-1]
    new_point = applydelta(old_point)
    iterations.append(new_point)

def standard_map(xy, k):
    """
    Definition of the Chirikov Standard Map. Takes in an xy coordinate and a k-value.
    x and p coordinates are normalised and modulo 1.
    0.5 is added to the theta coordinate in order to shift the plot up by 0.5.
    """
    theta_old = xy[0]
    p_old = xy[1]
    # applying standard map on old coordinates to get new coordinates
    p = p_old + (k/(2*np.pi))*np.sin(2*np.pi*theta_old)
    theta = theta_old + p + 0.5
    # return
    return np.array([theta, p])

k=1
point = np.array([0.8, 0.8])
print(standard_map(point, k))