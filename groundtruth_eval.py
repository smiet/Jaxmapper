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
from methods import step_NM, step_AGTNMx, apply_step, fixed_point_finder, mapping_vector, find_unique_fixed_points

from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod

map2 = Nmap(standard_map, 2)

k=0.5
map = map2
map_modulo = modulo
step = step_NM
step_modulo = no_p_mod

# find fixed points from a 1000x1000 grid
grid = grid_starting_points((0,0), (1,1), x_points=1000, y_points=1000)
map_unique_fixed_points = find_unique_fixed_points(map=map, map_modulo=map_modulo, Niter=50)
round1_fixed_points = map_unique_fixed_points(grid, step, step_modulo=step_modulo, k=k)

# use fixed point algorithm on the points above to iterate them to fixed points
map_fixed_point_finder = fixed_point_finder(map=map, map_modulo=map_modulo, 
                                            step=step, step_modulo=step_modulo, 
                                            Niter=50)
rolled_map_fixed_point_finder = lambda xy: map_fixed_point_finder(xy, k=k)
vmapped_map_fixed_point_finder = vmap(rolled_map_fixed_point_finder)
round2_fixed_points = vmapped_map_fixed_point_finder(round1_fixed_points)
print(round2_fixed_points)

# find f(x) - x for the points above
map_mapping_vector = mapping_vector(map=map, modulo=map_modulo)
rolled_mapping_vector = lambda xy: map_mapping_vector(xy, k=k)
vmapped_mapping_vector = vmap(rolled_mapping_vector)
print(vmapped_mapping_vector(round2_fixed_points))


