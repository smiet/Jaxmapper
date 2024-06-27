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
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section
from methods import step_NM, apply_step, fixed_point_finder, fixed_point_trajectory
from methods import mapping_vector, theta, test_isotrope, isotrope, theta_comparison
from methods import newton_fractal

from maps import standard_map_modulo as modulo

starts = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=1000)
start2 = np.array([0.1, 0.1])

map6 = Nmap(standard_map, 6)

test = newton_fractal(start2, map6, modulo, step_NM, 1, k=0.5)

points = calculate_poincare_section(starts, 10000, mapping=standard_map, k = 0.5)
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(points.shape[0]):
    ax.scatter(points[i,0,:], points[i,1,:], 
               #color=onp.random.random(3), 
               color='gray',
               #color=colors[i],
               s=0.0001, marker ='.')
    
for i in range(test.shape[0]):
    ax.scatter(test[i,0], test[i, 1], color='orange', s=5, marker='.')

plt.xlim([0,1])
plt.ylim([0,1])
plt.show()