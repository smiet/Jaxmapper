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
from methods import step_NM

from maps import standard_map_modulo as modulo

starts = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=1000)
grid = grid_starting_points(xy_start=(0,0), xy_end=(1,1), x_points=10, y_points=10)

initial_points = np.array([[0.2, 0.5]])

from plotting import plot_newtons_fractal, plot_fixed_points, plot_point_trajectories_to_fixed_points
plot_newtons_fractal((0,0), (1,1), standard_map, modulo, step_NM, 1000, 1000, 15, k=0.5)
plot_fixed_points(grid, (0,0), (1,1), standard_map, modulo, step_NM, k=0.5)
    
plot_point_trajectories_to_fixed_points(initial_points, standard_map, modulo, step_NM, 30, k=0.5)

plt.xlim([0,1])
plt.ylim([0,1])
plt.show()