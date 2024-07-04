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
from maps import original_standard_map, standard_map, Nmap
from maps import standard_map_modulo as modulo
from methods import grid_starting_points, linear_starting_points, step_NM

from plotting import plot_newtons_fractal, plot_fixed_points

grid = grid_starting_points(xy_start=(0,0), xy_end=(1,1), x_points=10, y_points=10)
initial_points = linear_starting_points((0.35,0.75), (0.35, 0.85), npoints=4)

k=1

map2 = Nmap(standard_map, 2)

plot_newtons_fractal((0,0), (1,1), 100, 100, map2, modulo, step_NM, 15, k=k)
plot_fixed_points(grid, (0,0), (1,1), map2, modulo, step_NM, k=k)

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title(f'Newtons Fractal of Standard Map. k={k}')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()