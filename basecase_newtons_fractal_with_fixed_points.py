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
from maps import basecase, no_modulo, Nmap
from methods import grid_starting_points, linear_starting_points, step_NM

from plotting import plot_newtons_fractal, plot_fixed_points

grid = grid_starting_points(xy_start=(-1,-1), xy_end=(1,1), x_points=10, y_points=10)
initial_points = linear_starting_points((0.35,0.75), (0.35, 0.85), npoints=4)

plot_newtons_fractal((-1,-1), (1,1), basecase, no_modulo, step_NM, 1000, 1000, 15)
plot_fixed_points(grid, (-1,-1), (1,1), basecase, no_modulo, step_NM)

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(f'Newtons Fractal of z^3 -1')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.tight_layout
plt.show()