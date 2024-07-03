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
from maps import standard_map_modulo as modulo
from methods import grid_starting_points, linear_starting_points, step_NM

from plotting import plot_poincare_section, plot_point_trajectories_to_fixed_points

k=0.5

starts = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=500)
initial_points = linear_starting_points((0.35,0.75), (0.35, 0.85), npoints=4)

map2 = Nmap(standard_map, 2)

plot_poincare_section(starts, 10000, map=standard_map, k=k)
plot_point_trajectories_to_fixed_points(initial_points, map2, modulo, step_NM, niter=50, k=k)

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title(f'NM to find fixed points of 2-cycle. k={k}.')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()