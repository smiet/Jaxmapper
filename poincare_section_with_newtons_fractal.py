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

map6 = Nmap(standard_map, 6)

from plotting import plot_poincare_section, plot_newtons_fractal
plot_poincare_section(starts, 10000, standard_map, k=0.5)
plot_newtons_fractal(map6, modulo, step_NM, 1000, 1000, 10, k=0.5)
    

plt.xlim([0,1])
plt.ylim([0,1])
plt.show()