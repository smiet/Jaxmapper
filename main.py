from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as onp

import math

from tests import run_test
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section
from methods import step_NM, apply_step, fixed_point_finder, fixed_point_trajectory
from methods import mapping_vector, theta, test_isotrope, isotrope, theta_comparison
from methods import newton_fractal

from maps import standard_map_modulo as modulo

starts = grid_starting_points(xy_start=(-1,-1), xy_end=(1,1), x_points=50, y_points=50)
starts1 = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=10)
# starts2 = linear_starting_points(xy_start=(0.6,0), xy_end=(0.6,1), npoints=1000)
# starts3 = linear_starting_points(xy_start=(0.4,0), xy_end=(0.4,1), npoints=1000)
# starts = np.append(starts1, starts2, axis=0)
# starts = np.append(starts, starts3, axis=0)

from methods import find_unique_fixed_points
from maps import basecase, no_modulo

grid = grid_starting_points((-1,-1), (1,1), 10, 10)


basecase_fixed_points = find_unique_fixed_points(basecase, no_modulo)
unique_fixed_points = basecase_fixed_points(grid, step_NM)

cmap = colormaps['gist_rainbow']
colours = cmap(np.linspace(0, 1, unique_fixed_points.shape[0]))

test = newton_fractal((-1, -1), (1,1), 300, 300, 
                      basecase, no_modulo, step_NM, niter=50,
                      test_grid=grid_starting_points((-1,1), (1,1), 10,10))


plt.imshow(test, origin = 'lower', extent=(-1, 1, -1, 1))
plt.scatter(unique_fixed_points[:, 0], unique_fixed_points[:, 1], facecolor=colours, marker='o', 
            edgecolor='black', linewidth = 2)

# plt.xlim([0,1])
# plt.ylim([0,1])
plt.title('Newtons Fractal for roots of z^3 - 1')
plt.savefig(fname='base case.pdf', bbox_inches='tight', dpi=300)

#TODO: find a fix for sympy mod issue (or ask chris if sympy can be abandoned)
# F = sym_standard_map(0.5)
# x, y = sym.symbols('x y')
# X=sym.Matrix([x,y])
# sym_jac = sym.simplify(F.jacobian(X))
# x, y = sym.symbols('x y')
# first_num_jac = sym.lambdify([x,y], sym_jac)
# print(sym_jac)
# #print(first_num_jac(0.1,0.1))


#TODO: plot fixed points on the same diagram
#TODO: use colours for the fixed points in a neater way
#TODO: compare different number of iterations
#TODO: check if the points all make it to the fixed points

#TODO: deliverables 
#TODO: TODO analytic expressions from sympy
#TODO: TODO numerical outputs from sympy_jac and jacfwd for 2, 3, 4maps
#TODO: TODO numerical outputs from step
#DONE: split up map.py
#DONE: make sympy part of step work properly and do comparison
#DONE: convert step into a functional transformation
#DONE: make a folder called tests with a few simple tests (sanity checks. eg f(x + dx) = f(x) + Mdx, make dx random)
#DONE: switch to imshow
#DONE: add simple case (z^3 - 1) for baseline