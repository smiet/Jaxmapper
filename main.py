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
from methods import mapping_vector, theta, test_isotrope, isotrope, theta_comparison
from methods import newton_fractal

from maps import standard_map_modulo as modulo

starts = grid_starting_points(xy_start=(0,0), xy_end=(1,1), x_points=10, y_points=10)
line = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=500)
# starts2 = linear_starting_points(xy_start=(0.6,0), xy_end=(0.6,1), npoints=1000)
# starts3 = linear_starting_points(xy_start=(0.4,0), xy_end=(0.4,1), npoints=1000)
# starts = np.append(starts1, starts2, axis=0)
# starts = np.append(starts, starts3, axis=0)

#map2 = Nmap(standard_map, 2)

#point = np.array([-0.1, 0.4])

#test = step_NM(map2)
#test = apply_step(step_NM(map2), modulo)

#print(test(point, k=0.5))

from plotting import plot_newtons_fractal, plot_poincare_section
plot_newtons_fractal((0,0), (1,1), standard_map, modulo, step_NM, 100, 100, 15, k=0.5)
#plot_poincare_section(line, 10000, standard_map, k=0.5)
plt.show()


# test = mapping_vector(basecase)
# length = lambda xy: np.linalg.norm(test(xy))
# jac = jacfwd(length)
# test2 = optimize.minimize(length, np.array([-0.5, -0.1]), method='L-BFGS-B', jac=jac)
# print(test2)


#TODO: find a fix for sympy mod issue (or ask chris if sympy can be abandoned)
# F = sym_standard_map(0.5)
# x, y = sym.symbols('x y')
# X=sym.Matrix([x,y])
# sym_jac = sym.simplify(F.jacobian(X))
# x, y = sym.symbols('x y')
# first_num_jac = sym.lambdify([x,y], sym_jac)
# print(sym_jac)
# #print(first_num_jac(0.1,0.1))


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
#DONE: plot fixed points on the same diagram