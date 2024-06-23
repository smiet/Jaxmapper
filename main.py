from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

from tests import run_test
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section, step_NM, NM

#starts = new_starts(xy_start=(0,0), xy_end=(1,1), x_points=10, y_points=10)
# starts1 = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=1000)
# starts2 = linear_starting_points(xy_start=(0.6,0), xy_end=(0.6,1), npoints=1000)
# starts3 = linear_starting_points(xy_start=(0.4,0), xy_end=(0.4,1), npoints=1000)
# starts = np.append(starts1, starts2, axis=0)
# starts = np.append(starts, starts3, axis=0)
starts = np.array([0.1, 0.1])

map2 = Nmap(standard_map, 4, k=2.18)

points = NM(starts, map2, 10000)

#points = calculate_poincare_section(starts, 10000, mapping=standard_map, k = 2.18)

#plot_save_points(points, name='fig', colors='random')

fig, ax = plt.subplots(figsize=(12, 6))
#i = round(100*onp.random.rand())
final_iter = points.shape[2]
for i in range(points.shape[0]):
    ax.scatter(points[i,0,final_iter], points[i,1,final_iter], color=onp.random.random(3), s=10, marker ='.')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()

#TODO: find a fix for sympy mod issue (or ask chris if sympy can be abandoned)
# F = sym_standard_map(0.5)
# x, y = sym.symbols('x y')
# X=sym.Matrix([x,y])
# sym_jac = sym.simplify(F.jacobian(X))
# x, y = sym.symbols('x y')
# first_num_jac = sym.lambdify([x,y], sym_jac)
# print(sym_jac)
# #print(first_num_jac(0.1,0.1))


#TODO: implement NM. plot n steps on 2d plot
#TODO: deliverables 
#TODO: TODO analytic expressions from sympy
#TODO: TODO numerical outputs from sympy_jac and jacfwd for 2, 3, 4maps
#TODO: TODO numerical outputs from step
#DONE: split up map.py
#DONE: make sympy part of step work properly and do comparison
#DONE: convert step into a functional transformation
#DONE: make a folder called tests with a few simple tests (sanity checks. eg f(x + dx) = f(x) + Mdx, make dx random)

#NOTE: Should I vmap step_NM?