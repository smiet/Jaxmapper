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
from methods import new_starts, linear_starting_points, calculate_poincare_section, step_NM, NM

#starts = starts_nontwist(200)
#starts = new_starts(xy_start=(0,0), xy_end=(1,1), x_points=10, y_points=10)
starts = linear_starting_points(xy_start=(0,0.1), xy_end=(1,0.1), npoints=100)
#starts = np.array([0.1, 0.1])
#start = np.array([0.98436717, 0.5])
#starts = np.array([[0.4, 0.4], [0.6, 0.6]])


points = NM(starts, standard_map, 10000, k=1)

# delta = step_NM(standard_map, method='jax')
# step = delta(starts, k=0.5)

#points = calculate_poincare_section(starts, 10000, mapping=standard_map, k = 1)

#plot_save_points(points, name='fig', colors='random')

fig, ax = plt.subplots(figsize=(12, 6))
#i = round(100*onp.random.rand())
for i in range(points.shape[0]):
    ax.scatter(points[i,0, points.shape[2]], points[i,1, points.shape[2]], color=onp.random.random(3), s=10, marker ='.')
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