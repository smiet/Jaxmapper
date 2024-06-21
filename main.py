from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

from maps import standard_map, sym_standard_map, sym_jac_func, Nmap
from methods import step_NM



#starts = starts_nontwist(200)
#starts = new_starts(xy_start=(0,0), xy_end=(1,1), x_points=10, y_points=10)
starts = np.array([0.1, 0.1])
#print(test_jac(starts))

F = sym_standard_map(k=0.5)
test_jac = sym_jac_func(F)
rolled_map = lambda xy : standard_map(xy, k=0.5)
jax_test = jacfwd(rolled_map)(starts)
jax_delta = step_NM(standard_map, method='jax')
sym_delta = step_NM(sym_standard_map, method='sympy')
print(jax_delta(starts,k=0.5))
print(sym_delta(starts, k=0.5))

#delta = step_NM(standard_map, starts)

#points = calculate_poincare_section(starts, 10000, mapping=standard_nontwist, b=0.2, a=0.53)
# points = calculate_poincare_section(starts, 100, mapping=standard_map, k = 0.5)

#plot_save_points(points, name='fig', colors='random')

# fig, ax = plt.subplots(figsize=(12, 6))
# for i in range(points.shape[0]):
#     ax.scatter(points[i,0, :], points[i,1,:], color=onp.random.random(3), s=10, marker ='.')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.tight_layout
# plt.show()


#TODO: make a folder called tests with a few simple tests (sanity checks. eg f(x + dx) = f(x) + Mdx, make dx random)
#TODO: implement NM. plot n steps on 2d plot
#TODO: deliverables 
#TODO: TODO analytic expressions from sympy
#TODO: TODO numerical outputs from sympy_jac and jacfwd for 2, 3, 4maps
#TODO: TODO numerical outputs from step
#DONE: split up map.py
#DONE: make sympy part of step work properly and do comparison
#DONE: convert step into a functional transformation