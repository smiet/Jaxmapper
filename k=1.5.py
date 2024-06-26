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
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section, step_NM, NM

from maps import standard_map_modulo as modulo

starts = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=1000)
initial_point = np.array([[0.4, 0.7]])

map3 = Nmap(standard_map, 3, k=1.5)

points = calculate_poincare_section(starts, 10000, mapping=standard_map, k = 1.5)
steps = NM(initial_point, map3, modulo=modulo, niter=1000)

fig, ax = plt.subplots(figsize=(12, 6))
#i = round(100*onp.random.rand())
colors = plt.cm.binary(np.linspace(0,1,points.shape[0]))
final_iter = points.shape[2]
for i in range(points.shape[0]):
    ax.scatter(points[i,0,:], points[i,1,:], 
               #color=onp.random.random(3), 
               color='gray',
               #color=colors[i],
               s=0.0001, marker ='.')

cmap = colormaps['gist_heat']
colors = cmap(np.linspace(0.5, 1, steps.shape[2]))

for i in range(steps.shape[2]):
    ax.plot(steps[0,0,i:i+2], steps[0,1,i:i+2],
            color='black',
            ms=10, marker ='.', markerfacecolor=colors[i])

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title('NM to find fixed points of 3-cycle. k=1.5.')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()